//===----------------- Wasm.cpp - Wasm Interpreter --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements interpreter support for code execution in WebAssembly.
//
//===----------------------------------------------------------------------===//

#include "Wasm.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>

#include <clang/Interpreter/Interpreter.h>

#include <string>

namespace lld {
enum Flavor {
  Invalid,
  Gnu,     // -flavor gnu
  MinGW,   // -flavor gnu MinGW
  WinLink, // -flavor link
  Darwin,  // -flavor darwin
  Wasm,    // -flavor wasm
};

using Driver = bool (*)(llvm::ArrayRef<const char *>, llvm::raw_ostream &,
                        llvm::raw_ostream &, bool, bool);

struct DriverDef {
  Flavor f;
  Driver d;
};

struct Result {
  int retCode;
  bool canRunAgain;
};

Result lldMain(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
               llvm::raw_ostream &stderrOS, llvm::ArrayRef<DriverDef> drivers);

namespace wasm {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
} // namespace wasm
} // namespace lld

#include <dlfcn.h>

namespace clang {

WasmIncrementalExecutor::WasmIncrementalExecutor(llvm::Error &Err) {
  llvm::ErrorAsOutParameter EAO(&Err);

  if (Err)
    return;

  if (auto EC =
          llvm::sys::fs::createUniqueDirectory("clang-wasm-exec-", TempDir))
    Err = llvm::make_error<llvm::StringError>(
        "Failed to create temporary directory for Wasm executor: " +
            EC.message(),
        llvm::inconvertibleErrorCode());
}

WasmIncrementalExecutor::~WasmIncrementalExecutor() = default;

llvm::Error WasmIncrementalExecutor::addModule(PartialTranslationUnit &PTU) {
  std::string ErrorString;

  const llvm::Target *Target = llvm::TargetRegistry::lookupTarget(
      PTU.TheModule->getTargetTriple(), ErrorString);
  if (!Target) {
    return llvm::make_error<llvm::StringError>("Failed to create Wasm Target: ",
                                               llvm::inconvertibleErrorCode());
  }

  llvm::TargetOptions TO = llvm::TargetOptions();
  llvm::TargetMachine *TargetMachine = Target->createTargetMachine(
      PTU.TheModule->getTargetTriple(), "", "", TO, llvm::Reloc::Model::PIC_);
  PTU.TheModule->setDataLayout(TargetMachine->createDataLayout());

  llvm::SmallString<256> ObjectFileName(TempDir);
  llvm::sys::path::append(ObjectFileName, PTU.TheModule->getName() + ".o");

  llvm::SmallString<256> BinaryFileName(TempDir);
  llvm::sys::path::append(BinaryFileName, PTU.TheModule->getName() + ".wasm");

  std::error_code EC;
  llvm::raw_fd_ostream ObjectFileOutput(ObjectFileName, EC);

  if (EC)
    return llvm::errorCodeToError(EC);

  llvm::legacy::PassManager PM;
  if (TargetMachine->addPassesToEmitFile(PM, ObjectFileOutput, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    return llvm::make_error<llvm::StringError>(
        "Wasm backend cannot produce object.", llvm::inconvertibleErrorCode());
  }

  if (!PM.run(*PTU.TheModule)) {

    return llvm::make_error<llvm::StringError>("Failed to emit Wasm object.",
                                               llvm::inconvertibleErrorCode());
  }

  ObjectFileOutput.close();

  std::vector<const char *> LinkerArgs = {"wasm-ld",
                                          "-shared",
                                          "--import-memory",
                                          "--experimental-pic",
                                          "--stack-first",
                                          "--allow-undefined",
                                          ObjectFileName.c_str(),
                                          "-o",
                                          BinaryFileName.c_str()};

  const lld::DriverDef WasmDriver = {lld::Flavor::Wasm, &lld::wasm::link};
  std::vector<lld::DriverDef> WasmDriverArgs;
  WasmDriverArgs.push_back(WasmDriver);
  lld::Result Result =
      lld::lldMain(LinkerArgs, llvm::outs(), llvm::errs(), WasmDriverArgs);

  if (Result.retCode)
    return llvm::make_error<llvm::StringError>(
        "Failed to link incremental module", llvm::inconvertibleErrorCode());

  void *LoadedLibModule =
      dlopen(BinaryFileName.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (LoadedLibModule == nullptr) {
    llvm::errs() << dlerror() << '\n';
    return llvm::make_error<llvm::StringError>(
        "Failed to load incremental module", llvm::inconvertibleErrorCode());
  }

  return llvm::Error::success();
}

llvm::Error WasmIncrementalExecutor::removeModule(PartialTranslationUnit &PTU) {
  return llvm::make_error<llvm::StringError>("Not implemented yet",
                                             llvm::inconvertibleErrorCode());
}

llvm::Error WasmIncrementalExecutor::runCtors() const {
  // This seems to be automatically done when using dlopen()
  return llvm::Error::success();
}

llvm::Error WasmIncrementalExecutor::cleanUp() {
  // Can't call cleanUp through IncrementalExecutor as it
  // tries to deinitialize JIT which hasn't been initialized
  return llvm::Error::success();
}

llvm::Expected<llvm::orc::ExecutorAddr>
WasmIncrementalExecutor::getSymbolAddress(llvm::StringRef Name,
                                          SymbolNameKind NameKind) const {
  void *Sym = dlsym(RTLD_DEFAULT, Name.str().c_str());
  if (!Sym) {
    return llvm::make_error<llvm::StringError>("dlsym failed for symbol: " +
                                                   Name.str(),
                                               llvm::inconvertibleErrorCode());
  }

  return llvm::orc::ExecutorAddr::fromPtr(Sym);
}

llvm::Error WasmIncrementalExecutor::LoadDynamicLibrary(const char *name) {
  void *handle = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    llvm::errs() << dlerror() << '\n';
    return llvm::make_error<llvm::StringError>("Failed to load dynamic library",
                                               llvm::inconvertibleErrorCode());
  }
  return llvm::Error::success();
}
} // namespace clang
