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
#include "IncrementalExecutor.h"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>

#include <clang/Interpreter/Interpreter.h>

#include <string>

namespace lld {
namespace wasm {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
} // namespace wasm
} // namespace lld

#include <dlfcn.h>

namespace clang {

WasmIncrementalExecutor::WasmIncrementalExecutor(
    llvm::orc::ThreadSafeContext &TSC)
    : IncrementalExecutor(TSC) {}

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
  std::string OutputFileName = PTU.TheModule->getName().str() + ".wasm";

  std::error_code Error;
  llvm::raw_fd_ostream OutputFile(llvm::StringRef(OutputFileName), Error);

  llvm::legacy::PassManager PM;
  if (TargetMachine->addPassesToEmitFile(PM, OutputFile, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    return llvm::make_error<llvm::StringError>(
        "Wasm backend cannot produce object.", llvm::inconvertibleErrorCode());
  }

  if (!PM.run(*PTU.TheModule)) {

    return llvm::make_error<llvm::StringError>("Failed to emit Wasm object.",
                                               llvm::inconvertibleErrorCode());
  }

  OutputFile.close();

  std::vector<const char *> LinkerArgs = {"wasm-ld",
                                          "-pie",
                                          "--import-memory",
                                          "--no-entry",
                                          "--export-all",
                                          "--experimental-pic",
                                          "--no-export-dynamic",
                                          "--stack-first",
                                          OutputFileName.c_str(),
                                          "-o",
                                          OutputFileName.c_str()};
  int Result =
      lld::wasm::link(LinkerArgs, llvm::outs(), llvm::errs(), false, false);
  if (!Result)
    return llvm::make_error<llvm::StringError>(
        "Failed to link incremental module", llvm::inconvertibleErrorCode());

  void *LoadedLibModule =
      dlopen(OutputFileName.c_str(), RTLD_NOW | RTLD_GLOBAL);
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

WasmIncrementalExecutor::~WasmIncrementalExecutor() = default;

} // namespace clang
