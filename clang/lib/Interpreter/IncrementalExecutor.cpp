//===--- IncrementalExecutor.cpp - Incremental Execution --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class which performs incremental code execution.
//
//===----------------------------------------------------------------------===//

#include "IncrementalExecutor.h"

#include "clang/Interpreter/PartialTranslationUnit.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"

namespace clang {

IncrementalExecutor::IncrementalExecutor(llvm::orc::ThreadSafeContext &TSC,
                                         llvm::Error &Err,
                                         const llvm::Triple &Triple)
    : TSCtx(TSC) {
  using namespace llvm::orc;
  llvm::ErrorAsOutParameter EAO(&Err);

  auto JTMB = JITTargetMachineBuilder(Triple);
  if (auto JitOrErr = LLJITBuilder().setJITTargetMachineBuilder(JTMB).create())
    Jit = std::move(*JitOrErr);
  else {
    Err = JitOrErr.takeError();
    return;
  }

  const char Pref = Jit->getDataLayout().getGlobalPrefix();
  // Discover symbols from the process as a fallback.
  if (auto PSGOrErr = DynamicLibrarySearchGenerator::GetForCurrentProcess(Pref))
    Jit->getMainJITDylib().addGenerator(std::move(*PSGOrErr));
  else {
    Err = PSGOrErr.takeError();
    return;
  }
}

IncrementalExecutor::~IncrementalExecutor() {}

llvm::Error IncrementalExecutor::addModule(PartialTranslationUnit &PTU) {
  llvm::orc::ResourceTrackerSP RT =
      Jit->getMainJITDylib().createResourceTracker();
  ResourceTrackers[&PTU] = RT;

  return Jit->addIRModule(RT, {std::move(PTU.TheModule), TSCtx});
}

llvm::Error IncrementalExecutor::removeModule(PartialTranslationUnit &PTU) {

  llvm::orc::ResourceTrackerSP RT = std::move(ResourceTrackers[&PTU]);
  if (!RT)
    return llvm::Error::success();

  ResourceTrackers.erase(&PTU);
  if (llvm::Error Err = RT->remove())
    return Err;
  return llvm::Error::success();
}

llvm::Error IncrementalExecutor::runCtors() const {
  return Jit->initialize(Jit->getMainJITDylib());
}

llvm::Expected<llvm::JITTargetAddress>
IncrementalExecutor::getSymbolAddress(llvm::StringRef Name,
                                      SymbolNameKind NameKind) const {
  auto Sym = (NameKind == LinkerName) ? Jit->lookupLinkerMangled(Name)
                                      : Jit->lookup(Name);

  if (!Sym)
    return Sym.takeError();
  return Sym->getValue();
}

} // end namespace clang
