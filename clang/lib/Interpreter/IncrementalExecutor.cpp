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

llvm::Error IncrementalExecutor::addModule(std::unique_ptr<llvm::Module> M) {
  return Jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(M), TSCtx));
}

llvm::Error IncrementalExecutor::runCtors() const {
  return Jit->initialize(Jit->getMainJITDylib());
}

llvm::Expected<llvm::JITTargetAddress>
IncrementalExecutor::getSymbolAddress(llvm::StringRef UnmangledName) const {
  auto Sym = Jit->lookup(UnmangledName);
  if (!Sym)
    return Sym.takeError();
  return Sym->getAddress();
}

} // end namespace clang
