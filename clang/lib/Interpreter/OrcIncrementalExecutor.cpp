//===--- OrcIncrementalExecutor.cpp - Orc Incremental Execution -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an Orc-based incremental code execution.
//
//===----------------------------------------------------------------------===//

#include "OrcIncrementalExecutor.h"
#include "clang/Interpreter/IncrementalExecutor.h"
#include "clang/Interpreter/PartialTranslationUnit.h"
#include "clang/Interpreter/ThreadSafeContext.h"

#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#ifdef LLVM_ON_UNIX
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

// Force linking some of the runtimes that helps attaching to a debugger.
LLVM_ATTRIBUTE_USED void linkComponents() {
  llvm::errs() << (void *)&llvm_orc_registerJITLoaderGDBAllocAction;
}

namespace clang {
OrcIncrementalExecutor::OrcIncrementalExecutor(
    clang::ThreadSafeContext &TSC, llvm::orc::LLJITBuilder &JITBuilder,
    llvm::Error &Err)
    : TSCtx(TSC), OrcTSCtx(nullptr) {
  using namespace llvm::orc;
  llvm::ErrorAsOutParameter EAO(&Err);

  // Create an ORC ThreadSafeContext wrapping the same LLVMContext as the
  // clang::ThreadSafeContext. This ensures all modules (compiled by clang
  // using TSC's context) are compatible with the ORC JIT.
  // We use a no-op deleter since the context is owned by clang::ThreadSafeContext.
  TSC.withContextDo([this](llvm::LLVMContext *Ctx) {
    this->OrcTSCtx = std::make_unique<llvm::orc::ThreadSafeContext>(
        std::unique_ptr<llvm::LLVMContext>(Ctx, [](llvm::LLVMContext *) {}));
  });

  if (auto JitOrErr = JITBuilder.create())
    Jit = std::move(*JitOrErr);
  else {
    Err = JitOrErr.takeError();
    return;
  }
}

OrcIncrementalExecutor::~OrcIncrementalExecutor() {}

llvm::Error OrcIncrementalExecutor::addModule(PartialTranslationUnit &PTU) {
  llvm::orc::ResourceTrackerSP RT =
      Jit->getMainJITDylib().createResourceTracker();
  ResourceTrackers[&PTU] = RT;

  return Jit->addIRModule(RT, {std::move(PTU.TheModule), OrcTSCtx});
}

llvm::Error OrcIncrementalExecutor::removeModule(PartialTranslationUnit &PTU) {

  llvm::orc::ResourceTrackerSP RT = std::move(ResourceTrackers[&PTU]);
  if (!RT)
    return llvm::Error::success();

  ResourceTrackers.erase(&PTU);
  if (llvm::Error Err = RT->remove())
    return Err;
  return llvm::Error::success();
}

// Clean up the JIT instance.
llvm::Error OrcIncrementalExecutor::cleanUp() {
  // This calls the global dtors of registered modules.
  return Jit->deinitialize(Jit->getMainJITDylib());
}

llvm::Error OrcIncrementalExecutor::runCtors() const {
  return Jit->initialize(Jit->getMainJITDylib());
}

llvm::Expected<clang::ExecutorAddress>
OrcIncrementalExecutor::getSymbolAddress(llvm::StringRef Name,
                                         SymbolNameKind NameKind) const {
  using namespace llvm::orc;
  auto SO = makeJITDylibSearchOrder({&Jit->getMainJITDylib(),
                                     Jit->getPlatformJITDylib().get(),
                                     Jit->getProcessSymbolsJITDylib().get()});

  ExecutionSession &ES = Jit->getExecutionSession();

  auto SymOrErr = ES.lookup(SO, (NameKind == SymbolNameKind::LinkerName)
                                    ? ES.intern(Name)
                                    : Jit->mangleAndIntern(Name));
  if (auto Err = SymOrErr.takeError())
    return std::move(Err);
  return clang::ExecutorAddress(SymOrErr->getAddress().getValue());
}

llvm::Error OrcIncrementalExecutor::LoadDynamicLibrary(const char *name) {
  // FIXME: Eventually we should put each library in its own JITDylib and
  //        turn off process symbols by default.
  llvm::orc::ExecutionSession &ES = Jit->getExecutionSession();
  auto DLSGOrErr = llvm::orc::EPCDynamicLibrarySearchGenerator::Load(ES, name);
  if (!DLSGOrErr)
    return DLSGOrErr.takeError();

  Jit->getProcessSymbolsJITDylib()->addGenerator(std::move(*DLSGOrErr));

  return llvm::Error::success();
}

} // namespace clang
