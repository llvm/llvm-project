//===--- IncrementalExecutor.h - Incremental Execution ----------*- C++ -*-===//
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

#ifndef LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H
#define LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H

#include "clang/Interpreter/Interpreter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>

namespace llvm {
class Error;
namespace orc {
class JITTargetMachineBuilder;
class LLJIT;
class LLJITBuilder;
class ThreadSafeContext;
} // namespace orc
} // namespace llvm

namespace clang {

struct PartialTranslationUnit;
class TargetInfo;

class IncrementalExecutor {
  using CtorDtorIterator = llvm::orc::CtorDtorIterator;
  std::unique_ptr<llvm::orc::LLJIT> Jit;
  llvm::orc::ThreadSafeContext &TSCtx;
  uint32_t OutOfProcessChildPid = -1;

  llvm::DenseMap<const PartialTranslationUnit *, llvm::orc::ResourceTrackerSP>
      ResourceTrackers;

protected:
  IncrementalExecutor(llvm::orc::ThreadSafeContext &TSC);

public:
  enum SymbolNameKind { IRName, LinkerName };

  IncrementalExecutor(llvm::orc::ThreadSafeContext &TSC,
                      llvm::orc::LLJITBuilder &JITBuilder,
                      Interpreter::JITConfig Config, llvm::Error &Err);
  virtual ~IncrementalExecutor();

  virtual llvm::Error addModule(PartialTranslationUnit &PTU);
  virtual llvm::Error removeModule(PartialTranslationUnit &PTU);
  virtual llvm::Error runCtors() const;
  virtual llvm::Error cleanUp();
  virtual llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddress(llvm::StringRef Name, SymbolNameKind NameKind) const;

  llvm::orc::LLJIT &GetExecutionEngine() { return *Jit; }

  uint32_t getOutOfProcessChildPid() const { return OutOfProcessChildPid; }

  static llvm::Expected<std::unique_ptr<llvm::orc::LLJITBuilder>>
  createDefaultJITBuilder(llvm::orc::JITTargetMachineBuilder JTMB);

  static llvm::Expected<
      std::pair<std::unique_ptr<llvm::orc::SimpleRemoteEPC>, uint32_t>>
  launchExecutor(llvm::StringRef ExecutablePath, bool UseSharedMemory,
                 unsigned SlabAllocateSize,
                 std::function<void()> CustomizeFork = nullptr);

#if LLVM_ON_UNIX && LLVM_ENABLE_THREADS
  static llvm::Expected<std::unique_ptr<llvm::orc::SimpleRemoteEPC>>
  connectTCPSocket(llvm::StringRef NetworkAddress, bool UseSharedMemory,
                   unsigned SlabAllocateSize);
#endif
};

} // end namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H
