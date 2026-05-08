//===--- IncrementalExecutor.h - Base Incremental Execution -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the base class that performs incremental code execution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H
#define LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H

#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace orc {
class ExecutorAddr;
class LLJITBuilder;
class ThreadSafeContext;
} // namespace orc
} // namespace llvm

namespace clang {
class IncrementalExecutor;
class TargetInfo;
namespace driver {
class Compilation;
} // namespace driver

class IncrementalExecutorBuilder {
public:
  /// Indicates whether out-of-process JIT execution is enabled.
  bool IsOutOfProcess = false;
  /// Path to the out-of-process JIT executor.
  std::string OOPExecutor = "";
  std::string OOPExecutorConnect = "";
  /// Indicates whether to use shared memory for communication.
  bool UseSharedMemory = false;
  /// Representing the slab allocation size for memory management in kb.
  unsigned SlabAllocateSize = 0;
  /// Path to the ORC runtime library.
  std::string OrcRuntimePath = "";
  /// PID of the out-of-process JIT executor.
  uint32_t ExecutorPID = 0;
  /// Custom lambda to be executed inside child process/executor
  std::function<void()> CustomizeFork = nullptr;
  /// An optional code model to provide to the JITTargetMachineBuilder
  std::optional<llvm::CodeModel::Model> CM = std::nullopt;
  /// An optional external IncrementalExecutor
  std::unique_ptr<IncrementalExecutor> IE;
  /// An optional external orc jit builder
  std::unique_ptr<llvm::orc::LLJITBuilder> JITBuilder;
  /// A default callback that can be used in the IncrementalCompilerBuilder to
  /// retrieve the path to the orc runtime.
  std::function<llvm::Error(const driver::Compilation &)>
      UpdateOrcRuntimePathCB = [this](const driver::Compilation &C) {
        return UpdateOrcRuntimePath(C);
      };

  ~IncrementalExecutorBuilder();

  llvm::Expected<std::unique_ptr<IncrementalExecutor>>
  create(llvm::orc::ThreadSafeContext &TSC, const clang::TargetInfo &TI);

private:
  llvm::Error UpdateOrcRuntimePath(const driver::Compilation &C);
};

struct PartialTranslationUnit;

class IncrementalExecutor {
public:
  enum SymbolNameKind { IRName, LinkerName };

  virtual ~IncrementalExecutor() = default;

  virtual llvm::Error addModule(PartialTranslationUnit &PTU) = 0;
  virtual llvm::Error removeModule(PartialTranslationUnit &PTU) = 0;
  virtual llvm::Error runCtors() const = 0;
  virtual llvm::Error cleanUp() = 0;

  virtual llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddress(llvm::StringRef Name, SymbolNameKind NameKind) const = 0;
  virtual llvm::Error LoadDynamicLibrary(const char *name) = 0;
};

} // namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H