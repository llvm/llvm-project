//===-- OrcIncrementalExecutorBuilder.h - ORC Builder -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the builder for ORC incremental execution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_ORCINCREMENTALEXECUTORBUILDER_H
#define LLVM_CLANG_INTERPRETER_ORCINCREMENTALEXECUTORBUILDER_H

#include "clang/Interpreter/IncrementalExecutor.h"
#include "llvm/Support/CodeGen.h"

#include <cstdint>
#include <functional>
#include <optional>

namespace llvm::orc {
class LLJITBuilder;
}

namespace clang {
namespace driver {
class Compilation;
}

/// Configuration for in-process and out-of-process ORC execution.
class OrcIncrementalExecutorBuilder : public IncrementalExecutorBuilder {
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
  /// An optional external ORC JIT builder.
  std::unique_ptr<llvm::orc::LLJITBuilder> JITBuilder;
  /// A default callback that can be used in the IncrementalCompilerBuilder to
  /// retrieve the path to the orc runtime.
  std::function<llvm::Error(const driver::Compilation &)>
      UpdateOrcRuntimePathCB = [this](const driver::Compilation &C) {
        return UpdateOrcRuntimePath(C);
      };

  OrcIncrementalExecutorBuilder();
  ~OrcIncrementalExecutorBuilder() override;

private:
  llvm::Expected<std::unique_ptr<IncrementalExecutor>>
  createExecutor(llvm::orc::ThreadSafeContext &TSC,
                 const clang::TargetInfo &TI) override;
  llvm::Error UpdateOrcRuntimePath(const driver::Compilation &C);
};

} // namespace clang

#endif // LLVM_CLANG_INTERPRETER_ORCINCREMENTALEXECUTORBUILDER_H
