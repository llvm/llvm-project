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

#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <vector>

namespace llvm {
class Triple;
namespace orc {
class ExecutorAddr;
class ThreadSafeContext;
} // namespace orc
} // namespace llvm

namespace clang {
class IncrementalExecutor;
class IncrementalCompilerBuilder;
class TargetInfo;

/// Common configuration and interface for incremental executor builders.
class IncrementalExecutorBuilder {
public:
  /// Options passed to the backend when configuring execution.
  struct Options {
    bool IsOutOfProcess = false;
    std::string ExecutorPath;
    std::string RuntimePath;
    unsigned SlabAllocateSize = 0;
    bool UseSharedMemory = false;
  };

  /// An optional external IncrementalExecutor.
  std::unique_ptr<IncrementalExecutor> IE;
  /// Frontend -mllvm arguments for backends that need to restore LLVM options.
  std::vector<std::string> LLVMArgs;

  virtual ~IncrementalExecutorBuilder();

  /// Create the default builder for the platform clangInterpreter is built for.
  /// The selected backend provides this definition.
  static std::unique_ptr<IncrementalExecutorBuilder> createDefault();

  /// Apply execution options and configure the compiler for this backend.
  virtual llvm::Error configure(IncrementalCompilerBuilder &CB,
                                const Options &Opts) = 0;

  /// Probe whether this backend can JIT code for the host.
  virtual bool supportsJIT() const = 0;

  /// Return the host triple used by this backend.
  virtual llvm::Expected<llvm::Triple> getHostJITTriple() const = 0;

  /// Return the supplied executor, or create one using the selected backend.
  llvm::Expected<std::unique_ptr<IncrementalExecutor>>
  create(llvm::orc::ThreadSafeContext &TSC, const clang::TargetInfo &TI);

private:
  virtual llvm::Expected<std::unique_ptr<IncrementalExecutor>>
  createExecutor(llvm::orc::ThreadSafeContext &TSC,
                 const clang::TargetInfo &TI) = 0;
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
