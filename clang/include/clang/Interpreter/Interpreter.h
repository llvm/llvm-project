//===--- Interpreter.h - Incremental Compilation and Execution---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the component which performs incremental code
// compilation and execution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_INTERPRETER_H
#define LLVM_CLANG_INTERPRETER_INTERPRETER_H

#include "clang/Interpreter/PartialTranslationUnit.h"

#include "clang/AST/GlobalDecl.h"

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <vector>

namespace llvm {
namespace orc {
class ThreadSafeContext;
}
} // namespace llvm

namespace clang {

class CompilerInstance;
class IncrementalExecutor;
class IncrementalParser;

/// Create a pre-configured \c CompilerInstance for incremental processing.
class IncrementalCompilerBuilder {
public:
  static llvm::Expected<std::unique_ptr<CompilerInstance>>
  create(std::vector<const char *> &ClangArgv);
};

/// Provides top-level interfaces for incremental compilation and execution.
class Interpreter {
  std::unique_ptr<llvm::orc::ThreadSafeContext> TSCtx;
  std::unique_ptr<IncrementalParser> IncrParser;
  std::unique_ptr<IncrementalExecutor> IncrExecutor;

  Interpreter(std::unique_ptr<CompilerInstance> CI, llvm::Error &Err);

public:
  ~Interpreter();
  static llvm::Expected<std::unique_ptr<Interpreter>>
  create(std::unique_ptr<CompilerInstance> CI);
  const CompilerInstance *getCompilerInstance() const;
  llvm::Expected<PartialTranslationUnit &> Parse(llvm::StringRef Code);
  llvm::Error Execute(PartialTranslationUnit &T);
  llvm::Error ParseAndExecute(llvm::StringRef Code) {
    auto PTU = Parse(Code);
    if (!PTU)
      return PTU.takeError();
    if (PTU->TheModule)
      return Execute(*PTU);
    return llvm::Error::success();
  }

  /// \returns the \c JITTargetAddress of a \c GlobalDecl. This interface uses
  /// the CodeGenModule's internal mangling cache to avoid recomputing the
  /// mangled name.
  llvm::Expected<llvm::JITTargetAddress> getSymbolAddress(GlobalDecl GD) const;

  /// \returns the \c JITTargetAddress of a given name as written in the IR.
  llvm::Expected<llvm::JITTargetAddress>
  getSymbolAddress(llvm::StringRef IRName) const;

  /// \returns the \c JITTargetAddress of a given name as written in the object
  /// file.
  llvm::Expected<llvm::JITTargetAddress>
  getSymbolAddressFromLinkerName(llvm::StringRef LinkerName) const;
};
} // namespace clang

#endif // LLVM_CLANG_INTERPRETER_INTERPRETER_H
