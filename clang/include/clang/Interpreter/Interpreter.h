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

#include "clang/Interpreter/Transaction.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <vector>

namespace llvm {
namespace orc {
class ThreadSafeContext;
}
class Module;
} // namespace llvm

namespace clang {

class CompilerInstance;
class DeclGroupRef;
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
  llvm::Expected<Transaction &> Parse(llvm::StringRef Code);
  llvm::Error Execute(Transaction &T);
  llvm::Error ParseAndExecute(llvm::StringRef Code) {
    auto ErrOrTransaction = Parse(Code);
    if (auto Err = ErrOrTransaction.takeError())
      return Err;
    if (ErrOrTransaction->TheModule)
      return Execute(*ErrOrTransaction);
    return llvm::Error::success();
  }
};
} // namespace clang

#endif // LLVM_CLANG_INTERPRETER_INTERPRETER_H
