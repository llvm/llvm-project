//===--- ExceptionAnalyzer.h - clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_EXCEPTION_ANALYZER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_EXCEPTION_ANALYZER_H

#include "clang/AST/ASTContext.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"

namespace clang {
namespace tidy {
namespace utils {

/// This class analysis if a `FunctionDecl` can in principle throw an exception,
/// either directly or indirectly.
/// It can be configured to ignore custom exception types.
class ExceptionAnalyzer {
public:
  ExceptionAnalyzer() = default;

  bool throwsException(const FunctionDecl *Func);
  void ignoreExceptions(llvm::StringSet<> ExceptionNames) {
    IgnoredExceptions = std::move(ExceptionNames);
  }

private:
  using TypeVec = llvm::SmallVector<const Type *, 8>;

  TypeVec throwsException(const FunctionDecl *Func,
                          llvm::SmallSet<const FunctionDecl *, 32> &CallStack);
  TypeVec throwsException(const Stmt *St, const TypeVec &Caught,
                          llvm::SmallSet<const FunctionDecl *, 32> &CallStack);
  bool isIgnoredExceptionType(const Type *Exception);

  llvm::StringSet<> IgnoredExceptions;
  llvm::DenseMap<const FunctionDecl *, bool> FunctionCache;
};
} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_EXCEPTION_ANALYZER_H
