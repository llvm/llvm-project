//===--- TypeMismatchCheck.h - clang-tidy------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MPI_TYPE_MISMATCH_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MPI_TYPE_MISMATCH_H

#include "../ClangTidy.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace tidy {
namespace mpi {

/// This check verifies if buffer type and MPI (Message Passing Interface)
/// datatype pairs match. All MPI datatypes defined by the MPI standard (3.1)
/// are verified by this check. User defined typedefs, custom MPI datatypes and
/// null pointer constants are skipped, in the course of verification.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/mpi-type-mismatch.html
class TypeMismatchCheck : public ClangTidyCheck {
public:
  TypeMismatchCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  /// Check if the buffer type MPI datatype pairs match.
  ///
  /// \param BufferTypes buffer types
  /// \param BufferExprs buffer arguments as expressions
  /// \param MPIDatatypes MPI datatype
  /// \param LO language options
  void checkArguments(ArrayRef<const Type *> BufferTypes,
                      ArrayRef<const Expr *> BufferExprs,
                      ArrayRef<StringRef> MPIDatatypes, const LangOptions &LO);
};

} // namespace mpi
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MPI_TYPE_MISMATCH_H
