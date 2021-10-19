//===--- BufferDerefCheck.h - clang-tidy-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MPI_BUFFER_DEREF_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MPI_BUFFER_DEREF_H

#include "../ClangTidyCheck.h"
#include "clang/StaticAnalyzer/Checkers/MPIFunctionClassifier.h"

namespace clang {
namespace tidy {
namespace mpi {

/// This check verifies if a buffer passed to an MPI (Message Passing Interface)
/// function is sufficiently dereferenced. Buffers should be passed as a single
/// pointer or array. As MPI function signatures specify void * for their buffer
/// types, insufficiently dereferenced buffers can be passed, like for example
/// as double pointers or multidimensional arrays, without a compiler warning
/// emitted.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/mpi-buffer-deref.html
class BufferDerefCheck : public ClangTidyCheck {
public:
  BufferDerefCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;

private:
  /// Checks for all buffers in an MPI call if they are sufficiently
  /// dereferenced.
  ///
  /// \param BufferTypes buffer types
  /// \param BufferExprs buffer arguments as expressions
  void checkBuffers(ArrayRef<const Type *> BufferTypes,
                    ArrayRef<const Expr *> BufferExprs);

  enum class IndirectionType : unsigned char { Pointer, Array };

  Optional<ento::mpi::MPIFunctionClassifier> FuncClassifier;
};

} // namespace mpi
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MPI_BUFFER_DEREF_H
