//===--- UnrollLoopsCheck.h - clang-tidy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_UNROLLLOOPSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_UNROLLLOOPSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::altera {

/// Finds inner loops that have not been unrolled, as well as fully unrolled
/// loops with unknown loop bounds or a large number of iterations.
///
/// Unrolling inner loops could improve the performance of OpenCL kernels.
/// However, if they have unknown loop bounds or a large number of iterations,
/// they cannot be fully unrolled, and should be partially unrolled.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/altera/unroll-loops.html
class UnrollLoopsCheck : public ClangTidyCheck {
public:
  UnrollLoopsCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  /// Recommend partial unrolling if number of loop iterations is greater than
  /// MaxLoopIterations.
  const unsigned MaxLoopIterations;
  /// The kind of unrolling, if any, applied to a given loop.
  enum UnrollType {
    // This loop has no #pragma unroll directive associated with it.
    NotUnrolled,
    // This loop has a #pragma unroll directive associated with it.
    FullyUnrolled,
    // This loop has a #pragma unroll <num> directive associated with it.
    PartiallyUnrolled
  };
  /// Attempts to extract an integer value from either side of the
  /// BinaryOperator. Returns true and saves the result to &value if successful,
  /// returns false otherwise.
  bool extractValue(int &Value, const BinaryOperator *Op,
                    const ASTContext *Context);
  /// Returns true if the given loop statement has a large number of iterations,
  /// as determined by the integer value in the loop's condition expression,
  /// if one exists.
  bool hasLargeNumIterations(const Stmt *Statement,
                             const IntegerLiteral *CXXLoopBound,
                             const ASTContext *Context);
  /// Checks one hand side of the binary operator to ascertain if the upper
  /// bound on the number of loops is greater than max_loop_iterations or not.
  /// If the expression is not evaluatable or not an integer, returns false.
  bool exprHasLargeNumIterations(const Expr *Expression,
                                 const ASTContext *Context);
  /// Returns the type of unrolling, if any, associated with the given
  /// statement.
  enum UnrollType unrollType(const Stmt *Statement, ASTContext *Context);
  /// Returns the condition expression within a given for statement. If there is
  /// none, or if the Statement is not a loop, then returns a NULL pointer.
  const Expr *getCondExpr(const Stmt *Statement);
  /// Returns True if the loop statement has known bounds.
  bool hasKnownBounds(const Stmt *Statement, const IntegerLiteral *CXXLoopBound,
                      const ASTContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
};

} // namespace clang::tidy::altera

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_UNROLLLOOPSCHECK_H
