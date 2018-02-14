//===--- RedundantExpressionCheck.h - clang-tidy-----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_REDUNDANT_EXPRESSION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_REDUNDANT_EXPRESSION_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// The checker detects expressions that are redundant, because they contain
/// ineffective, useless parts.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-redundant-expression.html
class RedundantExpressionCheck : public ClangTidyCheck {
public:
  RedundantExpressionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void checkArithmeticExpr(const ast_matchers::MatchFinder::MatchResult &R);
  void checkBitwiseExpr(const ast_matchers::MatchFinder::MatchResult &R);
  void checkRelationalExpr(const ast_matchers::MatchFinder::MatchResult &R);
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_REDUNDANT_EXPRESSION_H
