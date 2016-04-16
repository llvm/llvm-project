//===--- SizeofExpressionCheck.h - clang-tidy--------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SIZEOF_EXPRESSION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SIZEOF_EXPRESSION_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// Find suspicious usages of sizeof expression.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-sizeof-expression.html
class SizeofExpressionCheck : public ClangTidyCheck {
public:
  SizeofExpressionCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool WarnOnSizeOfConstant;
  const bool WarnOnSizeOfThis;
  const bool WarnOnSizeOfCompareToConstant;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SIZEOF_EXPRESSION_H
