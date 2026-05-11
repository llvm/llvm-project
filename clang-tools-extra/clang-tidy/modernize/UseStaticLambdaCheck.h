//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESTATICLAMBDACHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESTATICLAMBDACHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Finds non-capturing lambdas that can be marked ``static`` (C++23).
///
/// A lambda with an empty capture list (``[]``) has no dependency on any
/// enclosing state. Marking it ``static`` communicates this clearly and
/// may allow the compiler to skip generating an implicit conversion-to-
/// function-pointer operator.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/modernize/use-static-lambda.html
class UseStaticLambdaCheck : public ClangTidyCheck {
public:
  UseStaticLambdaCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus23;
  }
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESTATICLAMBDACHECK_H
