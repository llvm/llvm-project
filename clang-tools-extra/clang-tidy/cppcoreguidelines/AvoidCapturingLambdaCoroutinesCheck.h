//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_AVOIDCAPTURINGLAMBDACOROUTINESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_AVOIDCAPTURINGLAMBDACOROUTINESCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::cppcoreguidelines {

/// Flags C++20 coroutine lambdas with non-empty capture lists that may cause
/// use-after-free errors and suggests avoiding captures or ensuring the lambda
/// closure object has a guaranteed lifetime.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines/avoid-capturing-lambda-coroutines.html
class AvoidCapturingLambdaCoroutinesCheck : public ClangTidyCheck {
public:
  AvoidCapturingLambdaCoroutinesCheck(StringRef Name,
                                      ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;

private:
  const bool AllowExplicitObjectParameters;
};

} // namespace clang::tidy::cppcoreguidelines

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_AVOIDCAPTURINGLAMBDACOROUTINESCHECK_H
