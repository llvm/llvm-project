//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONFACTORYSCALECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONFACTORYSCALECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::abseil {

/// This check finds cases where the incorrect `Duration` factory function is
/// being used by looking for scaling constants inside the factory argument
/// and suggesting a more appropriate factory.  It also looks for the special
/// case of zero and suggests `ZeroDuration()`.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil/duration-factory-scale.html
class DurationFactoryScaleCheck : public ClangTidyCheck {
public:
  DurationFactoryScaleCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::abseil

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONFACTORYSCALECHECK_H
