//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_CRTPCONSTRUCTORACCESSIBILITYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_CRTPCONSTRUCTORACCESSIBILITYCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Detects error-prone Curiously Recurring Template Pattern usage, when the
/// CRTP can be constructed outside itself and the derived class.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/crtp-constructor-accessibility.html
class CrtpConstructorAccessibilityCheck : public ClangTidyCheck {
public:
  CrtpConstructorAccessibilityCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_CRTPCONSTRUCTORACCESSIBILITYCHECK_H
