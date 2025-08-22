//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: 2025 Siemens Corporation and/or its affiliates
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_METHODHIDINGCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_METHODHIDINGCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Checks that a derived class does not define the same (non virtual) method as
/// a base class
///
/// This anti-pattern is a Liskov violation.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/method-hiding.html
class MethodHidingCheck : public ClangTidyCheck {
public:
  MethodHidingCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_METHODHIDINGCHECK_H
