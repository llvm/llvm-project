//===--- TriviallyDestructibleCheck.h - clang-tidy --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_TRIVIALLYDESTRUCTIBLECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_TRIVIALLYDESTRUCTIBLECHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace performance {

/// A check that finds classes that would be trivial if not for the defaulted
/// destructors declared out-of-line:
/// struct A: TrivialClass {
///   ~A();
///   TrivialClass trivial_fields;
/// };
/// A::~A() = default;
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/trivially-destructible.html
class TriviallyDestructibleCheck : public ClangTidyCheck {
public:
  TriviallyDestructibleCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace performance
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_TRIVIALLYDESTRUCTIBLECHECK_H
