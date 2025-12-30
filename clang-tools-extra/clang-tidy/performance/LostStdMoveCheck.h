//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_LOSTSTDMOVECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_LOSTSTDMOVECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::performance {

/// Warns if copy constructor is used instead of std::move() and suggests a fix.
/// It honours cycles, lambdas, and unspecified call order in compound
/// expressions.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/performance/lost-std-move.html
class LostStdMoveCheck : public ClangTidyCheck {
public:
  LostStdMoveCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }

private:
  const bool StrictMode;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_LOSTSTDMOVECHECK_H
