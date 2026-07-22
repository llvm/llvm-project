//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_ANALYZERUNUSEDPROGRAMSTATEREFCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_ANALYZERUNUSEDPROGRAMSTATEREFCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::llvm_check {

/// Finds unused local `clang::ento::ProgramStateRef` variables. These are not
/// diagnosed by `-Wunused-variable` because `ProgramStateRef` has a non-trivial
/// destructor, even though it carries no meaningful RAII side effects.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/llvm/analyzer-unused-program-state-ref.html
class AnalyzerUnusedProgramStateRefCheck : public ClangTidyCheck {
public:
  AnalyzerUnusedProgramStateRefCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_ANALYZERUNUSEDPROGRAMSTATEREFCHECK_H
