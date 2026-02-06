//===--- InefficientCopyAssign.h - clang-tidy -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_USESTDMOVECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_USESTDMOVECHECK_H

#include "../ClangTidyCheck.h"

#include "clang/Analysis/CFG.h"

namespace clang::tidy::performance {

class UseStdMoveCheck : public ClangTidyCheck {
public:
  UseStdMoveCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  llvm::DenseMap<const FunctionDecl *, std::unique_ptr<CFG>> CFGCache;
  const CFG *getCFG(const FunctionDecl *, ASTContext *);
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_USESTDMOVECHECK_H
