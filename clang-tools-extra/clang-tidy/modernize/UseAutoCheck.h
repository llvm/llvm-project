//===--- UseAutoCheck.h - clang-tidy-----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_AUTO_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_AUTO_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

class UseAutoCheck : public ClangTidyCheck {
public:
  UseAutoCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void replaceIterators(const DeclStmt *D, ASTContext *Context);
  void replaceExpr(const DeclStmt *D, ASTContext *Context,
                   llvm::function_ref<QualType(const Expr *)> GetType,
                   StringRef Message);

  const unsigned int MinTypeNameLength;
  const bool RemoveStars;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_AUTO_H
