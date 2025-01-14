//===--- ConstCorrectnessCheck.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_CONSTCORRECTNESSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_CONSTCORRECTNESSCHECK_H

#include "../ClangTidyCheck.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "llvm/ADT/DenseSet.h"

namespace clang::tidy::misc {

/// This check warns on variables which could be declared const but are not.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc/const-correctness.html
class ConstCorrectnessCheck : public ClangTidyCheck {
public:
  ConstCorrectnessCheck(StringRef Name, ClangTidyContext *Context);

  // The rules for C and 'const' are different and incompatible for this check.
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void registerScope(const Stmt *LocalScope, ASTContext *Context);

  using MutationAnalyzer = std::unique_ptr<ExprMutationAnalyzer>;
  llvm::DenseMap<const Stmt *, MutationAnalyzer> ScopesCache;
  llvm::DenseSet<SourceLocation> TemplateDiagnosticsCache;

  const bool AnalyzeValues;
  const bool AnalyzeReferences;
  const bool WarnPointersAsValues;

  const bool TransformValues;
  const bool TransformReferences;
  const bool TransformPointersAsValues;

  const std::vector<StringRef> AllowedTypes;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_CONSTCORRECTNESSCHECK_H
