//===--- RedundantLookupCheck.h - clang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_REDUNDANTLOOKUPCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_REDUNDANTLOOKUPCHECK_H

#include "../ClangTidyCheck.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace clang {
class SourceManager;
} // namespace clang

namespace clang::tidy::performance {

/// Detects redundant container lookups.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/redundant-lookup.html
class RedundantLookupCheck : public ClangTidyCheck {
public:
  RedundantLookupCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

private:
  llvm::DenseMap<unsigned, llvm::SmallPtrSet<const CallExpr *, 2>>
      RegisteredLookups;
  const StringRef ContainerNameRegex;
  const std::vector<StringRef> LookupMethodNames;
  const SourceManager *SM = nullptr;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_REDUNDANTLOOKUPCHECK_H
