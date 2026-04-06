//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEINTEGERSIGNCOMPARISONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEINTEGERSIGNCOMPARISONCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang::tidy::modernize {

/// Replace comparisons between signed and unsigned integers with ``std::cmp_*``
/// and manual numeric_limits range checks with ``std::in_range``.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/modernize/use-integer-sign-comparison.html
class UseIntegerSignComparisonCheck : public ClangTidyCheck {
public:
  UseIntegerSignComparisonCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus20 || (LangOpts.CPlusPlus17 && EnableQtSupport);
  }

private:
  utils::IncludeInserter IncludeInserter;
  const bool EnableQtSupport;

  // Two-pass state for sign-comparison diagnostics: collect during check(),
  // emit in onEndOfTranslationUnit() after filtering out range-check children.
  struct PendingSignCmp {
    const BinaryOperator *BinaryOp;
  };
  llvm::SmallVector<PendingSignCmp, 8> PendingCmps;
  llvm::SmallVector<SourceRange, 4> RangeCheckRanges;
  const SourceManager *SrcMgr = nullptr;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEINTEGERSIGNCOMPARISONCHECK_H
