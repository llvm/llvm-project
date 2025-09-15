//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_UPGRADEDURATIONCONVERSIONSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_UPGRADEDURATIONCONVERSIONSCHECK_H

#include "../ClangTidyCheck.h"

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseSet.h"

namespace clang::tidy::abseil {

/// Finds deprecated uses of `absl::Duration` arithmetic operators and
/// factories.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil/upgrade-duration-conversions.html
class UpgradeDurationConversionsCheck : public ClangTidyCheck {
public:
  UpgradeDurationConversionsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  llvm::DenseSet<SourceLocation> MatchedTemplateLocations;
};

} // namespace clang::tidy::abseil

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_UPGRADEDURATIONCONVERSIONSCHECK_H
