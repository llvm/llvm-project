//===--- UpgradeDurationConversionsCheck.h - clang-tidy ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_UPGRADEDURATIONCONVERSIONSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_UPGRADEDURATIONCONVERSIONSCHECK_H

#include "../ClangTidy.h"

#include <unordered_set>

namespace clang {
namespace tidy {
namespace abseil {

/// Finds deprecated uses of `absl::Duration` arithmetic operators and factories.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil-upgrade-duration-conversions.html
class UpgradeDurationConversionsCheck : public ClangTidyCheck {
public:
  UpgradeDurationConversionsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  std::unordered_set<unsigned> MatchedTemplateLocations;
};

} // namespace abseil
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_UPGRADEDURATIONCONVERSIONSCHECK_H
