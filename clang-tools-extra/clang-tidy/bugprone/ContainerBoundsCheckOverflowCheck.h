//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_CONTAINERBOUNDSCHECKOVERFLOWCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_CONTAINERBOUNDSCHECKOVERFLOWCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds potential overflow in unsigned integer addition before comparison with
/// a container's size() method. For example a + b > v.size() can overflow if a
/// and b are large enough, leading to incorrect behavior.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/container-bounds-check-overflow.html
class ContainerBoundsCheckOverflowCheck : public ClangTidyCheck {
public:
  ContainerBoundsCheckOverflowCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

private:
  const std::vector<StringRef> IgnoredContainers;
  const std::vector<StringRef> SizeMethodNames;
  const std::vector<StringRef> IncludedFreeStandingSizeFuncNames;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_CONTAINERBOUNDSCHECKOVERFLOWCHECK_H
