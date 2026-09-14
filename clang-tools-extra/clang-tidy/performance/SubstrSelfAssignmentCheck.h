//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_SUBSTRSELFASSIGNMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_SUBSTRSELFASSIGNMENTCHECK_H

#include "../ClangTidyCheck.h"

#include <vector>

namespace clang::tidy::performance {

/// Finds cases where a string variable is assigned the result of calling
/// ``substr()`` on itself (e.g., ``s = s.substr(x, y)``). This pattern creates
/// an unnecessary temporary string; the same effect can be achieved in-place
/// using ``erase()``.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/performance/substr-self-assignment.html
class SubstrSelfAssignmentCheck : public ClangTidyCheck {
public:
  SubstrSelfAssignmentCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const std::vector<StringRef> StringLikeClasses;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_SUBSTRSELFASSIGNMENTCHECK_H
