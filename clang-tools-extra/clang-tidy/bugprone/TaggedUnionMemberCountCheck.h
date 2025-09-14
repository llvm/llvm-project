//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_TAGGEDUNIONMEMBERCOUNTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_TAGGEDUNIONMEMBERCOUNTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Gives warnings for tagged unions, where the number of tags is
/// different from the number of data members inside the union.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/tagged-union-member-count.html
class TaggedUnionMemberCountCheck : public ClangTidyCheck {
public:
  TaggedUnionMemberCountCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool StrictMode;
  const bool EnableCountingEnumHeuristic;
  const std::vector<StringRef> CountingEnumPrefixes;
  const std::vector<StringRef> CountingEnumSuffixes;

  std::pair<const std::size_t, const EnumConstantDecl *>
  getNumberOfEnumValues(const EnumDecl *ED);
  bool isCountingEnumLikeName(StringRef Name) const;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_TAGGEDUNIONMEMBERCOUNTCHECK_H
