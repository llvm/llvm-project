//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSSTRINGVIEWDATAUSAGECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSSTRINGVIEWDATAUSAGECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Identifies suspicious usages of std::string_view::data() that could lead to
/// reading out-of-bounds data due to inadequate or incorrect string null
/// termination.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/suspicious-stringview-data-usage.html
class SuspiciousStringviewDataUsageCheck : public ClangTidyCheck {
public:
  SuspiciousStringviewDataUsageCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  std::optional<TraversalKind> getCheckTraversalKind() const override;

private:
  std::vector<llvm::StringRef> StringViewTypes;
  std::vector<llvm::StringRef> AllowedCallees;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSSTRINGVIEWDATAUSAGECHECK_H
