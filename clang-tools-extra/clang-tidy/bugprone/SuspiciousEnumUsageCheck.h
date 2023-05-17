//===--- SuspiciousEnumUsageCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSENUMUSAGECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSENUMUSAGECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// The checker detects various cases when an enum is probably misused (as a
/// bitmask).
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/suspicious-enum-usage.html
class SuspiciousEnumUsageCheck : public ClangTidyCheck {
public:
  SuspiciousEnumUsageCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  void checkSuspiciousBitmaskUsage(const Expr*, const EnumDecl*);
  const bool StrictMode;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSENUMUSAGECHECK_H
