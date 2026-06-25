//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_EXPENSIVEVALUEORCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_EXPENSIVEVALUEORCHECK_H

#include "../ClangTidyCheck.h"
#include <vector>

namespace clang::tidy::performance {

/// Warns when 'value_or' is called on an optional type whose underlying type
/// is expensive to copy (not trivially copyable, or larger than a threshold).
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/performance/expensive-value-or.html
class ExpensiveValueOrCheck : public ClangTidyCheck {
public:
  ExpensiveValueOrCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const unsigned SizeThreshold;
  const std::vector<StringRef> OptionalTypes;
  const bool WarnOnOwnershipTaking;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_EXPENSIVEVALUEORCHECK_H
