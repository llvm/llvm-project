//===--- AvoidPlatformSpecificFundamentalTypesCheck.h - clang-tidy -------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPLATFORMSPECIFICFUNDAMENTALTYPESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPLATFORMSPECIFICFUNDAMENTALTYPESCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::portability {

/// Find fundamental integer types and recommend using typedefs or fixed-width
/// types.
///
/// Detects fundamental integer types (int, short, long, long long, and their
/// unsigned variants) and warns against their use due to platform-dependent
/// behavior. Excludes semantic types like char, bool, wchar_t, char16_t,
/// char32_t, size_t, and ptrdiff_t.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/portability/avoid-platform-specific-fundamental-types.html
class AvoidPlatformSpecificFundamentalTypesCheck : public ClangTidyCheck {
public:
  AvoidPlatformSpecificFundamentalTypesCheck(StringRef Name,
                                             ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }
};

} // namespace clang::tidy::portability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPLATFORMSPECIFICFUNDAMENTALTYPESCHECK_H
