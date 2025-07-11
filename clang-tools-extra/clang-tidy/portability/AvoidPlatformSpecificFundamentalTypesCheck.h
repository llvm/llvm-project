//==-- AvoidPlatformSpecificFundamentalTypesCheck.h - clang-tidy -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPLATFORMSPECIFICFUNDAMENTALTYPESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPLATFORMSPECIFICFUNDAMENTALTYPESCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"

namespace clang::tidy::portability {

/// Find fundamental platform-dependent types and recommend using typedefs or
/// fixed-width types.
///
/// Detects fundamental types (int, short, long, long long, char, float, etc)
/// and warns against their use due to platform-dependent behavior.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/portability/avoid-platform-specific-fundamental-types.html
class AvoidPlatformSpecificFundamentalTypesCheck : public ClangTidyCheck {
public:
  AvoidPlatformSpecificFundamentalTypesCheck(StringRef Name,
                                             ClangTidyContext *Context);
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const bool WarnOnFloats;
  const bool WarnOnInts;
  const bool WarnOnChars;
  utils::IncludeInserter IncludeInserter;
};

} // namespace clang::tidy::portability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_AVOIDPLATFORMSPECIFICFUNDAMENTALTYPESCHECK_H
