//===--- UseAnyOfAllOfCheck.h - clang-tidy-----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USEALGORITHMCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USEALGORITHMCHECK_H

#include "../ClangTidy.h"
#include "../utils/IncludeInserter.h"

namespace clang {
namespace tidy {
namespace readability {

/// Finds ranged-based for loops that can be replaced by a call to std::any_of
/// or std::all_of.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-use-anyofallof.html
class UseAnyOfAllOfCheck : public ClangTidyCheck {
public:
  using ClangTidyCheck::ClangTidyCheck;

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USEALGORITHMCHECK_H
