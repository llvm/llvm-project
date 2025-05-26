//===--- ConstructReusableObjectsOnceCheck.h - clang-tidy -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_CONSTRUCTREUSABLEOBJECTSONCECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_CONSTRUCTREUSABLEOBJECTSONCECHECK_H

#include "../ClangTidyCheck.h"
#include <optional>
#include <vector>

namespace clang::tidy::performance {

/// Finds variable declarations of expensive-to-construct classes that are
/// constructed from only constant literals and so can be reused.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/construct-reusable-objects-once.html
class ConstructReusableObjectsOnceCheck : public ClangTidyCheck {
public:
  ConstructReusableObjectsOnceCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  std::vector<StringRef> CheckedClasses;
  std::vector<StringRef> IgnoredClasses;
  std::vector<StringRef> IgnoredFunctions;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_CONSTRUCTREUSABLEOBJECTSONCECHECK_H
