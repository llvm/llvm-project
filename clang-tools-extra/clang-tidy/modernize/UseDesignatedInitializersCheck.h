//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEDESIGNATEDINITIALIZERSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEDESIGNATEDINITIALIZERSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Finds initializer lists for aggregate type that could be
/// written as designated initializers instead.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-designated-initializers.html
class UseDesignatedInitializersCheck : public ClangTidyCheck {
public:
  UseDesignatedInitializersCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus20 || LangOpts.C99 ||
           (LangOpts.CPlusPlus && !StrictCppStandardCompliance) ||
           (!LangOpts.CPlusPlus && !LangOpts.ObjC &&
            !StrictCStandardCompliance);
  }

private:
  bool IgnoreSingleElementAggregates;
  bool RestrictToPODTypes;
  bool IgnoreMacros;
  bool StrictCStandardCompliance;
  bool StrictCppStandardCompliance;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEDESIGNATEDINITIALIZERSCHECK_H
