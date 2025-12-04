//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_STRINGFINDSTARTSWITHCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_STRINGFINDSTARTSWITHCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include <memory>
#include <string>
#include <vector>

namespace clang::tidy::abseil {

// Find string.find(...) == 0 comparisons and suggest replacing with StartsWith.
// FIXME(niko): Add similar check for EndsWith
class StringFindStartswithCheck : public ClangTidyCheck {
public:
  using ClangTidyCheck::ClangTidyCheck;
  StringFindStartswithCheck(StringRef Name, ClangTidyContext *Context);
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    // Prefer modernize-use-starts-ends-with when C++20 is available.
    return LangOpts.CPlusPlus && !LangOpts.CPlusPlus20;
  }

private:
  const std::vector<StringRef> StringLikeClasses;
  utils::IncludeInserter IncludeInserter;
  const StringRef AbseilStringsMatchHeader;
};

} // namespace clang::tidy::abseil

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_STRINGFINDSTARTSWITHCHECK_H
