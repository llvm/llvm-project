//===--- StringFindStartswithCheck.h - clang-tidy----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_STRINGFINDSTARTSWITHCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_STRINGFINDSTARTSWITHCHECK_H

#include "../ClangTidy.h"
#include "../utils/IncludeInserter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace tidy {
namespace abseil {

// Find string.find(...) == 0 comparisons and suggest replacing with StartsWith.
// FIXME(niko): Add similar check for EndsWith
// FIXME(niko): Add equivalent modernize checks for C++20's std::starts_With
class StringFindStartswithCheck : public ClangTidyCheck {
public:
  using ClangTidyCheck::ClangTidyCheck;
  StringFindStartswithCheck(StringRef Name, ClangTidyContext *Context);
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  std::unique_ptr<clang::tidy::utils::IncludeInserter> IncludeInserter;
  const std::vector<std::string> StringLikeClasses;
  const utils::IncludeSorter::IncludeStyle IncludeStyle;
  const std::string AbseilStringsMatchHeader;
};

} // namespace abseil
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_STRINGFINDSTARTSWITHCHECK_H
