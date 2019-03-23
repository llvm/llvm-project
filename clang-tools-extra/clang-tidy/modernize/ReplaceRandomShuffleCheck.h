//===--- ReplaceRandomShuffleCheck.h - clang-tidy----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_RANDOM_SHUFFLE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_RANDOM_SHUFFLE_H

#include "../ClangTidy.h"
#include "../utils/IncludeInserter.h"

namespace clang {
namespace tidy {
namespace modernize {

/// std::random_shuffle will be removed as of C++17. This check will find and
/// replace all occurrences of std::random_shuffle with std::shuffle.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-replace-random-shuffle.html
class ReplaceRandomShuffleCheck : public ClangTidyCheck {
public:
  ReplaceRandomShuffleCheck(StringRef Name, ClangTidyContext *Context);
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  std::unique_ptr<utils::IncludeInserter> IncludeInserter;
  const utils::IncludeSorter::IncludeStyle IncludeStyle;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_RANDOM_SHUFFLE_H
