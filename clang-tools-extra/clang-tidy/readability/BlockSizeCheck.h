//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_BLOCKSIZECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_BLOCKSIZECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Warns about large if blocks
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/readability/block-size.html
class BlockSizeCheck : public ClangTidyCheck {
public:
  BlockSizeCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus || LangOpts.C99;
  }

private:
  const unsigned IfLineCountThreshold;
  const unsigned ForLineCountThreshold;
  const unsigned WhileLineCountThreshold;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_BLOCKSIZECHECK_H
