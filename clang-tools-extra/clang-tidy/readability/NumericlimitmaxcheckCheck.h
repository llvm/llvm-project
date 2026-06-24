//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NUMERICLIMITMAXCHECKCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NUMERICLIMITMAXCHECKCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"

namespace clang::tidy::readability {

/// Suggest replacement of values like -1 / `~0` (when used as unsigned)
/// with std::numeric_limits<T>::max() for unsigned integer types.
class NumericlimitmaxcheckCheck : public ClangTidyCheck {
public:
  NumericlimitmaxcheckCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpander) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  utils::IncludeInserter Inserter;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NUMERICLIMITMAXCHECKCHECK_H