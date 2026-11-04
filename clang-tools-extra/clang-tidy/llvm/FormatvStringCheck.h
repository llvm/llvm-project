//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_FORMATVSTRINGCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_FORMATVSTRINGCHECK_H

#include "../ClangTidyCheck.h"
#include <vector>

namespace clang::tidy::llvm_check {

/// Validates llvm::formatv format strings against the provided arguments.
///
/// Checks that:
/// - The number of format indices matches the number of arguments.
/// - Every argument is used by the format string.
/// - Automatic and explicit indices are not mixed.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/llvm/formatv-string.html
class FormatvStringCheck : public ClangTidyCheck {
public:
  FormatvStringCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  std::vector<StringRef> Functions;
  const StringRef AdditionalFunctions;
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_FORMATVSTRINGCHECK_H
