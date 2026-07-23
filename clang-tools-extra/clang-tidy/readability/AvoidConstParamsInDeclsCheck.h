//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOIDCONSTPARAMSINDECLSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOIDCONSTPARAMSINDECLSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

// Detect function declarations that have const value parameters and discourage
// them.
//
// For the user-facing documentation see:
// https://clang.llvm.org/extra/clang-tidy/checks/readability/avoid-const-params-in-decls.html
class AvoidConstParamsInDeclsCheck : public ClangTidyCheck {
public:
  AvoidConstParamsInDeclsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        IgnoreMacros(Options.get("IgnoreMacros", true)) {}

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const bool IgnoreMacros;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOIDCONSTPARAMSINDECLSCHECK_H
