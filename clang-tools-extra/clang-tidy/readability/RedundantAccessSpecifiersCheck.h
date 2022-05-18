//===--- RedundantAccessSpecifiersCheck.h - clang-tidy ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTACCESSSPECIFIERSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTACCESSSPECIFIERSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace readability {

/// Detects redundant access specifiers inside classes, structs, and unions.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/redundant-access-specifiers.html
class RedundantAccessSpecifiersCheck : public ClangTidyCheck {
public:
  RedundantAccessSpecifiersCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        CheckFirstDeclaration(
            Options.getLocalOrGlobal("CheckFirstDeclaration", false)) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool CheckFirstDeclaration;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTACCESSSPECIFIERSCHECK_H
