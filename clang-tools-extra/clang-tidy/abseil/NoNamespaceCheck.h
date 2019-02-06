//===--- NoNamespaceCheck.h - clang-tidy-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_NONAMESPACECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_NONAMESPACECHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace abseil {

/// This check ensures users don't open namespace absl, as that violates
/// Abseil's compatibility guidelines.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil-no-namespace.html
class NoNamespaceCheck : public ClangTidyCheck {
public:
  NoNamespaceCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace abseil
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_NONAMESPACECHECK_H
