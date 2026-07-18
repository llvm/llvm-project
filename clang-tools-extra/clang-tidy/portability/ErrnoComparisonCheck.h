//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_ERRNOCOMPARISONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_ERRNOCOMPARISONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::portability {

/// Flags comparisons of 'errno' against an integer literal. The values of the
/// error constants are implementation-defined, so a literal such as
/// 'errno == 5' is not portable; the 'E'-prefixed macros should be used.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/portability/errno-comparison.html
class ErrnoComparisonCheck : public ClangTidyCheck {
public:
  ErrnoComparisonCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    // Match the code as written so a comparison in a template is reported once,
    // not for every instantiation.
    return TK_IgnoreUnlessSpelledInSource;
  }
};

} // namespace clang::tidy::portability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_ERRNOCOMPARISONCHECK_H
