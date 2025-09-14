//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MULTIPLE_STATEMENT_MACRO_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MULTIPLE_STATEMENT_MACRO_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Detect multiple statement macros that are used in unbraced conditionals.
/// Only the first statement of the macro will be inside the conditional and the
/// other ones will be executed unconditionally.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/multiple-statement-macro.html
class MultipleStatementMacroCheck : public ClangTidyCheck {
public:
  MultipleStatementMacroCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MULTIPLE_STATEMENT_MACRO_H
