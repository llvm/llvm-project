//===--- MultipleStatementMacroCheck.h - clang-tidy--------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MULTIPLE_STATEMENT_MACRO_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MULTIPLE_STATEMENT_MACRO_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Detect multiple statement macros that are used in unbraced conditionals.
/// Only the first statement of the macro will be inside the conditional and the
/// other ones will be executed unconditionally.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-multiple-statement-macro.html
class MultipleStatementMacroCheck : public ClangTidyCheck {
public:
  MultipleStatementMacroCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MULTIPLE_STATEMENT_MACRO_H
