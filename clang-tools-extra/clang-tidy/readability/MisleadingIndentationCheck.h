//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MISLEADING_INDENTATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MISLEADING_INDENTATION_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Checks the code for dangling else, and possible misleading indentations due
/// to missing braces. Note that this check only works as expected when the tabs
/// or spaces are used consistently and not mixed.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/misleading-indentation.html
class MisleadingIndentationCheck : public ClangTidyCheck {
public:
  MisleadingIndentationCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  void danglingElseCheck(const SourceManager &SM, ASTContext *Context,
                         const IfStmt *If);
  void missingBracesCheck(const SourceManager &SM, const CompoundStmt *CStmt,
                          const LangOptions &LangOpts);
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MISLEADING_INDENTATION_H
