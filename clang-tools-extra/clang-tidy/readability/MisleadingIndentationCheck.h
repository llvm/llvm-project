//===--- MisleadingIndentationCheck.h - clang-tidy---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MISLEADING_INDENTATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MISLEADING_INDENTATION_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// Checks the code for dangling else, and possible misleading indentations due
/// to missing braces. Note that this check only works as expected when the tabs
/// or spaces are used consistently and not mixed.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-misleading-indentation.html
class MisleadingIndentationCheck : public ClangTidyCheck {
public:
  MisleadingIndentationCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void danglingElseCheck(const SourceManager &SM, const IfStmt *If);
  void missingBracesCheck(const SourceManager &SM, const CompoundStmt *CStmt);
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MISLEADING_INDENTATION_H
