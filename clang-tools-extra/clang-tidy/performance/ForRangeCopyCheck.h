//===--- ForRangeCopyCheck.h - clang-tidy------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_FORRANGECOPYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_FORRANGECOPYCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace performance {

/// A check that detects copied loop variables and suggests using const
/// references.
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/for-range-copy.html
class ForRangeCopyCheck : public ClangTidyCheck {
public:
  ForRangeCopyCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override{
    return LangOpts.CPlusPlus11;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  // Checks if the loop variable is a const value and expensive to copy. If so
  // suggests it be converted to a const reference.
  bool handleConstValueCopy(const VarDecl &LoopVar, ASTContext &Context);

  // Checks if the loop variable is a non-const value and whether only
  // const methods are invoked on it or whether it is only used as a const
  // reference argument. If so it suggests it be made a const reference.
  bool handleCopyIsOnlyConstReferenced(const VarDecl &LoopVar,
                                       const CXXForRangeStmt &ForRange,
                                       ASTContext &Context);

  const bool WarnOnAllAutoCopies;
  const std::vector<StringRef> AllowedTypes;
};

} // namespace performance
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_FORRANGECOPYCHECK_H
