//===--- SortConstructorInitializersCheck.h - clang-tidy --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SOCIALPOINT_SORTCONSTRUCTORINITIALIZERSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SOCIALPOINT_SORTCONSTRUCTORINITIALIZERSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace socialpoint {

/// FIXME: Write a short description.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-sort-constructor-initializers.html
class SortConstructorInitializersCheck : public ClangTidyCheck {
public:
  SortConstructorInitializersCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
private:
    void process(const CXXConstructorDecl* Decl, SourceManager& SourceMgr, const LangOptions& LangOpts);
};

} // namespace socialpoint
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SOCIALPOINT_SORTCONSTRUCTORINITIALIZERSCHECK_H
