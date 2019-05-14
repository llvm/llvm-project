//===--- UseDefaultNoneCheck.h - clang-tidy ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OPENMP_USEDEFAULTNONECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OPENMP_USEDEFAULTNONECHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace openmp {

/// Finds OpenMP directives that are allowed to contain a ``default`` clause,
/// but either don't specify it or the clause is specified but with the kind
/// other than ``none``, and suggests to use the ``default(none)`` clause.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/openmp-use-default-none.html
class UseDefaultNoneCheck : public ClangTidyCheck {
public:
  UseDefaultNoneCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace openmp
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OPENMP_USEDEFAULTNONECHECK_H
