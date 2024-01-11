//===--- ConditionaltostdminmaxCheck.h - clang-tidy -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONDITIONALTOSTDMINMAXCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONDITIONALTOSTDMINMAXCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// FIXME: replaces certain conditional statements with equivalent std::min or
/// std::max expressions, improving readability and promoting the use of
/// standard library functions."
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/ConditionalToStdMinMax.html
class ConditionaltostdminmaxCheck : public ClangTidyCheck {
public:
  ConditionaltostdminmaxCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONDITIONALTOSTDMINMAXCHECK_H
