//===--- UseBuiltinLiteralsCheck.h - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USEBUILTINLITERALSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USEBUILTINLITERALSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Finds literals explicitly casted to a type that could be expressed using
/// builtin prefixes or suffixes. Defined for character, integer, and floating
/// literals. Removes any suffixes or prefixes before applying the one that
/// corresponds to the type of the cast.
///
/// An explicit cast within a macro will be matched, but will only yield a
/// suggestion for a manual fix. Otherwise, if either the destination type or
/// the literal was substituted from a macro, no warning will be produced.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/use-builtin-literals.html
class UseBuiltinLiteralsCheck : public ClangTidyCheck {
public:
  UseBuiltinLiteralsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USEBUILTINLITERALSCHECK_H
