//===--- IncorrectIteratorsCheck.h - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCORRECTITERATORSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCORRECTITERATORSCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy::bugprone {

/// Detects calls to iterator algorithms that are called with potentially
/// invalid arguments.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/incorrect-iterators.html
class IncorrectIteratorsCheck : public ClangTidyCheck {
public:
  IncorrectIteratorsCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;

private:
  std::vector<StringRef> BeginFree;
  std::vector<StringRef> EndFree;
  std::vector<StringRef> BeginMethod;
  std::vector<StringRef> EndMethod;
  std::vector<StringRef> RBeginFree;
  std::vector<StringRef> REndFree;
  std::vector<StringRef> RBeginMethod;
  std::vector<StringRef> REndMethod;
  std::vector<StringRef> MakeReverseIterator;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCORRECTITERATORSCHECK_H
