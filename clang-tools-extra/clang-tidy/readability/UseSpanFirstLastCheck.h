//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USESPANFIRSTLASTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USESPANFIRSTLASTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Suggests using clearer 'std::span' member functions 'first()'/'last()'
/// instead of equivalent 'subspan()' calls where applicable.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/readability/use-span-first-last.html
///
/// For example:
/// \code
///   std::span<int> s = ...;
///   auto sub1 = s.subspan(0, n);           // ->  auto sub1 = s.first(n);
///   auto sub2 = s.subspan(s.size() - n);   // ->  auto sub2 = s.last(n);
///   auto sub3 = s.subspan(1, n);           // not changed
///   auto sub4 = s.subspan(n);              // not changed
/// \endcode
///
/// The check is only active in C++20 mode.
class UseSpanFirstLastCheck : public ClangTidyCheck {
public:
  UseSpanFirstLastCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus20;
  }
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USESPANFIRSTLASTCHECK_H
