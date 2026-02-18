//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEINITSTATEMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEINITSTATEMENTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Finds variable declarations immediately before an ``if`` or ``switch``
/// statement where the variable is only used inside the conditional, and
/// suggests moving the declaration into the init-statement (C++17).
///
/// For example:
/// \code
///   auto it = map.find(key);
///   if (it != map.end()) { use(it); }
/// \endcode
/// becomes:
/// \code
///   if (auto it = map.find(key); it != map.end()) { use(it); }
/// \endcode
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/modernize/use-init-statement.html
class UseInitStatementCheck : public ClangTidyCheck {
public:
  UseInitStatementCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus17;
  }
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEINITSTATEMENTCHECK_H
