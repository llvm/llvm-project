//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNCHECKEDEXPECTEDACCESSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNCHECKEDEXPECTEDACCESSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Warns when the code is accessing the value (or error) of a
/// `std::expected<T, E>` object without assuring that it contains a value
/// (or error).
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/unchecked-expected-access.html
class UncheckedExpectedAccessCheck : public ClangTidyCheck {
public:
  UncheckedExpectedAccessCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  // TODO: Consider limiting to C++23 and later
  // unless pre-C++23 implementations (e.g. `tl::expected`) get supported.
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNCHECKEDEXPECTEDACCESSCHECK_H
