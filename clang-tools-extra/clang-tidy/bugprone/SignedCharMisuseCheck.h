//===--- SignedCharMisuseCheck.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SIGNEDCHARMISUSECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SIGNEDCHARMISUSECHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Finds those ``signed char`` -> integer conversions which might indicate a
/// programming error. The basic problem with the ``signed char``, that it might
/// store the non-ASCII characters as negative values. This behavior can cause a
/// misunderstanding of the written code both when an explicit and when an
/// implicit conversion happens.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/signed-char-misuse.html
class SignedCharMisuseCheck : public ClangTidyCheck {
public:
  SignedCharMisuseCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  ast_matchers::internal::BindableMatcher<clang::Stmt> charCastExpression(
      bool IsSigned,
      const ast_matchers::internal::Matcher<clang::QualType> &IntegerType,
      const std::string &CastBindName) const;

  const StringRef CharTypdefsToIgnoreList;
  const bool DiagnoseSignedUnsignedCharComparisons;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SIGNEDCHARMISUSECHECK_H
