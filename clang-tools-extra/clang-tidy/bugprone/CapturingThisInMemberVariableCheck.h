//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_CAPTURINGTHISINMEMBERVARIABLECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_CAPTURINGTHISINMEMBERVARIABLECHECK_H

#include "../ClangTidyCheck.h"
#include "clang/AST/ASTTypeTraits.h"
#include <optional>

namespace clang::tidy::bugprone {

/// Finds lambda captures that capture the ``this`` pointer and store it as
/// class members without handle the copy and move constructors and the
/// assignments.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/capturing-this-in-member-variable.html
class CapturingThisInMemberVariableCheck : public ClangTidyCheck {
public:
  CapturingThisInMemberVariableCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TraversalKind::TK_IgnoreUnlessSpelledInSource;
  }

private:
  ///< store the function wrapper types
  const std::vector<StringRef> FunctionWrapperTypes;
  const std::vector<StringRef> BindFunctions;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_CAPTURINGTHISINMEMBERVARIABLECHECK_H
