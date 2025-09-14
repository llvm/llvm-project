//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NON_CONST_PARAMETER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NON_CONST_PARAMETER_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Warn when a pointer function parameter can be const.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/non-const-parameter.html
class NonConstParameterCheck : public ClangTidyCheck {
public:
  NonConstParameterCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  /// Parameter info.
  struct ParmInfo {
    /// Is function parameter referenced?
    bool IsReferenced;

    /// Can function parameter be const?
    bool CanBeConst;
  };

  /// Track all nonconst integer/float parameters.
  std::map<const ParmVarDecl *, ParmInfo> Parameters;

  /// Add function parameter.
  void addParm(const ParmVarDecl *Parm);

  /// Set IsReferenced.
  void setReferenced(const DeclRefExpr *Ref);

  /// Set CanNotBeConst.
  /// Visits sub expressions recursively. If a DeclRefExpr is found
  /// and CanNotBeConst is true the Parameter is marked as not-const.
  /// The CanNotBeConst is updated as sub expressions are visited.
  void markCanNotBeConst(const Expr *E, bool CanNotBeConst);

  /// Diagnose non const parameters.
  void diagnoseNonConstParameters();
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NON_CONST_PARAMETER_H
