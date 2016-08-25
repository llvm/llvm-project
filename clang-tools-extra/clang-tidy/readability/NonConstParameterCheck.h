//===--- NonConstParameterCheck.h - clang-tidy-------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NON_CONST_PARAMETER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NON_CONST_PARAMETER_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// Warn when a pointer function parameter can be const.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-non-const-parameter.html
class NonConstParameterCheck : public ClangTidyCheck {
public:
  NonConstParameterCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;

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

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NON_CONST_PARAMETER_H
