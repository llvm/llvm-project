//===--- UnnecessaryCopyInitialization.cpp - clang-tidy--------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnnecessaryCopyInitialization.h"

#include "../utils/DeclRefExprUtils.h"
#include "../utils/FixItHintUtils.h"
#include "../utils/Matchers.h"

namespace clang {
namespace tidy {
namespace performance {
namespace {

void recordFixes(const VarDecl &Var, ASTContext &Context,
                 DiagnosticBuilder &Diagnostic) {
  // Do not propose fixes in macros since we cannot place them correctly.
  if (Var.getLocation().isMacroID())
    return;

  Diagnostic << utils::fixit::changeVarDeclToReference(Var, Context);
  if (!Var.getType().isLocalConstQualified())
    Diagnostic << utils::fixit::changeVarDeclToConst(Var);
}

} // namespace


using namespace ::clang::ast_matchers;
using utils::decl_ref_expr::isOnlyUsedAsConst;

void UnnecessaryCopyInitialization::registerMatchers(MatchFinder *Finder) {
  auto ConstReference = referenceType(pointee(qualType(isConstQualified())));
  auto ConstOrConstReference =
      allOf(anyOf(ConstReference, isConstQualified()),
            unless(allOf(pointerType(), unless(pointerType(pointee(
                                            qualType(isConstQualified())))))));

  // Match method call expressions where the this argument is a const
  // type or const reference. This returned const reference is highly likely to
  // outlive the local const reference of the variable being declared.
  // The assumption is that the const reference being returned either points
  // to a global static variable or to a member of the called object.
  auto ConstRefReturningMethodCallOfConstParam = cxxMemberCallExpr(
      callee(cxxMethodDecl(returns(ConstReference))),
      on(declRefExpr(to(varDecl(hasType(qualType(ConstOrConstReference)))))));
  auto ConstRefReturningFunctionCall =
      callExpr(callee(functionDecl(returns(ConstReference))),
               unless(callee(cxxMethodDecl())));

  auto localVarCopiedFrom = [](const internal::Matcher<Expr> &CopyCtorArg) {
    return compoundStmt(
               forEachDescendant(
                   varDecl(hasLocalStorage(),
                           hasType(matchers::isExpensiveToCopy()),
                           hasInitializer(cxxConstructExpr(
                                              hasDeclaration(cxxConstructorDecl(
                                                  isCopyConstructor())),
                                              hasArgument(0, CopyCtorArg))
                                              .bind("ctorCall")))
                       .bind("newVarDecl")))
        .bind("blockStmt");
  };

  Finder->addMatcher(
      localVarCopiedFrom(anyOf(ConstRefReturningFunctionCall,
                               ConstRefReturningMethodCallOfConstParam)),
      this);

  Finder->addMatcher(localVarCopiedFrom(declRefExpr(
                         to(varDecl(hasLocalStorage()).bind("oldVarDecl")))),
                     this);
}

void UnnecessaryCopyInitialization::check(
    const MatchFinder::MatchResult &Result) {
  const auto *NewVar = Result.Nodes.getNodeAs<VarDecl>("newVarDecl");
  const auto *OldVar = Result.Nodes.getNodeAs<VarDecl>("oldVarDecl");
  const auto *BlockStmt = Result.Nodes.getNodeAs<Stmt>("blockStmt");
  const auto *CtorCall = Result.Nodes.getNodeAs<CXXConstructExpr>("ctorCall");

  // A constructor that looks like T(const T& t, bool arg = false) counts as a
  // copy only when it is called with default arguments for the arguments after
  // the first.
  for (unsigned int i = 1; i < CtorCall->getNumArgs(); ++i)
    if (!CtorCall->getArg(i)->isDefaultArgument())
      return;

  if (OldVar == nullptr) {
    handleCopyFromMethodReturn(*NewVar, *BlockStmt, *Result.Context);
  } else {
    handleCopyFromLocalVar(*NewVar, *OldVar, *BlockStmt, *Result.Context);
  }
}

void UnnecessaryCopyInitialization::handleCopyFromMethodReturn(
    const VarDecl &Var, const Stmt &BlockStmt, ASTContext &Context) {
  bool IsConstQualified = Var.getType().isConstQualified();
  if (!IsConstQualified && !isOnlyUsedAsConst(Var, BlockStmt, Context))
    return;

  auto Diagnostic =
      diag(Var.getLocation(),
           IsConstQualified ? "the const qualified variable %0 is "
                              "copy-constructed from a const reference; "
                              "consider making it a const reference"
                            : "the variable %0 is copy-constructed from a "
                              "const reference but is only used as const "
                              "reference; consider making it a const reference")
      << &Var;
  recordFixes(Var, Context, Diagnostic);
}

void UnnecessaryCopyInitialization::handleCopyFromLocalVar(
    const VarDecl &NewVar, const VarDecl &OldVar, const Stmt &BlockStmt,
    ASTContext &Context) {
  if (!isOnlyUsedAsConst(NewVar, BlockStmt, Context) ||
      !isOnlyUsedAsConst(OldVar, BlockStmt, Context))
    return;

  auto Diagnostic = diag(NewVar.getLocation(),
                         "local copy %0 of the variable %1 is never modified; "
                         "consider avoiding the copy")
                    << &NewVar << &OldVar;
  recordFixes(NewVar, Context, Diagnostic);
}

} // namespace performance
} // namespace tidy
} // namespace clang
