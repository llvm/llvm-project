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

using namespace ::clang::ast_matchers;

void UnnecessaryCopyInitialization::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
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
  Finder->addMatcher(
      compoundStmt(
          forEachDescendant(
              varDecl(
                  hasLocalStorage(), hasType(matchers::isExpensiveToCopy()),
                  hasInitializer(cxxConstructExpr(
                      hasDeclaration(cxxConstructorDecl(isCopyConstructor())),
                      hasArgument(
                          0, anyOf(ConstRefReturningFunctionCall,
                                   ConstRefReturningMethodCallOfConstParam)))))
                  .bind("varDecl"))).bind("blockStmt"),
      this);
}

void UnnecessaryCopyInitialization::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto *Var = Result.Nodes.getNodeAs<VarDecl>("varDecl");
  const auto *BlockStmt = Result.Nodes.getNodeAs<Stmt>("blockStmt");
  bool IsConstQualified = Var->getType().isConstQualified();
  if (!IsConstQualified &&
      !decl_ref_expr_utils::isOnlyUsedAsConst(*Var, *BlockStmt,
                                              *Result.Context))
    return;
  DiagnosticBuilder Diagnostic =
      diag(Var->getLocation(),
           IsConstQualified ? "the const qualified variable '%0' is "
                              "copy-constructed from a const reference; "
                              "consider making it a const reference"
                            : "the variable '%0' is copy-constructed from a "
                              "const reference but is only used as const "
                              "reference; consider making it a const reference")
      << Var->getName();
  // Do not propose fixes in macros since we cannot place them correctly.
  if (Var->getLocStart().isMacroID())
    return;
  Diagnostic << utils::create_fix_it::changeVarDeclToReference(*Var,
                                                               *Result.Context);
  if (!IsConstQualified)
    Diagnostic << utils::create_fix_it::changeVarDeclToConst(*Var);
}

} // namespace performance
} // namespace tidy
} // namespace clang
