//===--- DeclRefExprUtils.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeclRefExprUtils.h"
#include "Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang::tidy::utils::decl_ref_expr {

using namespace ::clang::ast_matchers;
using llvm::SmallPtrSet;

namespace {

template <typename S> bool isSetDifferenceEmpty(const S &S1, const S &S2) {
  for (auto E : S1)
    if (S2.count(E) == 0)
      return false;
  return true;
}

// Extracts all Nodes keyed by ID from Matches and inserts them into Nodes.
template <typename Node>
void extractNodesByIdTo(ArrayRef<BoundNodes> Matches, StringRef ID,
                        SmallPtrSet<const Node *, 16> &Nodes) {
  for (const auto &Match : Matches)
    Nodes.insert(Match.getNodeAs<Node>(ID));
}

} // namespace

// Finds all DeclRefExprs where a const method is called on VarDecl or VarDecl
// is the a const reference or value argument to a CallExpr or CXXConstructExpr.
SmallPtrSet<const DeclRefExpr *, 16>
constReferenceDeclRefExprs(const VarDecl &VarDecl, const Stmt &Stmt,
                           ASTContext &Context) {
  auto DeclRefToVar =
      declRefExpr(to(varDecl(equalsNode(&VarDecl)))).bind("declRef");
  auto MemberExprOfVar = memberExpr(hasObjectExpression(DeclRefToVar));
  auto DeclRefToVarOrMemberExprOfVar =
      stmt(anyOf(DeclRefToVar, MemberExprOfVar));
  auto ConstMethodCallee = callee(cxxMethodDecl(isConst()));
  // Match method call expressions where the variable is referenced as the this
  // implicit object argument and operator call expression for member operators
  // where the variable is the 0-th argument.
  auto Matches = match(
      findAll(expr(anyOf(
          cxxMemberCallExpr(ConstMethodCallee,
                            on(DeclRefToVarOrMemberExprOfVar)),
          cxxOperatorCallExpr(ConstMethodCallee,
                              hasArgument(0, DeclRefToVarOrMemberExprOfVar))))),
      Stmt, Context);
  SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  auto ConstReferenceOrValue =
      qualType(anyOf(matchers::isReferenceToConst(),
                     unless(anyOf(referenceType(), pointerType(),
                                  substTemplateTypeParmType()))));
  auto ConstReferenceOrValueOrReplaced = qualType(anyOf(
      ConstReferenceOrValue,
      substTemplateTypeParmType(hasReplacementType(ConstReferenceOrValue))));
  auto UsedAsConstRefOrValueArg = forEachArgumentWithParam(
      DeclRefToVarOrMemberExprOfVar,
      parmVarDecl(hasType(ConstReferenceOrValueOrReplaced)));
  Matches = match(findAll(invocation(UsedAsConstRefOrValueArg)), Stmt, Context);
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  // References and pointers to const assignments.
  Matches = match(
      findAll(declStmt(has(varDecl(
          hasType(qualType(matchers::isReferenceToConst())),
          hasInitializer(ignoringImpCasts(DeclRefToVarOrMemberExprOfVar)))))),
      Stmt, Context);
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  Matches = match(findAll(declStmt(has(varDecl(
                      hasType(qualType(matchers::isPointerToConst())),
                      hasInitializer(ignoringImpCasts(unaryOperator(
                          hasOperatorName("&"),
                          hasUnaryOperand(DeclRefToVarOrMemberExprOfVar)))))))),
                  Stmt, Context);
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  return DeclRefs;
}

bool isOnlyUsedAsConst(const VarDecl &Var, const Stmt &Stmt,
                       ASTContext &Context) {
  // Collect all DeclRefExprs to the loop variable and all CallExprs and
  // CXXConstructExprs where the loop variable is used as argument to a const
  // reference parameter.
  // If the difference is empty it is safe for the loop variable to be a const
  // reference.
  auto AllDeclRefs = allDeclRefExprs(Var, Stmt, Context);
  auto ConstReferenceDeclRefs = constReferenceDeclRefExprs(Var, Stmt, Context);
  return isSetDifferenceEmpty(AllDeclRefs, ConstReferenceDeclRefs);
}

SmallPtrSet<const DeclRefExpr *, 16>
allDeclRefExprs(const VarDecl &VarDecl, const Stmt &Stmt, ASTContext &Context) {
  auto Matches = match(
      findAll(declRefExpr(to(varDecl(equalsNode(&VarDecl)))).bind("declRef")),
      Stmt, Context);
  SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  return DeclRefs;
}

SmallPtrSet<const DeclRefExpr *, 16>
allDeclRefExprs(const VarDecl &VarDecl, const Decl &Decl, ASTContext &Context) {
  auto Matches = match(
      decl(forEachDescendant(
          declRefExpr(to(varDecl(equalsNode(&VarDecl)))).bind("declRef"))),
      Decl, Context);
  SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  return DeclRefs;
}

bool isCopyConstructorArgument(const DeclRefExpr &DeclRef, const Decl &Decl,
                               ASTContext &Context) {
  auto UsedAsConstRefArg = forEachArgumentWithParam(
      declRefExpr(equalsNode(&DeclRef)),
      parmVarDecl(hasType(matchers::isReferenceToConst())));
  auto Matches = match(
      decl(hasDescendant(
          cxxConstructExpr(UsedAsConstRefArg, hasDeclaration(cxxConstructorDecl(
                                                  isCopyConstructor())))
              .bind("constructExpr"))),
      Decl, Context);
  return !Matches.empty();
}

bool isCopyAssignmentArgument(const DeclRefExpr &DeclRef, const Decl &Decl,
                              ASTContext &Context) {
  auto UsedAsConstRefArg = forEachArgumentWithParam(
      declRefExpr(equalsNode(&DeclRef)),
      parmVarDecl(hasType(matchers::isReferenceToConst())));
  auto Matches = match(
      decl(hasDescendant(
          cxxOperatorCallExpr(UsedAsConstRefArg, hasOverloadedOperatorName("="),
                              callee(cxxMethodDecl(isCopyAssignmentOperator())))
              .bind("operatorCallExpr"))),
      Decl, Context);
  return !Matches.empty();
}

} // namespace clang::tidy::utils::decl_ref_expr
