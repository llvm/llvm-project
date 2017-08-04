//===--- DeclRefExprUtils.cpp - clang-tidy---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DeclRefExprUtils.h"
#include "Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace tidy {
namespace utils {
namespace decl_ref_expr {

using namespace ::clang::ast_matchers;
using llvm::SmallPtrSet;

namespace {

template <typename S> bool isSetDifferenceEmpty(const S &S1, const S &S2) {
  for (const auto &E : S1)
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
  auto ConstMethodCallee = callee(cxxMethodDecl(isConst()));
  // Match method call expressions where the variable is referenced as the this
  // implicit object argument and opertor call expression for member operators
  // where the variable is the 0-th argument.
  auto Matches = match(
      findAll(expr(anyOf(cxxMemberCallExpr(ConstMethodCallee, on(DeclRefToVar)),
                         cxxOperatorCallExpr(ConstMethodCallee,
                                             hasArgument(0, DeclRefToVar))))),
      Stmt, Context);
  SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  auto ConstReferenceOrValue =
      qualType(anyOf(referenceType(pointee(qualType(isConstQualified()))),
                     unless(anyOf(referenceType(), pointerType()))));
  auto UsedAsConstRefOrValueArg = forEachArgumentWithParam(
      DeclRefToVar, parmVarDecl(hasType(ConstReferenceOrValue)));
  Matches = match(findAll(callExpr(UsedAsConstRefOrValueArg)), Stmt, Context);
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  Matches =
      match(findAll(cxxConstructExpr(UsedAsConstRefOrValueArg)), Stmt, Context);
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  return DeclRefs;
}

// Finds all DeclRefExprs where a const method is called on VarDecl or VarDecl
// is the a const reference or value argument to a CallExpr or CXXConstructExpr.
SmallPtrSet<const DeclRefExpr *, 16>
constReferenceDeclRefExprs(const VarDecl &VarDecl, const Decl &Decl,
                           ASTContext &Context) {
  auto DeclRefToVar =
      declRefExpr(to(varDecl(equalsNode(&VarDecl)))).bind("declRef");
  auto ConstMethodCallee = callee(cxxMethodDecl(isConst()));
  // Match method call expressions where the variable is referenced as the this
  // implicit object argument and opertor call expression for member operators
  // where the variable is the 0-th argument.
  auto Matches =
      match(decl(forEachDescendant(expr(
                anyOf(cxxMemberCallExpr(ConstMethodCallee, on(DeclRefToVar)),
                      cxxOperatorCallExpr(ConstMethodCallee,
                                          hasArgument(0, DeclRefToVar)))))),
            Decl, Context);
  SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  auto ConstReferenceOrValue =
      qualType(anyOf(referenceType(pointee(qualType(isConstQualified()))),
                     unless(anyOf(referenceType(), pointerType()))));
  auto UsedAsConstRefOrValueArg = forEachArgumentWithParam(
      DeclRefToVar, parmVarDecl(hasType(ConstReferenceOrValue)));
  Matches = match(decl(forEachDescendant(callExpr(UsedAsConstRefOrValueArg))),
                  Decl, Context);
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  Matches =
      match(decl(forEachDescendant(cxxConstructExpr(UsedAsConstRefOrValueArg))),
            Decl, Context);
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

} // namespace decl_ref_expr
} // namespace utils
} // namespace tidy
} // namespace clang
