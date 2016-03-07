//===--- DeclRefExprUtils.cpp - clang-tidy---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DeclRefExprUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace clang {
namespace tidy {
namespace decl_ref_expr_utils {

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

// Finds all DeclRefExprs to VarDecl in Stmt.
SmallPtrSet<const DeclRefExpr *, 16>
declRefExprs(const VarDecl &VarDecl, const Stmt &Stmt, ASTContext &Context) {
  auto Matches = match(
      findAll(declRefExpr(to(varDecl(equalsNode(&VarDecl)))).bind("declRef")),
      Stmt, Context);
  SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  return DeclRefs;
}

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

} // namespace

bool isOnlyUsedAsConst(const VarDecl &Var, const Stmt &Stmt,
                       ASTContext &Context) {
  // Collect all DeclRefExprs to the loop variable and all CallExprs and
  // CXXConstructExprs where the loop variable is used as argument to a const
  // reference parameter.
  // If the difference is empty it is safe for the loop variable to be a const
  // reference.
  auto AllDeclRefs = declRefExprs(Var, Stmt, Context);
  auto ConstReferenceDeclRefs = constReferenceDeclRefExprs(Var, Stmt, Context);
  return isSetDifferenceEmpty(AllDeclRefs, ConstReferenceDeclRefs);
}

} // namespace decl_ref_expr_utils
} // namespace tidy
} // namespace clang
