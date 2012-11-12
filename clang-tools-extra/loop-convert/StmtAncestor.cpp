//===-- loop-convert/StmtAncestor.cpp - AST property visitors ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of several RecursiveASTVisitors used to
// build and check data structures used in loop migration.
//
//===----------------------------------------------------------------------===//
#include "StmtAncestor.h"

namespace clang {
namespace loop_migrate {

/// \brief Tracks a stack of parent statements during traversal.
///
/// All this really does is inject push_back() before running
/// RecursiveASTVisitor::TraverseStmt() and pop_back() afterwards. The Stmt atop
/// the stack is the parent of the current statement (NULL for the topmost
/// statement).
bool StmtAncestorASTVisitor::TraverseStmt(Stmt *Statement) {
  StmtAncestors.insert(std::make_pair(Statement, StmtStack.back()));
  StmtStack.push_back(Statement);
  RecursiveASTVisitor<StmtAncestorASTVisitor>::TraverseStmt(Statement);
  StmtStack.pop_back();
  return true;
}

/// \brief Keep track of the DeclStmt associated with each VarDecl.
///
/// Combined with StmtAncestors, this provides roughly the same information as
/// Scope, as we can map a VarDecl to its DeclStmt, then walk up the parent tree
/// using StmtAncestors.
bool StmtAncestorASTVisitor::VisitDeclStmt(DeclStmt *Decls) {
  for (DeclStmt::const_decl_iterator I = Decls->decl_begin(),
                                     E = Decls->decl_end(); I != E; ++I)
    if (const VarDecl *VD = dyn_cast<VarDecl>(*I))
      DeclParents.insert(std::make_pair(VD, Decls));
  return true;
}

/// \brief record the DeclRefExpr as part of the parent expression.
bool ComponentFinderASTVisitor::VisitDeclRefExpr(DeclRefExpr *E) {
  Components.push_back(E);
  return true;
}

/// \brief record the MemberExpr as part of the parent expression.
bool ComponentFinderASTVisitor::VisitMemberExpr(MemberExpr *Member) {
  Components.push_back(Member);
  return true;
}

/// \brief Forward any DeclRefExprs to a check on the referenced variable
/// declaration.
bool DependencyFinderASTVisitor::VisitDeclRefExpr(DeclRefExpr *DRE) {
  if (VarDecl *VD = dyn_cast_or_null<VarDecl>(DRE->getDecl()))
    return VisitVarDecl(VD);
  return true;
}

/// \brief Determine if any this variable is declared inside the ContainingStmt.
bool DependencyFinderASTVisitor::VisitVarDecl(VarDecl *VD) {
  const Stmt *Curr = DeclParents->lookup(VD);
  // First, see if the variable was declared within an inner scope of the loop.
  while (Curr != NULL) {
    if (Curr == ContainingStmt) {
      DependsOnInsideVariable = true;
      return false;
    }
    Curr = StmtParents->lookup(Curr);
  }

  // Next, check if the variable was removed from existence by an earlier
  // iteration.
  for (ReplacedVarsMap::const_iterator I = ReplacedVars->begin(),
                                       E = ReplacedVars->end(); I != E; ++I)
    if ((*I).second == VD) {
      DependsOnInsideVariable = true;
      return false;
    }
  return true;
}

/// \brief If we already created a variable for TheLoop, check to make sure
/// that the name was not already taken.
bool DeclFinderASTVisitor::VisitForStmt(ForStmt *TheLoop) {
  StmtGeneratedVarNameMap::const_iterator I = GeneratedDecls->find(TheLoop);
  if (I != GeneratedDecls->end() && I->second == Name) {
    Found = true;
    return false;
  }
  return true;
}

/// \brief If any named declaration within the AST subtree has the same name,
/// then consider Name already taken.
bool DeclFinderASTVisitor::VisitNamedDecl(NamedDecl *ND) {
  const IdentifierInfo *Ident = ND->getIdentifier();
  if (Ident && Ident->getName() == Name) {
    Found = true;
    return false;
  }
  return true;
}

/// \brief Forward any declaration references to the actual check on the
/// referenced declaration.
bool DeclFinderASTVisitor::VisitDeclRefExpr(DeclRefExpr *DRE) {
  if (NamedDecl *ND = dyn_cast<NamedDecl>(DRE->getDecl()))
    return VisitNamedDecl(ND);
  return true;
}

} // namespace clang
} // namespace for_migrate
