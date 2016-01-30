//== PseudoConstantAnalysis.cpp - Find Pseudoconstants in the AST-*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tracks the usage of variables in a Decl body to see if they are
// never written to, implying that they constant. This is useful in static
// analysis to see if a developer might have intended a variable to be const.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/PseudoConstantAnalysis.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <deque>

using namespace clang;

typedef llvm::SmallPtrSet<const VarDecl*, 32> VarDeclSet;

PseudoConstantAnalysis::PseudoConstantAnalysis(const Stmt *DeclBody) :
      DeclBody(DeclBody), Analyzed(false) {
  NonConstantsImpl = new VarDeclSet;
  UsedVarsImpl = new VarDeclSet;
}

PseudoConstantAnalysis::~PseudoConstantAnalysis() {
  delete (VarDeclSet*)NonConstantsImpl;
  delete (VarDeclSet*)UsedVarsImpl;
}

// Returns true if the given ValueDecl is never written to in the given DeclBody
bool PseudoConstantAnalysis::isPseudoConstant(const VarDecl *VD) {
  // Only local and static variables can be pseudoconstants
  if (!VD->hasLocalStorage() && !VD->isStaticLocal())
    return false;

  if (!Analyzed) {
    RunAnalysis();
    Analyzed = true;
  }

  VarDeclSet *NonConstants = (VarDeclSet*)NonConstantsImpl;

  return !NonConstants->count(VD);
}

// Returns true if the variable was used (self assignments don't count)
bool PseudoConstantAnalysis::wasReferenced(const VarDecl *VD) {
  if (!Analyzed) {
    RunAnalysis();
    Analyzed = true;
  }

  VarDeclSet *UsedVars = (VarDeclSet*)UsedVarsImpl;

  return UsedVars->count(VD);
}

// Returns a Decl from a (Block)DeclRefExpr (if any)
const Decl *PseudoConstantAnalysis::getDecl(const Expr *E) {
  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(E))
    return DR->getDecl();
  else
    return nullptr;
}

void PseudoConstantAnalysis::RunAnalysis() {
  std::deque<const Stmt *> WorkList;
  VarDeclSet *NonConstants = (VarDeclSet*)NonConstantsImpl;
  VarDeclSet *UsedVars = (VarDeclSet*)UsedVarsImpl;

  // Start with the top level statement of the function
  WorkList.push_back(DeclBody);

  while (!WorkList.empty()) {
    const Stmt *Head = WorkList.front();
    WorkList.pop_front();

    if (const Expr *Ex = dyn_cast<Expr>(Head))
      Head = Ex->IgnoreParenCasts();

    switch (Head->getStmtClass()) {
    // Case 1: Assignment operators modifying VarDecls
    case Stmt::BinaryOperatorClass: {
      const BinaryOperator *BO = cast<BinaryOperator>(Head);
      // Look for a Decl on the LHS
      const Decl *LHSDecl = getDecl(BO->getLHS()->IgnoreParenCasts());
      if (!LHSDecl)
        break;

      // We found a binary operator with a DeclRefExpr on the LHS. We now check
      // for any of the assignment operators, implying that this Decl is being
      // written to.
      switch (BO->getOpcode()) {
      // Self-assignments don't count as use of a variable
      case BO_Assign: {
        // Look for a DeclRef on the RHS
        const Decl *RHSDecl = getDecl(BO->getRHS()->IgnoreParenCasts());

        // If the Decls match, we have self-assignment
        if (LHSDecl == RHSDecl)
          // Do not visit the children
          continue;

      }
      case BO_AddAssign:
      case BO_SubAssign:
      case BO_MulAssign:
      case BO_DivAssign:
      case BO_AndAssign:
      case BO_OrAssign:
      case BO_XorAssign:
      case BO_ShlAssign:
      case BO_ShrAssign: {
        const VarDecl *VD = dyn_cast<VarDecl>(LHSDecl);
        // The DeclRefExpr is being assigned to - mark it as non-constant
        if (VD)
          NonConstants->insert(VD);
        break;
      }

      default:
        break;
      }
      break;
    }

    // Case 2: Pre/post increment/decrement and address of
    case Stmt::UnaryOperatorClass: {
      const UnaryOperator *UO = cast<UnaryOperator>(Head);

      // Look for a DeclRef in the subexpression
      const Decl *D = getDecl(UO->getSubExpr()->IgnoreParenCasts());
      if (!D)
        break;

      // We found a unary operator with a DeclRef as a subexpression. We now
      // check for any of the increment/decrement operators, as well as
      // addressOf.
      switch (UO->getOpcode()) {
      case UO_PostDec:
      case UO_PostInc:
      case UO_PreDec:
      case UO_PreInc:
        // The DeclRef is being changed - mark it as non-constant
      case UO_AddrOf: {
        // If we are taking the address of the DeclRefExpr, assume it is
        // non-constant.
        const VarDecl *VD = dyn_cast<VarDecl>(D);
        if (VD)
          NonConstants->insert(VD);
        break;
      }

      default:
        break;
      }
      break;
    }

    // Case 3: Reference Declarations
    case Stmt::DeclStmtClass: {
      const DeclStmt *DS = cast<DeclStmt>(Head);
      // Iterate over each decl and see if any of them contain reference decls
      for (const auto *I : DS->decls()) {
        // We only care about VarDecls
        const VarDecl *VD = dyn_cast<VarDecl>(I);
        if (!VD)
          continue;

        // We found a VarDecl; make sure it is a reference type
        if (!VD->getType().getTypePtr()->isReferenceType())
          continue;

        // Try to find a Decl in the initializer
        const Decl *D = getDecl(VD->getInit()->IgnoreParenCasts());
        if (!D)
          break;

        // If the reference is to another var, add the var to the non-constant
        // list
        if (const VarDecl *RefVD = dyn_cast<VarDecl>(D)) {
          NonConstants->insert(RefVD);
          continue;
        }
      }
      break;
    }

    // Case 4: Variable references
    case Stmt::DeclRefExprClass: {
      const DeclRefExpr *DR = cast<DeclRefExpr>(Head);
      if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
        // Add the Decl to the used list
        UsedVars->insert(VD);
        continue;
      }
      break;
    }

    // Case 5: Block expressions
    case Stmt::BlockExprClass: {
      const BlockExpr *B = cast<BlockExpr>(Head);
      // Add the body of the block to the list
      WorkList.push_back(B->getBody());
      continue;
    }

    default:
      break;
    } // switch (head->getStmtClass())

    // Add all substatements to the worklist
    for (const Stmt *SubStmt : Head->children())
      if (SubStmt)
        WorkList.push_back(SubStmt);
  } // while (!WorkList.empty())
}
