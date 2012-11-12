//===--- CFGStmtVisitor.h - Visitor for Stmts in a CFG ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CFGStmtVisitor interface, which extends
//  StmtVisitor.  This interface is useful for visiting statements in a CFG
//  where some statements have implicit control-flow and thus should
//  be treated specially.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_CFGSTMTVISITOR_H
#define LLVM_CLANG_ANALYSIS_CFGSTMTVISITOR_H

#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/CFG.h"

namespace clang {

#define DISPATCH_CASE(CLASS) \
case Stmt::CLASS ## Class: return \
static_cast<ImplClass*>(this)->BlockStmt_Visit ## CLASS(static_cast<CLASS*>(S));

#define DEFAULT_BLOCKSTMT_VISIT(CLASS) RetTy BlockStmt_Visit ## CLASS(CLASS *S)\
{ return\
  static_cast<ImplClass*>(this)->BlockStmt_VisitImplicitControlFlowExpr(\
  cast<Expr>(S)); }

template <typename ImplClass, typename RetTy=void>
class CFGStmtVisitor : public StmtVisitor<ImplClass,RetTy> {
  Stmt *CurrentBlkStmt;

  struct NullifyStmt {
    Stmt*& S;

    NullifyStmt(Stmt*& s) : S(s) {}
    ~NullifyStmt() { S = NULL; }
  };

public:
  CFGStmtVisitor() : CurrentBlkStmt(NULL) {}

  Stmt *getCurrentBlkStmt() const { return CurrentBlkStmt; }

  RetTy Visit(Stmt *S) {
    if (S == CurrentBlkStmt ||
        !static_cast<ImplClass*>(this)->getCFG().isBlkExpr(S))
      return StmtVisitor<ImplClass,RetTy>::Visit(S);
    else
      return RetTy();
  }
  
  /// VisitConditionVariableInit - Handle the initialization of condition
  ///  variables at branches.  Valid statements include IfStmt, ForStmt,
  ///  WhileStmt, and SwitchStmt.
  RetTy VisitConditionVariableInit(Stmt *S) {
    return RetTy();
  }

  /// BlockVisit_XXX - Visitor methods for visiting the "root" statements in
  /// CFGBlocks.  Root statements are the statements that appear explicitly in
  /// the list of statements in a CFGBlock.  For substatements, or when there
  /// is no implementation provided for a BlockStmt_XXX method, we default
  /// to using StmtVisitor's Visit method.
  RetTy BlockStmt_Visit(Stmt *S) {
    CurrentBlkStmt = S;
    NullifyStmt cleanup(CurrentBlkStmt);

    switch (S->getStmtClass()) {
      case Stmt::IfStmtClass:
      case Stmt::ForStmtClass:
      case Stmt::WhileStmtClass:
      case Stmt::SwitchStmtClass:
        return static_cast<ImplClass*>(this)->VisitConditionVariableInit(S);

      DISPATCH_CASE(StmtExpr)
      DISPATCH_CASE(ConditionalOperator)
      DISPATCH_CASE(BinaryConditionalOperator)
      DISPATCH_CASE(ObjCForCollectionStmt)
      DISPATCH_CASE(CXXForRangeStmt)

      case Stmt::BinaryOperatorClass: {
        BinaryOperator* B = cast<BinaryOperator>(S);
        if (B->isLogicalOp())
          return static_cast<ImplClass*>(this)->BlockStmt_VisitLogicalOp(B);
        else if (B->getOpcode() == BO_Comma)
          return static_cast<ImplClass*>(this)->BlockStmt_VisitComma(B);
        // Fall through.
      }

      default:
        if (isa<Expr>(S))
          return
            static_cast<ImplClass*>(this)->BlockStmt_VisitExpr(cast<Expr>(S));
        else
          return static_cast<ImplClass*>(this)->BlockStmt_VisitStmt(S);
    }
  }

  DEFAULT_BLOCKSTMT_VISIT(StmtExpr)
  DEFAULT_BLOCKSTMT_VISIT(ConditionalOperator)
  DEFAULT_BLOCKSTMT_VISIT(BinaryConditionalOperator)

  RetTy BlockStmt_VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
    return static_cast<ImplClass*>(this)->BlockStmt_VisitStmt(S);
  }

  RetTy BlockStmt_VisitCXXForRangeStmt(CXXForRangeStmt *S) {
    return static_cast<ImplClass*>(this)->BlockStmt_VisitStmt(S);
  }

  RetTy BlockStmt_VisitImplicitControlFlowExpr(Expr *E) {
    return static_cast<ImplClass*>(this)->BlockStmt_VisitExpr(E);
  }

  RetTy BlockStmt_VisitExpr(Expr *E) {
    return static_cast<ImplClass*>(this)->BlockStmt_VisitStmt(E);
  }

  RetTy BlockStmt_VisitStmt(Stmt *S) {
    return static_cast<ImplClass*>(this)->Visit(S);
  }

  RetTy BlockStmt_VisitLogicalOp(BinaryOperator* B) {
    return
     static_cast<ImplClass*>(this)->BlockStmt_VisitImplicitControlFlowExpr(B);
  }

  RetTy BlockStmt_VisitComma(BinaryOperator* B) {
    return
     static_cast<ImplClass*>(this)->BlockStmt_VisitImplicitControlFlowExpr(B);
  }

  //===--------------------------------------------------------------------===//
  // Utility methods.  Not called by default (but subclasses may use them).
  //===--------------------------------------------------------------------===//

  /// VisitChildren: Call "Visit" on each child of S.
  void VisitChildren(Stmt *S) {

    switch (S->getStmtClass()) {
      default:
        break;

      case Stmt::StmtExprClass: {
        CompoundStmt *CS = cast<StmtExpr>(S)->getSubStmt();
        if (CS->body_empty()) return;
        static_cast<ImplClass*>(this)->Visit(CS->body_back());
        return;
      }

      case Stmt::BinaryOperatorClass: {
        BinaryOperator* B = cast<BinaryOperator>(S);
        if (B->getOpcode() != BO_Comma) break;
        static_cast<ImplClass*>(this)->Visit(B->getRHS());
        return;
      }
    }

    for (Stmt::child_range I = S->children(); I; ++I)
      if (*I) static_cast<ImplClass*>(this)->Visit(*I);
  }
};

#undef DEFAULT_BLOCKSTMT_VISIT
#undef DISPATCH_CASE

}  // end namespace clang

#endif
