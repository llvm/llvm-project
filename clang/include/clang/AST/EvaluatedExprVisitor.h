//===--- EvaluatedExprVisitor.h - Evaluated expression visitor --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the EvaluatedExprVisitor class template, which visits
//  the potentially-evaluated subexpressions of a potentially-evaluated
//  expression.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_EVALUATEDEXPRVISITOR_H
#define LLVM_CLANG_AST_EVALUATEDEXPRVISITOR_H

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"

namespace clang {
  
class ASTContext;
  
/// \brief Given a potentially-evaluated expression, this visitor visits all
/// of its potentially-evaluated subexpressions, recursively.
template<typename ImplClass>
class EvaluatedExprVisitor : public StmtVisitor<ImplClass> {
  ASTContext &Context;
  
public:
  explicit EvaluatedExprVisitor(ASTContext &Context) : Context(Context) { }
  
  // Expressions that have no potentially-evaluated subexpressions (but may have
  // other sub-expressions).
  void VisitDeclRefExpr(DeclRefExpr *E) { }
  void VisitOffsetOfExpr(OffsetOfExpr *E) { }
  void VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E) { }
  void VisitExpressionTraitExpr(ExpressionTraitExpr *E) { }
  void VisitBlockExpr(BlockExpr *E) { }
  void VisitCXXUuidofExpr(CXXUuidofExpr *E) { }  
  void VisitCXXNoexceptExpr(CXXNoexceptExpr *E) { }
  
  void VisitMemberExpr(MemberExpr *E) {
    // Only the base matters.
    return this->Visit(E->getBase());
  }
  
  void VisitChooseExpr(ChooseExpr *E) {
    // Only the selected subexpression matters; the other one is not evaluated.
    return this->Visit(E->getChosenSubExpr(Context));
  }
                 
  void VisitDesignatedInitExpr(DesignatedInitExpr *E) {
    // Only the actual initializer matters; the designators are all constant
    // expressions.
    return this->Visit(E->getInit());
  }
  
  void VisitCXXTypeidExpr(CXXTypeidExpr *E) {
    // typeid(expression) is potentially evaluated when the argument is
    // a glvalue of polymorphic type. (C++ 5.2.8p2-3)
    if (!E->isTypeOperand() && E->Classify(Context).isGLValue())
      if (const RecordType *Record 
                 = E->getExprOperand()->getType()->template getAs<RecordType>())
        if (cast<CXXRecordDecl>(Record->getDecl())->isPolymorphic())
          return this->Visit(E->getExprOperand());
  }
  
  /// \brief The basis case walks all of the children of the statement or
  /// expression, assuming they are all potentially evaluated.
  void VisitStmt(Stmt *S) {
    for (Stmt::child_range C = S->children(); C; ++C)
      if (*C)
        this->Visit(*C);
  }
};

}

#endif // LLVM_CLANG_AST_EVALUATEDEXPRVISITOR_H
