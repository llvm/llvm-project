//==- CFGRecStmtVisitor - Recursive visitor of CFG statements ---*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the template class CFGRecStmtVisitor, which extends
// CFGStmtVisitor by implementing a default recursive visit of all statements.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_CFG_REC_STMT_VISITOR_H
#define LLVM_CLANG_ANALYSIS_CFG_REC_STMT_VISITOR_H

#include "clang/Analysis/Visitors/CFGStmtVisitor.h"

namespace clang {
template <typename ImplClass>
class CFGRecStmtVisitor : public CFGStmtVisitor<ImplClass,void> {
public:

  void VisitStmt(Stmt *S) {
    static_cast< ImplClass* >(this)->VisitChildren(S);
  }
  
  void VisitCompoundStmt(CompoundStmt *S) {
    // Do nothing.  Everything in a CompoundStmt is inlined
    // into the CFG.
  }
  
  void VisitConditionVariableInit(Stmt *S) {
    assert(S == this->getCurrentBlkStmt());
    VarDecl *CondVar = 0;
    switch (S->getStmtClass()) {
#define CONDVAR_CASE(CLASS) \
case Stmt::CLASS ## Class:\
CondVar = cast<CLASS>(S)->getConditionVariable();\
break;
        CONDVAR_CASE(IfStmt)
        CONDVAR_CASE(ForStmt)
        CONDVAR_CASE(SwitchStmt)
        CONDVAR_CASE(WhileStmt)
#undef CONDVAR_CASE
      default:
        llvm_unreachable("Infeasible");
    }    
    static_cast<ImplClass*>(this)->Visit(CondVar->getInit());
  }

  // Defining operator() allows the visitor to be used as a C++ style functor.
  void operator()(Stmt *S) { static_cast<ImplClass*>(this)->BlockStmt_Visit(S);}
};

} // end namespace clang

#endif
