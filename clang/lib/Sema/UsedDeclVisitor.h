//===- UsedDeclVisitor.h - ODR-used declarations visitor --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
//  This file defines UsedDeclVisitor, a CRTP class which visits all the
//  declarations that are ODR-used by an expression or statement.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_SEMA_USEDDECLVISITOR_H
#define LLVM_CLANG_LIB_SEMA_USEDDECLVISITOR_H

#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/Sema/SemaInternal.h"

namespace clang {
template <class Derived>
class UsedDeclVisitor : public EvaluatedExprVisitor<Derived> {
protected:
  Sema &S;

public:
  typedef EvaluatedExprVisitor<Derived> Inherited;

  UsedDeclVisitor(Sema &S) : Inherited(S.Context), S(S) {}

  Derived &asImpl() { return *static_cast<Derived *>(this); }

  void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
    asImpl().visitUsedDecl(
        E->getBeginLoc(),
        const_cast<CXXDestructorDecl *>(E->getTemporary()->getDestructor()));
    asImpl().Visit(E->getSubExpr());
  }

  void VisitCXXNewExpr(CXXNewExpr *E) {
    if (E->getOperatorNew())
      asImpl().visitUsedDecl(E->getBeginLoc(), E->getOperatorNew());
    if (E->getOperatorDelete())
      asImpl().visitUsedDecl(E->getBeginLoc(), E->getOperatorDelete());
    Inherited::VisitCXXNewExpr(E);
  }

  void VisitCXXDeleteExpr(CXXDeleteExpr *E) {
    if (E->getOperatorDelete())
      asImpl().visitUsedDecl(E->getBeginLoc(), E->getOperatorDelete());
    QualType Destroyed = S.Context.getBaseElementType(E->getDestroyedType());
    if (const RecordType *DestroyedRec = Destroyed->getAs<RecordType>()) {
      CXXRecordDecl *Record = cast<CXXRecordDecl>(DestroyedRec->getDecl());
      asImpl().visitUsedDecl(E->getBeginLoc(), S.LookupDestructor(Record));
    }

    Inherited::VisitCXXDeleteExpr(E);
  }

  void VisitCXXConstructExpr(CXXConstructExpr *E) {
    asImpl().visitUsedDecl(E->getBeginLoc(), E->getConstructor());
    Inherited::VisitCXXConstructExpr(E);
  }

  void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
    asImpl().Visit(E->getExpr());
  }
};
} // end namespace clang

#endif // LLVM_CLANG_LIB_SEMA_USEDDECLVISITOR_H
