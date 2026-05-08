//===---------- SubobjectVisitor.h - Subobject Visitor ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the SubobjectVisitor interface, which recursively
//  traverses subobjects within a type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_SUBOBJECTVISITOR_H
#define LLVM_CLANG_AST_SUBOBJECTVISITOR_H

#include "clang/AST/Type.h"

namespace clang {
template <typename Derived> class SubobjectVisitor {
  ASTContext &Ctx;

  public:
  SubobjectVisitor(ASTContext &Ctx) : Ctx(Ctx) {}
  /// Return a reference to the derived class.
  Derived &getDerived() { return *static_cast<Derived *>(this); }

  bool enterRecord(CXXRecordDecl *Record, FieldDecl *Parent) {
    return getDerived().visitRecord(Record, Parent);
  }

  bool leaveRecord(CXXRecordDecl *Record, FieldDecl *Parent) {
    return true;
  }

  bool enterUnion(CXXRecordDecl *Record, FieldDecl *Parent) {
    return getDerived().visitUnion(Record, Parent);
  }

  bool enterArray(QualType ArrayTy, FieldDecl *Parent) {
    return getDerived().visitArrayTy(ArrayTy, Parent);
  }

  bool visitRecord(CXXRecordDecl *Record, FieldDecl *Parent) {
    return true;
  }

  bool visitUnion(CXXRecordDecl *Record, FieldDecl *Parent) {
    return true;
  }

  bool visitReferenceType(QualType Ty, FieldDecl *Parent) {
    return true;
  }

  bool visitPointerType(QualType Ty, FieldDecl *Parent) {
    return true;
  }

  bool visitScalarType(QualType Ty, FieldDecl *Parent) {
    return true;
  }

  bool visitOtherType(QualType Ty, FieldDecl *Parent) {
    return true;
  }

  bool visitArrayTy(QualType Ty, FieldDecl *Parent) {
    return true;
  }

  bool traverseRecord(CXXRecordDecl *Record) {
    FieldDecl* Parent = nullptr;
    getDerived().traverseRecord(Record, Parent);
    return true;
  }

  bool traverseRecord(CXXRecordDecl *Record, FieldDecl *Parent) {
    getDerived().enterRecord(Record, Parent);

    for (const auto &Base : Record->bases()) {
      QualType BaseTy = Base.getType();
      getDerived().traverseType(BaseTy, Parent);
    }
    for (const auto Field : Record->fields()) {
      QualType FieldTy = Field->getType();
      getDerived().traverseType(FieldTy, Field);
    }

    getDerived().leaveRecord(Record, Parent);
    return true;
  }

  bool traverseUnion(CXXRecordDecl *Record, FieldDecl *Parent) {
    getDerived().enterUnion(Record, Parent);

    for (const auto Field : Record->fields()) {
      QualType FieldTy = Field->getType();
      getDerived().traverseType(FieldTy, Field);
    }
    return true;
  }

  bool traverseArray(QualType Ty, FieldDecl *Parent) {
    getDerived().enterArray(Ty, Parent);
    const ConstantArrayType *CAT = Ctx.getAsConstantArrayType(Ty);
    if (!CAT)
      return true;

    QualType ET = CAT->getElementType();
    uint64_t ElemCount = CAT->getSize().getZExtValue();
    for (uint64_t Index = 0; Index < ElemCount; ++Index)
      getDerived().traverseType(ET, Parent);

    return true;
  }

  bool traverseType(QualType Ty, FieldDecl *Parent) {
    if (Ty->isStructureOrClassType()) {
      CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
      getDerived().traverseRecord(RD, Parent);
    } else if (Ty->isUnionType()) {
      CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
      getDerived().traverseUnion(RD, Parent);
    } else if (Ty->isReferenceType())
      getDerived().visitReferenceType(Ty, Parent);
    else if (Ty->isPointerType())
      getDerived().visitPointerType(Ty, Parent);
    else if (Ty->isArrayType())
      getDerived().traverseArray(Ty, Parent);
    else if (Ty->isScalarType() || Ty->isVectorType())
      getDerived().visitScalarType(Ty, Parent);
    else
      getDerived().visitOtherType(Ty, Parent);
    return true;
  }

};
} // end namespace clang
#endif // LLVM_CLANG_AST_SUBOBJECTVISITOR
