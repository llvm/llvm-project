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

/// Given a type, subobject visitors visit all subobjects of the type in depth
/// first order. Both pre-order and post-order visitation are performed so that
/// derived classes can maintain an access path to the visited elements.
/// Subobjects include all base classes and non-static data members, including
/// those that are not subobjects according to the C++standard like data
/// members with a reference type. Virtual base classes are visited each time
/// they appear in a class hierarchy despite there being only one actual
/// subobject present in an object of a most derived type. Array elements are
/// not individually visited; only their containing array is.
template <template <typename> class Ptr, typename Derived>
class SubobjectVisitorBase {
  ASTContext &Ctx;
  template <typename Class> using ptr_t = typename Ptr<Class>::type;

public:
  SubobjectVisitorBase(ASTContext &Ctx) : Ctx(Ctx) {}

  /// Return a reference to the derived class.
  Derived &getDerived() { return *static_cast<Derived *>(this); }

  void visit(QualType QT) {
    assert(!QT->isDependentType());
    QT = QT.getDesugaredType(Ctx);

    if (auto *ResAtomicType = QT->getAs<AtomicType>())
      QT = ResAtomicType->getValueType();

    // If the type is an array, visit its element type. Separate traversal of
    // arrays is not needed because the array will be encountered as a
    // FieldDecl.
    if (QT->isArrayType()) {
      QualType ElTy = Ctx.getAsArrayType(QT)->getElementType();
      getDerived().visit(ElTy);
      return;
    }

    if (ptr_t<RecordDecl> RD = QT->getAsRecordDecl()) {
      getDerived().traverseRecord(RD);
      return;
    }
  }

  void traverseRecord(ptr_t<RecordDecl> RD) {
    if (ptr_t<CXXRecordDecl> CRD = dyn_cast<CXXRecordDecl>(RD)) {
      for (auto &BS : CRD->bases()) {
        if (getDerived().visitBaseSpecifierPre(&BS))
          getDerived().visit(BS.getType());
        getDerived().visitBaseSpecifierPost(&BS);
      }
    }
    for (ptr_t<FieldDecl> FD : RD->fields()) {
      if (getDerived().visitFieldDeclPre(FD))
        getDerived().visit(FD->getType());
      getDerived().visitFieldDeclPost(FD);
    }
  }

  // Default base class specifier pre-order visitor.
  bool visitBaseSpecifierPre(ptr_t<CXXBaseSpecifier> BS) { return true; }

  // Default base class specifier post-order visitor.
  void visitBaseSpecifierPost(ptr_t<CXXBaseSpecifier> BS) {}

  // Default field pre-order visitor.
  bool visitFieldDeclPre(ptr_t<FieldDecl> FD) { return true; }

  // Default field post-order visitor.
  void visitFieldDeclPost(ptr_t<FieldDecl> FD) {}
};

template <typename Derived>
class SubobjectVisitor
    : public SubobjectVisitorBase<std::add_pointer, Derived> {
public:
  SubobjectVisitor(ASTContext &Ctx)
      : SubobjectVisitorBase<std::add_pointer, Derived>(Ctx) {}
};

template <typename Derived>
class ConstSubobjectVisitor
    : public SubobjectVisitorBase<llvm::make_const_ptr, Derived> {
public:
  ConstSubobjectVisitor(ASTContext &Ctx)
      : SubobjectVisitorBase<llvm::make_const_ptr, Derived>(Ctx) {}
};

} // end namespace clang

#endif // LLVM_CLANG_AST_SUBOBJECTVISITOR
