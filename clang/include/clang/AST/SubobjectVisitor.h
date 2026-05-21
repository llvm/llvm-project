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

#define DISPATCH(CLASS)                                                        \
  return static_cast<Derived *>(this)->visit##CLASS(                           \
      static_cast<const CLASS *>(T))

template <template <typename> class Ptr, typename Derived>
class SubobjectVisitorBase {
  ASTContext &Ctx;
  template <typename Class> using ptr_t = typename Ptr<Class>::type;
  template <typename Class>
  using non_ptr_t = typename std::remove_pointer<ptr_t<Class>>::type;

public:
  SubobjectVisitorBase(ASTContext &Ctx) : Ctx(Ctx) {}

  /// Return a reference to the derived class.
  Derived &getDerived() { return *static_cast<Derived *>(this); }

  void visit(QualType QT) {
    // If the type is an array, visit its element type. Separate traversal of
    // arrays is not needed because the array will be encountered as a
    // FieldDecl.

    if (QT->isArrayType()) {
      QualType ElTy =
          cast<ConstantArrayType>(Ctx.getAsArrayType(QT))->getElementType();
      visit(ElTy);
      return;
    }

    if (ptr_t<RecordDecl> RD = QT->getAsRecordDecl()) {
      traverseRecord(RD);
      return;
    }

    visitGenericType(QT.getCanonicalType().getTypePtr());
  }

  void traverseRecord(ptr_t<RecordDecl> RD) {
    if (ptr_t<CXXRecordDecl> CRD = dyn_cast<CXXRecordDecl>(RD)) {
      for (non_ptr_t<CXXBaseSpecifier> BS : CRD->bases()) {
        if (!static_cast<Derived *>(this)->visitBaseSpecifierPre(BS))
          continue;
        visit(BS.getType());
        static_cast<Derived *>(this)->visitBaseSpecifierPost(BS);
      }
    }
    for (ptr_t<FieldDecl> FD : RD->fields()) {
      if (!static_cast<Derived *>(this)->visitFieldDeclPre(FD))
        continue;
      visit(FD->getType());
      static_cast<Derived *>(this)->visitFieldDeclPost(FD);
    }
  }

  // Default base class specifier pre-order visitor.
  bool visitBaseSpecifierPre(non_ptr_t<CXXBaseSpecifier> BS) { return true; }

  // Default base class specifier post-order visitor.
  void visitBaseSpecifierPost(non_ptr_t<CXXBaseSpecifier> BS) {}

  // Default field pre-order visitor.
  bool visitFieldDeclPre(ptr_t<FieldDecl> FD) { return true; }

  // Default field post-order visitor.
  void visitFieldDeclPost(ptr_t<FieldDecl> FD) {}
  bool visitGenericType(const Type *T) {
    // Top switch stmt: dispatch to VisitFooType for each FooType.
    switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT)                                                    \
  case Type::CLASS:                                                            \
    DISPATCH(CLASS##Type);
#include "clang/AST/TypeNodes.inc"
    }
    llvm_unreachable("Unknown type class!");
  }

  // If the implementation chooses not to implement a certain visit method, fall
  // back on superclass.
#define TYPE(CLASS, PARENT)                                                    \
  bool visit##CLASS##Type(const CLASS##Type *T) { DISPATCH(PARENT); }
#include "clang/AST/TypeNodes.inc"

  /// Method called if \c ImpClass doesn't provide specific handler
  /// for some type class.
  bool visitType(const Type *) { return true; }
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
      : SubobjectVisitorBase<std::add_pointer, Derived>(Ctx) {}
};

#undef DISPATCH

} // end namespace clang

#endif // LLVM_CLANG_AST_SUBOBJECTVISITOR
