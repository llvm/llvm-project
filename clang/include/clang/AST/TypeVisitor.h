//===--- TypeVisitor.h - Visitor for Type subclasses ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TypeVisitor interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TYPEVISITOR_H
#define LLVM_CLANG_AST_TYPEVISITOR_H

#include "clang/AST/Type.h"

namespace clang {

#define DISPATCH(CLASS) \
  return static_cast<ImplClass*>(this)-> \
           Visit##CLASS(static_cast<const CLASS*>(T))

template<typename ImplClass, typename RetTy=void>
class TypeVisitor {
public:
  RetTy Visit(const Type *T) {
    // Top switch stmt: dispatch to VisitFooType for each FooType.
    switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) case Type::CLASS: DISPATCH(CLASS##Type);
#include "clang/AST/TypeNodes.def"
    }
    llvm_unreachable("Unknown type class!");
  }

  // If the implementation chooses not to implement a certain visit method, fall
  // back on superclass.
#define TYPE(CLASS, PARENT) RetTy Visit##CLASS##Type(const CLASS##Type *T) { \
  DISPATCH(PARENT);                                                          \
}
#include "clang/AST/TypeNodes.def"

  // Base case, ignore it. :)
  RetTy VisitType(const Type*) { return RetTy(); }
};

#undef DISPATCH

}  // end namespace clang

#endif
