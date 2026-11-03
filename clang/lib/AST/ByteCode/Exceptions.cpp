//===-------------------------- Exceptions.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Exceptions.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"

using namespace clang;
using namespace clang::interp;

static bool isPointerOrMemberPointerType(const Type *T) {
  return T->isPointerType() || T->isMemberPointerType();
}
static bool isPointerOrMemberPointerType(QualType T) {
  return isPointerOrMemberPointerType(T.getTypePtr());
}

bool ExceptionTableEntry::canCatch(const Type *ThrowType,
                                   const ASTContext &ASTCtx) const {
  const Type *CatchType = this->CatchType;
  // "A handler is a match for an exception object of type E if ..."

  // "The handler is of type cv T or cv T& and E and T are the same type
  // (ignoring the top-level cv-qualifiers)"
  if (!CatchType || ASTContext::hasSameType(CatchType, ThrowType) ||
      (CatchType->isReferenceType() &&
       ASTContext::hasSameType(CatchType->getPointeeType().getTypePtr(),
                               ThrowType)))
    return true;

  assert(CatchType);

  // "the handler is of type cv T or const T& where T is a pointer or
  // pointer-to-member type and E is std::nullptr_t"
  if (ThrowType->isNullPtrType()) {
    if (isPointerOrMemberPointerType(CatchType))
      return true;

    if (CatchType->isReferenceType() &&
        (CatchType->getPointeeType()->isPointerType() ||
         CatchType->getPointeeType()->isMemberPointerType()))
      return true;
  }

  // void* can catch all thrown pointer types.
  if (ThrowType->isPointerType() && CatchType->isVoidPointerType())
    return true;

  // "the handler is of type cv T or const T& where T is a pointer or
  // pointer-to-member type and E is a pointer or pointer-to-member type that
  // can be converted to T by one or more of ..."
  if ((isPointerOrMemberPointerType(CatchType) ||
       (CatchType->isReferenceType() &&
        isPointerOrMemberPointerType(CatchType->getPointeeType()))) &&
      isPointerOrMemberPointerType(ThrowType)) {

    // "a function pointer conversion"
    if (CatchType->isFunctionPointerType() &&
        ThrowType->isFunctionPointerType()) {
      const auto *FuncT =
          CatchType->getPointeeType()->castAs<FunctionProtoType>();
      const auto *FuncE =
          ThrowType->getPointeeType()->castAs<FunctionProtoType>();

      // We can catch a noexcept function as non-noexcept, but not the other way
      // around.
      if (FuncT->hasNoexceptExceptionSpec() &&
          !FuncE->hasNoexceptExceptionSpec())
        return false;
      // Both being noexcept is also fine.
      return true;
    }

    // "a qualification conversion"
    if (CatchType->isPointerType() && ThrowType->isPointerType()) {
      QualType PointeeT = CatchType->getPointeeType();
      QualType PointeeE = ThrowType->getPointeeType();

      if (ASTContext::hasSameType(PointeeT, PointeeE))
        return true;

      // We can catch T* as const T*, not not the other way around.
      if (!PointeeT.isConstQualified() && PointeeE.isConstQualified())
        return false;
      if (ASTCtx.hasSimilarType(PointeeT, PointeeE))
        return true;
    }
  }

  // "the handler is of type cv T or cv T& and T is an unambiguous public base
  // class of E"
  if (CatchType->isReferenceType())
    CatchType = CatchType->getPointeeType().getTypePtr();

  // T* can only catch T*, not T.
  if (CatchType->isPointerType() && ThrowType->isPointerType()) {
    CatchType = CatchType->getPointeeType().getTypePtr();
    ThrowType = ThrowType->getPointeeType().getTypePtr();
  }

  if (CatchType->isRecordType() && ThrowType->isRecordType()) {
    const CXXRecordDecl *CatchDecl = CatchType->getAsCXXRecordDecl();
    const CXXRecordDecl *ThrowDecl = ThrowType->getAsCXXRecordDecl();
    assert(CatchDecl);
    assert(ThrowDecl);

    if (CatchDecl == ThrowDecl)
      return true;

    CXXBasePaths Paths;
    if (ThrowDecl->isDerivedFrom(CatchDecl, Paths)) {
      if (Paths.isAmbiguous(ASTCtx.getCanonicalTagType(CatchDecl)))
        return false;

      if (Paths.front().Access != AS_public)
        return false;

      return true;
    }
  }

  return false;
}
