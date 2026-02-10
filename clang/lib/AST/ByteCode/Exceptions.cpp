//===-------------------------- Exceptions.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Exceptions.h"
#include "clang/AST/ASTContext.h"

using namespace clang;
using namespace clang::interp;

bool ExceptionTableEntry::canCatch(const Type *ThrowType) const {
  const Type *CatchType = this->CatchType;

  if (!CatchType || ASTContext::hasSameType(CatchType, ThrowType))
    return true;

  assert(CatchType);

  // nullptr_t can be caught by any pointer type.
  if (ThrowType->isNullPtrType() && CatchType->isPointerType())
    return true;

  // void* can catch all thown pointer types.
  if (ThrowType->isPointerType() && CatchType->isVoidPointerType())
    return true;

  if (CatchType->isPointerType() && !ThrowType->isPointerType())
    return false;

  if (CatchType->isPointerOrReferenceType())
    CatchType = CatchType->getPointeeType().getTypePtr();
  if (ThrowType->isPointerOrReferenceType())
    ThrowType = ThrowType->getPointeeType().getTypePtr();

  if (CatchType == ThrowType)
    return true;

  if (CatchType->isRecordType() && ThrowType->isRecordType()) {
    const CXXRecordDecl *CatchDecl = CatchType->getAsCXXRecordDecl();
    const CXXRecordDecl *ThrowDecl = ThrowType->getAsCXXRecordDecl();
    assert(CatchDecl);
    assert(ThrowDecl);

    if (CatchDecl == ThrowDecl)
      return true;

    if (ThrowDecl->isDerivedFrom(CatchDecl))
      return true;
  }

  return false;
}
