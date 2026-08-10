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

static bool isRecordOrPointerToRecordType(const Type *T) {
  return T->isRecordType() ||
         (T->isPointerOrReferenceType() && T->getPointeeType()->isRecordType());
}

bool ExceptionTableEntry::canCatch(const Type *ThrowType,
                                   const ASTContext &ASTCtx) const {
  const Type *CatchType = this->CatchType;

  if (!CatchType || ASTContext::hasSameType(CatchType, ThrowType))
    return true;

  assert(CatchType);

  llvm::errs() << __PRETTY_FUNCTION__ << '\n';
  ThrowType->dump();
  CatchType->dump();

  // nullptr_t can be caught by any pointer type (including member pointers)
  // and references of pointer types.
  if (ThrowType->isNullPtrType()) {
    if (CatchType->isPointerType() || CatchType->isMemberPointerType())
      return true;
    if (CatchType->isReferenceType() &&
        CatchType->getPointeeType()->isPointerType())
      return true;
  }

  // void* can catch all thown pointer types.
  if (ThrowType->isPointerType() && CatchType->isVoidPointerType())
    return true;

  // T& can catch T.
  if (CatchType->isReferenceType() &&
      ASTContext::hasSameType(CatchType->getPointeeType().getTypePtr(),
                              ThrowType))
    return true;

  // From this point foward, we only care about T, T* or T& where T is a record
  // type.
  if (!isRecordOrPointerToRecordType(ThrowType) ||
      !isRecordOrPointerToRecordType(CatchType))
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

    // "T is an unambiguous public base class of E."
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
