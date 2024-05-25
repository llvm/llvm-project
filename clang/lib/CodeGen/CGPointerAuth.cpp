//===--- CGPointerAuth.cpp - IR generation for pointer authentication -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common routines relating to the emission of
// pointer authentication operations.
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"
#include "CGCall.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/PointerAuthOptions.h"
#include "clang/CodeGen/CodeGenABITypes.h"
#include "clang/CodeGen/ConstantInitBuilder.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Analysis/ValueTracking.h"
#include <vector>

using namespace clang;
using namespace CodeGen;

/// Return the abstract pointer authentication schema for a pointer to the given
/// function type.
CGPointerAuthInfo CodeGenModule::getFunctionPointerAuthInfo(QualType T) {
  auto &Schema = getCodeGenOpts().PointerAuth.FunctionPointers;
  if (!Schema)
    return CGPointerAuthInfo();

  assert(!Schema.isAddressDiscriminated() &&
         "function pointers cannot use address-specific discrimination");

  assert(!Schema.hasOtherDiscrimination() &&
         "function pointers don't support any discrimination yet");

  return CGPointerAuthInfo(Schema.getKey(), Schema.getAuthenticationMode(),
                           /*IsaPointer=*/false, /*AuthenticatesNull=*/false,
                           /*Discriminator=*/nullptr);
}

/// Build a signed-pointer "ptrauth" constant.
static llvm::ConstantPtrAuth *
buildConstantAddress(CodeGenModule &CGM, llvm::Constant *pointer, unsigned key,
                     llvm::Constant *storageAddress,
                     llvm::Constant *otherDiscriminator) {
  llvm::Constant *addressDiscriminator = nullptr;
  if (storageAddress) {
    addressDiscriminator = storageAddress;
    assert(storageAddress->getType() == CGM.UnqualPtrTy);
  } else {
    addressDiscriminator = llvm::Constant::getNullValue(CGM.UnqualPtrTy);
  }

  llvm::ConstantInt *integerDiscriminator = nullptr;
  if (otherDiscriminator) {
    assert(otherDiscriminator->getType() == CGM.Int64Ty);
    integerDiscriminator = cast<llvm::ConstantInt>(otherDiscriminator);
  } else {
    integerDiscriminator = llvm::ConstantInt::get(CGM.Int64Ty, 0);
  }

  return llvm::ConstantPtrAuth::get(
    pointer, llvm::ConstantInt::get(CGM.Int32Ty, key), integerDiscriminator,
    addressDiscriminator);
}

llvm::Constant *
CodeGenModule::getConstantSignedPointer(llvm::Constant *pointer,
                                        unsigned key,
                                        llvm::Constant *storageAddress,
                                        llvm::Constant *otherDiscriminator) {
  // Unique based on the underlying value, not a signing of it.
  auto stripped = pointer->stripPointerCasts();

  // Build the constant.
  return buildConstantAddress(*this, stripped, key, storageAddress,
                              otherDiscriminator);
}

llvm::Constant *
CodeGen::getConstantSignedPointer(CodeGenModule &CGM,
                                  llvm::Constant *pointer, unsigned key,
                                  llvm::Constant *storageAddress,
                                  llvm::Constant *otherDiscriminator) {
  return CGM.getConstantSignedPointer(pointer, key, storageAddress,
                                      otherDiscriminator);
}

/// If applicable, sign a given constant function pointer with the ABI rules for
/// functionType.
llvm::Constant *CodeGenModule::getFunctionPointer(llvm::Constant *pointer,
                                                  QualType functionType,
                                                  GlobalDecl GD) {
  assert(functionType->isFunctionType() ||
         functionType->isFunctionReferenceType() ||
         functionType->isFunctionPointerType());

  if (auto pointerAuth = getFunctionPointerAuthInfo(functionType)) {
    return getConstantSignedPointer(
      pointer, pointerAuth.getKey(), nullptr,
      cast_or_null<llvm::Constant>(pointerAuth.getDiscriminator()));
  }

  return pointer;
}

llvm::Constant *CodeGenModule::getFunctionPointer(GlobalDecl GD,
                                                  llvm::Type *Ty) {
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());

  // Annoyingly, K&R functions have prototypes in the clang AST, but
  // expressions referring to them are unprototyped.
  QualType FuncType = FD->getType();
  if (!FD->hasPrototype())
    if (const auto *Proto = FuncType->getAs<FunctionProtoType>())
      FuncType = Context.getFunctionNoProtoType(Proto->getReturnType(),
                                                Proto->getExtInfo());

  return getFunctionPointer(getRawFunctionPointer(GD, Ty), FuncType, GD);
}
