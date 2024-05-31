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
