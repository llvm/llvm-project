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

#include "CodeGenModule.h"
#include "clang/CodeGen/CodeGenABITypes.h"

using namespace clang;
using namespace CodeGen;

/// Build a signed-pointer "ptrauth" constant.
static llvm::ConstantPtrAuth *
buildConstantAddress(CodeGenModule &CGM, llvm::Constant *Pointer, unsigned Key,
                     llvm::Constant *StorageAddress,
                     llvm::Constant *OtherDiscriminator) {
  llvm::Constant *AddressDiscriminator = nullptr;
  if (StorageAddress) {
    AddressDiscriminator = StorageAddress;
    assert(StorageAddress->getType() == CGM.UnqualPtrTy);
  } else {
    AddressDiscriminator = llvm::Constant::getNullValue(CGM.UnqualPtrTy);
  }

  llvm::ConstantInt *IntegerDiscriminator = nullptr;
  if (OtherDiscriminator) {
    assert(OtherDiscriminator->getType() == CGM.Int64Ty);
    IntegerDiscriminator = cast<llvm::ConstantInt>(OtherDiscriminator);
  } else {
    IntegerDiscriminator = llvm::ConstantInt::get(CGM.Int64Ty, 0);
  }

  return llvm::ConstantPtrAuth::get(Pointer,
                                    llvm::ConstantInt::get(CGM.Int32Ty, Key),
                                    IntegerDiscriminator, AddressDiscriminator);
}

llvm::Constant *
CodeGenModule::getConstantSignedPointer(llvm::Constant *Pointer, unsigned Key,
                                        llvm::Constant *StorageAddress,
                                        llvm::Constant *OtherDiscriminator) {
  llvm::Constant *Stripped = Pointer->stripPointerCasts();

  // Build the constant.
  return buildConstantAddress(*this, Stripped, Key, StorageAddress,
                              OtherDiscriminator);
}

llvm::Constant *
CodeGen::getConstantSignedPointer(CodeGenModule &CGM, llvm::Constant *Pointer,
                                  unsigned Key, llvm::Constant *StorageAddress,
                                  llvm::Constant *OtherDiscriminator) {
  return CGM.getConstantSignedPointer(Pointer, Key, StorageAddress,
                                      OtherDiscriminator);
}
