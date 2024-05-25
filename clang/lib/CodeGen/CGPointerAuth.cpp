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

llvm::Constant *
CodeGenModule::getConstantSignedPointer(llvm::Constant *Pointer, unsigned Key,
                                        llvm::Constant *StorageAddress,
                                        llvm::ConstantInt *OtherDiscriminator) {
  llvm::Constant *AddressDiscriminator;
  if (StorageAddress) {
    assert(StorageAddress->getType() == UnqualPtrTy);
    AddressDiscriminator = StorageAddress;
  } else {
    AddressDiscriminator = llvm::Constant::getNullValue(UnqualPtrTy);
  }

  llvm::ConstantInt *IntegerDiscriminator;
  if (OtherDiscriminator) {
    assert(OtherDiscriminator->getType() == Int64Ty);
    IntegerDiscriminator = OtherDiscriminator;
  } else {
    IntegerDiscriminator = llvm::ConstantInt::get(Int64Ty, 0);
  }

  return llvm::ConstantPtrAuth::get(Pointer,
                                    llvm::ConstantInt::get(Int32Ty, Key),
                                    IntegerDiscriminator, AddressDiscriminator);
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
