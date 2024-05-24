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
#include "llvm/Support/SipHash.h"
#include <vector>

using namespace clang;
using namespace CodeGen;

/// Given a pointer-authentication schema, return a concrete "other"
/// discriminator for it.
llvm::Constant *CodeGenModule::getPointerAuthOtherDiscriminator(
    const PointerAuthSchema &schema, GlobalDecl decl, QualType type) {
  switch (schema.getOtherDiscrimination()) {
  case PointerAuthSchema::Discrimination::None:
    return nullptr;

  case PointerAuthSchema::Discrimination::Type:
    llvm_unreachable("type discrimination not implemented yet");

  case PointerAuthSchema::Discrimination::Decl:
    assert(decl.getDecl() &&
           "declaration not provided for decl-discriminated schema");
    return llvm::ConstantInt::get(IntPtrTy,
                                  getPointerAuthDeclDiscriminator(decl));

  case PointerAuthSchema::Discrimination::Constant:
    return llvm::ConstantInt::get(IntPtrTy, schema.getConstantDiscrimination());
  }
  llvm_unreachable("bad discrimination kind");
}

uint16_t CodeGen::getPointerAuthDeclDiscriminator(CodeGenModule &CGM,
                                                  GlobalDecl declaration) {
  return CGM.getPointerAuthDeclDiscriminator(declaration);
}

/// Return the "other" decl-specific discriminator for the given decl.
uint16_t
CodeGenModule::getPointerAuthDeclDiscriminator(GlobalDecl declaration) {
  uint16_t &entityHash = PtrAuthDiscriminatorHashes[declaration];

  if (entityHash == 0) {
    StringRef name = getMangledName(declaration);
    entityHash = llvm::getPointerAuthStableSipHash16(name);
  }

  return entityHash;
}

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

llvm::Value *
CodeGenFunction::EmitPointerAuthBlendDiscriminator(llvm::Value *storageAddress,
                                                   llvm::Value *discriminator) {
  storageAddress = Builder.CreatePtrToInt(storageAddress, IntPtrTy);
  auto intrinsic = CGM.getIntrinsic(llvm::Intrinsic::ptrauth_blend);
  return Builder.CreateCall(intrinsic, {storageAddress, discriminator});
}

/// Emit the concrete pointer authentication informaton for the
/// given authentication schema.
CGPointerAuthInfo CodeGenFunction::EmitPointerAuthInfo(
    const PointerAuthSchema &schema, llvm::Value *storageAddress,
    GlobalDecl schemaDecl, QualType schemaType) {
  if (!schema)
    return CGPointerAuthInfo();

  llvm::Value *discriminator =
      CGM.getPointerAuthOtherDiscriminator(schema, schemaDecl, schemaType);

  if (schema.isAddressDiscriminated()) {
    assert(storageAddress &&
           "address not provided for address-discriminated schema");

    if (discriminator)
      discriminator =
          EmitPointerAuthBlendDiscriminator(storageAddress, discriminator);
    else
      discriminator = Builder.CreatePtrToInt(storageAddress, IntPtrTy);
  }

  return CGPointerAuthInfo(schema.getKey(), schema.getAuthenticationMode(),
                           schema.isIsaPointer(),
                           schema.authenticatesNullValues(), discriminator);
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

/// Does a given PointerAuthScheme require us to sign a value
static bool shouldSignPointer(const PointerAuthSchema &schema) {
  auto authenticationMode = schema.getAuthenticationMode();
  return authenticationMode == PointerAuthenticationMode::SignAndStrip ||
         authenticationMode == PointerAuthenticationMode::SignAndAuth;
}

/// Sign a constant pointer using the given scheme, producing a constant
/// with the same IR type.
llvm::Constant *CodeGenModule::getConstantSignedPointer(
    llvm::Constant *pointer, const PointerAuthSchema &schema,
    llvm::Constant *storageAddress, GlobalDecl schemaDecl,
    QualType schemaType) {
  assert(shouldSignPointer(schema));
  llvm::Constant *otherDiscriminator =
      getPointerAuthOtherDiscriminator(schema, schemaDecl, schemaType);

  return getConstantSignedPointer(pointer, schema.getKey(), storageAddress,
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

/// Sign the given pointer and add it to the constant initializer
/// currently being built.
void ConstantAggregateBuilderBase::addSignedPointer(
    llvm::Constant *pointer, const PointerAuthSchema &schema,
    GlobalDecl calleeDecl, QualType calleeType) {
  if (!schema || !shouldSignPointer(schema))
    return add(pointer);

  llvm::Constant *storageAddress = nullptr;
  if (schema.isAddressDiscriminated()) {
    storageAddress = getAddrOfCurrentPosition(pointer->getType());
  }

  llvm::Constant *signedPointer = Builder.CGM.getConstantSignedPointer(
      pointer, schema, storageAddress, calleeDecl, calleeType);
  add(signedPointer);
}

void ConstantAggregateBuilderBase::addSignedPointer(
    llvm::Constant *pointer, unsigned key, bool useAddressDiscrimination,
    llvm::Constant *otherDiscriminator) {
  llvm::Constant *storageAddress = nullptr;
  if (useAddressDiscrimination) {
    storageAddress = getAddrOfCurrentPosition(pointer->getType());
  }

  llvm::Constant *signedPointer = Builder.CGM.getConstantSignedPointer(
      pointer, key, storageAddress, otherDiscriminator);
  add(signedPointer);
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

std::optional<PointerAuthQualifier>
CodeGenModule::computeVTPointerAuthentication(const CXXRecordDecl *thisClass) {
  auto defaultAuthentication = getCodeGenOpts().PointerAuth.CXXVTablePointers;
  if (!defaultAuthentication)
    return std::nullopt;
  const CXXRecordDecl *primaryBase =
      Context.baseForVTableAuthentication(thisClass);

  unsigned key = defaultAuthentication.getKey();
  bool addressDiscriminated = defaultAuthentication.isAddressDiscriminated();
  auto defaultDiscrimination = defaultAuthentication.getOtherDiscrimination();
  unsigned typeBasedDiscriminator =
      Context.getPointerAuthVTablePointerDiscriminator(primaryBase);
  unsigned discriminator;
  if (defaultDiscrimination == PointerAuthSchema::Discrimination::Type) {
    discriminator = typeBasedDiscriminator;
  } else if (defaultDiscrimination ==
             PointerAuthSchema::Discrimination::Constant) {
    discriminator = defaultAuthentication.getConstantDiscrimination();
  } else {
    assert(defaultDiscrimination == PointerAuthSchema::Discrimination::None);
    discriminator = 0;
  }
  if (auto explicitAuthentication =
          primaryBase->getAttr<VTablePointerAuthenticationAttr>()) {
    auto explicitKey = explicitAuthentication->getKey();
    auto explicitAddressDiscrimination =
        explicitAuthentication->getAddressDiscrimination();
    auto explicitDiscriminator =
        explicitAuthentication->getExtraDiscrimination();
    if (explicitKey == VTablePointerAuthenticationAttr::NoKey) {
      return std::nullopt;
    }
    if (explicitKey != VTablePointerAuthenticationAttr::DefaultKey) {
      if (explicitKey == VTablePointerAuthenticationAttr::ProcessIndependent)
        key = (unsigned)PointerAuthSchema::ARM8_3Key::ASDA;
      else {
        assert(explicitKey ==
               VTablePointerAuthenticationAttr::ProcessDependent);
        key = (unsigned)PointerAuthSchema::ARM8_3Key::ASDB;
      }
    }

    if (explicitAddressDiscrimination !=
        VTablePointerAuthenticationAttr::DefaultAddressDiscrimination) {
      addressDiscriminated =
          explicitAddressDiscrimination ==
          VTablePointerAuthenticationAttr::AddressDiscrimination;
    }

    if (explicitDiscriminator ==
        VTablePointerAuthenticationAttr::TypeDiscrimination) {
      discriminator = typeBasedDiscriminator;
    } else if (explicitDiscriminator ==
               VTablePointerAuthenticationAttr::CustomDiscrimination) {
      discriminator = explicitAuthentication->getCustomDiscriminationValue();
    } else if (explicitDiscriminator == VTablePointerAuthenticationAttr::NoExtraDiscrimination) {
      discriminator = 0;
    }
  }
  return PointerAuthQualifier::Create(key, addressDiscriminated, discriminator,
                                      PointerAuthenticationMode::SignAndAuth,
                                      /* isIsaPointer */ false,
                                      /* authenticatesNullValues */ false);
}

std::optional<PointerAuthQualifier>
CodeGenModule::getVTablePointerAuthentication(const CXXRecordDecl *record) {
  if (!record->getDefinition() || !record->isPolymorphic())
    return std::nullopt;

  auto existing = VTablePtrAuthInfos.find(record);
  std::optional<PointerAuthQualifier> authentication;
  if (existing != VTablePtrAuthInfos.end()) {
    authentication = existing->getSecond();
  } else {
    authentication = computeVTPointerAuthentication(record);
    VTablePtrAuthInfos.insert(std::make_pair(record, authentication));
  }
  return authentication;
}

std::optional<CGPointerAuthInfo>
CodeGenModule::getVTablePointerAuthInfo(CodeGenFunction *CGF,
                                        const CXXRecordDecl *record,
                                        llvm::Value *storageAddress) {
  auto authentication = getVTablePointerAuthentication(record);
  if (!authentication)
    return std::nullopt;

  llvm::Value *discriminator = nullptr;
  if (auto extraDiscriminator = authentication->getExtraDiscriminator()) {
    discriminator = llvm::ConstantInt::get(IntPtrTy, extraDiscriminator);
  }
  if (authentication->isAddressDiscriminated()) {
    assert(storageAddress &&
           "address not provided for address-discriminated schema");
    if (discriminator)
      discriminator =
          CGF->EmitPointerAuthBlendDiscriminator(storageAddress, discriminator);
    else
      discriminator = CGF->Builder.CreatePtrToInt(storageAddress, IntPtrTy);
  }

  return CGPointerAuthInfo(authentication->getKey(),
                           PointerAuthenticationMode::SignAndAuth,
                           /* IsIsaPointer */ false,
                           /* authenticatesNullValues */ false, discriminator);
}
