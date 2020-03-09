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
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "CGCall.h"
#include "clang/AST/StableHash.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "clang/CodeGen/CodeGenABITypes.h"
#include "clang/Basic/PointerAuthOptions.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Analysis/ValueTracking.h"
#include <vector>

using namespace clang;
using namespace CodeGen;

/// Given a pointer-authentication schema, return a concrete "other"
/// discriminator for it.
llvm::Constant *
CodeGenModule::getPointerAuthOtherDiscriminator(const PointerAuthSchema &schema,
                                                GlobalDecl decl,
                                                QualType type) {
  switch (schema.getOtherDiscrimination()) {
  case PointerAuthSchema::Discrimination::None:
    return nullptr;

  case PointerAuthSchema::Discrimination::Type:
    assert(!type.isNull() &&
           "type not provided for type-discriminated schema");
    return llvm::ConstantInt::get(
        IntPtrTy, getContext().getPointerAuthTypeDiscriminator(type));

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

uint16_t CodeGen::getPointerAuthTypeDiscriminator(CodeGenModule &CGM,
                                                  QualType functionType) {
  return CGM.getContext().getPointerAuthTypeDiscriminator(functionType);
}

/// Compute an ABI-stable hash of the given string.
uint64_t CodeGen::computeStableStringHash(StringRef string) {
  return clang::getStableStringHash(string);
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
    entityHash = getPointerAuthStringDiscriminator(getContext(), name);
  }

  return entityHash;
}

/// Return the abstract pointer authentication schema for a
/// function pointer of the given type.
CGPointerAuthInfo
CodeGenModule::getFunctionPointerAuthInfo(QualType functionType) {
  // Check for a generic pointer authentication schema.
  auto &schema = getCodeGenOpts().PointerAuth.FunctionPointers;
  if (!schema) return CGPointerAuthInfo();

  assert(!schema.isAddressDiscriminated() &&
         "function pointers cannot use address-specific discrimination");

  auto discriminator =
    getPointerAuthOtherDiscriminator(schema, GlobalDecl(), functionType);
  return CGPointerAuthInfo(schema.getKey(), discriminator);
}

CGPointerAuthInfo
CodeGenModule::getMemberFunctionPointerAuthInfo(QualType functionType) {
  assert(functionType->getAs<MemberPointerType>() &&
         "MemberPointerType expected");
  auto &schema = getCodeGenOpts().PointerAuth.CXXMemberFunctionPointers;
  if (!schema)
    return CGPointerAuthInfo();

  assert(!schema.isAddressDiscriminated() &&
         "function pointers cannot use address-specific discrimination");

  auto discriminator =
      getPointerAuthOtherDiscriminator(schema, GlobalDecl(), functionType);
  return CGPointerAuthInfo(schema.getKey(), discriminator);
}

/// Return the natural pointer authentication for values of the given
/// pointer type.
static CGPointerAuthInfo getPointerAuthInfoForType(CodeGenModule &CGM,
                                                   QualType type) {
  assert(type->isPointerType());

  // Function pointers use the function-pointer schema by default.
  if (auto ptrTy = type->getAs<PointerType>()) {
    auto functionType = ptrTy->getPointeeType();
    if (functionType->isFunctionType()) {
      return CGM.getFunctionPointerAuthInfo(functionType);
    }
  }

  // Normal data pointers never use direct pointer authentication by default.
  return CGPointerAuthInfo();
}

llvm::Value *CodeGenFunction::EmitPointerAuthBlendDiscriminator(
    llvm::Value *storageAddress, llvm::Value *discriminator) {
  storageAddress = Builder.CreatePtrToInt(storageAddress, IntPtrTy);
  auto intrinsic = CGM.getIntrinsic(llvm::Intrinsic::ptrauth_blend,
                                    { CGM.IntPtrTy });
  return Builder.CreateCall(intrinsic, {storageAddress, discriminator});
}

/// Emit the concrete pointer authentication informaton for the
/// given authentication schema.
CGPointerAuthInfo
CodeGenFunction::EmitPointerAuthInfo(const PointerAuthSchema &schema,
                                     llvm::Value *storageAddress,
                                     GlobalDecl schemaDecl,
                                     QualType schemaType) {
  if (!schema) return CGPointerAuthInfo();

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

  return CGPointerAuthInfo(schema.getKey(), discriminator);
}

CGPointerAuthInfo
CodeGenFunction::EmitPointerAuthInfo(PointerAuthQualifier qualifier,
                                     Address storageAddress) {
  assert(qualifier &&
         "don't call this if you don't know that the qualifier is present");

  llvm::Value *discriminator = nullptr;
  if (unsigned extra = qualifier.getExtraDiscriminator()) {
    discriminator = llvm::ConstantInt::get(IntPtrTy, extra);
  }

  if (qualifier.isAddressDiscriminated()) {
    assert(storageAddress.isValid() &&
           "address discrimination without address");
    auto storagePtr = storageAddress.getPointer();
    if (discriminator) {
      discriminator =
        EmitPointerAuthBlendDiscriminator(storagePtr, discriminator);
    } else {
      discriminator = Builder.CreatePtrToInt(storagePtr, IntPtrTy);
    }
  }

  return CGPointerAuthInfo(qualifier.getKey(), discriminator);
}

static std::pair<llvm::Value *, CGPointerAuthInfo>
emitLoadOfOrigPointerRValue(CodeGenFunction &CGF, const LValue &lv,
                            SourceLocation loc) {
  auto value = CGF.EmitLoadOfScalar(lv, loc);
  CGPointerAuthInfo authInfo;
  if (auto ptrauth = lv.getQuals().getPointerAuth()) {
    authInfo = CGF.EmitPointerAuthInfo(ptrauth, lv.getAddress(CGF));
  } else {
    authInfo = getPointerAuthInfoForType(CGF.CGM, lv.getType());
  }
  return { value, authInfo };
}

std::pair<llvm::Value *, CGPointerAuthInfo>
CodeGenFunction::EmitOrigPointerRValue(const Expr *E) {
  assert(E->getType()->isPointerType());

  E = E->IgnoreParens();
  if (auto load = dyn_cast<ImplicitCastExpr>(E)) {
    if (load->getCastKind() == CK_LValueToRValue) {
      E = load->getSubExpr()->IgnoreParens();

      // We're semantically required to not emit loads of certain DREs naively.
      if (auto refExpr = dyn_cast<DeclRefExpr>(const_cast<Expr*>(E))) {
        if (auto result = tryEmitAsConstant(refExpr)) {
          // Fold away a use of an intermediate variable.
          if (!result.isReference())
            return { result.getValue(),
                      getPointerAuthInfoForType(CGM, refExpr->getType()) };

          // Fold away a use of an intermediate reference.
          auto lv = result.getReferenceLValue(*this, refExpr);
          return emitLoadOfOrigPointerRValue(*this, lv, refExpr->getLocation());
        }
      }

      // Otherwise, load and use the pointer
      auto lv = EmitCheckedLValue(E, CodeGenFunction::TCK_Load);
      return emitLoadOfOrigPointerRValue(*this, lv, E->getExprLoc());
    }
  }

  // Emit direct references to functions without authentication.
  if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    if (auto FD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
      return { CGM.getRawFunctionPointer(FD), CGPointerAuthInfo() };
    }
  } else if (auto ME = dyn_cast<MemberExpr>(E)) {
    if (auto FD = dyn_cast<FunctionDecl>(ME->getMemberDecl())) {
      EmitIgnoredExpr(ME->getBase());
      return { CGM.getRawFunctionPointer(FD), CGPointerAuthInfo() };
    }
  }

  // Fallback: just use the normal rules for the type.
  auto value = EmitScalarExpr(E);
  return { value, getPointerAuthInfoForType(CGM, E->getType()) };
}

llvm::Value *
CodeGenFunction::EmitPointerAuthQualify(PointerAuthQualifier destQualifier,
                                        const Expr *E,
                                        Address destStorageAddress) {
  assert(destQualifier);

  auto src = EmitOrigPointerRValue(E);
  auto value = src.first;
  auto curAuthInfo = src.second;

  auto destAuthInfo = EmitPointerAuthInfo(destQualifier, destStorageAddress);
  return EmitPointerAuthResign(value, E->getType(), curAuthInfo, destAuthInfo,
                               isPointerKnownNonNull(E));
}

llvm::Value *
CodeGenFunction::EmitPointerAuthQualify(PointerAuthQualifier destQualifier,
                                        llvm::Value *value,
                                        QualType pointerType,
                                        Address destStorageAddress,
                                        bool isKnownNonNull) {
  assert(destQualifier);

  auto curAuthInfo = getPointerAuthInfoForType(CGM, pointerType);
  auto destAuthInfo = EmitPointerAuthInfo(destQualifier, destStorageAddress);
  return EmitPointerAuthResign(value, pointerType, curAuthInfo, destAuthInfo,
                               isKnownNonNull);
}

llvm::Value *
CodeGenFunction::EmitPointerAuthUnqualify(PointerAuthQualifier curQualifier,
                                          llvm::Value *value,
                                          QualType pointerType,
                                          Address curStorageAddress,
                                          bool isKnownNonNull) {
  assert(curQualifier);

  auto curAuthInfo = EmitPointerAuthInfo(curQualifier, curStorageAddress);
  auto destAuthInfo = getPointerAuthInfoForType(CGM, pointerType);
  return EmitPointerAuthResign(value, pointerType, curAuthInfo, destAuthInfo,
                               isKnownNonNull);
}

static bool isZeroConstant(llvm::Value *value) {
  if (auto ci = dyn_cast<llvm::ConstantInt>(value))
    return ci->isZero();
  return false;
}

llvm::Value *
CodeGenFunction::EmitPointerAuthResign(llvm::Value *value, QualType type,
                                       const CGPointerAuthInfo &curAuthInfo,
                                       const CGPointerAuthInfo &newAuthInfo,
                                       bool isKnownNonNull) {
  // Fast path: if neither schema wants a signature, we're done.
  if (!curAuthInfo && !newAuthInfo)
    return value;

  // If the value is obviously null, we're done.
  auto null =
    CGM.getNullPointer(cast<llvm::PointerType>(value->getType()), type);
  if (value == null) {
    return value;
  }

  // If both schemas sign the same way, we're done.
  if (curAuthInfo && newAuthInfo &&
      curAuthInfo.getKey() == newAuthInfo.getKey()) {
    auto curD = curAuthInfo.getDiscriminator();
    auto newD = newAuthInfo.getDiscriminator();
    if (curD == newD ||
        (curD == nullptr && isZeroConstant(newD)) ||
        (newD == nullptr && isZeroConstant(curD)))
      return value;
  }

  llvm::BasicBlock *initBB = Builder.GetInsertBlock();
  llvm::BasicBlock *resignBB = nullptr, *contBB = nullptr;

  // Null pointers have to be mapped to null, and the ptrauth_resign
  // intrinsic doesn't do that.
  if (!isKnownNonNull && !llvm::isKnownNonZero(value, CGM.getDataLayout())) {
    contBB = createBasicBlock("resign.cont");
    resignBB = createBasicBlock("resign.nonnull");

    auto isNonNull = Builder.CreateICmpNE(value, null);
    Builder.CreateCondBr(isNonNull, resignBB, contBB);
    EmitBlock(resignBB);
  }

  // Perform the auth/sign/resign operation.
  if (!newAuthInfo) {
    value = EmitPointerAuthAuth(curAuthInfo, value);
  } else if (!curAuthInfo) {
    value = EmitPointerAuthSign(newAuthInfo, value);
  } else {
    value = EmitPointerAuthResignCall(value, curAuthInfo, newAuthInfo);
  }

  // Clean up with a phi if we branched before.
  if (contBB) {
    EmitBlock(contBB);
    auto phi = Builder.CreatePHI(value->getType(), 2);
    phi->addIncoming(null, initBB);
    phi->addIncoming(value, resignBB);
    value = phi;
  }

  return value;
}

void CodeGenFunction::EmitPointerAuthCopy(PointerAuthQualifier qualifier,
                                          QualType type,
                                          Address destAddress,
                                          Address srcAddress) {
  assert(qualifier);

  llvm::Value *value = Builder.CreateLoad(srcAddress);

  // If we're using address-discrimination, we have to re-sign the value.
  if (qualifier.isAddressDiscriminated()) {
    auto srcPtrAuth = EmitPointerAuthInfo(qualifier, srcAddress);
    auto destPtrAuth = EmitPointerAuthInfo(qualifier, destAddress);
    value = EmitPointerAuthResign(value, type, srcPtrAuth, destPtrAuth,
                                  /*is known nonnull*/ false);
  }

  Builder.CreateStore(value, destAddress);
}

/// We use an abstract, side-allocated cache for signed function pointers
/// because (1) most compiler invocations will not need this cache at all,
/// since they don't use signed function pointers, and (2) the
/// representation is pretty complicated (an llvm::ValueMap) and we don't
/// want to have to include that information in CodeGenModule.h.
template <class CacheTy>
static CacheTy &getOrCreateCache(void *&abstractStorage) {
  auto cache = static_cast<CacheTy*>(abstractStorage);
  if (cache) return *cache;

  abstractStorage = cache = new CacheTy();
  return *cache;
}

template <class CacheTy>
static void destroyCache(void *&abstractStorage) {
  delete static_cast<CacheTy*>(abstractStorage);
  abstractStorage = nullptr;
}

namespace {
struct PointerAuthConstantEntry {
  unsigned Key;
  llvm::Constant *OtherDiscriminator;
  llvm::GlobalVariable *Global;
};

using PointerAuthConstantEntries =
  std::vector<PointerAuthConstantEntry>;
using ByConstantCacheTy =
  llvm::ValueMap<llvm::Constant*, PointerAuthConstantEntries>;
using ByDeclCacheTy =
  llvm::DenseMap<const Decl *, llvm::Constant*>;
}

/// Build a global signed-pointer constant.
static llvm::GlobalVariable *
buildConstantSignedPointer(CodeGenModule &CGM,
                           llvm::Constant *pointer,
                           unsigned key,
                           llvm::Constant *storageAddress,
                           llvm::Constant *otherDiscriminator) {
  ConstantInitBuilder builder(CGM);
  auto values = builder.beginStruct();
  values.addBitCast(pointer, CGM.Int8PtrTy);
  values.addInt(CGM.Int32Ty, key);
  if (storageAddress) {
    if (isa<llvm::ConstantInt>(storageAddress)) {
      assert(!storageAddress->isNullValue() &&
             "expecting pointer or special address-discriminator indicator");
      values.add(storageAddress);
    } else {
      values.add(llvm::ConstantExpr::getPtrToInt(storageAddress, CGM.IntPtrTy));
    }
  } else {
    values.addInt(CGM.SizeTy, 0);
  }
  if (otherDiscriminator) {
    assert(otherDiscriminator->getType() == CGM.SizeTy);
    values.add(otherDiscriminator);
  } else {
    values.addInt(CGM.SizeTy, 0);
  }

  auto *stripped = pointer->stripPointerCasts();
  StringRef name;
  if (const auto *origGlobal = dyn_cast<llvm::GlobalValue>(stripped))
    name = origGlobal->getName();
  else if (const auto *ce = dyn_cast<llvm::ConstantExpr>(stripped))
    if (ce->getOpcode() == llvm::Instruction::GetElementPtr)
      name = cast<llvm::GEPOperator>(ce)->getPointerOperand()->getName();

  auto global = values.finishAndCreateGlobal(
      name + ".ptrauth",
      CGM.getPointerAlign(),
      /*constant*/ true,
      llvm::GlobalVariable::PrivateLinkage);
  global->setSection("llvm.ptrauth");

  return global;
}

llvm::Constant *
CodeGenModule::getConstantSignedPointer(llvm::Constant *pointer,
                                        unsigned key,
                                        llvm::Constant *storageAddress,
                                        llvm::Constant *otherDiscriminator) {
  // Unique based on the underlying value, not a signing of it.
  auto stripped = pointer->stripPointerCasts();

  PointerAuthConstantEntries *entries = nullptr;

  // We can cache this for discriminators that aren't defined in terms
  // of globals.  Discriminators defined in terms of globals (1) would
  // require additional tracking to be safe and (2) only come up with
  // address-specific discrimination, where this entry is almost certainly
  // unique to the use-site anyway.
  if (!storageAddress &&
      (!otherDiscriminator ||
       isa<llvm::ConstantInt>(otherDiscriminator))) {

    // Get or create the cache.
    auto &cache =
      getOrCreateCache<ByConstantCacheTy>(ConstantSignedPointersByConstant);

    // Check for an existing entry.
    entries = &cache[stripped];
    for (auto &entry : *entries) {
      if (entry.Key == key && entry.OtherDiscriminator == otherDiscriminator) {
        auto global = entry.Global;
        return llvm::ConstantExpr::getBitCast(global, pointer->getType());
      }
    }
  }

  // Build the constant.
  auto global =
    buildConstantSignedPointer(*this, stripped, key, storageAddress,
                               otherDiscriminator);

  // Cache if applicable.
  if (entries) {
    entries->push_back({ key, otherDiscriminator, global });
  }

  // Cast to the original type.
  return llvm::ConstantExpr::getBitCast(global, pointer->getType());
}

/// Sign a constant pointer using the given scheme, producing a constant
/// with the same IR type.
llvm::Constant *
CodeGenModule::getConstantSignedPointer(llvm::Constant *pointer,
                                        const PointerAuthSchema &schema,
                                        llvm::Constant *storageAddress,
                                        GlobalDecl schemaDecl,
                                        QualType schemaType) {
  llvm::Constant *otherDiscriminator =
    getPointerAuthOtherDiscriminator(schema, schemaDecl, schemaType);

  return getConstantSignedPointer(pointer, schema.getKey(),
                                  storageAddress, otherDiscriminator);
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
  if (!schema) return add(pointer);

  llvm::Constant *storageAddress = nullptr;
  if (schema.isAddressDiscriminated()) {
    storageAddress = getAddrOfCurrentPosition(pointer->getType());
  }

  llvm::Constant *signedPointer =
    Builder.CGM.getConstantSignedPointer(pointer, schema, storageAddress,
                                         calleeDecl, calleeType);
  add(signedPointer);
}

void ConstantAggregateBuilderBase::addSignedPointer(
    llvm::Constant *pointer, unsigned key,
    bool useAddressDiscrimination, llvm::Constant *otherDiscriminator) {
  llvm::Constant *storageAddress = nullptr;
  if (useAddressDiscrimination) {
    storageAddress = getAddrOfCurrentPosition(pointer->getType());
  }

  llvm::Constant *signedPointer =
    Builder.CGM.getConstantSignedPointer(pointer, key, storageAddress,
                                         otherDiscriminator);
  add(signedPointer);
}

void CodeGenModule::destroyConstantSignedPointerCaches() {
  destroyCache<ByConstantCacheTy>(ConstantSignedPointersByConstant);
  destroyCache<ByDeclCacheTy>(ConstantSignedPointersByDecl);
  destroyCache<ByDeclCacheTy>(SignedThunkPointers);
}

llvm::Constant *CodeGenModule::getFunctionPointer(llvm::Constant *pointer,
                                                  QualType functionType,
                                                  GlobalDecl GD) {
  if (auto pointerAuth = getFunctionPointerAuthInfo(functionType)) {
    // Check a cache that, for now, just has entries for functions signed
    // with the standard function-pointer scheme.
    // Cache function pointers based on their decl.  Anything without a decl is
    // going to be a one-off that doesn't need to be cached anyway.
    llvm::Constant **entry = nullptr;
    if (GD) {
      auto FD = cast<FunctionDecl>(GD.getDecl());
      auto &cache =
          getOrCreateCache<ByDeclCacheTy>(ConstantSignedPointersByDecl);
      entry = &cache[FD->getCanonicalDecl()];
      if (*entry)
        return llvm::ConstantExpr::getBitCast(*entry, pointer->getType());
    }

    // If the cache misses, build a new constant.  It's not a *problem* to
    // have more than one of these for a particular function, but it's nice
    // to avoid it.
    pointer = getConstantSignedPointer(
        pointer, pointerAuth.getKey(), nullptr,
        cast_or_null<llvm::Constant>(pointerAuth.getDiscriminator()));

    // Store the result back into the cache, if any.
    if (entry)
      *entry = pointer;
  }

  return pointer;
}

llvm::Constant *CodeGenModule::getFunctionPointer(GlobalDecl GD,
                                                  llvm::Type *Ty) {
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());
  return getFunctionPointer(getRawFunctionPointer(GD, Ty), FD->getType(), FD);
}

llvm::Constant *
CodeGenModule::getMemberFunctionPointer(llvm::Constant *pointer,
                                        QualType functionType,
                                        const FunctionDecl *FD) {
  if (auto pointerAuth = getMemberFunctionPointerAuthInfo(functionType)) {
    llvm::Constant **entry = nullptr;
    if (FD) {
      auto &cache =
          getOrCreateCache<ByDeclCacheTy>(SignedThunkPointers);
      entry = &cache[FD->getCanonicalDecl()];
      if (*entry)
        return llvm::ConstantExpr::getBitCast(*entry, pointer->getType());
    }

    pointer = getConstantSignedPointer(
        pointer, pointerAuth.getKey(), nullptr,
        cast_or_null<llvm::Constant>(pointerAuth.getDiscriminator()));

    if (entry)
      *entry = pointer;
  }

  return pointer;
}

llvm::Constant *
CodeGenModule::getMemberFunctionPointer(const FunctionDecl *FD, llvm::Type *Ty) {
  QualType functionType = FD->getType();
  functionType = getContext().getMemberPointerType(
      functionType, cast<CXXMethodDecl>(FD)->getParent()->getTypeForDecl());
  return getMemberFunctionPointer(getRawFunctionPointer(FD, Ty), functionType,
                                  FD);
}
