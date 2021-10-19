//===-- CGBuilder.h - Choose IRBuilder implementation  ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGBUILDER_H
#define LLVM_CLANG_LIB_CODEGEN_CGBUILDER_H

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "Address.h"
#include "CodeGenTypeCache.h"

namespace clang {
namespace CodeGen {

class CodeGenFunction;

/// This is an IRBuilder insertion helper that forwards to
/// CodeGenFunction::InsertHelper, which adds necessary metadata to
/// instructions.
class CGBuilderInserter final : public llvm::IRBuilderDefaultInserter {
public:
  CGBuilderInserter() = default;
  explicit CGBuilderInserter(CodeGenFunction *CGF) : CGF(CGF) {}

  /// This forwards to CodeGenFunction::InsertHelper.
  void InsertHelper(llvm::Instruction *I, const llvm::Twine &Name,
                    llvm::BasicBlock *BB,
                    llvm::BasicBlock::iterator InsertPt) const override;
private:
  CodeGenFunction *CGF = nullptr;
};

typedef CGBuilderInserter CGBuilderInserterTy;

typedef llvm::IRBuilder<llvm::ConstantFolder, CGBuilderInserterTy>
    CGBuilderBaseTy;

class CGBuilderTy : public CGBuilderBaseTy {
  /// Storing a reference to the type cache here makes it a lot easier
  /// to build natural-feeling, target-specific IR.
  const CodeGenTypeCache &TypeCache;
public:
  CGBuilderTy(const CodeGenTypeCache &TypeCache, llvm::LLVMContext &C)
    : CGBuilderBaseTy(C), TypeCache(TypeCache) {}
  CGBuilderTy(const CodeGenTypeCache &TypeCache,
              llvm::LLVMContext &C, const llvm::ConstantFolder &F,
              const CGBuilderInserterTy &Inserter)
    : CGBuilderBaseTy(C, F, Inserter), TypeCache(TypeCache) {}
  CGBuilderTy(const CodeGenTypeCache &TypeCache, llvm::Instruction *I)
    : CGBuilderBaseTy(I), TypeCache(TypeCache) {}
  CGBuilderTy(const CodeGenTypeCache &TypeCache, llvm::BasicBlock *BB)
    : CGBuilderBaseTy(BB), TypeCache(TypeCache) {}

  llvm::ConstantInt *getSize(CharUnits N) {
    return llvm::ConstantInt::get(TypeCache.SizeTy, N.getQuantity());
  }
  llvm::ConstantInt *getSize(uint64_t N) {
    return llvm::ConstantInt::get(TypeCache.SizeTy, N);
  }

  // Note that we intentionally hide the CreateLoad APIs that don't
  // take an alignment.
  llvm::LoadInst *CreateLoad(Address Addr, const llvm::Twine &Name = "") {
    return CreateAlignedLoad(Addr.getElementType(), Addr.getPointer(),
                             Addr.getAlignment().getAsAlign(), Name);
  }
  llvm::LoadInst *CreateLoad(Address Addr, const char *Name) {
    // This overload is required to prevent string literals from
    // ending up in the IsVolatile overload.
    return CreateAlignedLoad(Addr.getElementType(), Addr.getPointer(),
                             Addr.getAlignment().getAsAlign(), Name);
  }
  llvm::LoadInst *CreateLoad(Address Addr, bool IsVolatile,
                             const llvm::Twine &Name = "") {
    return CreateAlignedLoad(Addr.getElementType(), Addr.getPointer(),
                             Addr.getAlignment().getAsAlign(), IsVolatile,
                             Name);
  }

  using CGBuilderBaseTy::CreateAlignedLoad;
  llvm::LoadInst *CreateAlignedLoad(llvm::Type *Ty, llvm::Value *Addr,
                                    CharUnits Align,
                                    const llvm::Twine &Name = "") {
    assert(Addr->getType()->getPointerElementType() == Ty);
    return CreateAlignedLoad(Ty, Addr, Align.getAsAlign(), Name);
  }

  // Note that we intentionally hide the CreateStore APIs that don't
  // take an alignment.
  llvm::StoreInst *CreateStore(llvm::Value *Val, Address Addr,
                               bool IsVolatile = false) {
    return CreateAlignedStore(Val, Addr.getPointer(),
                              Addr.getAlignment().getAsAlign(), IsVolatile);
  }

  using CGBuilderBaseTy::CreateAlignedStore;
  llvm::StoreInst *CreateAlignedStore(llvm::Value *Val, llvm::Value *Addr,
                                      CharUnits Align, bool IsVolatile = false) {
    return CreateAlignedStore(Val, Addr, Align.getAsAlign(), IsVolatile);
  }

  // FIXME: these "default-aligned" APIs should be removed,
  // but I don't feel like fixing all the builtin code right now.
  llvm::StoreInst *CreateDefaultAlignedStore(llvm::Value *Val,
                                             llvm::Value *Addr,
                                             bool IsVolatile = false) {
    return CGBuilderBaseTy::CreateStore(Val, Addr, IsVolatile);
  }

  /// Emit a load from an i1 flag variable.
  llvm::LoadInst *CreateFlagLoad(llvm::Value *Addr,
                                 const llvm::Twine &Name = "") {
    assert(Addr->getType()->getPointerElementType() == getInt1Ty());
    return CreateAlignedLoad(getInt1Ty(), Addr, CharUnits::One(), Name);
  }

  /// Emit a store to an i1 flag variable.
  llvm::StoreInst *CreateFlagStore(bool Value, llvm::Value *Addr) {
    assert(Addr->getType()->getPointerElementType() == getInt1Ty());
    return CreateAlignedStore(getInt1(Value), Addr, CharUnits::One());
  }

  // Temporarily use old signature; clang will be updated to an Address overload
  // in a subsequent patch.
  llvm::AtomicCmpXchgInst *
  CreateAtomicCmpXchg(llvm::Value *Ptr, llvm::Value *Cmp, llvm::Value *New,
                      llvm::AtomicOrdering SuccessOrdering,
                      llvm::AtomicOrdering FailureOrdering,
                      llvm::SyncScope::ID SSID = llvm::SyncScope::System) {
    return CGBuilderBaseTy::CreateAtomicCmpXchg(
        Ptr, Cmp, New, llvm::MaybeAlign(), SuccessOrdering, FailureOrdering,
        SSID);
  }

  // Temporarily use old signature; clang will be updated to an Address overload
  // in a subsequent patch.
  llvm::AtomicRMWInst *
  CreateAtomicRMW(llvm::AtomicRMWInst::BinOp Op, llvm::Value *Ptr,
                  llvm::Value *Val, llvm::AtomicOrdering Ordering,
                  llvm::SyncScope::ID SSID = llvm::SyncScope::System) {
    return CGBuilderBaseTy::CreateAtomicRMW(Op, Ptr, Val, llvm::MaybeAlign(),
                                            Ordering, SSID);
  }

  using CGBuilderBaseTy::CreateBitCast;
  Address CreateBitCast(Address Addr, llvm::Type *Ty,
                        const llvm::Twine &Name = "") {
    return Address(CreateBitCast(Addr.getPointer(), Ty, Name),
                   Addr.getAlignment());
  }

  using CGBuilderBaseTy::CreateAddrSpaceCast;
  Address CreateAddrSpaceCast(Address Addr, llvm::Type *Ty,
                              const llvm::Twine &Name = "") {
    return Address(CreateAddrSpaceCast(Addr.getPointer(), Ty, Name),
                   Addr.getAlignment());
  }

  /// Cast the element type of the given address to a different type,
  /// preserving information like the alignment and address space.
  Address CreateElementBitCast(Address Addr, llvm::Type *Ty,
                               const llvm::Twine &Name = "") {
    auto PtrTy = Ty->getPointerTo(Addr.getAddressSpace());
    return CreateBitCast(Addr, PtrTy, Name);
  }

  using CGBuilderBaseTy::CreatePointerBitCastOrAddrSpaceCast;
  Address CreatePointerBitCastOrAddrSpaceCast(Address Addr, llvm::Type *Ty,
                                              const llvm::Twine &Name = "") {
    llvm::Value *Ptr =
      CreatePointerBitCastOrAddrSpaceCast(Addr.getPointer(), Ty, Name);
    return Address(Ptr, Addr.getAlignment());
  }

  /// Given
  ///   %addr = {T1, T2...}* ...
  /// produce
  ///   %name = getelementptr inbounds %addr, i32 0, i32 index
  ///
  /// This API assumes that drilling into a struct like this is always an
  /// inbounds operation.
  using CGBuilderBaseTy::CreateStructGEP;
  Address CreateStructGEP(Address Addr, unsigned Index,
                          const llvm::Twine &Name = "") {
    llvm::StructType *ElTy = cast<llvm::StructType>(Addr.getElementType());
    const llvm::DataLayout &DL = BB->getParent()->getParent()->getDataLayout();
    const llvm::StructLayout *Layout = DL.getStructLayout(ElTy);
    auto Offset = CharUnits::fromQuantity(Layout->getElementOffset(Index));

    return Address(CreateStructGEP(Addr.getElementType(),
                                   Addr.getPointer(), Index, Name),
                   Addr.getAlignment().alignmentAtOffset(Offset));
  }

  /// Given
  ///   %addr = [n x T]* ...
  /// produce
  ///   %name = getelementptr inbounds %addr, i64 0, i64 index
  /// where i64 is actually the target word size.
  ///
  /// This API assumes that drilling into an array like this is always
  /// an inbounds operation.
  Address CreateConstArrayGEP(Address Addr, uint64_t Index,
                              const llvm::Twine &Name = "") {
    llvm::ArrayType *ElTy = cast<llvm::ArrayType>(Addr.getElementType());
    const llvm::DataLayout &DL = BB->getParent()->getParent()->getDataLayout();
    CharUnits EltSize =
        CharUnits::fromQuantity(DL.getTypeAllocSize(ElTy->getElementType()));

    return Address(
        CreateInBoundsGEP(Addr.getElementType(), Addr.getPointer(),
                          {getSize(CharUnits::Zero()), getSize(Index)}, Name),
        Addr.getAlignment().alignmentAtOffset(Index * EltSize));
  }

  /// Given
  ///   %addr = T* ...
  /// produce
  ///   %name = getelementptr inbounds %addr, i64 index
  /// where i64 is actually the target word size.
  Address CreateConstInBoundsGEP(Address Addr, uint64_t Index,
                                 const llvm::Twine &Name = "") {
    llvm::Type *ElTy = Addr.getElementType();
    const llvm::DataLayout &DL = BB->getParent()->getParent()->getDataLayout();
    CharUnits EltSize = CharUnits::fromQuantity(DL.getTypeAllocSize(ElTy));

    return Address(CreateInBoundsGEP(Addr.getElementType(), Addr.getPointer(),
                                     getSize(Index), Name),
                   Addr.getAlignment().alignmentAtOffset(Index * EltSize));
  }

  /// Given
  ///   %addr = T* ...
  /// produce
  ///   %name = getelementptr inbounds %addr, i64 index
  /// where i64 is actually the target word size.
  Address CreateConstGEP(Address Addr, uint64_t Index,
                         const llvm::Twine &Name = "") {
    const llvm::DataLayout &DL = BB->getParent()->getParent()->getDataLayout();
    CharUnits EltSize =
        CharUnits::fromQuantity(DL.getTypeAllocSize(Addr.getElementType()));

    return Address(CreateGEP(Addr.getElementType(), Addr.getPointer(),
                             getSize(Index), Name),
                   Addr.getAlignment().alignmentAtOffset(Index * EltSize));
  }

  /// Given a pointer to i8, adjust it by a given constant offset.
  Address CreateConstInBoundsByteGEP(Address Addr, CharUnits Offset,
                                     const llvm::Twine &Name = "") {
    assert(Addr.getElementType() == TypeCache.Int8Ty);
    return Address(CreateInBoundsGEP(Addr.getElementType(), Addr.getPointer(),
                                     getSize(Offset), Name),
                   Addr.getAlignment().alignmentAtOffset(Offset));
  }
  Address CreateConstByteGEP(Address Addr, CharUnits Offset,
                             const llvm::Twine &Name = "") {
    assert(Addr.getElementType() == TypeCache.Int8Ty);
    return Address(CreateGEP(Addr.getElementType(), Addr.getPointer(),
                             getSize(Offset), Name),
                   Addr.getAlignment().alignmentAtOffset(Offset));
  }

  using CGBuilderBaseTy::CreateConstInBoundsGEP2_32;
  Address CreateConstInBoundsGEP2_32(Address Addr, unsigned Idx0, unsigned Idx1,
                                     const llvm::Twine &Name = "") {
    const llvm::DataLayout &DL = BB->getParent()->getParent()->getDataLayout();

    auto *GEP = cast<llvm::GetElementPtrInst>(CreateConstInBoundsGEP2_32(
        Addr.getElementType(), Addr.getPointer(), Idx0, Idx1, Name));
    llvm::APInt Offset(
        DL.getIndexSizeInBits(Addr.getType()->getPointerAddressSpace()), 0,
        /*isSigned=*/true);
    if (!GEP->accumulateConstantOffset(DL, Offset))
      llvm_unreachable("offset of GEP with constants is always computable");
    return Address(GEP, Addr.getAlignment().alignmentAtOffset(
                            CharUnits::fromQuantity(Offset.getSExtValue())));
  }

  using CGBuilderBaseTy::CreateMemCpy;
  llvm::CallInst *CreateMemCpy(Address Dest, Address Src, llvm::Value *Size,
                               bool IsVolatile = false) {
    return CreateMemCpy(Dest.getPointer(), Dest.getAlignment().getAsAlign(),
                        Src.getPointer(), Src.getAlignment().getAsAlign(), Size,
                        IsVolatile);
  }
  llvm::CallInst *CreateMemCpy(Address Dest, Address Src, uint64_t Size,
                               bool IsVolatile = false) {
    return CreateMemCpy(Dest.getPointer(), Dest.getAlignment().getAsAlign(),
                        Src.getPointer(), Src.getAlignment().getAsAlign(), Size,
                        IsVolatile);
  }

  using CGBuilderBaseTy::CreateMemCpyInline;
  llvm::CallInst *CreateMemCpyInline(Address Dest, Address Src, uint64_t Size) {
    return CreateMemCpyInline(
        Dest.getPointer(), Dest.getAlignment().getAsAlign(), Src.getPointer(),
        Src.getAlignment().getAsAlign(), getInt64(Size));
  }

  using CGBuilderBaseTy::CreateMemMove;
  llvm::CallInst *CreateMemMove(Address Dest, Address Src, llvm::Value *Size,
                                bool IsVolatile = false) {
    return CreateMemMove(Dest.getPointer(), Dest.getAlignment().getAsAlign(),
                         Src.getPointer(), Src.getAlignment().getAsAlign(),
                         Size, IsVolatile);
  }

  using CGBuilderBaseTy::CreateMemSet;
  llvm::CallInst *CreateMemSet(Address Dest, llvm::Value *Value,
                               llvm::Value *Size, bool IsVolatile = false) {
    return CreateMemSet(Dest.getPointer(), Value, Size,
                        Dest.getAlignment().getAsAlign(), IsVolatile);
  }

  using CGBuilderBaseTy::CreatePreserveStructAccessIndex;
  Address CreatePreserveStructAccessIndex(Address Addr,
                                          unsigned Index,
                                          unsigned FieldIndex,
                                          llvm::MDNode *DbgInfo) {
    llvm::StructType *ElTy = cast<llvm::StructType>(Addr.getElementType());
    const llvm::DataLayout &DL = BB->getParent()->getParent()->getDataLayout();
    const llvm::StructLayout *Layout = DL.getStructLayout(ElTy);
    auto Offset = CharUnits::fromQuantity(Layout->getElementOffset(Index));

    return Address(CreatePreserveStructAccessIndex(ElTy, Addr.getPointer(),
                                                   Index, FieldIndex, DbgInfo),
                   Addr.getAlignment().alignmentAtOffset(Offset));
  }
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
