//===--- Descriptor.cpp - Types for the constexpr VM ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Descriptor.h"
#include "Boolean.h"
#include "FixedPoint.h"
#include "Floating.h"
#include "IntegralAP.h"
#include "MemberPointer.h"
#include "Pointer.h"
#include "PrimType.h"
#include "Record.h"
#include "Source.h"
#include "clang/AST/ExprCXX.h"

using namespace clang;
using namespace clang::interp;

template <typename T> static constexpr bool needsCtor() {
  if constexpr (std::is_same_v<T, Integral<8, true>> ||
                std::is_same_v<T, Integral<8, false>> ||
                std::is_same_v<T, Integral<16, true>> ||
                std::is_same_v<T, Integral<16, false>> ||
                std::is_same_v<T, Integral<32, true>> ||
                std::is_same_v<T, Integral<32, false>> ||
                std::is_same_v<T, Integral<64, true>> ||
                std::is_same_v<T, Integral<64, false>> ||
                std::is_same_v<T, Boolean>)
    return false;

  return true;
}

template <typename T>
static void ctorTy(Block *, std::byte *Ptr, bool, bool, bool, bool, bool,
                   const Descriptor *) {
  static_assert(needsCtor<T>());
  new (Ptr) T();
}

template <typename T>
static void dtorTy(Block *, std::byte *Ptr, const Descriptor *) {
  static_assert(needsCtor<T>());
  reinterpret_cast<T *>(Ptr)->~T();
}

template <typename T>
static void moveTy(Block *, std::byte *Src, std::byte *Dst,
                   const Descriptor *) {
  auto *SrcPtr = reinterpret_cast<T *>(Src);
  auto *DstPtr = reinterpret_cast<T *>(Dst);
  new (DstPtr) T(std::move(*SrcPtr));
}

template <typename T>
static void ctorArrayTy(Block *, std::byte *Ptr, bool, bool, bool, bool, bool,
                        const Descriptor *D) {
  new (Ptr) InitMapPtr(std::nullopt);

  if constexpr (needsCtor<T>()) {
    Ptr += sizeof(InitMapPtr);
    for (unsigned I = 0, NE = D->getNumElems(); I < NE; ++I) {
      new (&reinterpret_cast<T *>(Ptr)[I]) T();
    }
  }
}

template <typename T>
static void dtorArrayTy(Block *, std::byte *Ptr, const Descriptor *D) {
  InitMapPtr &IMP = *reinterpret_cast<InitMapPtr *>(Ptr);

  if (IMP)
    IMP = std::nullopt;

  if constexpr (needsCtor<T>()) {
    Ptr += sizeof(InitMapPtr);
    for (unsigned I = 0, NE = D->getNumElems(); I < NE; ++I) {
      reinterpret_cast<T *>(Ptr)[I].~T();
    }
  }
}

template <typename T>
static void moveArrayTy(Block *, std::byte *Src, std::byte *Dst,
                        const Descriptor *D) {
  InitMapPtr &SrcIMP = *reinterpret_cast<InitMapPtr *>(Src);
  if (SrcIMP) {
    // We only ever invoke the moveFunc when moving block contents to a
    // DeadBlock. DeadBlocks don't need InitMaps, so we destroy them here.
    SrcIMP = std::nullopt;
  }
  Src += sizeof(InitMapPtr);
  Dst += sizeof(InitMapPtr);
  if constexpr (!needsCtor<T>()) {
    std::memcpy(Dst, Src, D->getNumElems() * D->getElemSize());
  } else {
    for (unsigned I = 0, NE = D->getNumElems(); I < NE; ++I) {
      auto *SrcPtr = &reinterpret_cast<T *>(Src)[I];
      auto *DstPtr = &reinterpret_cast<T *>(Dst)[I];
      new (DstPtr) T(std::move(*SrcPtr));
    }
  }
}

static void ctorArrayDesc(Block *B, std::byte *Ptr, bool IsConst,
                          bool IsMutable, bool IsVolatile, bool IsActive,
                          bool InUnion, const Descriptor *D) {
  const unsigned NumElems = D->getNumElems();
  const unsigned ElemSize =
      D->ElemDesc->getAllocSize() + sizeof(InlineDescriptor);

  unsigned ElemOffset = 0;
  for (unsigned I = 0; I < NumElems; ++I, ElemOffset += ElemSize) {
    auto *ElemPtr = Ptr + ElemOffset;
    auto *Desc = reinterpret_cast<InlineDescriptor *>(ElemPtr);
    auto *ElemLoc = reinterpret_cast<std::byte *>(Desc + 1);
    auto *SD = D->ElemDesc;

    Desc->Offset = ElemOffset + sizeof(InlineDescriptor);
    Desc->Desc = SD;
    Desc->IsInitialized = true;
    Desc->IsBase = false;
    Desc->IsActive = IsActive;
    Desc->IsConst = IsConst || D->IsConst;
    Desc->IsFieldMutable = IsMutable || D->IsMutable;
    Desc->InUnion = InUnion;
    Desc->IsArrayElement = true;
    Desc->IsVolatile = IsVolatile;

    if (auto Fn = D->ElemDesc->CtorFn)
      Fn(B, ElemLoc, Desc->IsConst, Desc->IsFieldMutable, IsVolatile, IsActive,
         Desc->InUnion || SD->isUnion(), D->ElemDesc);
  }
}

static void dtorArrayDesc(Block *B, std::byte *Ptr, const Descriptor *D) {
  const unsigned NumElems = D->getNumElems();
  const unsigned ElemSize =
      D->ElemDesc->getAllocSize() + sizeof(InlineDescriptor);

  unsigned ElemOffset = 0;
  for (unsigned I = 0; I < NumElems; ++I, ElemOffset += ElemSize) {
    auto *ElemPtr = Ptr + ElemOffset;
    auto *Desc = reinterpret_cast<InlineDescriptor *>(ElemPtr);
    auto *ElemLoc = reinterpret_cast<std::byte *>(Desc + 1);
    if (auto Fn = D->ElemDesc->DtorFn)
      Fn(B, ElemLoc, D->ElemDesc);
  }
}

static void initField(Block *B, std::byte *Ptr, bool IsConst, bool IsMutable,
                      bool IsVolatile, bool IsActive, bool IsUnionField,
                      bool InUnion, const Descriptor *D, unsigned FieldOffset) {
  auto *Desc = reinterpret_cast<InlineDescriptor *>(Ptr + FieldOffset) - 1;
  Desc->Offset = FieldOffset;
  Desc->Desc = D;
  Desc->IsInitialized = D->IsArray;
  Desc->IsBase = false;
  Desc->IsActive = IsActive && !IsUnionField;
  Desc->InUnion = InUnion;
  Desc->IsConst = IsConst || D->IsConst;
  Desc->IsFieldMutable = IsMutable || D->IsMutable;
  Desc->IsVolatile = IsVolatile || D->IsVolatile;
  // True if this field is const AND the parent is mutable.
  Desc->IsConstInMutable = Desc->IsConst && IsMutable;

  if (auto Fn = D->CtorFn)
    Fn(B, Ptr + FieldOffset, Desc->IsConst, Desc->IsFieldMutable,
       Desc->IsVolatile, Desc->IsActive, InUnion || D->isUnion(), D);
}

static void initBase(Block *B, std::byte *Ptr, bool IsConst, bool IsMutable,
                     bool IsVolatile, bool IsActive, bool InUnion,
                     const Descriptor *D, unsigned FieldOffset,
                     bool IsVirtualBase) {
  assert(D);
  assert(D->ElemRecord);
  assert(!D->ElemRecord->isUnion()); // Unions cannot be base classes.

  auto *Desc = reinterpret_cast<InlineDescriptor *>(Ptr + FieldOffset) - 1;
  Desc->Offset = FieldOffset;
  Desc->Desc = D;
  Desc->IsInitialized = D->IsArray;
  Desc->IsBase = true;
  Desc->IsVirtualBase = IsVirtualBase;
  Desc->IsActive = IsActive && !InUnion;
  Desc->IsConst = IsConst || D->IsConst;
  Desc->IsFieldMutable = IsMutable || D->IsMutable;
  Desc->InUnion = InUnion;
  Desc->IsVolatile = false;

  for (const auto &V : D->ElemRecord->bases())
    initBase(B, Ptr + FieldOffset, IsConst, IsMutable, IsVolatile, IsActive,
             InUnion, V.Desc, V.Offset, false);
  for (const auto &F : D->ElemRecord->fields())
    initField(B, Ptr + FieldOffset, IsConst, IsMutable, IsVolatile, IsActive,
              InUnion, InUnion, F.Desc, F.Offset);
}

static void ctorRecord(Block *B, std::byte *Ptr, bool IsConst, bool IsMutable,
                       bool IsVolatile, bool IsActive, bool InUnion,
                       const Descriptor *D) {
  for (const auto &V : D->ElemRecord->bases())
    initBase(B, Ptr, IsConst, IsMutable, IsVolatile, IsActive, InUnion, V.Desc,
             V.Offset,
             /*IsVirtualBase=*/false);
  for (const auto &F : D->ElemRecord->fields()) {
    bool IsUnionField = D->isUnion();
    initField(B, Ptr, IsConst, IsMutable, IsVolatile, IsActive, IsUnionField,
              InUnion || IsUnionField, F.Desc, F.Offset);
  }
  for (const auto &V : D->ElemRecord->virtual_bases())
    initBase(B, Ptr, IsConst, IsMutable, IsVolatile, IsActive, InUnion, V.Desc,
             V.Offset,
             /*IsVirtualBase=*/true);
}

static void destroyField(Block *B, std::byte *Ptr, const Descriptor *D,
                         unsigned FieldOffset) {
  if (auto Fn = D->DtorFn)
    Fn(B, Ptr + FieldOffset, D);
}

static void destroyBase(Block *B, std::byte *Ptr, const Descriptor *D,
                        unsigned FieldOffset) {
  assert(D);
  assert(D->ElemRecord);

  for (const auto &V : D->ElemRecord->bases())
    destroyBase(B, Ptr + FieldOffset, V.Desc, V.Offset);
  for (const auto &F : D->ElemRecord->fields())
    destroyField(B, Ptr + FieldOffset, F.Desc, F.Offset);
}

static void dtorRecord(Block *B, std::byte *Ptr, const Descriptor *D) {
  for (const auto &F : D->ElemRecord->bases())
    destroyBase(B, Ptr, F.Desc, F.Offset);
  for (const auto &F : D->ElemRecord->fields())
    destroyField(B, Ptr, F.Desc, F.Offset);
  for (const auto &F : D->ElemRecord->virtual_bases())
    destroyBase(B, Ptr, F.Desc, F.Offset);
}

static BlockCtorFn getCtorPrim(PrimType Type) {
  // Floating types are special. They are primitives, but need their
  // constructor called.
  if (Type == PT_Float)
    return ctorTy<PrimConv<PT_Float>::T>;
  if (Type == PT_IntAP)
    return ctorTy<PrimConv<PT_IntAP>::T>;
  if (Type == PT_IntAPS)
    return ctorTy<PrimConv<PT_IntAPS>::T>;
  if (Type == PT_MemberPtr)
    return ctorTy<PrimConv<PT_MemberPtr>::T>;

  COMPOSITE_TYPE_SWITCH(Type, return ctorTy<T>, return nullptr);
}

static BlockDtorFn getDtorPrim(PrimType Type) {
  // Floating types are special. They are primitives, but need their
  // destructor called, since they might allocate memory.
  if (Type == PT_Float)
    return dtorTy<PrimConv<PT_Float>::T>;
  if (Type == PT_IntAP)
    return dtorTy<PrimConv<PT_IntAP>::T>;
  if (Type == PT_IntAPS)
    return dtorTy<PrimConv<PT_IntAPS>::T>;
  if (Type == PT_MemberPtr)
    return dtorTy<PrimConv<PT_MemberPtr>::T>;

  COMPOSITE_TYPE_SWITCH(Type, return dtorTy<T>, return nullptr);
}

static BlockCtorFn getCtorArrayPrim(PrimType Type) {
  TYPE_SWITCH(Type, return ctorArrayTy<T>);
  llvm_unreachable("unknown Expr");
}

static BlockDtorFn getDtorArrayPrim(PrimType Type) {
  TYPE_SWITCH(Type, return dtorArrayTy<T>);
  llvm_unreachable("unknown Expr");
}

/// Primitives.
Descriptor::Descriptor(const DeclTy &D, const Type *SourceTy, PrimType Type,
                       MetadataSize MD, bool IsConst, bool IsTemporary,
                       bool IsMutable, bool IsVolatile)
    : Source(D), SourceType(SourceTy), ElemSize(primSize(Type)), Size(ElemSize),
      MDSize(MD.value_or(0)), AllocSize(align(Size + MDSize)), PrimT(Type),
      IsConst(IsConst), IsMutable(IsMutable), IsTemporary(IsTemporary),
      IsVolatile(IsVolatile), CtorFn(getCtorPrim(Type)),
      DtorFn(getDtorPrim(Type)) {
  assert(AllocSize >= Size);
  assert(Source && "Missing source");
}

/// Primitive arrays.
Descriptor::Descriptor(const DeclTy &D, PrimType Type, MetadataSize MD,
                       size_t NumElems, bool IsConst, bool IsTemporary,
                       bool IsMutable)
    : Source(D), ElemSize(primSize(Type)), Size(ElemSize * NumElems),
      MDSize(MD.value_or(0)),
      AllocSize(align(MDSize) + align(Size) + sizeof(InitMapPtr)), PrimT(Type),
      IsConst(IsConst), IsMutable(IsMutable), IsTemporary(IsTemporary),
      IsArray(true), CtorFn(getCtorArrayPrim(Type)),
      DtorFn(getDtorArrayPrim(Type)) {
  assert(Source && "Missing source");
  assert(NumElems <= (MaxArrayElemBytes / ElemSize));
}

/// Primitive unknown-size arrays.
Descriptor::Descriptor(const DeclTy &D, PrimType Type, MetadataSize MD,
                       bool IsTemporary, bool IsConst, UnknownSize)
    : Source(D), ElemSize(primSize(Type)), Size(UnknownSizeMark),
      MDSize(MD.value_or(0)),
      AllocSize(MDSize + sizeof(InitMapPtr) + alignof(void *)), PrimT(Type),
      IsConst(IsConst), IsMutable(false), IsTemporary(IsTemporary),
      IsArray(true), CtorFn(getCtorArrayPrim(Type)),
      DtorFn(getDtorArrayPrim(Type)) {
  assert(Source && "Missing source");
}

/// Arrays of composite elements.
Descriptor::Descriptor(const DeclTy &D, const Type *SourceTy,
                       const Descriptor *Elem, MetadataSize MD,
                       unsigned NumElems, bool IsConst, bool IsTemporary,
                       bool IsMutable)
    : Source(D), SourceType(SourceTy),
      ElemSize(Elem->getAllocSize() + sizeof(InlineDescriptor)),
      Size(ElemSize * NumElems), MDSize(MD.value_or(0)),
      AllocSize(std::max<size_t>(alignof(void *), Size) + MDSize),
      ElemDesc(Elem), IsConst(IsConst), IsMutable(IsMutable),
      IsTemporary(IsTemporary), IsArray(true), CtorFn(ctorArrayDesc),
      DtorFn(dtorArrayDesc) {
  assert(Source && "Missing source");
}

/// Unknown-size arrays of composite elements.
Descriptor::Descriptor(const DeclTy &D, const Descriptor *Elem, MetadataSize MD,
                       bool IsTemporary, UnknownSize)
    : Source(D), ElemSize(Elem->getAllocSize() + sizeof(InlineDescriptor)),
      Size(UnknownSizeMark), MDSize(MD.value_or(0)),
      AllocSize(MDSize + alignof(void *)), ElemDesc(Elem), IsConst(true),
      IsMutable(false), IsTemporary(IsTemporary), IsArray(true),
      CtorFn(ctorArrayDesc), DtorFn(dtorArrayDesc) {
  assert(Source && "Missing source");
}

/// Composite records.
Descriptor::Descriptor(const DeclTy &D, const Record *R, MetadataSize MD,
                       bool IsConst, bool IsTemporary, bool IsMutable,
                       bool IsVolatile)
    : Source(D), ElemSize(std::max<size_t>(alignof(void *), R->getFullSize())),
      Size(ElemSize), MDSize(MD.value_or(0)), AllocSize(Size + MDSize),
      ElemRecord(R), IsConst(IsConst), IsMutable(IsMutable),
      IsTemporary(IsTemporary), IsVolatile(IsVolatile), CtorFn(ctorRecord),
      DtorFn(dtorRecord) {
  assert(Source && "Missing source");
}

/// Dummy.
Descriptor::Descriptor(const DeclTy &D, MetadataSize MD)
    : Source(D), ElemSize(1), Size(1), MDSize(MD.value_or(0)),
      AllocSize(MDSize), ElemRecord(nullptr), IsConst(true), IsMutable(false),
      IsTemporary(false) {
  assert(Source && "Missing source");
}

QualType Descriptor::getType() const {
  if (SourceType)
    return QualType(SourceType, 0);
  if (const auto *D = asValueDecl())
    return D->getType();
  if (const auto *T = dyn_cast_if_present<TypeDecl>(asDecl()))
    return T->getASTContext().getTypeDeclType(T);

  // The Source sometimes has a different type than the once
  // we really save. Try to consult the Record first.
  if (isRecord()) {
    const RecordDecl *RD = ElemRecord->getDecl();
    return RD->getASTContext().getCanonicalTagType(RD);
  }
  if (const auto *E = asExpr())
    return E->getType();
  llvm_unreachable("Invalid descriptor type");
}

QualType Descriptor::getElemQualType() const {
  assert(isArray());
  QualType T = getType();
  if (T->isPointerOrReferenceType())
    T = T->getPointeeType();

  if (const auto *AT = T->getAsArrayTypeUnsafe()) {
    // For primitive arrays, we don't save a QualType at all,
    // just a PrimType. Try to figure out the QualType here.
    if (isPrimitiveArray()) {
      while (T->isArrayType())
        T = T->getAsArrayTypeUnsafe()->getElementType();
      return T;
    }
    return AT->getElementType();
  }
  if (const auto *CT = T->getAs<ComplexType>())
    return CT->getElementType();
  if (const auto *CT = T->getAs<VectorType>())
    return CT->getElementType();

  return T;
}

QualType Descriptor::getDataType(const ASTContext &Ctx) const {
  auto MakeArrayType = [&](QualType ElemType) -> QualType {
    if (IsArray)
      return Ctx.getConstantArrayType(
          ElemType, APInt(64, static_cast<uint64_t>(getNumElems()), false),
          nullptr, ArraySizeModifier::Normal, 0);
    return ElemType;
  };

  if (const auto *E = asExpr()) {
    if (isa<CXXNewExpr>(E))
      return MakeArrayType(E->getType()->getPointeeType());

    // std::allocator.allocate() call.
    if (const auto *ME = dyn_cast<CXXMemberCallExpr>(E);
        ME && ME->getRecordDecl()->getName() == "allocator" &&
        ME->getMethodDecl()->getName() == "allocate")
      return MakeArrayType(E->getType()->getPointeeType());
    return E->getType();
  }

  return getType();
}

SourceLocation Descriptor::getLocation() const {
  if (auto *D = dyn_cast<const Decl *>(Source))
    return D->getLocation();
  if (auto *E = dyn_cast<const Expr *>(Source))
    return E->getExprLoc();
  llvm_unreachable("Invalid descriptor type");
}

SourceInfo Descriptor::getLoc() const {
  if (const auto *D = dyn_cast<const Decl *>(Source))
    return SourceInfo(D);
  if (const auto *E = dyn_cast<const Expr *>(Source))
    return SourceInfo(E);
  llvm_unreachable("Invalid descriptor type");
}

bool Descriptor::hasTrivialDtor() const {
  if (isPrimitive() || isPrimitiveArray())
    return true;

  if (isRecord()) {
    assert(ElemRecord);
    const CXXDestructorDecl *Dtor = ElemRecord->getDestructor();
    return !Dtor || Dtor->isTrivial();
  }

  if (!ElemDesc)
    return true;
  // Composite arrays.
  return ElemDesc->hasTrivialDtor();
}

bool Descriptor::isUnion() const { return isRecord() && ElemRecord->isUnion(); }

InitMap::InitMap(unsigned N)
    : UninitFields(N), Data(std::make_unique<T[]>(numFields(N))) {}

bool InitMap::initializeElement(unsigned I) {
  unsigned Bucket = I / PER_FIELD;
  T Mask = T(1) << (I % PER_FIELD);
  if (!(data()[Bucket] & Mask)) {
    data()[Bucket] |= Mask;
    UninitFields -= 1;
  }
  return UninitFields == 0;
}

bool InitMap::isElementInitialized(unsigned I) const {
  unsigned Bucket = I / PER_FIELD;
  return data()[Bucket] & (T(1) << (I % PER_FIELD));
}
