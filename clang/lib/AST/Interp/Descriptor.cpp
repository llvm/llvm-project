//===--- Descriptor.cpp - Types for the constexpr VM ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Descriptor.h"
#include "Boolean.h"
#include "Floating.h"
#include "FunctionPointer.h"
#include "Pointer.h"
#include "PrimType.h"
#include "Record.h"

using namespace clang;
using namespace clang::interp;

template <typename T>
static void ctorTy(Block *, std::byte *Ptr, bool, bool, bool,
                   const Descriptor *) {
  new (Ptr) T();
}

template <typename T>
static void dtorTy(Block *, std::byte *Ptr, const Descriptor *) {
  reinterpret_cast<T *>(Ptr)->~T();
}

template <typename T>
static void moveTy(Block *, const std::byte *Src, std::byte *Dst,
                   const Descriptor *) {
  const auto *SrcPtr = reinterpret_cast<const T *>(Src);
  auto *DstPtr = reinterpret_cast<T *>(Dst);
  new (DstPtr) T(std::move(*SrcPtr));
}

template <typename T>
static void ctorArrayTy(Block *, std::byte *Ptr, bool, bool, bool,
                        const Descriptor *D) {
  for (unsigned I = 0, NE = D->getNumElems(); I < NE; ++I) {
    new (&reinterpret_cast<T *>(Ptr)[I]) T();
  }
}

template <typename T>
static void dtorArrayTy(Block *, std::byte *Ptr, const Descriptor *D) {
  InitMap *IM = *reinterpret_cast<InitMap **>(Ptr);
  if (IM != (InitMap *)-1)
    free(IM);

  Ptr += sizeof(InitMap *);
  for (unsigned I = 0, NE = D->getNumElems(); I < NE; ++I) {
    reinterpret_cast<T *>(Ptr)[I].~T();
  }
}

template <typename T>
static void moveArrayTy(Block *, const std::byte *Src, std::byte *Dst,
                        const Descriptor *D) {
  for (unsigned I = 0, NE = D->getNumElems(); I < NE; ++I) {
    const auto *SrcPtr = &reinterpret_cast<const T *>(Src)[I];
    auto *DstPtr = &reinterpret_cast<T *>(Dst)[I];
    new (DstPtr) T(std::move(*SrcPtr));
  }
}

static void ctorArrayDesc(Block *B, std::byte *Ptr, bool IsConst,
                          bool IsMutable, bool IsActive, const Descriptor *D) {
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
    if (auto Fn = D->ElemDesc->CtorFn)
      Fn(B, ElemLoc, Desc->IsConst, Desc->IsFieldMutable, IsActive,
         D->ElemDesc);
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

static void moveArrayDesc(Block *B, const std::byte *Src, std::byte *Dst,
                          const Descriptor *D) {
  const unsigned NumElems = D->getNumElems();
  const unsigned ElemSize =
      D->ElemDesc->getAllocSize() + sizeof(InlineDescriptor);

  unsigned ElemOffset = 0;
  for (unsigned I = 0; I < NumElems; ++I, ElemOffset += ElemSize) {
    const auto *SrcPtr = Src + ElemOffset;
    auto *DstPtr = Dst + ElemOffset;

    const auto *SrcDesc = reinterpret_cast<const InlineDescriptor *>(SrcPtr);
    const auto *SrcElemLoc = reinterpret_cast<const std::byte *>(SrcDesc + 1);
    auto *DstDesc = reinterpret_cast<InlineDescriptor *>(DstPtr);
    auto *DstElemLoc = reinterpret_cast<std::byte *>(DstDesc + 1);

    *DstDesc = *SrcDesc;
    if (auto Fn = D->ElemDesc->MoveFn)
      Fn(B, SrcElemLoc, DstElemLoc, D->ElemDesc);
  }
}

static void ctorRecord(Block *B, std::byte *Ptr, bool IsConst, bool IsMutable,
                       bool IsActive, const Descriptor *D) {
  const bool IsUnion = D->ElemRecord->isUnion();
  auto CtorSub = [=](unsigned SubOff, Descriptor *F, bool IsBase) {
    auto *Desc = reinterpret_cast<InlineDescriptor *>(Ptr + SubOff) - 1;
    Desc->Offset = SubOff;
    Desc->Desc = F;
    Desc->IsInitialized = F->IsArray && !IsBase;
    Desc->IsBase = IsBase;
    Desc->IsActive = IsActive && !IsUnion;
    Desc->IsConst = IsConst || F->IsConst;
    Desc->IsFieldMutable = IsMutable || F->IsMutable;
    if (auto Fn = F->CtorFn)
      Fn(B, Ptr + SubOff, Desc->IsConst, Desc->IsFieldMutable, Desc->IsActive,
         F);
  };
  for (const auto &B : D->ElemRecord->bases())
    CtorSub(B.Offset, B.Desc, /*isBase=*/true);
  for (const auto &F : D->ElemRecord->fields())
    CtorSub(F.Offset, F.Desc, /*isBase=*/false);
  for (const auto &V : D->ElemRecord->virtual_bases())
    CtorSub(V.Offset, V.Desc, /*isBase=*/true);
}

static void dtorRecord(Block *B, std::byte *Ptr, const Descriptor *D) {
  auto DtorSub = [=](unsigned SubOff, Descriptor *F) {
    if (auto Fn = F->DtorFn)
      Fn(B, Ptr + SubOff, F);
  };
  for (const auto &F : D->ElemRecord->bases())
    DtorSub(F.Offset, F.Desc);
  for (const auto &F : D->ElemRecord->fields())
    DtorSub(F.Offset, F.Desc);
  for (const auto &F : D->ElemRecord->virtual_bases())
    DtorSub(F.Offset, F.Desc);
}

static void moveRecord(Block *B, const std::byte *Src, std::byte *Dst,
                       const Descriptor *D) {
  for (const auto &F : D->ElemRecord->fields()) {
    auto FieldOff = F.Offset;
    auto *FieldDesc = F.Desc;

    if (auto Fn = FieldDesc->MoveFn)
      Fn(B, Src + FieldOff, Dst + FieldOff, FieldDesc);
  }
}

static BlockCtorFn getCtorPrim(PrimType Type) {
  // Floating types are special. They are primitives, but need their
  // constructor called.
  if (Type == PT_Float)
    return ctorTy<PrimConv<PT_Float>::T>;

  COMPOSITE_TYPE_SWITCH(Type, return ctorTy<T>, return nullptr);
}

static BlockDtorFn getDtorPrim(PrimType Type) {
  // Floating types are special. They are primitives, but need their
  // destructor called, since they might allocate memory.
  if (Type == PT_Float)
    return dtorTy<PrimConv<PT_Float>::T>;

  COMPOSITE_TYPE_SWITCH(Type, return dtorTy<T>, return nullptr);
}

static BlockMoveFn getMovePrim(PrimType Type) {
  COMPOSITE_TYPE_SWITCH(Type, return moveTy<T>, return nullptr);
}

static BlockCtorFn getCtorArrayPrim(PrimType Type) {
  COMPOSITE_TYPE_SWITCH(Type, return ctorArrayTy<T>, return nullptr);
}

static BlockDtorFn getDtorArrayPrim(PrimType Type) {
  TYPE_SWITCH(Type, return dtorArrayTy<T>);
  llvm_unreachable("unknown Expr");
}

static BlockMoveFn getMoveArrayPrim(PrimType Type) {
  COMPOSITE_TYPE_SWITCH(Type, return moveArrayTy<T>, return nullptr);
}

Descriptor::Descriptor(const DeclTy &D, PrimType Type, MetadataSize MD,
                       bool IsConst, bool IsTemporary, bool IsMutable)
    : Source(D), ElemSize(primSize(Type)), Size(ElemSize),
      MDSize(MD.value_or(0)), AllocSize(align(Size + MDSize)), IsConst(IsConst),
      IsMutable(IsMutable), IsTemporary(IsTemporary), CtorFn(getCtorPrim(Type)),
      DtorFn(getDtorPrim(Type)), MoveFn(getMovePrim(Type)) {
  assert(AllocSize >= Size);
  assert(Source && "Missing source");
}

Descriptor::Descriptor(const DeclTy &D, PrimType Type, MetadataSize MD,
                       size_t NumElems, bool IsConst, bool IsTemporary,
                       bool IsMutable)
    : Source(D), ElemSize(primSize(Type)), Size(ElemSize * NumElems),
      MDSize(MD.value_or(0)),
      AllocSize(align(Size) + sizeof(InitMap *) + MDSize), IsConst(IsConst),
      IsMutable(IsMutable), IsTemporary(IsTemporary), IsArray(true),
      CtorFn(getCtorArrayPrim(Type)), DtorFn(getDtorArrayPrim(Type)),
      MoveFn(getMoveArrayPrim(Type)) {
  assert(Source && "Missing source");
}

Descriptor::Descriptor(const DeclTy &D, PrimType Type, bool IsTemporary,
                       UnknownSize)
    : Source(D), ElemSize(primSize(Type)), Size(UnknownSizeMark), MDSize(0),
      AllocSize(alignof(void *)), IsConst(true), IsMutable(false),
      IsTemporary(IsTemporary), IsArray(true), CtorFn(getCtorArrayPrim(Type)),
      DtorFn(getDtorArrayPrim(Type)), MoveFn(getMoveArrayPrim(Type)) {
  assert(Source && "Missing source");
}

Descriptor::Descriptor(const DeclTy &D, Descriptor *Elem, MetadataSize MD,
                       unsigned NumElems, bool IsConst, bool IsTemporary,
                       bool IsMutable)
    : Source(D), ElemSize(Elem->getAllocSize() + sizeof(InlineDescriptor)),
      Size(ElemSize * NumElems), MDSize(MD.value_or(0)),
      AllocSize(std::max<size_t>(alignof(void *), Size) + MDSize),
      ElemDesc(Elem), IsConst(IsConst), IsMutable(IsMutable),
      IsTemporary(IsTemporary), IsArray(true), CtorFn(ctorArrayDesc),
      DtorFn(dtorArrayDesc), MoveFn(moveArrayDesc) {
  assert(Source && "Missing source");
}

Descriptor::Descriptor(const DeclTy &D, Descriptor *Elem, bool IsTemporary,
                       UnknownSize)
    : Source(D), ElemSize(Elem->getAllocSize() + sizeof(InlineDescriptor)),
      Size(UnknownSizeMark), MDSize(0), AllocSize(alignof(void *)),
      ElemDesc(Elem), IsConst(true), IsMutable(false), IsTemporary(IsTemporary),
      IsArray(true), CtorFn(ctorArrayDesc), DtorFn(dtorArrayDesc),
      MoveFn(moveArrayDesc) {
  assert(Source && "Missing source");
}

Descriptor::Descriptor(const DeclTy &D, Record *R, MetadataSize MD,
                       bool IsConst, bool IsTemporary, bool IsMutable)
    : Source(D), ElemSize(std::max<size_t>(alignof(void *), R->getFullSize())),
      Size(ElemSize), MDSize(MD.value_or(0)), AllocSize(Size + MDSize),
      ElemRecord(R), IsConst(IsConst), IsMutable(IsMutable),
      IsTemporary(IsTemporary), CtorFn(ctorRecord), DtorFn(dtorRecord),
      MoveFn(moveRecord) {
  assert(Source && "Missing source");
}

QualType Descriptor::getType() const {
  if (auto *E = asExpr())
    return E->getType();
  if (auto *D = asValueDecl())
    return D->getType();
  if (auto *T = dyn_cast<TypeDecl>(asDecl()))
    return QualType(T->getTypeForDecl(), 0);
  llvm_unreachable("Invalid descriptor type");
}

QualType Descriptor::getElemQualType() const {
  assert(isArray());
  const auto *AT = cast<ArrayType>(getType());
  return AT->getElementType();
}

SourceLocation Descriptor::getLocation() const {
  if (auto *D = Source.dyn_cast<const Decl *>())
    return D->getLocation();
  if (auto *E = Source.dyn_cast<const Expr *>())
    return E->getExprLoc();
  llvm_unreachable("Invalid descriptor type");
}

InitMap::InitMap(unsigned N) : UninitFields(N) {
  std::fill_n(data(), (N + PER_FIELD - 1) / PER_FIELD, 0);
}

InitMap::T *InitMap::data() {
  auto *Start = reinterpret_cast<char *>(this) + align(sizeof(InitMap));
  return reinterpret_cast<T *>(Start);
}

const InitMap::T *InitMap::data() const {
  auto *Start = reinterpret_cast<const char *>(this) + align(sizeof(InitMap));
  return reinterpret_cast<const T *>(Start);
}

bool InitMap::initialize(unsigned I) {
  unsigned Bucket = I / PER_FIELD;
  T Mask = T(1) << (I % PER_FIELD);
  if (!(data()[Bucket] & Mask)) {
    data()[Bucket] |= Mask;
    UninitFields -= 1;
  }
  return UninitFields == 0;
}

bool InitMap::isInitialized(unsigned I) const {
  unsigned Bucket = I / PER_FIELD;
  return data()[Bucket] & (T(1) << (I % PER_FIELD));
}

InitMap *InitMap::allocate(unsigned N) {
  const size_t NumFields = ((N + PER_FIELD - 1) / PER_FIELD);
  const size_t Size = align(sizeof(InitMap)) + NumFields * PER_FIELD;
  return new (malloc(Size)) InitMap(N);
}
