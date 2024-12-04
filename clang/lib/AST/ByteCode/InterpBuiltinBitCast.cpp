//===-------------------- InterpBuiltinBitCast.cpp --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "InterpBuiltinBitCast.h"
#include "BitcastBuffer.h"
#include "Boolean.h"
#include "Context.h"
#include "Floating.h"
#include "Integral.h"
#include "InterpState.h"
#include "MemberPointer.h"
#include "Pointer.h"
#include "Record.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/TargetInfo.h"

using namespace clang;
using namespace clang::interp;

/// Implement __builtin_bit_cast and related operations.
/// Since our internal representation for data is more complex than
/// something we can simply memcpy or memcmp, we first bitcast all the data
/// into a buffer, which we then later use to copy the data into the target.

// TODO:
//  - Try to minimize heap allocations.
//  - Optimize the common case of only pushing and pulling full
//    bytes to/from the buffer.

/// Used to iterate over pointer fields.
using DataFunc = llvm::function_ref<bool(const Pointer &P, PrimType Ty,
                                         Bits BitOffset, bool PackedBools)>;

#define BITCAST_TYPE_SWITCH(Expr, B)                                           \
  do {                                                                         \
    switch (Expr) {                                                            \
      TYPE_SWITCH_CASE(PT_Sint8, B)                                            \
      TYPE_SWITCH_CASE(PT_Uint8, B)                                            \
      TYPE_SWITCH_CASE(PT_Sint16, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint16, B)                                           \
      TYPE_SWITCH_CASE(PT_Sint32, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint32, B)                                           \
      TYPE_SWITCH_CASE(PT_Sint64, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint64, B)                                           \
      TYPE_SWITCH_CASE(PT_IntAP, B)                                            \
      TYPE_SWITCH_CASE(PT_IntAPS, B)                                           \
      TYPE_SWITCH_CASE(PT_Bool, B)                                             \
    default:                                                                   \
      llvm_unreachable("Unhandled bitcast type");                              \
    }                                                                          \
  } while (0)

#define BITCAST_TYPE_SWITCH_FIXED_SIZE(Expr, B)                                \
  do {                                                                         \
    switch (Expr) {                                                            \
      TYPE_SWITCH_CASE(PT_Sint8, B)                                            \
      TYPE_SWITCH_CASE(PT_Uint8, B)                                            \
      TYPE_SWITCH_CASE(PT_Sint16, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint16, B)                                           \
      TYPE_SWITCH_CASE(PT_Sint32, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint32, B)                                           \
      TYPE_SWITCH_CASE(PT_Sint64, B)                                           \
      TYPE_SWITCH_CASE(PT_Uint64, B)                                           \
      TYPE_SWITCH_CASE(PT_Bool, B)                                             \
    default:                                                                   \
      llvm_unreachable("Unhandled bitcast type");                              \
    }                                                                          \
  } while (0)

static void swapBytes(std::byte *M, size_t N) {
  for (size_t I = 0; I != (N / 2); ++I)
    std::swap(M[I], M[N - 1 - I]);
}

/// We use this to recursively iterate over all fields and elements of a pointer
/// and extract relevant data for a bitcast.
static bool enumerateData(const Pointer &P, const Context &Ctx, Bits Offset,
                          Bits BitsToRead, DataFunc F) {
  const Descriptor *FieldDesc = P.getFieldDesc();
  assert(FieldDesc);

  // Primitives.
  if (FieldDesc->isPrimitive())
    return F(P, FieldDesc->getPrimType(), Offset, /*PackedBools=*/false);

  // Primitive arrays.
  if (FieldDesc->isPrimitiveArray()) {
    QualType ElemType = FieldDesc->getElemQualType();
    size_t ElemSizeInBits = Ctx.getASTContext().getTypeSize(ElemType);
    PrimType ElemT = *Ctx.classify(ElemType);
    // Special case, since the bools here are packed.
    bool PackedBools = FieldDesc->getType()->isExtVectorBoolType();
    unsigned NumElems = FieldDesc->getNumElems();
    bool Ok = true;
    for (unsigned I = P.getIndex(); I != NumElems; ++I) {
      Ok = Ok && F(P.atIndex(I), ElemT, Offset, PackedBools);
      Offset += PackedBools ? 1 : ElemSizeInBits;
      if (Offset >= BitsToRead)
        break;
    }
    return Ok;
  }

  // Composite arrays.
  if (FieldDesc->isCompositeArray()) {
    QualType ElemType = FieldDesc->getElemQualType();
    size_t ElemSizeInBits = Ctx.getASTContext().getTypeSize(ElemType);
    for (unsigned I = 0; I != FieldDesc->getNumElems(); ++I) {
      enumerateData(P.atIndex(I).narrow(), Ctx, Offset, BitsToRead, F);
      Offset += ElemSizeInBits;
      if (Offset >= BitsToRead)
        break;
    }
    return true;
  }

  // Records.
  if (FieldDesc->isRecord()) {
    const Record *R = FieldDesc->ElemRecord;
    const ASTRecordLayout &Layout =
        Ctx.getASTContext().getASTRecordLayout(R->getDecl());
    bool Ok = true;

    for (const Record::Field &Fi : R->fields()) {
      Pointer Elem = P.atField(Fi.Offset);
      Bits BitOffset =
          Offset + Bits(Layout.getFieldOffset(Fi.Decl->getFieldIndex()));
      Ok = Ok && enumerateData(Elem, Ctx, BitOffset, BitsToRead, F);
    }
    for (const Record::Base &B : R->bases()) {
      Pointer Elem = P.atField(B.Offset);
      CharUnits ByteOffset =
          Layout.getBaseClassOffset(cast<CXXRecordDecl>(B.Decl));
      Bits BitOffset = Offset + Bits(Ctx.getASTContext().toBits(ByteOffset));
      Ok = Ok && enumerateData(Elem, Ctx, BitOffset, BitsToRead, F);
    }

    return Ok;
  }

  llvm_unreachable("Unhandled data type");
}

static bool enumeratePointerFields(const Pointer &P, const Context &Ctx,
                                   Bits BitsToRead, DataFunc F) {
  return enumerateData(P, Ctx, Bits::zero(), BitsToRead, F);
}

//  This function is constexpr if and only if To, From, and the types of
//  all subobjects of To and From are types T such that...
//  (3.1) - is_union_v<T> is false;
//  (3.2) - is_pointer_v<T> is false;
//  (3.3) - is_member_pointer_v<T> is false;
//  (3.4) - is_volatile_v<T> is false; and
//  (3.5) - T has no non-static data members of reference type
//
// NOTE: This is a version of checkBitCastConstexprEligibilityType() in
// ExprConstant.cpp.
static bool CheckBitcastType(InterpState &S, CodePtr OpPC, QualType T,
                             bool IsToType) {
  enum {
    E_Union = 0,
    E_Pointer,
    E_MemberPointer,
    E_Volatile,
    E_Reference,
  };
  enum { C_Member, C_Base };

  auto diag = [&](int Reason) -> bool {
    const Expr *E = S.Current->getExpr(OpPC);
    S.FFDiag(E, diag::note_constexpr_bit_cast_invalid_type)
        << static_cast<int>(IsToType) << (Reason == E_Reference) << Reason
        << E->getSourceRange();
    return false;
  };
  auto note = [&](int Construct, QualType NoteType, SourceRange NoteRange) {
    S.Note(NoteRange.getBegin(), diag::note_constexpr_bit_cast_invalid_subtype)
        << NoteType << Construct << T.getUnqualifiedType() << NoteRange;
    return false;
  };

  T = T.getCanonicalType();

  if (T->isUnionType())
    return diag(E_Union);
  if (T->isPointerType())
    return diag(E_Pointer);
  if (T->isMemberPointerType())
    return diag(E_MemberPointer);
  if (T.isVolatileQualified())
    return diag(E_Volatile);

  if (const RecordDecl *RD = T->getAsRecordDecl()) {
    if (const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
      for (const CXXBaseSpecifier &BS : CXXRD->bases()) {
        if (!CheckBitcastType(S, OpPC, BS.getType(), IsToType))
          return note(C_Base, BS.getType(), BS.getBeginLoc());
      }
    }
    for (const FieldDecl *FD : RD->fields()) {
      if (FD->getType()->isReferenceType())
        return diag(E_Reference);
      if (!CheckBitcastType(S, OpPC, FD->getType(), IsToType))
        return note(C_Member, FD->getType(), FD->getSourceRange());
    }
  }

  if (T->isArrayType() &&
      !CheckBitcastType(S, OpPC, S.getASTContext().getBaseElementType(T),
                        IsToType))
    return false;

  return true;
}

static bool readPointerToBuffer(const Context &Ctx, const Pointer &FromPtr,
                                BitcastBuffer &Buffer, bool ReturnOnUninit) {
  const ASTContext &ASTCtx = Ctx.getASTContext();
  Endian TargetEndianness =
      ASTCtx.getTargetInfo().isLittleEndian() ? Endian::Little : Endian::Big;

  return enumeratePointerFields(
      FromPtr, Ctx, Buffer.size(),
      [&](const Pointer &P, PrimType T, Bits BitOffset,
          bool PackedBools) -> bool {
        // if (!P.isInitialized()) {
        // assert(false && "Implement uninitialized value tracking");
        // return ReturnOnUninit;
        // }

        // assert(P.isInitialized());
        // nullptr_t is a PT_Ptr for us, but it's still not std::is_pointer_v.
        if (T == PT_Ptr)
          assert(false && "Implement casting to pointer types");

        CharUnits ObjectReprChars = ASTCtx.getTypeSizeInChars(P.getType());
        Bits BitWidth = Bits(ASTCtx.toBits(ObjectReprChars));
        Bits FullBitWidth = BitWidth;
        auto Buff =
            std::make_unique<std::byte[]>(ObjectReprChars.getQuantity());
        // Work around floating point types that contain unused padding bytes.
        // This is really just `long double` on x86, which is the only
        // fundamental type with padding bytes.
        if (T == PT_Float) {
          const Floating &F = P.deref<Floating>();
          Bits NumBits = Bits(
              llvm::APFloatBase::getSizeInBits(F.getAPFloat().getSemantics()));
          assert(NumBits.isFullByte());
          assert(NumBits.getQuantity() <= FullBitWidth.getQuantity());
          F.bitcastToMemory(Buff.get());
          // Now, only (maybe) swap the actual size of the float, excluding the
          // padding bits.
          if (llvm::sys::IsBigEndianHost)
            swapBytes(Buff.get(), NumBits.roundToBytes());

        } else {
          if (const FieldDecl *FD = P.getField(); FD && FD->isBitField())
            BitWidth = Bits(std::min(FD->getBitWidthValue(ASTCtx),
                                     (unsigned)FullBitWidth.getQuantity()));
          else if (T == PT_Bool && PackedBools)
            BitWidth = Bits(1);

          BITCAST_TYPE_SWITCH(T, { P.deref<T>().bitcastToMemory(Buff.get()); });

          if (llvm::sys::IsBigEndianHost)
            swapBytes(Buff.get(), FullBitWidth.roundToBytes());
        }

        Buffer.pushData(Buff.get(), BitOffset, BitWidth, TargetEndianness);
        return true;
      });
}

bool clang::interp::DoBitCast(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                              std::byte *Buff, size_t BuffSize,
                              bool &HasIndeterminateBits) {
  assert(Ptr.isLive());
  assert(Ptr.isBlockPointer());
  assert(Buff);

  Bits BitSize = Bytes(BuffSize).toBits();
  BitcastBuffer Buffer(BitSize);
  if (!CheckBitcastType(S, OpPC, Ptr.getType(), /*IsToType=*/false))
    return false;

  bool Success = readPointerToBuffer(S.getContext(), Ptr, Buffer,
                                     /*ReturnOnUninit=*/false);
  HasIndeterminateBits = !Buffer.allInitialized();

  const ASTContext &ASTCtx = S.getASTContext();
  Endian TargetEndianness =
      ASTCtx.getTargetInfo().isLittleEndian() ? Endian::Little : Endian::Big;
  auto B = Buffer.copyBits(Bits::zero(), BitSize, BitSize, TargetEndianness);

  std::memcpy(Buff, B.get(), BuffSize);

  if (llvm::sys::IsBigEndianHost)
    swapBytes(Buff, BuffSize);

  return Success;
}

bool clang::interp::DoBitCastPtr(InterpState &S, CodePtr OpPC,
                                 const Pointer &FromPtr, Pointer &ToPtr) {
  assert(FromPtr.isLive());
  assert(FromPtr.isBlockPointer());
  assert(ToPtr.isBlockPointer());

  QualType FromType = FromPtr.getType();
  QualType ToType = ToPtr.getType();

  if (!CheckBitcastType(S, OpPC, ToType, /*IsToType=*/true))
    return false;
  if (!CheckBitcastType(S, OpPC, FromType, /*IsToType=*/false))
    return false;

  const ASTContext &ASTCtx = S.getASTContext();

  CharUnits ObjectReprChars = ASTCtx.getTypeSizeInChars(ToType);
  BitcastBuffer Buffer(Bits(ASTCtx.toBits(ObjectReprChars)));
  readPointerToBuffer(S.getContext(), FromPtr, Buffer,
                      /*ReturnOnUninit=*/false);

  // Now read the values out of the buffer again and into ToPtr.
  Endian TargetEndianness =
      ASTCtx.getTargetInfo().isLittleEndian() ? Endian::Little : Endian::Big;
  bool Success = enumeratePointerFields(
      ToPtr, S.getContext(), Buffer.size(),
      [&](const Pointer &P, PrimType T, Bits BitOffset,
          bool PackedBools) -> bool {
        CharUnits ObjectReprChars = ASTCtx.getTypeSizeInChars(P.getType());
        Bits FullBitWidth = Bits(ASTCtx.toBits(ObjectReprChars));
        if (T == PT_Float) {
          const auto &Semantics = ASTCtx.getFloatTypeSemantics(P.getType());
          Bits NumBits = Bits(llvm::APFloatBase::getSizeInBits(Semantics));
          assert(NumBits.isFullByte());
          assert(NumBits.getQuantity() <= FullBitWidth.getQuantity());
          auto M = Buffer.copyBits(BitOffset, NumBits, FullBitWidth,
                                   TargetEndianness);

          if (llvm::sys::IsBigEndianHost)
            swapBytes(M.get(), NumBits.roundToBytes());

          P.deref<Floating>() = Floating::bitcastFromMemory(M.get(), Semantics);
          P.initialize();
          return true;
        }

        Bits BitWidth;
        if (const FieldDecl *FD = P.getField(); FD && FD->isBitField())
          BitWidth = Bits(std::min(FD->getBitWidthValue(ASTCtx),
                                   (unsigned)FullBitWidth.getQuantity()));
        else if (T == PT_Bool && PackedBools)
          BitWidth = Bits(1);
        else
          BitWidth = FullBitWidth;

        auto Memory = Buffer.copyBits(BitOffset, BitWidth, FullBitWidth,
                                      TargetEndianness);
        if (llvm::sys::IsBigEndianHost)
          swapBytes(Memory.get(), FullBitWidth.roundToBytes());

        BITCAST_TYPE_SWITCH_FIXED_SIZE(T, {
          if (BitWidth.nonZero())
            P.deref<T>() = T::bitcastFromMemory(Memory.get(), T::bitWidth())
                               .truncate(BitWidth.getQuantity());
          else
            P.deref<T>() = T::zero();
        });
        P.initialize();
        return true;
      });

  return Success;
}
