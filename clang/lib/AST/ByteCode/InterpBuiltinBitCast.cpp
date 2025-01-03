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
using DataFunc =
    llvm::function_ref<bool(const Pointer &P, PrimType Ty, Bits BitOffset,
                            Bits FullBitWidth, bool PackedBools)>;

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

/// We use this to recursively iterate over all fields and elements of a pointer
/// and extract relevant data for a bitcast.
static bool enumerateData(const Pointer &P, const Context &Ctx, Bits Offset,
                          Bits BitsToRead, DataFunc F) {
  const Descriptor *FieldDesc = P.getFieldDesc();
  assert(FieldDesc);

  // Primitives.
  if (FieldDesc->isPrimitive()) {
    Bits FullBitWidth =
        Bits(Ctx.getASTContext().getTypeSize(FieldDesc->getType()));
    return F(P, FieldDesc->getPrimType(), Offset, FullBitWidth,
             /*PackedBools=*/false);
  }

  // Primitive arrays.
  if (FieldDesc->isPrimitiveArray()) {
    QualType ElemType = FieldDesc->getElemQualType();
    Bits ElemSize = Bits(Ctx.getASTContext().getTypeSize(ElemType));
    PrimType ElemT = *Ctx.classify(ElemType);
    // Special case, since the bools here are packed.
    bool PackedBools = FieldDesc->getType()->isExtVectorBoolType();
    unsigned NumElems = FieldDesc->getNumElems();
    bool Ok = true;
    for (unsigned I = P.getIndex(); I != NumElems; ++I) {
      Ok = Ok && F(P.atIndex(I), ElemT, Offset, ElemSize, PackedBools);
      Offset += PackedBools ? Bits(1) : ElemSize;
      if (Offset >= BitsToRead)
        break;
    }
    return Ok;
  }

  // Composite arrays.
  if (FieldDesc->isCompositeArray()) {
    QualType ElemType = FieldDesc->getElemQualType();
    Bits ElemSize = Bits(Ctx.getASTContext().getTypeSize(ElemType));
    for (unsigned I = P.getIndex(); I != FieldDesc->getNumElems(); ++I) {
      enumerateData(P.atIndex(I).narrow(), Ctx, Offset, BitsToRead, F);
      Offset += ElemSize;
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
      if (Fi.isUnnamedBitField())
        continue;
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
      // FIXME: We should only (need to) do this when bitcasting OUT of the
      // buffer, not when copying data into it.
      if (Ok)
        Elem.initialize();
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

  if (const auto *VT = T->getAs<VectorType>()) {
    const ASTContext &ASTCtx = S.getASTContext();
    QualType EltTy = VT->getElementType();
    unsigned NElts = VT->getNumElements();
    unsigned EltSize =
        VT->isExtVectorBoolType() ? 1 : ASTCtx.getTypeSize(EltTy);

    if ((NElts * EltSize) % ASTCtx.getCharWidth() != 0) {
      // The vector's size in bits is not a multiple of the target's byte size,
      // so its layout is unspecified. For now, we'll simply treat these cases
      // as unsupported (this should only be possible with OpenCL bool vectors
      // whose element count isn't a multiple of the byte size).
      const Expr *E = S.Current->getExpr(OpPC);
      S.FFDiag(E, diag::note_constexpr_bit_cast_invalid_vector)
          << QualType(VT, 0) << EltSize << NElts << ASTCtx.getCharWidth();
      return false;
    }

    if (EltTy->isRealFloatingType() &&
        &ASTCtx.getFloatTypeSemantics(EltTy) == &APFloat::x87DoubleExtended()) {
      // The layout for x86_fp80 vectors seems to be handled very inconsistently
      // by both clang and LLVM, so for now we won't allow bit_casts involving
      // it in a constexpr context.
      const Expr *E = S.Current->getExpr(OpPC);
      S.FFDiag(E, diag::note_constexpr_bit_cast_unsupported_type) << EltTy;
      return false;
    }
  }

  return true;
}

bool clang::interp::readPointerToBuffer(const Context &Ctx,
                                        const Pointer &FromPtr,
                                        BitcastBuffer &Buffer,
                                        bool ReturnOnUninit) {
  const ASTContext &ASTCtx = Ctx.getASTContext();
  Endian TargetEndianness =
      ASTCtx.getTargetInfo().isLittleEndian() ? Endian::Little : Endian::Big;

  return enumeratePointerFields(
      FromPtr, Ctx, Buffer.size(),
      [&](const Pointer &P, PrimType T, Bits BitOffset, Bits FullBitWidth,
          bool PackedBools) -> bool {
        Bits BitWidth = FullBitWidth;

        if (const FieldDecl *FD = P.getField(); FD && FD->isBitField())
          BitWidth = Bits(std::min(FD->getBitWidthValue(ASTCtx),
                                   (unsigned)FullBitWidth.getQuantity()));
        else if (T == PT_Bool && PackedBools)
          BitWidth = Bits(1);

        if (BitWidth.isZero())
          return true;

        // Bits will be left uninitialized and diagnosed when reading.
        if (!P.isInitialized())
          return true;

        if (T == PT_Ptr) {
          assert(P.getType()->isNullPtrType());
          // Clang treats nullptr_t has having NO bits in its value
          // representation. So, we accept it here and leave its bits
          // uninitialized.
          return true;
        }

        assert(P.isInitialized());
        auto Buff = std::make_unique<std::byte[]>(FullBitWidth.roundToBytes());
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

          Buffer.markInitialized(BitOffset, NumBits);
        } else {
          BITCAST_TYPE_SWITCH(T, { P.deref<T>().bitcastToMemory(Buff.get()); });

          if (llvm::sys::IsBigEndianHost)
            swapBytes(Buff.get(), FullBitWidth.roundToBytes());
          Buffer.markInitialized(BitOffset, BitWidth);
        }

        Buffer.pushData(Buff.get(), BitOffset, BitWidth, TargetEndianness);
        return true;
      });
}

bool clang::interp::DoBitCast(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                              std::byte *Buff, Bits BitWidth, Bits FullBitWidth,
                              bool &HasIndeterminateBits) {
  assert(Ptr.isLive());
  assert(Ptr.isBlockPointer());
  assert(Buff);
  assert(BitWidth <= FullBitWidth);
  assert(FullBitWidth.isFullByte());
  assert(BitWidth.isFullByte());

  BitcastBuffer Buffer(FullBitWidth);
  size_t BuffSize = FullBitWidth.roundToBytes();
  if (!CheckBitcastType(S, OpPC, Ptr.getType(), /*IsToType=*/false))
    return false;

  bool Success = readPointerToBuffer(S.getContext(), Ptr, Buffer,
                                     /*ReturnOnUninit=*/false);
  HasIndeterminateBits = !Buffer.rangeInitialized(Bits::zero(), BitWidth);

  const ASTContext &ASTCtx = S.getASTContext();
  Endian TargetEndianness =
      ASTCtx.getTargetInfo().isLittleEndian() ? Endian::Little : Endian::Big;
  auto B =
      Buffer.copyBits(Bits::zero(), BitWidth, FullBitWidth, TargetEndianness);

  std::memcpy(Buff, B.get(), BuffSize);

  if (llvm::sys::IsBigEndianHost)
    swapBytes(Buff, BitWidth.roundToBytes());

  return Success;
}
bool clang::interp::DoBitCastPtr(InterpState &S, CodePtr OpPC,
                                 const Pointer &FromPtr, Pointer &ToPtr) {
  const ASTContext &ASTCtx = S.getASTContext();
  CharUnits ObjectReprChars = ASTCtx.getTypeSizeInChars(ToPtr.getType());

  return DoBitCastPtr(S, OpPC, FromPtr, ToPtr, ObjectReprChars.getQuantity());
}

bool clang::interp::DoBitCastPtr(InterpState &S, CodePtr OpPC,
                                 const Pointer &FromPtr, Pointer &ToPtr,
                                 size_t Size) {
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
  BitcastBuffer Buffer(Bytes(Size).toBits());
  readPointerToBuffer(S.getContext(), FromPtr, Buffer,
                      /*ReturnOnUninit=*/false);

  // Now read the values out of the buffer again and into ToPtr.
  Endian TargetEndianness =
      ASTCtx.getTargetInfo().isLittleEndian() ? Endian::Little : Endian::Big;
  bool Success = enumeratePointerFields(
      ToPtr, S.getContext(), Buffer.size(),
      [&](const Pointer &P, PrimType T, Bits BitOffset, Bits FullBitWidth,
          bool PackedBools) -> bool {
        QualType PtrType = P.getType();
        if (T == PT_Float) {
          const auto &Semantics = ASTCtx.getFloatTypeSemantics(PtrType);
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

        // If any of the bits are uninitialized, we need to abort unless the
        // target type is std::byte or unsigned char.
        bool Initialized = Buffer.rangeInitialized(BitOffset, BitWidth);
        if (!Initialized) {
          if (!PtrType->isStdByteType() &&
              !PtrType->isSpecificBuiltinType(BuiltinType::UChar) &&
              !PtrType->isSpecificBuiltinType(BuiltinType::Char_U)) {
            const Expr *E = S.Current->getExpr(OpPC);
            S.FFDiag(E, diag::note_constexpr_bit_cast_indet_dest)
                << PtrType << S.getLangOpts().CharIsSigned
                << E->getSourceRange();

            return false;
          }
          return true;
        }

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

bool clang::interp::DoMemcpy(InterpState &S, CodePtr OpPC,
                             const Pointer &SrcPtr, const Pointer &DestPtr,
                             Bits Size) {
  assert(SrcPtr.isBlockPointer());
  assert(DestPtr.isBlockPointer());

  unsigned SrcStartOffset = SrcPtr.getByteOffset();
  unsigned DestStartOffset = DestPtr.getByteOffset();

  enumeratePointerFields(SrcPtr, S.getContext(), Size,
                         [&](const Pointer &P, PrimType T, Bits BitOffset,
                             Bits FullBitWidth, bool PackedBools) -> bool {
                           unsigned SrcOffsetDiff =
                               P.getByteOffset() - SrcStartOffset;

                           Pointer DestP =
                               Pointer(DestPtr.asBlockPointer().Pointee,
                                       DestPtr.asBlockPointer().Base,
                                       DestStartOffset + SrcOffsetDiff);

                           TYPE_SWITCH(T, {
                             DestP.deref<T>() = P.deref<T>();
                             DestP.initialize();
                           });

                           return true;
                         });

  return true;
}
