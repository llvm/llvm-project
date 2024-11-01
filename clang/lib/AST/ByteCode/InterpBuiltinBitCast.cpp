//===-------------------- InterpBuiltinBitCast.cpp --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "InterpBuiltinBitCast.h"
#include "Boolean.h"
#include "Context.h"
#include "FixedPoint.h"
#include "Floating.h"
#include "Integral.h"
#include "IntegralAP.h"
#include "InterpState.h"
#include "MemberPointer.h"
#include "Pointer.h"
#include "Record.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/BitVector.h"
#include <bitset>

using namespace clang;
using namespace clang::interp;

/// Used to iterate over pointer fields.
using DataFunc =
    llvm::function_ref<bool(const Pointer &P, PrimType Ty, size_t BitOffset)>;

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

/// Float is a special case that sometimes needs the floating point semantics
/// to be available.
#define BITCAST_TYPE_SWITCH_WITH_FLOAT(Expr, B)                                \
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
      TYPE_SWITCH_CASE(PT_Float, B)                                            \
    default:                                                                   \
      llvm_unreachable("Unhandled bitcast type");                              \
    }                                                                          \
  } while (0)

static bool bitof(std::byte B, unsigned BitIndex) {
  return (B & (std::byte{1} << BitIndex)) != std::byte{0};
}

static void swapBytes(std::byte *M, size_t N) {
  for (size_t I = 0; I != (N / 2); ++I)
    std::swap(M[I], M[N - 1 - I]);
}

/// Track what bits have been initialized to known values and which ones
/// have indeterminate value.
/// All offsets are in bits.
struct BitcastBuffer {
  llvm::BitVector Data;

  BitcastBuffer() = default;

  size_t size() const { return Data.size(); }

  const std::byte *data() const {
    unsigned NBytes = Data.size() / 8;
    unsigned BitVectorWordSize = sizeof(uintptr_t);
    bool FullWord = (NBytes % BitVectorWordSize == 0);

    // llvm::BitVector uses 64-bit fields internally, so when we have
    // fewer bytes than that, we need to compensate for that on
    // big endian hosts.
    unsigned DataPlus;
    if (llvm::sys::IsBigEndianHost)
      DataPlus = BitVectorWordSize - (NBytes % BitVectorWordSize);
    else
      DataPlus = 0;

    return reinterpret_cast<const std::byte *>(Data.getData().data()) +
           (FullWord ? 0 : DataPlus);
  }

  bool allInitialized() const {
    // FIXME: Implement.
    return true;
  }

  void pushData(const std::byte *data, size_t BitOffset, size_t BitWidth,
                bool BigEndianTarget) {
    Data.reserve(BitOffset + BitWidth);

    bool OnlyFullBytes = BitWidth % 8 == 0;
    unsigned NBytes = BitWidth / 8;

    size_t BitsHandled = 0;
    // Read all full bytes first
    for (size_t I = 0; I != NBytes; ++I) {
      std::byte B =
          BigEndianTarget ? data[NBytes - OnlyFullBytes - I] : data[I];
      for (unsigned X = 0; X != 8; ++X) {
        Data.push_back(bitof(B, X));
        ++BitsHandled;
      }
    }

    if (BitsHandled == BitWidth)
      return;

    // Rest of the bits.
    assert((BitWidth - BitsHandled) < 8);
    std::byte B = BigEndianTarget ? data[0] : data[NBytes];
    for (size_t I = 0, E = (BitWidth - BitsHandled); I != E; ++I) {
      Data.push_back(bitof(B, I));
      ++BitsHandled;
    }

    assert(BitsHandled == BitWidth);
  }
};

/// We use this to recursively iterate over all fields and elemends of a pointer
/// and extract relevant data for a bitcast.
static bool enumerateData(const Pointer &P, const Context &Ctx, size_t Offset,
                          DataFunc F) {
  const Descriptor *FieldDesc = P.getFieldDesc();
  assert(FieldDesc);

  // Primitives.
  if (FieldDesc->isPrimitive())
    return F(P, FieldDesc->getPrimType(), Offset);

  // Primitive arrays.
  if (FieldDesc->isPrimitiveArray()) {
    bool BigEndianTarget = Ctx.getASTContext().getTargetInfo().isBigEndian();
    QualType ElemType = FieldDesc->getElemQualType();
    size_t ElemSizeInBits = Ctx.getASTContext().getTypeSize(ElemType);
    PrimType ElemT = *Ctx.classify(ElemType);
    bool Ok = true;
    for (unsigned I = 0; I != FieldDesc->getNumElems(); ++I) {
      unsigned Index = BigEndianTarget ? (FieldDesc->getNumElems() - 1 - I) : I;
      Ok = Ok && F(P.atIndex(Index), ElemT, Offset);
      Offset += ElemSizeInBits;
    }
    return Ok;
  }

  // Composite arrays.
  if (FieldDesc->isCompositeArray()) {
    bool BigEndianTarget = Ctx.getASTContext().getTargetInfo().isBigEndian();
    QualType ElemType = FieldDesc->getElemQualType();
    size_t ElemSizeInBits = Ctx.getASTContext().getTypeSize(ElemType);
    for (unsigned I = 0; I != FieldDesc->getNumElems(); ++I) {
      unsigned Index = BigEndianTarget ? (FieldDesc->getNumElems() - 1 - I) : I;
      enumerateData(P.atIndex(Index).narrow(), Ctx, Offset, F);
      Offset += ElemSizeInBits;
    }
    return true;
  }

  // Records.
  if (FieldDesc->isRecord()) {
    bool BigEndianTarget = Ctx.getASTContext().getTargetInfo().isBigEndian();
    const Record *R = FieldDesc->ElemRecord;
    const ASTRecordLayout &Layout =
        Ctx.getASTContext().getASTRecordLayout(R->getDecl());
    bool Ok = true;

    auto enumerateFields = [&]() -> void {
      for (unsigned I = 0, N = R->getNumFields(); I != N; ++I) {
        const Record::Field *Fi =
            R->getField(BigEndianTarget ? (N - 1 - I) : I);
        Pointer Elem = P.atField(Fi->Offset);
        size_t BitOffset =
            Offset + Layout.getFieldOffset(Fi->Decl->getFieldIndex());
        Ok = Ok && enumerateData(Elem, Ctx, BitOffset, F);
      }
    };
    auto enumerateBases = [&]() -> void {
      for (unsigned I = 0, N = R->getNumBases(); I != N; ++I) {
        const Record::Base *B = R->getBase(BigEndianTarget ? (N - 1 - I) : I);
        Pointer Elem = P.atField(B->Offset);
        CharUnits ByteOffset =
            Layout.getBaseClassOffset(cast<CXXRecordDecl>(B->Decl));
        size_t BitOffset = Offset + Ctx.getASTContext().toBits(ByteOffset);
        Ok = Ok && enumerateData(Elem, Ctx, BitOffset, F);
      }
    };

    if (BigEndianTarget) {
      enumerateFields();
      enumerateBases();
    } else {
      enumerateBases();
      enumerateFields();
    }

    return Ok;
  }

  llvm_unreachable("Unhandled data type");
}

static bool enumeratePointerFields(const Pointer &P, const Context &Ctx,
                                   DataFunc F) {
  return enumerateData(P, Ctx, 0, F);
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
        << NoteType << Construct << T << NoteRange;
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
  bool SwapData = (ASTCtx.getTargetInfo().isLittleEndian() !=
                   llvm::sys::IsLittleEndianHost);
  bool BigEndianTarget = ASTCtx.getTargetInfo().isBigEndian();

  return enumeratePointerFields(
      FromPtr, Ctx,
      [&](const Pointer &P, PrimType T, size_t BitOffset) -> bool {
        if (!P.isInitialized()) {
          assert(false && "Implement uninitialized value tracking");
          return ReturnOnUninit;
        }

        assert(P.isInitialized());
        // nullptr_t is a PT_Ptr for us, but it's still not std::is_pointer_v.
        if (T == PT_Ptr)
          assert(false && "Implement casting to pointer types");

        CharUnits ObjectReprChars = ASTCtx.getTypeSizeInChars(P.getType());
        unsigned BitWidth;
        if (const FieldDecl *FD = P.getField(); FD && FD->isBitField())
          BitWidth = FD->getBitWidthValue(ASTCtx);
        else
          BitWidth = ASTCtx.toBits(ObjectReprChars);

        llvm::SmallVector<std::byte> Buff(ObjectReprChars.getQuantity());
        BITCAST_TYPE_SWITCH_WITH_FLOAT(T, {
          T Val = P.deref<T>();
          Val.bitcastToMemory(Buff.data());
        });
        if (SwapData)
          swapBytes(Buff.data(), ObjectReprChars.getQuantity());

        if (BitWidth != (Buff.size() * 8) && BigEndianTarget) {
          Buffer.pushData(Buff.data() + (Buff.size() - 1 - (BitWidth / 8)),
                          BitOffset, BitWidth, BigEndianTarget);
        } else {
          Buffer.pushData(Buff.data(), BitOffset, BitWidth, BigEndianTarget);
        }
        return true;
      });
}

bool clang::interp::DoBitCast(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                              std::byte *Buff, size_t BuffSize,
                              bool &HasIndeterminateBits) {
  assert(Ptr.isLive());
  assert(Ptr.isBlockPointer());
  assert(Buff);

  BitcastBuffer Buffer;
  if (!CheckBitcastType(S, OpPC, Ptr.getType(), /*IsToType=*/false))
    return false;

  bool Success = readPointerToBuffer(S.getContext(), Ptr, Buffer,
                                     /*ReturnOnUninit=*/false);
  assert(Buffer.size() == BuffSize * 8);

  HasIndeterminateBits = !Buffer.allInitialized();
  std::memcpy(Buff, Buffer.data(), BuffSize);

  return Success;
}
