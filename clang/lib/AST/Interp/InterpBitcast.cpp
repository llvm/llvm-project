//===--- InterpBitcast.cpp - Interpreter for the constexpr VM ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Boolean.h"
#include "Interp.h"
#include "PrimType.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/BitVector.h"

namespace clang {
namespace interp {

// TODO: Try to e-duplicate the primitive and composite versions.

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
      TYPE_SWITCH_CASE(PT_Bool, B)                                             \
      TYPE_SWITCH_CASE(PT_Float, B)                                            \
    default:                                                                   \
      llvm_unreachable("Unhandled bitcast type");                              \
    }                                                                          \
  } while (0)

/// Rotate things around for big endian targets.
static void swapBytes(std::byte *M, size_t N) {
  for (size_t I = 0; I != (N / 2); ++I)
    std::swap(M[I], M[N - 1 - I]);
}

/// Track what bits have been initialized to known values and which ones
/// have indeterminate value.
/// All offsets are in bits.
struct BitTracker {
  llvm::BitVector Initialized;
  std::vector<std::byte> Data;

  BitTracker() = default;

  size_t size() const {
    assert(Initialized.size() == Data.size());
    return Initialized.size();
  }

  std::byte *getBytes(size_t Offset) { return Data.data() + Offset; }
  bool allInitialized(size_t Offset, size_t Size) const {
    for (size_t I = Offset; I != (Size + Offset); ++I) {
      if (!Initialized[I])
        return false;
    }
    return true;
  }

  std::byte *getWritableBytes(size_t Offset, size_t Size, bool InitValue) {
    assert(Offset >= Data.size());
    assert(Size > 0);

    size_t OldSize = Data.size();
    Data.resize(Offset + Size);

    // Everything from the old size to the new offset is indeterminate.
    for (size_t I = OldSize; I != Offset; ++I)
      Initialized.push_back(false);
    for (size_t I = Offset; I != Offset + Size; ++I)
      Initialized.push_back(InitValue);

    return Data.data() + Offset;
  }

  void markUninitializedUntil(size_t Offset) {
    assert(Offset >= Data.size());

    size_t NBytes = Offset - Data.size();
    for (size_t I = 0; I != NBytes; ++I)
      Initialized.push_back(false);
    Data.resize(Offset);
  }

  void zeroUntil(size_t Offset) {
    assert(Offset >= Data.size());

    assert(Data.size() == Initialized.size());
    size_t NBytes = Offset - Data.size();
    for (size_t I = 0; I != NBytes; ++I) {
      Initialized.push_back(true);
      Data.push_back(std::byte{0});
    }
  }
};

struct BitcastBuffer {
  llvm::BitVector Data;
  std::byte *Buff;
  size_t BitOffset = 0;
  size_t BuffSize;
  unsigned IndeterminateBits = 0;

  BitcastBuffer(std::byte *Buff, size_t BuffSize)
      : Buff(Buff), BuffSize(BuffSize) {}

  void pushData(const std::byte *data, size_t BitOffset, size_t BitWidth) {
    assert(BitOffset >= Data.size());
    // First, fill up the bit vector until BitOffset. The bits are all 0
    // but we record them as indeterminate.
    {
      size_t FillBits = BitOffset - Data.size();
      IndeterminateBits += FillBits;
      Data.resize(BitOffset, false);
    }

    size_t BitsHandled = 0;
    // Read all full bytes first
    for (size_t I = 0; I != BitWidth / 8; ++I) {
      for (unsigned X = 0; X != 8; ++X) {
        Data.push_back((data[I] & std::byte(1 << X)) != std::byte{0});
        ++BitsHandled;
      }
    }

    // Rest of the bits.
    assert((BitWidth - BitsHandled) < 8);
    for (size_t I = 0, E = (BitWidth - BitsHandled); I != E; ++I) {
      Data.push_back((data[BitWidth / 8] & std::byte(1 << I)) != std::byte{0});
      ++BitsHandled;
    }
  }

  void pushZeroes(size_t Amount) { Data.resize(Data.size() + Amount, false); }

  void finish() {
    // Fill up with zeroes until the buffer is BuffSize in size.
    // The added bits are of indeterminate value.
    assert(Data.size() <= (BuffSize * 8));
    size_t Remainder = (BuffSize * 8) - Data.size();
    for (size_t I = 0; I != Remainder; ++I)
      Data.push_back(false);

    IndeterminateBits += Remainder;
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
    return F(P, *Ctx.classify(FieldDesc->getType()), Offset);

  // Primitive arrays.
  if (FieldDesc->isPrimitiveArray()) {
    QualType ElemType =
        FieldDesc->getType()->getAsArrayTypeUnsafe()->getElementType();
    size_t ElemSizeInBits = Ctx.getASTContext().getTypeSize(ElemType);
    PrimType ElemT = *Ctx.classify(ElemType);
    for (unsigned I = 0; I != FieldDesc->getNumElems(); ++I) {
      if (!F(P.atIndex(I), ElemT, Offset))
        return false;
      Offset += ElemSizeInBits;
    }
    return true;
  }

  // Composite arrays.
  if (FieldDesc->isCompositeArray()) {
    QualType ElemType =
        FieldDesc->getType()->getAsArrayTypeUnsafe()->getElementType();
    size_t ElemSizeInBits = Ctx.getASTContext().getTypeSize(ElemType);
    for (unsigned I = 0; I != FieldDesc->getNumElems(); ++I) {
      enumerateData(P.atIndex(I).narrow(), Ctx, Offset, F);
      Offset += ElemSizeInBits;
    }
    return true;
  }

  // Records.
  if (FieldDesc->isRecord()) {
    const Record *R = FieldDesc->ElemRecord;
    const ASTRecordLayout &Layout =
        Ctx.getASTContext().getASTRecordLayout(R->getDecl());
    for (const auto &B : R->bases()) {
      Pointer Elem = P.atField(B.Offset);
      CharUnits ByteOffset =
          Layout.getBaseClassOffset(cast<CXXRecordDecl>(B.Decl));
      size_t BitOffset = Offset + Ctx.getASTContext().toBits(ByteOffset);
      if (!enumerateData(Elem, Ctx, BitOffset, F))
        return false;
    }
    // TODO: Virtual bases?

    for (unsigned I = 0; I != R->getNumFields(); ++I) {
      const Record::Field *Fi = R->getField(I);
      Pointer Elem = P.atField(Fi->Offset);
      size_t BitOffset = Offset + Layout.getFieldOffset(I);
      if (!enumerateData(Elem, Ctx, BitOffset, F))
        return false;
    }
    return true;
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
      !CheckBitcastType(S, OpPC, S.getCtx().getBaseElementType(T), IsToType))
    return false;

  return true;
}

/// Bitcast all fields from \p P into \p Buff.
/// This is used for bitcasting TO a single primitive value.
bool DoBitCast(InterpState &S, CodePtr OpPC, const Pointer &P, std::byte *Buff,
               size_t BuffSize, unsigned &IndeterminateBits) {
  llvm::errs() << __PRETTY_FUNCTION__ << "\n";
  assert(P.isLive());
  assert(Buff);
  assert(BuffSize > 0);

  BitcastBuffer F(Buff, BuffSize);

  if (!CheckBitcastType(S, OpPC, P.getType(), /*IsToType=*/false))
    return false;

  const Context &Ctx = S.getContext();
  const ASTContext &ASTCtx = Ctx.getASTContext();
  uint64_t PointerSizeInBits =
      ASTCtx.getTargetInfo().getPointerWidth(LangAS::Default);

  bool Success = enumeratePointerFields(
      P, S.getContext(),
      [&](const Pointer &Ptr, PrimType T, size_t BitOffset) -> bool {
        if (!Ptr.isInitialized())
          return false;
        if (T == PT_Ptr) {
          assert(Ptr.getType()->isNullPtrType());
          F.pushZeroes(PointerSizeInBits);
          return true;
        }

        CharUnits ObjectReprChars = ASTCtx.getTypeSizeInChars(Ptr.getType());
        unsigned BitWidth;
        if (const FieldDecl *FD = Ptr.getField(); FD && FD->isBitField())
          BitWidth = FD->getBitWidthValue(ASTCtx);
        else
          BitWidth = ASTCtx.toBits(ObjectReprChars);

        BITCAST_TYPE_SWITCH_WITH_FLOAT(T, {
          T Val = Ptr.deref<T>();
          std::byte Buff[sizeof(T)];
          Val.bitcastToMemory(Buff);
          F.pushData(Buff, BitOffset, BitWidth);
        });
        return true;
      });

  F.finish();

  IndeterminateBits = F.IndeterminateBits;
  assert(F.Data.size() == BuffSize * 8);
  std::memcpy(Buff, F.Data.getData().data(), BuffSize);

  return Success;
}

//  This function is constexpr if and only if To, From, and the types of
//  all subobjects of To and From are types T such that...
//  (3.1) - is_union_v<T> is false;
//  (3.2) - is_pointer_v<T> is false;
//  (3.3) - is_member_pointer_v<T> is false;
//  (3.4) - is_volatile_v<T> is false; and
//  (3.5) - T has no non-static data members of reference type
bool DoBitCastToPtr(InterpState &S, const Pointer &P, Pointer &DestPtr,
                    CodePtr OpPC) {
  assert(P.isLive());
  assert(DestPtr.isLive());

  QualType FromType = P.getType();
  QualType ToType = DestPtr.getType();

  if (!CheckBitcastType(S, OpPC, FromType, /*IsToType=*/false))
    return false;

  if (!CheckBitcastType(S, OpPC, ToType, /*IsToType=*/true))
    return false;

  const Context &Ctx = S.getContext();
  const ASTContext &ASTCtx = Ctx.getASTContext();
  uint64_t PointerSize =
      ASTCtx
          .toCharUnitsFromBits(
              ASTCtx.getTargetInfo().getPointerWidth(LangAS::Default))
          .getQuantity();
  bool BigEndian = ASTCtx.getTargetInfo().isBigEndian();

  BitTracker Bytes;
  enumeratePointerFields(
      P, S.getContext(),
      [&](const Pointer &P, PrimType T, size_t ByteOffset) -> bool {
        ByteOffset /= 8;
        bool PtrInitialized = P.isInitialized();
        if (!PtrInitialized) {
          Bytes.markUninitializedUntil(ByteOffset + primSize(T));
          return true;
        }

        assert(P.isInitialized());
        // nullptr_t is a PT_Ptr for us, but it's still not std::is_pointer_v.
        if (T == PT_Ptr) {
          assert(P.getType()->isNullPtrType());
          std::byte *M = Bytes.getWritableBytes(ByteOffset, PointerSize,
                                                /*InitValue=*/true);
          std::memset(M, 0, PointerSize);
          return true;
        }
        BITCAST_TYPE_SWITCH_WITH_FLOAT(T, {
          T Val = P.deref<T>();
          unsigned ObjectReprBytes =
              ASTCtx.getTypeSizeInChars(P.getType()).getQuantity();
          unsigned ValueReprBytes = Val.valueReprBytes(ASTCtx);
          assert(ObjectReprBytes >= ValueReprBytes);

          std::byte *Dest = Bytes.getWritableBytes(ByteOffset, ValueReprBytes,
                                                   PtrInitialized);
          Val.bitcastToMemory(Dest);
          Bytes.zeroUntil(ByteOffset + ObjectReprBytes);

          if (BigEndian)
            swapBytes(Dest, ValueReprBytes);
        });
        return true;
      });

  bool Success = enumeratePointerFields(
      DestPtr, S.getContext(),
      [&](const Pointer &P, PrimType T, size_t ByteOffset) -> bool {
        ByteOffset /= 8;
        if (T == PT_Float) {
          const QualType FloatType = P.getFieldDesc()->getType();
          const auto &Sem = ASTCtx.getFloatTypeSemantics(FloatType);
          size_t ValueReprBytes =
              ASTCtx.toCharUnitsFromBits(APFloat::semanticsSizeInBits(Sem))
                  .getQuantity();

          std::byte *M = Bytes.getBytes(ByteOffset);

          if (BigEndian)
            swapBytes(M, ValueReprBytes);
          P.deref<Floating>() = Floating::bitcastFromMemory(M, Sem);
          P.initialize();
          return true;
        }
        if (T == PT_Ptr) {
          assert(P.getType()->isNullPtrType());
          // Just need to write out a nullptr.
          P.deref<Pointer>() = Pointer();
          P.initialize();
          return true;
        }

        BITCAST_TYPE_SWITCH(T, {
          T &Val = P.deref<T>();

          size_t ValueReprBytes = T::valueReprBytes(ASTCtx);
          // Check if any of the bits we're about to read are uninitialized.
          bool HasIndeterminateBytes =
              !Bytes.allInitialized(ByteOffset, ValueReprBytes);

          if (HasIndeterminateBytes) {
            // Always an error, unless the type of the field we're reading is
            // either unsigned char or std::byte.
            bool TargetIsUCharOrBytes =
                (ValueReprBytes == 1 &&
                 (P.getType()->isSpecificBuiltinType(BuiltinType::UChar) ||
                  P.getType()->isSpecificBuiltinType(BuiltinType::Char_U) ||
                  P.getType()->isStdByteType()));

            if (!TargetIsUCharOrBytes) {
              const Expr *E = S.Current->getExpr(OpPC);
              QualType ExprType = P.getType();
              S.FFDiag(E, diag::note_constexpr_bit_cast_indet_dest)
                  << ExprType << S.getLangOpts().CharIsSigned
                  << E->getSourceRange();
              return false;
            }
          }

          std::byte *M = Bytes.getBytes(ByteOffset);
          if (BigEndian)
            swapBytes(M, ValueReprBytes);
          Val = T::bitcastFromMemory(M);

          if (!HasIndeterminateBytes)
            P.initialize();
        });
        return true;
      });

  return Success;
}
} // namespace interp
} // namespace clang
