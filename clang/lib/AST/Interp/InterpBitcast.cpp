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
  llvm::BitVector Data;

  BitTracker() = default;

  size_t size() const {
    assert(Initialized.size() == Data.size());
    return Initialized.size();
  }

  const std::byte *getBytes(size_t BitOffset, int a) {
    assert(BitOffset % 8 == 0);
    return reinterpret_cast<const std::byte *>(Data.getData().data()) +
           (BitOffset / 8);
  }

  bool allInitialized(size_t Offset, size_t Size) const {
    return Initialized.find_first_unset_in(Offset, Offset + Size) == -1;
  }

  bool allInitialized() const { return Initialized.all(); }

  void pushData(const std::byte *data, size_t BitOffset, size_t BitWidth) {
    assert(BitOffset >= Data.size());
    Data.reserve(BitOffset + BitWidth);
    Initialized.reserve(BitOffset + BitWidth);

    // First, fill up the bit vector until BitOffset. The bits are all 0
    // but we record them as indeterminate.
    {
      Data.resize(BitOffset, false);
      Initialized.resize(BitOffset, false);
    }

    size_t BitsHandled = 0;
    // Read all full bytes first
    for (size_t I = 0; I != BitWidth / 8; ++I) {
      for (unsigned X = 0; X != 8; ++X) {
        Data.push_back((data[I] & std::byte(1 << X)) != std::byte{0});
        Initialized.push_back(true);
        ++BitsHandled;
      }
    }

    // Rest of the bits.
    assert((BitWidth - BitsHandled) < 8);
    for (size_t I = 0, E = (BitWidth - BitsHandled); I != E; ++I) {
      Data.push_back((data[BitWidth / 8] & std::byte(1 << I)) != std::byte{0});
      Initialized.push_back(true);
      ++BitsHandled;
    }
  }

  void pushZeroes(size_t Amount) {
    size_t N = Data.size();
    Data.resize(N + Amount, false);
    Initialized.resize(N + Amount, true);
  }

  void markUninitializedUntil(size_t Offset) {
    assert(Offset >= Data.size());
    Data.resize(Offset, false);
    Initialized.resize(Offset, false);
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
    bool Ok = true;
    for (unsigned I = 0; I != FieldDesc->getNumElems(); ++I) {
      Ok = Ok && F(P.atIndex(I), ElemT, Offset);
      Offset += ElemSizeInBits;
    }
    return Ok;
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
    bool Ok = true;
    for (const auto &B : R->bases()) {
      Pointer Elem = P.atField(B.Offset);
      CharUnits ByteOffset =
          Layout.getBaseClassOffset(cast<CXXRecordDecl>(B.Decl));
      size_t BitOffset = Offset + Ctx.getASTContext().toBits(ByteOffset);
      Ok = Ok && enumerateData(Elem, Ctx, BitOffset, F);
    }

    for (unsigned I = 0; I != R->getNumFields(); ++I) {
      const Record::Field *Fi = R->getField(I);
      Pointer Elem = P.atField(Fi->Offset);
      size_t BitOffset = Offset + Layout.getFieldOffset(I);
      Ok = Ok && enumerateData(Elem, Ctx, BitOffset, F);
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
      !CheckBitcastType(S, OpPC, S.getCtx().getBaseElementType(T), IsToType))
    return false;

  return true;
}

static bool readPointerToBuffer(const Context &Ctx, const Pointer &FromPtr,
                                BitTracker &Bits, bool ReturnOnUninit) {
  const ASTContext &ASTCtx = Ctx.getASTContext();
  uint64_t PointerSizeInBits =
      ASTCtx.getTargetInfo().getPointerWidth(LangAS::Default);
  bool BigEndian = ASTCtx.getTargetInfo().isBigEndian();

  return enumeratePointerFields(
      FromPtr, Ctx,
      [&](const Pointer &P, PrimType T, size_t BitOffset) -> bool {
        if (!P.isInitialized()) {
          Bits.markUninitializedUntil(BitOffset +
                                      (primSize(T) * 8)); /// primSize() usage.
          return ReturnOnUninit;
        }

        assert(P.isInitialized());
        // nullptr_t is a PT_Ptr for us, but it's still not std::is_pointer_v.
        if (T == PT_Ptr) {
          assert(P.getType()->isNullPtrType());
          Bits.pushZeroes(PointerSizeInBits);
          return true;
        }

        unsigned ObjectReprBytes =
            ASTCtx.getTypeSizeInChars(P.getType()).getQuantity();

        CharUnits ObjectReprChars = ASTCtx.getTypeSizeInChars(P.getType());
        unsigned BitWidth;
        if (const FieldDecl *FD = P.getField(); FD && FD->isBitField())
          BitWidth = FD->getBitWidthValue(ASTCtx);
        else
          BitWidth = ASTCtx.toBits(ObjectReprChars);
        BITCAST_TYPE_SWITCH_WITH_FLOAT(T, {
          T Val = P.deref<T>();
          unsigned ValueReprBytes = Val.valueReprBytes(ASTCtx);
          assert(ObjectReprBytes >= ValueReprBytes);

          llvm::SmallVector<std::byte> Buff(ObjectReprChars.getQuantity());
          Val.bitcastToMemory(Buff.data());
          if (BigEndian)
            swapBytes(Buff.data(), BitWidth / 8);
          Bits.pushData(Buff.data(), BitOffset, BitWidth);
        });
        return true;
      });
}

/// Bitcast all fields from \p P into \p Buff.
/// This is used for bitcasting TO a single primitive value.
bool DoBitCast(InterpState &S, CodePtr OpPC, const Pointer &P, std::byte *Buff,
               size_t BuffSize, bool &HasIndeterminateBits) {
  assert(P.isLive());
  assert(Buff);
  assert(BuffSize > 0);

  if (!CheckBitcastType(S, OpPC, P.getType(), /*IsToType=*/false))
    return false;

  const ASTContext &ASTCtx = S.getContext().getASTContext();
  bool BigEndian = ASTCtx.getTargetInfo().isBigEndian();

  BitTracker Bits;
  bool Success =
      readPointerToBuffer(S.getContext(), P, Bits, /*ReturnOnUninit=*/false);

  Bits.markUninitializedUntil(BuffSize * 8);
  assert(Bits.size() == BuffSize * 8);

  HasIndeterminateBits = !Bits.allInitialized();
  std::memcpy(Buff, Bits.Data.getData().data(), BuffSize);

  if (BigEndian)
    swapBytes(Buff, BuffSize);
  return Success;
}

/// Bitcast from a Pointer to a Pointer.
/// We read all fields from \p P into a buffer, then iterate
/// over the fields of \p DestPtr and read from the buffer.
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
  bool BigEndian = ASTCtx.getTargetInfo().isBigEndian();

  BitTracker Bytes;
  if (!readPointerToBuffer(Ctx, P, Bytes, /*ReturnOnUninit=*/true))
    return false;

  bool Success = enumeratePointerFields(
      DestPtr, S.getContext(),
      [&](const Pointer &P, PrimType T, size_t BitOffset) -> bool {
        if (T == PT_Float) {
          const QualType FloatType = P.getFieldDesc()->getType();
          const auto &Sem = ASTCtx.getFloatTypeSemantics(FloatType);

          CharUnits ObjectReprChars = ASTCtx.getTypeSizeInChars(P.getType());
          llvm::SmallVector<std::byte> Buff(ObjectReprChars.getQuantity());
          const std::byte *M =
              Bytes.getBytes(BitOffset, ObjectReprChars.getQuantity());
          std::memcpy(Buff.data(), M, ObjectReprChars.getQuantity());

          if (BigEndian)
            swapBytes(Buff.data(), ObjectReprChars.getQuantity());

          P.deref<Floating>() = Floating::bitcastFromMemory(Buff.data(), Sem);
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

          size_t ValueReprBits = Val.valueReprBits(ASTCtx);
          // Check if any of the bits we're about to read are uninitialized.
          bool HasIndeterminateBits =
              !Bytes.allInitialized(BitOffset, ValueReprBits);

          if (HasIndeterminateBits) {
            // Always an error, unless the type of the field we're reading is
            // either unsigned char or std::byte.
            bool TargetIsUCharOrBytes =
                (P.getType()->isSpecificBuiltinType(BuiltinType::UChar) ||
                 P.getType()->isSpecificBuiltinType(BuiltinType::Char_U) ||
                 P.getType()->isStdByteType());

            if (!TargetIsUCharOrBytes) {
              const Expr *E = S.Current->getExpr(OpPC);
              QualType ExprType = P.getType();
              S.FFDiag(E, diag::note_constexpr_bit_cast_indet_dest)
                  << ExprType << S.getLangOpts().CharIsSigned
                  << E->getSourceRange();
              return false;
            }
          }

          assert(!P.getField()->isBitField());

          llvm::SmallVector<std::byte> Copy(ValueReprBits / 8);
          const std::byte *M = Bytes.getBytes(BitOffset, ValueReprBits / 8);

          std::memcpy(Copy.data(), M, ValueReprBits / 8);

          if (BigEndian)
            swapBytes(Copy.data(), ValueReprBits / 8);
          Val = T::bitcastFromMemory(Copy.data(), Val.bitWidth());

          if (!HasIndeterminateBits)
            P.initialize();
        });
        return true;
      });

  return Success;
}
} // namespace interp
} // namespace clang
