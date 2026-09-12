//===------------- InterpBuiltinObjectSize.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implementation of the frontend part of the __builtin_object_size and
// __builtin_dynamic_object_size builtins.

#include "InterpHelpers.h"
#include "Pointer.h"
#include "Record.h"
#include "clang/AST/RecordLayout.h"

using namespace clang;
using namespace clang::interp;

enum : uint8_t {
  Regular = 1 << 0,
  IgnoreBaseCasts = 1 << 1,
  SurroundingArray = 1 << 2,
};

// Helper to check if a Type can be passed to
// ASTContext::getRecordLayout().
static bool validType(QualType T) {
  if (const RecordDecl *RD = T->getAsRecordDecl())
    return ASTContext::hasLayout(RD);
  return true;
}

static QualType computeFieldType(const ASTContext &ASTCtx,
                                 const OpaquePointer &OP,
                                 unsigned TypeModifier = 0) {
  QualType CurType = OP.getObjectType();

  unsigned Drop = 0;
  if (TypeModifier & IgnoreBaseCasts && OP.PathLength != 0 &&
      OP.path().back().Kind == PointerPathEntry::Base)
    Drop = 1;

  if (TypeModifier & SurroundingArray && OP.PathLength != 0 &&
      OP.path().back().Kind == PointerPathEntry::Array)
    Drop = 1;

  for (const PointerPathEntry &Entry : OP.path().drop_back(Drop)) {
    switch (Entry.Kind) {
    case PointerPathEntry::Base:
      CurType = ASTCtx.getCanonicalTagType(Entry.RD.getPointer());
      break;
    case PointerPathEntry::Field:
      CurType = Entry.FD->getType();
      break;
    case PointerPathEntry::Array:
    case PointerPathEntry::NegativeArray:
      if (!CurType->isArrayType())
        continue;
      CurType = CurType->getAsArrayTypeUnsafe()->getElementType();
    }
  }

  return CurType;
}

static std::optional<unsigned> computeFullDescSize(const ASTContext &ASTCtx,
                                                   const Descriptor *Desc) {
  if (Desc->isPrimitive() || Desc->isArray()) {
    QualType T = Desc->getType();
    if (!validType(T))
      return std::nullopt;
    return ASTCtx.getTypeSizeInChars(T).getQuantity();
  }

  if (Desc->isRecord()) {
    // Can't use Descriptor::getType() as that may return a pointer type. Look
    // at the decl directly.

    const RecordDecl *RD = Desc->ElemRecord->getDecl();
    if (!ASTContext::hasLayout(RD))
      return std::nullopt;

    return ASTCtx.getTypeSizeInChars(ASTCtx.getCanonicalTagType(RD))
        .getQuantity();
  }

  return std::nullopt;
}

/// Compute the byte offset of \p Ptr in the full declaration.
static unsigned computePointerOffset(const ASTContext &ASTCtx,
                                     const Pointer &Ptr) {
  return Ptr.computeLayoutOffset(ASTCtx).value_or(0);
}

/// Does Ptr point to the last subobject?
static bool pointsToLastObject(const Pointer &Ptr) {
  Pointer P = Ptr;
  while (!P.isRoot()) {

    if (P.isArrayElement()) {
      P = P.expand().getArray();
      continue;
    }
    if (P.isBaseClass()) {
      if (P.getRecord()->getNumFields() > 0)
        return false;
      P = P.getBase();
      continue;
    }

    Pointer Base = P.getBase();
    if (const Record *R = Base.getRecord()) {
      assert(P.getField());
      if (P.getField()->getFieldIndex() != R->getNumFields() - 1)
        return false;
    }
    P = Base;
  }

  return true;
}

/// Does Ptr point to the last object AND to a flexible array member?
static bool isUserWritingOffTheEnd(const ASTContext &Ctx, const Pointer &Ptr,
                                   bool InvalidBase) {
  auto isFlexibleArrayMember = [&](const Descriptor *FieldDesc) {
    using FAMKind = LangOptions::StrictFlexArraysLevelKind;
    FAMKind StrictFlexArraysLevel =
        Ctx.getLangOpts().getStrictFlexArraysLevel();

    if (StrictFlexArraysLevel == FAMKind::Default)
      return true;

    unsigned NumElems = FieldDesc->getNumElems();
    if (NumElems == 0 && StrictFlexArraysLevel != FAMKind::IncompleteOnly)
      return true;

    if (NumElems == 1 && StrictFlexArraysLevel == FAMKind::OneZeroOrIncomplete)
      return true;
    return false;
  };

  const Descriptor *FieldDesc = Ptr.getFieldDesc();
  if (!FieldDesc->isArray())
    return false;

  return InvalidBase && pointsToLastObject(Ptr) &&
         isFlexibleArrayMember(FieldDesc);
}

static bool isUserWritingOffTheEnd(const ASTContext &ASTCtx,
                                   const OpaquePointer &OP) {
  if (OP.PathLength == 0)
    return false;

  QualType CurType = OP.getObjectType();
  for (unsigned I = 0; I != OP.PathLength; ++I) {
    const PointerPathEntry &Entry = OP.Path[I];
    switch (Entry.Kind) {
    case PointerPathEntry::Base:
      return false;
    case PointerPathEntry::Field: {
      const FieldDecl *FD = OP.Path[I].FD;
      if (!FD->getParent()->isUnion() &&
          FD->getFieldIndex() != FD->getParent()->getNumFields() - 1)
        return false;
      CurType = FD->getType();
    } break;
    case PointerPathEntry::Array: {
      if (I == OP.PathLength - 1)
        break;

      if (!CurType->isArrayType())
        break;

      unsigned Index = OP.Path[I].Index;
      const ArrayType *AT = CurType->getAsArrayTypeUnsafe();
      assert(AT);
      if (const auto *CAT = dyn_cast<ConstantArrayType>(AT)) {
        if (Index != CAT->getLimitedSize() - 1)
          return false;
        CurType = CAT->getElementType();
      } else {
        return false;
      }
    } break;
    case PointerPathEntry::NegativeArray:
      return false;
    }
  }

  // We're pointing to the last field in the full object.
  // CurType is now the most derived type.
  if (!CurType->isArrayType())
    return false;

  if (isa<IncompleteArrayType>(CurType))
    return true;

  const auto *CAT = dyn_cast<ConstantArrayType>(CurType);
  if (!CAT)
    return false;

  using FAMKind = LangOptions::StrictFlexArraysLevelKind;
  FAMKind StrictFlexArraysLevel =
      ASTCtx.getLangOpts().getStrictFlexArraysLevel();

  if (StrictFlexArraysLevel == FAMKind::Default)
    return true;

  unsigned Size = CAT->getZExtSize();
  if (Size == 0 && StrictFlexArraysLevel != FAMKind::IncompleteOnly)
    return true;

  if (Size == 1 && StrictFlexArraysLevel == FAMKind::OneZeroOrIncomplete)
    return true;
  return false;
}

/// Determine the offset of the given pointer. Depending on \c
/// UseClosestSurroundingVariable, the offset is either relative to the full
/// object or to the closest surrounding field or array.
static std::optional<uint64_t>
computeOpaquePtrOffset(const ASTContext &ASTCtx, const Pointer &Ptr,
                       bool UseClosestSurroundingVariable,
                       bool &OffsetIsNegative) {
  const OpaquePointer &OP = Ptr.asOpaquePointer();

  uint64_t Offset = 0;
  std::optional<uint64_t> SurroundingArrayOffset;
  QualType CurType = OP.getObjectType();
  for (const PointerPathEntry &Entry : OP.path()) {
    switch (Entry.Kind) {
    case PointerPathEntry::Base: {
      const RecordDecl *RD = CurType->getAsRecordDecl();
      if (!ASTContext::hasLayout(RD))
        return std::nullopt;

      const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(RD);
      Offset += Layout.getBaseClassOffset(Entry.RD.getPointer()).getQuantity();

      CurType = ASTCtx.getCanonicalTagType(Entry.RD.getPointer());
    } break;

    case PointerPathEntry::Field: {
      const FieldDecl *FD = Entry.FD;
      const RecordDecl *RD = FD->getParent();
      if (!ASTContext::hasLayout(RD))
        return std::nullopt;

      const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(RD);
      Offset +=
          ASTCtx.toCharUnitsFromBits(Layout.getFieldOffset(FD->getFieldIndex()))
              .getQuantity();

      CurType = FD->getType();
    } break;
    case PointerPathEntry::Array:
    case PointerPathEntry::NegativeArray: {
      bool Add = (Entry.Kind == PointerPathEntry::Array);
      uint64_t Index = Entry.Index;
      if (!Add) {
        // NegativeArray is always > 0.
        OffsetIsNegative = true;
      }
      SurroundingArrayOffset = Offset;
      if (!CurType->isArrayType()) {
        if (Add)
          Offset += Index * ASTCtx.getTypeSizeInChars(CurType).getQuantity();
        else
          Offset -= Index * ASTCtx.getTypeSizeInChars(CurType).getQuantity();
        continue;
      }
      const ArrayType *AT = CurType->getAsArrayTypeUnsafe();
      assert(AT);
      QualType ElemTy = AT->getElementType();
      if (!validType(ElemTy) || isa<VariableArrayType>(AT))
        return std::nullopt;
      if (Add)
        Offset += Index * ASTCtx.getTypeSizeInChars(ElemTy).getQuantity();
      else
        Offset -= Index * ASTCtx.getTypeSizeInChars(ElemTy).getQuantity();
      CurType = AT->getElementType();
    } break;
    }
  }

  if (UseClosestSurroundingVariable && SurroundingArrayOffset)
    return Offset - *SurroundingArrayOffset;

  QualType Ty = CurType.getNonReferenceType();

  if (UseClosestSurroundingVariable &&
      (Ty->isIncompleteType() || Ty->isFunctionType()))
    return std::nullopt;

  if (isa<VariableArrayType>(Ty))
    return std::nullopt;

  if (OP.PathLength == 1 && OP.path().back().Kind == PointerPathEntry::Field &&
      isa<IncompleteArrayType>(CurType)) {
    return Offset;
  }

  if (UseClosestSurroundingVariable)
    return 0;

  return Offset;
}

/// Check if the given pointer points to the complete object, i.e. either to the
/// very beginning or after the end (into the flexible array member) of the
/// object.
static bool pointsToCompleteObject(const ASTContext &ASTCtx,
                                   const Pointer &Ptr) {
  const OpaquePointer &OP = Ptr.asOpaquePointer();
  if (OP.PathLength == 0)
    return true;

  QualType FieldType = computeFieldType(ASTCtx, OP);
  if (OP.isArrayElement())
    FieldType = OP.getSurroundingArray();
  return isa<IncompleteArrayType>(FieldType);
}

static std::optional<unsigned>
computeOpaqueSize(const ASTContext &ASTCtx, const Pointer &Ptr,
                  bool UseClosestSurroundingVariable, bool WritingOffTheEnd,
                  bool DetermineForCompleteObject) {
  const OpaquePointer &OP = Ptr.asOpaquePointer();

  CharUnits TypeSize;
  // NOTE: Clang does not consider base casts. GCC does.
  if (UseClosestSurroundingVariable) {
    QualType FieldTy =
        computeFieldType(ASTCtx, OP, SurroundingArray | IgnoreBaseCasts);
    if (!validType(FieldTy))
      return std::nullopt;
    TypeSize = ASTCtx.getTypeSizeInChars(FieldTy);
  } else {
    QualType ObjectTy = OP.getObjectType();
    if (!validType(ObjectTy))
      return std::nullopt;
    TypeSize = ASTCtx.getTypeSizeInChars(ObjectTy);
  }

  // The Flexible array member should only be checked if we're pointing to the
  // object as a whole, or if we're looking for the whole object size.
  if (!WritingOffTheEnd && !DetermineForCompleteObject)
    return TypeSize.getQuantity();

  // Check if we need to add the flexible array member size.
  const VarDecl *Base = dyn_cast<VarDecl>(OP.Base);
  if (!Base)
    return TypeSize.getQuantity();

  // If the base type is an incomplete array type (not a flexible array member
  // of a struct), and we're looking for the complete object... we can't.
  if (DetermineForCompleteObject && isa<IncompleteArrayType>(Base->getType()))
    return std::nullopt;

  if (!Base->getType()->isRecordType())
    return TypeSize.getQuantity();

  if (!Base->hasInit())
    return TypeSize.getQuantity();
  CharUnits FlexibleArraySize = Base->getFlexibleArrayInitChars(ASTCtx);
  return (TypeSize + FlexibleArraySize).getQuantity();
}

namespace clang {
namespace interp {

/// Evaluate __builtin_object_size or __builtin_dynamic_object_size for the
/// given pointer and Kind.
///
/// When computing the final result, the most important variable is
/// UseClosestSurroundingVariable. If it is true, we will use the field the
/// pointer points to, or the parent array of the element.
/// UseClosestSurroundingVariable is true for Kind 1 and 3.
UnsignedOrNone evaluateBuiltinObjectSize(const ASTContext &ASTCtx,
                                         unsigned Kind, Pointer &Ptr,
                                         const Expr *E, bool IsDynamic) {
  if (Ptr.isZero())
    return std::nullopt;

  bool InvalidBase = false;
  if (Ptr.isOpaquePointer()) {
    bool UseClosestSurroundingVariable = (Kind == 1) || (Kind == 3);
    const OpaquePointer &OP = Ptr.asOpaquePointer();
    InvalidBase = OP.Base->getType()->isPointerType();
    bool DetermineForCompleteObject = pointsToCompleteObject(ASTCtx, Ptr);
    bool WritingOffTheEnd = isUserWritingOffTheEnd(ASTCtx, OP);

    // Either the size of the full variable (Kind = 0 or 2) or the size of the
    // closest surrounding variable (Kind = 1 or 3).
    std::optional<unsigned> FullSize =
        computeOpaqueSize(ASTCtx, Ptr, UseClosestSurroundingVariable,
                          WritingOffTheEnd, DetermineForCompleteObject);

    if (!FullSize)
      return std::nullopt;

    // Similar to the FullSize above, the offset is relative either to the full
    // variable or to the closest surrounding variable.
    bool OffsetIsNegative = false;
    std::optional<uint64_t> Offset = computeOpaquePtrOffset(
        ASTCtx, Ptr, UseClosestSurroundingVariable, OffsetIsNegative);

    if (!Offset)
      return std::nullopt;

    if (OffsetIsNegative)
      return 0u;

    // For __builtin_dynamic_object_size on a counted_by-annotated flexible
    // array member, defer to IR generation (emitCountedBySize in CGBuiltin):
    // its runtime computation uses the live 'count' field and is more accurate
    // than the layout/initializer-derived size we'd produce here. Use the same
    // findStructFieldAccess form-recognition CGBuiltin does, so we refuse to
    // fold on exactly the shapes that path handles (and, importantly, *not*
    // on '&af.fam' which designates the array-as-a-whole and stays on the
    // layout-derived path to match GCC).
    if (IsDynamic) {
      const auto *ME =
          dyn_cast_if_present<MemberExpr>(findStructFieldAccess(E));
      const auto *FD = ME ? dyn_cast<FieldDecl>(ME->getMemberDecl()) : nullptr;
      if (FD && FD->getType()->isCountAttributedType())
        return std::nullopt;
    }

    if (!UseClosestSurroundingVariable || DetermineForCompleteObject) {
      // Kind=3 wants a lower bound, so we can't fall back to this.
      if (Kind == 3 && !DetermineForCompleteObject)
        return std::nullopt;

      if (InvalidBase)
        return std::nullopt;

      QualType ObjectTy = OP.getObjectType();
      if (ObjectTy->isIncompleteType() || isa<VariableArrayType>(ObjectTy) ||
          ObjectTy->isFunctionType())
        return std::nullopt;
    }

    *Offset += Ptr.getByteOffset();

    if (*Offset > *FullSize)
      return 0u;

    if (Kind == 1 && InvalidBase && WritingOffTheEnd)
      return std::nullopt;

    assert(*Offset <= *FullSize);
    return static_cast<unsigned>(*FullSize - *Offset);
  }

  // ----------------------------------------------------------------------------------------------------

  if (Ptr.isDummy() && Ptr.getType()->isPointerType())
    return std::nullopt;

  if (!Ptr.isBlockPointer())
    return std::nullopt;

  if (Ptr.isDummy()) {
    if (const VarDecl *VD = Ptr.getRootVarDecl();
        VD && VD->getType()->isPointerType())
      InvalidBase = true;
  }

  bool UseFieldDesc = (Kind & 1u);
  bool ReportMinimum = (Kind & 2u);

  // According to the GCC documentation, we want the size of the subobject
  // denoted by the pointer. But that's not quite right -- what we actually
  // want is the size of the immediately-enclosing array, if there is one.
  if (Ptr.isArrayElement())
    Ptr = Ptr.expand();

  bool DetermineForCompleteObject = Ptr.getFieldDesc() == Ptr.getDeclDesc();
  const Descriptor *DeclDesc = Ptr.getDeclDesc();
  assert(DeclDesc);

  if (!UseFieldDesc || DetermineForCompleteObject) {
    // Can't read beyond the pointer decl desc.
    if (!ReportMinimum && DeclDesc->getDataType(ASTCtx)->isPointerType())
      return std::nullopt;

    if (InvalidBase)
      return std::nullopt;
  } else {
    if (isUserWritingOffTheEnd(ASTCtx, Ptr, InvalidBase)) {
      // If we cannot determine the size of the initial allocation, then we
      // can't given an accurate upper-bound. However, we are still able to give
      // conservative lower-bounds for Type=3.
      if (Kind == 1)
        return std::nullopt;
    }
  }

  // The "closest surrounding subobject" is NOT a base class,
  // so strip the base class casts.
  if (UseFieldDesc && Ptr.isBaseClass())
    Ptr = Ptr.stripBaseCasts();

  const Descriptor *Desc = UseFieldDesc ? Ptr.getFieldDesc() : DeclDesc;
  assert(Desc);

  std::optional<unsigned> FullSize = computeFullDescSize(ASTCtx, Desc);
  if (!FullSize)
    return std::nullopt;

  unsigned ByteOffset;
  if (UseFieldDesc) {
    if (Ptr.isBaseClass()) {
      assert(computePointerOffset(ASTCtx, Ptr.getBase()) <=
             computePointerOffset(ASTCtx, Ptr));
      ByteOffset = computePointerOffset(ASTCtx, Ptr.getBase()) -
                   computePointerOffset(ASTCtx, Ptr);
    } else {
      if (Ptr.inArray())
        ByteOffset =
            computePointerOffset(ASTCtx, Ptr) -
            computePointerOffset(ASTCtx, Ptr.expand().atIndex(0).narrow());
      else
        ByteOffset = 0;
    }
  } else
    ByteOffset = computePointerOffset(ASTCtx, Ptr);

  assert(ByteOffset <= *FullSize);
  return *FullSize - ByteOffset;
}
} // namespace interp
} // namespace clang
