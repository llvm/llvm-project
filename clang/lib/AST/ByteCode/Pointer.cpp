//===--- Pointer.cpp - Types for the constexpr VM ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pointer.h"
#include "Boolean.h"
#include "Char.h"
#include "Context.h"
#include "Floating.h"
#include "Function.h"
#include "InitMap.h"
#include "Integral.h"
#include "InterpBlock.h"
#include "MemberPointer.h"
#include "PrimType.h"
#include "Record.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecordLayout.h"

using namespace clang;
using namespace clang::interp;

Pointer::Pointer(Block *Pointee)
    : Pointer(Pointee, Pointee->getDescriptor()->getMetadataSize(),
              Pointee->getDescriptor()->getMetadataSize()) {}

Pointer::Pointer(Block *Pointee, uint64_t BaseAndOffset)
    : Pointer(Pointee, BaseAndOffset, BaseAndOffset) {}

Pointer::Pointer(Block *Pointee, unsigned Base, uint64_t Offset)
    : Offset(Offset), StorageKind(Storage::Block) {
  assert(Pointee);
  assert(Base % alignof(void *) == 0 && "wrong base");
  assert(Base >= Pointee->getDescriptor()->getMetadataSize());

  BS = {Pointee, Base, nullptr, nullptr};
  Pointee->addPointer(this);
}

Pointer::Pointer(const Pointer &P)
    : Offset(P.Offset), StorageKind(P.StorageKind) {
  switch (StorageKind) {
  case Storage::Int:
    Int = P.Int;
    break;
  case Storage::Block:
    BS = P.BS;
    if (BS.Pointee)
      BS.Pointee->addPointer(this);
    break;
  case Storage::Fn:
    Fn = P.Fn;
    break;
  case Storage::Typeid:
    Typeid = P.Typeid;
    break;
  }
}

Pointer::Pointer(Pointer &&P) : Offset(P.Offset), StorageKind(P.StorageKind) {
  switch (StorageKind) {
  case Storage::Int:
    Int = P.Int;
    break;
  case Storage::Block:
    BS = P.BS;
    if (BS.Pointee)
      BS.Pointee->replacePointer(&P, this);
    break;
  case Storage::Fn:
    Fn = P.Fn;
    break;
  case Storage::Typeid:
    Typeid = P.Typeid;
    break;
  }
}

Pointer::~Pointer() {
  if (!isBlockPointer())
    return;

  if (Block *Pointee = BS.Pointee) {
    Pointee->removePointer(this);
    BS.Pointee = nullptr;
    Pointee->cleanup();
  }
}

Pointer &Pointer::operator=(const Pointer &P) {
  // If the current storage type is Block, we need to remove
  // this pointer from the block.
  if (isBlockPointer()) {
    if (P.isBlockPointer() && this->block() == P.block()) {
      Offset = P.Offset;
      BS.Base = P.BS.Base;
      return *this;
    }

    if (Block *Pointee = BS.Pointee) {
      Pointee->removePointer(this);
      BS.Pointee = nullptr;
      Pointee->cleanup();
    }
  }

  StorageKind = P.StorageKind;
  Offset = P.Offset;

  switch (StorageKind) {
  case Storage::Int:
    Int = P.Int;
    break;
  case Storage::Block:
    BS = P.BS;

    if (BS.Pointee)
      BS.Pointee->addPointer(this);
    break;
  case Storage::Fn:
    Fn = P.Fn;
    break;
  case Storage::Typeid:
    Typeid = P.Typeid;
  }
  return *this;
}

Pointer &Pointer::operator=(Pointer &&P) {
  // If the current storage type is Block, we need to remove
  // this pointer from the block.
  if (isBlockPointer()) {
    if (P.isBlockPointer() && this->block() == P.block()) {
      Offset = P.Offset;
      BS.Base = P.BS.Base;
      return *this;
    }

    if (Block *Pointee = BS.Pointee) {
      Pointee->removePointer(this);
      BS.Pointee = nullptr;
      Pointee->cleanup();
    }
  }

  StorageKind = P.StorageKind;
  Offset = P.Offset;

  switch (StorageKind) {
  case Storage::Int:
    Int = P.Int;
    break;
  case Storage::Block:
    BS = P.BS;

    if (BS.Pointee)
      BS.Pointee->addPointer(this);
    break;
  case Storage::Fn:
    Fn = P.Fn;
    break;
  case Storage::Typeid:
    Typeid = P.Typeid;
  }
  return *this;
}

APValue Pointer::toAPValue(const ASTContext &ASTCtx) const {
  llvm::SmallVector<APValue::LValuePathEntry, 5> Path;

  if (isZero())
    return APValue(static_cast<const Expr *>(nullptr), CharUnits::Zero(), Path,
                   /*IsOnePastEnd=*/false, /*IsNullPtr=*/true);
  if (isIntegralPointer())
    return APValue(static_cast<const Expr *>(nullptr),
                   CharUnits::fromQuantity(asIntPointer().Value + this->Offset),
                   Path,
                   /*IsOnePastEnd=*/false, /*IsNullPtr=*/false);
  if (isFunctionPointer()) {
    const FunctionPointer &FP = asFunctionPointer();
    if (const FunctionDecl *FD = FP.Func->getDecl())
      return APValue(FD, CharUnits::fromQuantity(Offset), {},
                     /*OnePastTheEnd=*/false, /*IsNull=*/false);
    return APValue(FP.Func->getExpr(), CharUnits::fromQuantity(Offset), {},
                   /*OnePastTheEnd=*/false, /*IsNull=*/false);
  }

  if (isTypeidPointer()) {
    TypeInfoLValue TypeInfo(Typeid.TypePtr);
    return APValue(APValue::LValueBase::getTypeInfo(
                       TypeInfo, QualType(Typeid.TypeInfoType, 0)),
                   CharUnits::Zero(), {},
                   /*OnePastTheEnd=*/false, /*IsNull=*/false);
  }

  // Build the lvalue base from the block.
  const Descriptor *Desc = getDeclDesc();
  APValue::LValueBase Base;
  if (const auto *VD = Desc->asValueDecl())
    Base = VD;
  else if (const auto *E = Desc->asExpr()) {
    if (block()->isDynamic()) {
      QualType AllocatedType = getDeclPtr().getFieldDesc()->getDataType(ASTCtx);
      DynamicAllocLValue DA(*block()->DynAllocId);
      Base = APValue::LValueBase::getDynamicAlloc(DA, AllocatedType);
    } else {
      Base = E;
    }
  } else
    llvm_unreachable("Invalid allocation type");

  CharUnits Offset = CharUnits::Zero();

  auto getFieldOffset = [&](const FieldDecl *FD) -> CharUnits {
    // This shouldn't happen, but if it does, don't crash inside
    // getASTRecordLayout.
    if (FD->getParent()->isInvalidDecl())
      return CharUnits::Zero();
    const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(FD->getParent());
    unsigned FieldIndex = FD->getFieldIndex();
    return ASTCtx.toCharUnitsFromBits(Layout.getFieldOffset(FieldIndex));
  };

  // Build the path into the object.
  bool OnePastEnd = isOnePastEnd() && !isZeroSizeArray();

  PtrView Ptr = view();
  while (Ptr.isField() || Ptr.isArrayElement()) {

    if (Ptr.isArrayRoot()) {
      // An array root may still be an array element itself.
      if (Ptr.isArrayElement()) {
        Ptr = Ptr.expand();
        const Descriptor *Desc = Ptr.getFieldDesc();
        unsigned Index = Ptr.getIndex();
        QualType ElemType = Desc->getElemQualType();
        Offset += (Index * ASTCtx.getTypeSizeInChars(ElemType));
        if (Ptr.getArray().getFieldDesc()->IsArray)
          Path.push_back(APValue::LValuePathEntry::ArrayIndex(Index));
        Ptr = Ptr.getArray();
      } else {
        const Descriptor *Desc = Ptr.getFieldDesc();
        const auto *Dcl = Desc->asDecl();
        Path.push_back(APValue::LValuePathEntry({Dcl, /*IsVirtual=*/false}));

        if (const auto *FD = dyn_cast_if_present<FieldDecl>(Dcl))
          Offset += getFieldOffset(FD);

        Ptr = Ptr.getBase();
      }
    } else if (Ptr.isArrayElement()) {
      Ptr = Ptr.expand();
      const Descriptor *Desc = Ptr.getFieldDesc();
      unsigned Index;
      if (Ptr.isOnePastEnd()) {
        Index = Ptr.getArray().getNumElems();
        OnePastEnd = false;
      } else
        Index = Ptr.getIndex();

      QualType ElemType = Desc->getElemQualType();
      if (const auto *RD = ElemType->getAsRecordDecl();
          RD && !RD->getDefinition()) {
        // Ignore this for the offset.
      } else {
        Offset += (Index * ASTCtx.getTypeSizeInChars(ElemType));
      }
      if (Ptr.getArray().getFieldDesc()->IsArray)
        Path.push_back(APValue::LValuePathEntry::ArrayIndex(Index));
      Ptr = Ptr.getArray();
    } else {
      const Descriptor *Desc = Ptr.getFieldDesc();

      // Create a path entry for the field.
      if (const auto *BaseOrMember = Desc->asDecl()) {
        bool IsVirtual = false;
        if (const auto *FD = dyn_cast<FieldDecl>(BaseOrMember)) {
          Ptr = Ptr.getBase();
          Offset += getFieldOffset(FD);
        } else if (const auto *RD = dyn_cast<CXXRecordDecl>(BaseOrMember)) {
          IsVirtual = Ptr.isVirtualBaseClass();
          Ptr = Ptr.getBase();
          const Record *BaseRecord = Ptr.getRecord();

          const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(
              cast<CXXRecordDecl>(BaseRecord->getDecl()));
          if (IsVirtual)
            Offset += Layout.getVBaseClassOffset(RD);
          else
            Offset += Layout.getBaseClassOffset(RD);

        } else {
          Ptr = Ptr.getBase();
        }
        Path.push_back(APValue::LValuePathEntry({BaseOrMember, IsVirtual}));
        continue;
      }
      llvm_unreachable("Invalid field type");
    }
  }

  // We assemble the LValuePath starting from the innermost pointer to the
  // outermost one. SO in a.b.c, the first element in Path will refer to
  // the field 'c', while later code expects it to refer to 'a'.
  // Just invert the order of the elements.
  std::reverse(Path.begin(), Path.end());

  auto Result = APValue(Base, Offset, Path, OnePastEnd);
  Result.setConstexprUnknown(isConstexprUnknown());
  return Result;
}

void Pointer::print(llvm::raw_ostream &OS) const {
  switch (StorageKind) {
  case Storage::Block: {
    const Block *B = BS.Pointee;
    OS << "(Block) " << B << " {";

    if (isRoot())
      OS << "rootptr(" << BS.Base << "), ";
    else
      OS << BS.Base << ", ";

    if (isElementPastEnd())
      OS << "pastend, ";
    else
      OS << Offset << ", ";

    if (B)
      OS << B->getSize();
    else
      OS << "nullptr";
    OS << "}";
  } break;
  case Storage::Int:
    OS << "(Int) {" << Int.Value << " + " << Offset << ", " << Int.Ty << "}";
    break;
  case Storage::Fn:
    OS << "(Fn) { " << Fn.Func << " + " << Offset << " }";
    break;
  case Storage::Typeid:
    OS << "(Typeid) { " << (const void *)asTypeidPointer().TypePtr << ", "
       << (const void *)asTypeidPointer().TypeInfoType << " + " << Offset
       << "}";
  }
}

/// Compute an offset that can be used to compare the pointer to another one
/// with the same base. To get accurate results, we basically _have to_ compute
/// the lvalue offset using the ASTRecordLayout.
///
/// This function will fail if we're trying to get the type size of a forward
/// declaration.
///
// FIXME: We're still mixing values from the record layout with our internal
// offsets, which will inevitably lead to cryptic errors.
std::optional<size_t>
Pointer::computeOffsetForComparison(const ASTContext &ASTCtx) const {
  switch (StorageKind) {
  case Storage::Int:
    return Int.Value + Offset;
  case Storage::Block:
    // See below.
    break;
  case Storage::Fn:
    return getIntegerRepresentation();
  case Storage::Typeid:
    return reinterpret_cast<uintptr_t>(asTypeidPointer().TypePtr) + Offset;
  }

  auto getTypeSize = [&](QualType T) -> std::optional<size_t> {
    if (const RecordType *RT = T->getAs<RecordType>()) {
      // We cannot get the type size of a forward declaration.
      if (!RT->getDecl()->getDefinition())
        return std::nullopt;
    }
    return ASTCtx.getTypeSizeInChars(T).getQuantity();
  };

  size_t Result = 0;
  PtrView P = view();
  while (true) {
    if (P.isVirtualBaseClass()) {
      Result += getInlineDesc()->Offset;
      P = P.getBase();
      continue;
    }

    if (P.isBaseClass()) {
      Result += P.getInlineDesc()->Offset - sizeof(InlineDescriptor);
      P = P.getBase();
      continue;
    }
    if (P.isArrayElement()) {
      P = P.expand();
      Result += (P.getIndex() * P.elemSize());
      P = P.getArray();
      continue;
    }

    if (P.isRoot()) {
      if (P.isOnePastEnd()) {
        if (auto Size = getTypeSize(P.getDeclDesc()->getType()))
          Result += *Size;
        else
          return std::nullopt;
      }
      break;
    }

    assert(P.getField());
    const Record *R = P.getBase().getRecord();
    assert(R);

    const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(R->getDecl());
    Result += ASTCtx
                  .toCharUnitsFromBits(
                      Layout.getFieldOffset(P.getField()->getFieldIndex()))
                  .getQuantity();

    if (P.isOnePastEnd()) {
      if (auto Size = getTypeSize(P.getField()->getType()))
        Result += *Size;
      else
        return std::nullopt;
    }

    P = P.getBase();
    if (P.isRoot())
      break;
  }
  return Result;
}

std::optional<size_t>
Pointer::computeLayoutOffset(const ASTContext &ASTCtx) const {
  switch (StorageKind) {
  case Storage::Int:
    return Int.Value + Offset;
  case Storage::Block:
    // See below.
    break;
  case Storage::Fn:
    return getIntegerRepresentation();
  case Storage::Typeid:
    return reinterpret_cast<uintptr_t>(asTypeidPointer().TypePtr) + Offset;
  }

  auto getTypeSize = [&](QualType T) -> std::optional<size_t> {
    if (const RecordType *RT = T->getAs<RecordType>()) {
      // We cannot get the type size of a forward declaration.
      if (!RT->getDecl()->getDefinition())
        return std::nullopt;
    }
    return ASTCtx.getTypeSizeInChars(T).getQuantity();
  };

  auto getRecordDecl = [&](PtrView P) -> const CXXRecordDecl * {
    if (const Record *R = P.getRecord())
      return cast<CXXRecordDecl>(R->getDecl());
    return cast<CXXRecordDecl>(P.getFieldDesc()->asDecl());
  };

  auto getRecordSize = [&](const RecordDecl *RD) -> unsigned {
    CanQualType RecordTy = ASTCtx.getCanonicalTagType(RD);
    return ASTCtx.getTypeSizeInChars(RecordTy).getQuantity();
  };

  size_t Result = 0;
  PtrView P = view();
  while (true) {
    if (P.isBaseClass()) {
      const ASTRecordLayout &Layout =
          ASTCtx.getASTRecordLayout(getRecordDecl(P.getBase()));
      const CXXRecordDecl *RD = getRecordDecl(P);
      if (P.isVirtualBaseClass())
        Result += Layout.getVBaseClassOffset(RD).getQuantity();
      else
        Result += Layout.getBaseClassOffset(RD).getQuantity();

      if (P.isOnePastEnd())
        Result += getRecordSize(RD);

      P = P.getBase();
      continue;
    }

    if (P.isArrayElement()) {
      P = P.expand();
      assert(P.getFieldDesc()->isArray());
      if (std::optional<size_t> ElemSize =
              getTypeSize(P.getFieldDesc()->getElemQualType()))
        Result += *ElemSize * P.getIndex();
      else
        return std::nullopt;

      P = P.getArray();
      continue;
    }

    if (P.isRoot()) {
      if (P.isPastEnd() || P.isOnePastEnd()) {
        if (std::optional<size_t> Size =
                getTypeSize(P.getDeclDesc()->getType()))
          Result += *Size * P.getIndex();
        else
          return std::nullopt;
      }
      break;
    }

    assert(P.getField());
    const FieldDecl *F = P.getField();
    const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(F->getParent());
    Result +=
        ASTCtx.toCharUnitsFromBits(Layout.getFieldOffset(F->getFieldIndex()))
            .getQuantity();

    if (P.isPastEnd() || P.isOnePastEnd()) {
      if (std::optional<size_t> Size = getTypeSize(F->getType()))
        Result += *Size * P.getIndex();
      else
        return std::nullopt;
    }

    P = P.getBase();
    if (P.isRoot())
      break;
  }
  return Result;
}

std::string Pointer::toDiagnosticString(const ASTContext &Ctx) const {
  if (isZero())
    return "nullptr";

  if (isIntegralPointer())
    return (Twine("&(") + Twine(asIntPointer().Value + Offset) + ")").str();

  QualType Ty = getType();
  if (Ty->isLValueReferenceType())
    Ty = Ty->getPointeeType();
  return toAPValue(Ctx).getAsString(Ctx, Ty);
}

bool Pointer::isInitialized() const {
  if (!isBlockPointer())
    return true;

  if (isRoot() && BS.Base == sizeof(GlobalInlineDescriptor) &&
      Offset == BS.Base) {
    const auto &GD = block()->getBlockDesc<GlobalInlineDescriptor>();
    return GD.InitState == GlobalInitState::Initialized;
  }

  assert(BS.Pointee && "Cannot check if null pointer was initialized");
  const Descriptor *Desc = getFieldDesc();
  assert(Desc);
  if (Desc->isPrimitiveArray())
    return isElementInitialized(getIndex());

  if (asBlockPointer().Base == 0)
    return true;
  // Field has its bit in an inline descriptor.
  return getInlineDesc()->IsInitialized;
}

bool PtrView::isElementInitialized(unsigned Index) const {
  const Descriptor *Desc = getFieldDesc();
  assert(Desc);

  if (Pointee->isStatic() && Base == 0)
    return true;

  if (isRoot() && Base == sizeof(GlobalInlineDescriptor) && Offset == Base) {
    const auto &GD = Pointee->getBlockDesc<GlobalInlineDescriptor>();
    return GD.InitState == GlobalInitState::Initialized;
  }

  if (Desc->isPrimitiveArray()) {
    InitMapPtr IM = getInitMap();

    if (IM.allInitialized())
      return true;

    if (!IM.hasInitMap())
      return false;
    return IM->isElementInitialized(Index);
  }
  return isInitialized();
}

bool Pointer::isElementAlive(unsigned Index) const {
  assert(getFieldDesc()->isPrimitiveArray());

  InitMapPtr &IM = getInitMap();
  if (!IM.hasInitMap())
    return true;

  if (IM.allInitialized())
    return true;

  return IM->isElementAlive(Index);
}

Lifetime PtrView::getLifetime() const {
  if (Base < sizeof(InlineDescriptor))
    return Lifetime::Started;

  if (inArray() && !isArrayRoot()) {
    InitMapPtr &IM = getInitMap();

    if (!IM.hasInitMap()) {
      if (IM.allInitialized())
        return Lifetime::Started;
      return getArray().getLifetime();
    }

    return IM->isElementAlive(getIndex()) ? Lifetime::Started : Lifetime::Ended;
  }

  return getInlineDesc()->LifeState;
}

void PtrView::setLifeState(Lifetime L) const {
  if (Base < sizeof(InlineDescriptor))
    return;

  if (inArray() && !isArrayRoot()) {
    assert(L == Lifetime::Started || L == Lifetime::Ended);
    const Descriptor *Desc = getFieldDesc();
    InitMapPtr &IM = getInitMap();
    if (!IM.hasInitMap())
      IM.setInitMap(new InitMap(Desc->getNumElems(), IM.allInitialized()));

    if (L == Lifetime::Ended)
      IM->endElementLifetime(getIndex());
    else if (L == Lifetime::Started)
      IM->startElementLifetime(getIndex());
    assert(isArrayRoot() || (this->getLifetime() == L));
    return;
  }

  getInlineDesc()->LifeState = L;
}

void PtrView::initialize() const {
  if (isRoot() && Base == sizeof(GlobalInlineDescriptor) && Offset == Base) {
    auto &GD = Pointee->getBlockDesc<GlobalInlineDescriptor>();
    GD.InitState = GlobalInitState::Initialized;
    return;
  }

  const Descriptor *Desc = getFieldDesc();
  assert(Desc);
  if (Desc->isPrimitiveArray()) {
    if (Desc->getNumElems() != 0)
      initializeElement(getIndex());
    return;
  }

  // Field has its bit in an inline descriptor.
  assert(Base != 0 && "Only composite fields can be initialised");
  getInlineDesc()->IsInitialized = true;
  getInlineDesc()->LifeState = Lifetime::Started;
}

void PtrView::initializeElement(unsigned Index) const {
  // Primitive global arrays don't have an initmap.
  if (Pointee->isStatic() && Base == 0)
    return;

  assert(Index < getFieldDesc()->getNumElems());

  InitMapPtr &IM = getInitMap();
  if (IM.allInitialized())
    return;

  if (!IM.hasInitMap()) {
    const Descriptor *Desc = getFieldDesc();
    IM.setInitMap(new InitMap(Desc->getNumElems()));
  }
  assert(IM.hasInitMap());

  if (IM->initializeElement(Index))
    IM.noteAllInitialized();
}

void Pointer::initializeAllElements() const {
  assert(getFieldDesc()->isPrimitiveArray());
  assert(isArrayRoot());

  getInitMap().noteAllInitialized();
}

bool PtrView::allElementsInitialized() const {
  assert(getFieldDesc()->isPrimitiveArray());
  assert(isArrayRoot());

  if (Pointee->isStatic() && Base == 0)
    return true;

  if (isRoot() && Base == sizeof(GlobalInlineDescriptor) && Offset == Base) {
    const auto &GD = Pointee->getBlockDesc<GlobalInlineDescriptor>();
    return GD.InitState == GlobalInitState::Initialized;
  }

  InitMapPtr IM = getInitMap();
  return IM.allInitialized();
}

bool Pointer::allElementsAlive() const {
  assert(getFieldDesc()->isPrimitiveArray());
  assert(isArrayRoot());

  if (isStatic() && BS.Base == 0)
    return true;

  if (isRoot() && BS.Base == sizeof(GlobalInlineDescriptor) &&
      Offset == BS.Base) {
    const auto &GD = block()->getBlockDesc<GlobalInlineDescriptor>();
    return GD.InitState == GlobalInitState::Initialized;
  }

  InitMapPtr &IM = getInitMap();
  return IM.allInitialized() || (IM.hasInitMap() && IM->allElementsAlive());
}

void PtrView::activate() const {
  // Field has its bit in an inline descriptor.
  assert(Base != 0 && "Only composite fields can be activated");

  if (isRoot() && Base == sizeof(GlobalInlineDescriptor))
    return;
  if (!getInlineDesc()->InUnion)
    return;

  std::function<void(PtrView P)> activate;
  activate = [&activate](PtrView P) -> void {
    P.getInlineDesc()->IsActive = true;
    P.startLifetime();
    if (const Record *R = P.getRecord(); R && !R->isUnion()) {
      for (const Record::Field &F : R->fields()) {
        PtrView FieldPtr = P.atField(F.Offset);
        if (!FieldPtr.getInlineDesc()->IsActive)
          activate(FieldPtr);
      }
      // FIXME: Bases?
    }
  };

  std::function<void(PtrView &)> deactivate;
  deactivate = [&deactivate](PtrView &P) -> void {
    P.getInlineDesc()->IsActive = false;

    if (const Record *R = P.getRecord()) {
      for (const Record::Field &F : R->fields()) {
        PtrView FieldPtr = P.atField(F.Offset);
        if (FieldPtr.getInlineDesc()->IsActive)
          deactivate(FieldPtr);
      }
      // FIXME: Bases?
    }
  };

  PtrView B = *this;
  // Primitive array elements can't be activated individually, so
  // look at the array root instead.
  if (B.getFieldDesc()->isPrimitiveArray() && B.isArrayElement())
    B = B.getArray();

  while (!B.isRoot() && B.inUnion()) {
    activate(B);

    // When walking up the pointer chain, deactivate
    // all union child pointers that aren't on our path.
    PtrView Cur = B;
    B = B.getBase();
    if (const Record *BR = B.getRecord(); BR && BR->isUnion()) {
      for (const Record::Field &F : BR->fields()) {
        PtrView FieldPtr = B.atField(F.Offset);
        if (FieldPtr != Cur)
          deactivate(FieldPtr);
      }
    }
  }
}

bool Pointer::hasSameBase(const Pointer &A, const Pointer &B) {
  // Two null pointers always have the same base.
  if (A.isZero() && B.isZero())
    return true;

  if (A.isIntegralPointer() && B.isIntegralPointer())
    return true;
  if (A.isFunctionPointer() && B.isFunctionPointer())
    return true;
  if (A.isTypeidPointer() && B.isTypeidPointer())
    return A.asTypeidPointer().TypePtr == B.asTypeidPointer().TypePtr;

  if (A.StorageKind != B.StorageKind)
    return false;

  return A.asBlockPointer().Pointee == B.asBlockPointer().Pointee;
}

bool Pointer::pointToSameBlock(const Pointer &A, const Pointer &B) {
  if (!A.isBlockPointer() || !B.isBlockPointer())
    return false;
  return A.block() == B.block();
}

bool Pointer::hasSameArray(const Pointer &A, const Pointer &B) {
  return hasSameBase(A, B) && A.BS.Base == B.BS.Base &&
         A.getFieldDesc()->IsArray;
}

bool Pointer::pointsToLiteral() const {
  if (isZero() || !isBlockPointer())
    return false;

  if (block()->isDynamic())
    return false;

  const Expr *E = block()->getDescriptor()->asExpr();
  return E && !isa<MaterializeTemporaryExpr, StringLiteral>(E);
}

bool Pointer::pointsToStringLiteral() const {
  if (isZero() || !isBlockPointer())
    return false;

  if (block()->isDynamic())
    return false;

  const Expr *E = block()->getDescriptor()->asExpr();
  return isa_and_nonnull<StringLiteral>(E);
}

bool Pointer::pointsToLabel() const {
  if (isZero() || !isBlockPointer())
    return false;

  if (const Expr *E = BS.Pointee->getDescriptor()->asExpr())
    return isa<AddrLabelExpr>(E);
  return false;
}

std::optional<std::pair<PtrView, PtrView>>
Pointer::computeSplitPoint(const Pointer &A, const Pointer &B) {
  if (!A.isBlockPointer() || !B.isBlockPointer())
    return std::nullopt;

  if (A.asBlockPointer().Pointee != B.asBlockPointer().Pointee)
    return std::nullopt;
  if (A.isRoot() && B.isRoot())
    return std::nullopt;

  if (A == B)
    return std::make_pair(A.view(), B.view());

  auto getBase = [](PtrView P) -> PtrView {
    if (P.isArrayElement())
      return P.expand().getArray();
    return P.getBase();
  };

  PtrView IterA = A.view();
  PtrView IterB = B.view();
  PtrView CurA = IterA;
  PtrView CurB = IterB;
  for (;;) {
    if (IterA.Base > IterB.Base) {
      CurA = IterA;
      IterA = getBase(IterA);
    } else {
      CurB = IterB;
      IterB = getBase(IterB);
    }

    if (IterA == IterB) {
      // If the Iter is an array, CurA and CurB are both elements of the same
      // array. That is fine, so return nullopt.
      if (IterA.getFieldDesc()->isArray())
        return std::nullopt;
      return std::make_pair(CurA, CurB);
    }

    if (IterA.isRoot() && IterB.isRoot())
      return std::nullopt;
  }

  llvm_unreachable("The loop above should've returned.");
}

std::optional<APValue> Pointer::toRValue(const Context &Ctx,
                                         QualType ResultType) const {
  const ASTContext &ASTCtx = Ctx.getASTContext();
  assert(!ResultType.isNull());
  // Method to recursively traverse composites.
  std::function<bool(QualType, PtrView, APValue &)> Composite;
  Composite = [&Composite, &Ctx, &ASTCtx](QualType Ty, PtrView Ptr,
                                          APValue &R) {
    if (const auto *AT = Ty->getAs<AtomicType>())
      Ty = AT->getValueType();

    // Invalid pointers.
    if (Ptr.isDummy() || !Ptr.isLive() || Ptr.isPastEnd())
      return false;

    // Primitives should never end up here.
    assert(!Ctx.canClassify(Ty));
    const Descriptor *FieldDesc = Ptr.getFieldDesc();
    assert(FieldDesc);

    if (const auto *RT = Ty->getAsCanonical<RecordType>()) {
      if (!FieldDesc->isRecord())
        return false;
      const auto *Record = Ptr.getRecord();
      assert(Record && "Missing record descriptor");

      bool Ok = true;
      if (RT->getDecl()->isUnion()) {
        const FieldDecl *ActiveField = nullptr;
        APValue Value;
        for (const auto &F : Record->fields()) {
          PtrView FP = Ptr.atField(F.Offset);
          if (FP.isActive()) {
            const Descriptor *Desc = F.Desc;
            if (Desc->isPrimitive()) {
              TYPE_SWITCH(Desc->getPrimType(),
                          Value = FP.deref<T>().toAPValue(ASTCtx));
            } else {
              QualType FieldTy = F.Decl->getType();
              Ok &= Composite(FieldTy, FP, Value);
            }
            ActiveField = FP.getFieldDesc()->asFieldDecl();
            break;
          }
        }
        R = APValue(ActiveField, Value);
      } else {
        unsigned NF = Record->getNumFields();
        unsigned NB = Record->getNumBases();
        unsigned NV = Ptr.isBaseClass() ? 0 : Record->getNumVirtualBases();

        R = APValue(APValue::UninitStruct(), NB, NF);

        for (unsigned I = 0; I != NF; ++I) {
          const Record::Field *FD = Record->getField(I);
          const Descriptor *Desc = FD->Desc;
          PtrView FP = Ptr.atField(FD->Offset);
          APValue &Value = R.getStructField(I);
          if (Desc->isPrimitive()) {
            TYPE_SWITCH(Desc->getPrimType(),
                        Value = FP.deref<T>().toAPValue(ASTCtx));
          } else {
            QualType FieldTy = FD->Decl->getType();
            Ok &= Composite(FieldTy, FP, Value);
          }
        }

        for (unsigned I = 0; I != NB; ++I) {
          const Record::Base *BD = Record->getBase(I);
          QualType BaseTy = Ctx.getASTContext().getCanonicalTagType(BD->Decl);
          PtrView BP = Ptr.atField(BD->Offset);
          Ok &= Composite(BaseTy, BP, R.getStructBase(I));
        }

        for (unsigned I = 0; I != NV; ++I) {
          const Record::Base *VD = Record->getVirtualBase(I);
          assert(VD);
          QualType VirtBaseTy =
              Ctx.getASTContext().getCanonicalTagType(VD->Decl);
          PtrView VP = Ptr.atField(VD->Offset);
          Ok &= Composite(VirtBaseTy, VP, R.getStructBase(NB + I));
        }
      }
      return Ok;
    }

    if (Ty->isIncompleteArrayType()) {
      R = APValue(APValue::UninitArray(), 0, 0);
      return true;
    }

    if (const auto *AT = Ty->getAsArrayTypeUnsafe()) {
      if (!FieldDesc->isArray())
        return false;
      const size_t NumElems = Ptr.getNumElems();
      QualType ElemTy = AT->getElementType();
      R = APValue(APValue::UninitArray{}, NumElems, NumElems);

      bool Ok = true;
      OptPrimType ElemT = Ctx.classify(ElemTy);
      for (unsigned I = 0; I != NumElems; ++I) {
        APValue &Slot = R.getArrayInitializedElt(I);
        if (ElemT) {
          TYPE_SWITCH(*ElemT, Slot = Ptr.elem<T>(I).toAPValue(ASTCtx));
        } else {
          Ok &= Composite(ElemTy, Ptr.atIndex(I).narrow(), Slot);
        }
      }
      return Ok;
    }

    // Complex types.
    if (Ty->isAnyComplexType()) {
      // Can happen via C casts.
      if (!FieldDesc->getType()->isAnyComplexType())
        return false;

      PrimType ElemT = FieldDesc->getPrimType();
      if (isIntegerOrBoolType(ElemT)) {
        INT_TYPE_SWITCH(ElemT, {
          auto V1 = Ptr.elem<T>(0);
          auto V2 = Ptr.elem<T>(1);
          R = APValue(V1.toAPSInt(), V2.toAPSInt());
          return true;
        });
      } else if (ElemT == PT_Float) {
        R = APValue(Ptr.elem<Floating>(0).getAPFloat(),
                    Ptr.elem<Floating>(1).getAPFloat());
        return true;
      }
      return false;
    }

    // Vector types.
    if (const auto *VT = Ty->getAs<VectorType>()) {
      if (!FieldDesc->isPrimitiveArray())
        return false;

      PrimType ElemT = FieldDesc->getPrimType();
      SmallVector<APValue> Values;
      Values.reserve(VT->getNumElements());
      for (unsigned I = 0; I != VT->getNumElements(); ++I) {
        TYPE_SWITCH(ElemT,
                    { Values.push_back(Ptr.elem<T>(I).toAPValue(ASTCtx)); });
      }

      assert(Values.size() == VT->getNumElements());
      R = APValue(Values.data(), Values.size());
      return true;
    }

    // Constant Matrix types.
    if (const auto *MT = Ty->getAs<ConstantMatrixType>()) {
      if (!FieldDesc->isPrimitiveArray())
        return false;
      PrimType ElemT = FieldDesc->getPrimType();
      unsigned NumElems = MT->getNumElementsFlattened();

      SmallVector<APValue> Values;
      Values.reserve(NumElems);
      for (unsigned I = 0; I != NumElems; ++I) {
        TYPE_SWITCH(ElemT,
                    { Values.push_back(Ptr.elem<T>(I).toAPValue(ASTCtx)); });
      }

      R = APValue(Values.data(), MT->getNumRows(), MT->getNumColumns());
      return true;
    }

    llvm_unreachable("invalid value to return");
  };

  // Can't return functions as rvalues.
  if (ResultType->isFunctionType())
    return std::nullopt;

  // Invalid to read from.
  if (isDummy() || !isLive() || isPastEnd() ||
      (isOnePastEnd() && !isZeroSizeArray()))
    return std::nullopt;

  // We can return these as rvalues, but we can't deref() them.
  if (isZero() || isIntegralPointer())
    return toAPValue(ASTCtx);

  // Just load primitive types.
  if (OptPrimType T = Ctx.classify(ResultType)) {
    if (!canDeref(*T))
      return std::nullopt;
    TYPE_SWITCH(*T, return this->deref<T>().toAPValue(ASTCtx));
  }

  // Return the composite type.
  APValue Result;
  if (!Composite(ResultType, view(), Result))
    return std::nullopt;
  return Result;
}

const VarDecl *Pointer::getRootVarDecl() const {
  if (isBlockPointer())
    return getDeclDesc()->asVarDecl();
  return nullptr;
}

std::optional<IntPointer> IntPointer::atOffset(const interp::Context &Ctx,
                                               unsigned Offset) const {
  QualType CurType = getPointeeType();
  if (CurType.isNull() || !CurType->isRecordType())
    return std::nullopt;

  const Record *R = Ctx.getRecord(CurType->getAsRecordDecl());
  if (!R)
    return *this;

  const Record::Field *F = nullptr;
  for (auto &It : R->fields()) {
    if (It.Offset == Offset) {
      F = &It;
      break;
    }
  }
  if (!F)
    return *this;

  const FieldDecl *FD = F->Decl;
  if (FD->getParent()->isInvalidDecl())
    return std::nullopt;

  const ASTContext &ASTCtx = Ctx.getASTContext();
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(FD->getParent());
  unsigned FieldIndex = FD->getFieldIndex();
  uint64_t FieldOffset =
      ASTCtx.toCharUnitsFromBits(Layout.getFieldOffset(FieldIndex))
          .getQuantity();

  return IntPointer{FD->getType().getTypePtr(), this->Value + FieldOffset};
}

IntPointer IntPointer::baseCast(const interp::Context &Ctx,
                                unsigned BaseOffset) const {
  if (!Ty)
    return *this;

  QualType CurType = getPointeeType();
  if (CurType.isNull() || !CurType->isRecordType())
    return *this;

  const Record *R = Ctx.getRecord(CurType->getAsRecordDecl());
  const Descriptor *BaseDesc = nullptr;

  // This iterates over bases and checks for the proper offset. That's
  // potentially slow but this case really shouldn't happen a lot.
  for (const Record::Base &B : R->bases()) {
    if (B.Offset == BaseOffset) {
      BaseDesc = B.Desc;
      break;
    }
  }
  assert(BaseDesc);

  // Adjust the offset value based on the information from the record layout.
  const ASTContext &ASTCtx = Ctx.getASTContext();
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(R->getDecl());
  CharUnits BaseLayoutOffset =
      Layout.getBaseClassOffset(cast<CXXRecordDecl>(BaseDesc->asDecl()));

  const RecordDecl *RD = BaseDesc->ElemRecord->getDecl();
  QualType T = RD->getASTContext().getTagType(ElaboratedTypeKeyword::None,
                                              std::nullopt, RD, false);
  return {T.getTypePtr(), Value + BaseLayoutOffset.getQuantity()};
}
