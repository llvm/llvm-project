//===--- Pointer.cpp - Types for the constexpr VM ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pointer.h"
#include "Boolean.h"
#include "Context.h"
#include "Floating.h"
#include "Function.h"
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
  assert((Base == RootPtrMark || Base % alignof(void *) == 0) && "wrong base");

  BS = {Pointee, Base, nullptr, nullptr};

  if (Pointee)
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
    if (const FunctionDecl *FD = FP.getFunction()->getDecl())
      return APValue(FD, CharUnits::fromQuantity(Offset), {},
                     /*OnePastTheEnd=*/false, /*IsNull=*/false);
    return APValue(FP.getFunction()->getExpr(), CharUnits::fromQuantity(Offset),
                   {},
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

  if (isUnknownSizeArray())
    return APValue(Base, CharUnits::Zero(), Path,
                   /*IsOnePastEnd=*/isOnePastEnd(), /*IsNullPtr=*/false);

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

  bool UsePath = true;
  if (const ValueDecl *VD = getDeclDesc()->asValueDecl();
      VD && VD->getType()->isReferenceType())
    UsePath = false;

  // Build the path into the object.
  bool OnePastEnd = isOnePastEnd() && !isZeroSizeArray();
  Pointer Ptr = *this;
  while (Ptr.isField() || Ptr.isArrayElement()) {

    if (Ptr.isArrayRoot()) {
      // An array root may still be an array element itself.
      if (Ptr.isArrayElement()) {
        Ptr = Ptr.expand();
        const Descriptor *Desc = Ptr.getFieldDesc();
        unsigned Index = Ptr.getIndex();
        QualType ElemType = Desc->getElemQualType();
        Offset += (Index * ASTCtx.getTypeSizeInChars(ElemType));
        if (Ptr.getArray().getType()->isArrayType())
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
      if (Ptr.getArray().getType()->isArrayType())
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

  if (UsePath)
    return APValue(Base, Offset, Path, OnePastEnd);

  return APValue(Base, Offset, APValue::NoLValuePath());
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
    OS << "(Int) {";
    OS << Int.Value << " + " << Offset << ", " << Int.Desc;
    OS << "}";
    break;
  case Storage::Fn:
    OS << "(Fn) { " << asFunctionPointer().getFunction() << " + " << Offset
       << " }";
    break;
  case Storage::Typeid:
    OS << "(Typeid) { " << (const void *)asTypeidPointer().TypePtr << ", "
       << (const void *)asTypeidPointer().TypeInfoType << " + " << Offset
       << "}";
  }
}

size_t Pointer::computeOffsetForComparison() const {
  switch (StorageKind) {
  case Storage::Int:
    return Int.Value + Offset;
  case Storage::Block:
    // See below.
    break;
  case Storage::Fn:
    return Fn.getIntegerRepresentation() + Offset;
  case Storage::Typeid:
    return reinterpret_cast<uintptr_t>(asTypeidPointer().TypePtr) + Offset;
  }

  size_t Result = 0;
  Pointer P = *this;
  while (true) {

    if (P.isVirtualBaseClass()) {
      Result += getInlineDesc()->Offset;
      P = P.getBase();
      continue;
    }

    if (P.isBaseClass()) {
      if (P.getRecord()->getNumVirtualBases() > 0)
        Result += P.getInlineDesc()->Offset;
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
      if (P.isOnePastEnd())
        ++Result;
      break;
    }

    if (const Record *R = P.getBase().getRecord(); R && R->isUnion()) {
      if (P.isOnePastEnd())
        ++Result;
      // Direct child of a union - all have offset 0.
      P = P.getBase();
      continue;
    }

    // Fields, etc.
    Result += P.getInlineDesc()->Offset;
    if (P.isOnePastEnd())
      ++Result;

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

  if (isFunctionPointer())
    return asFunctionPointer().toDiagnosticString(Ctx);

  return toAPValue(Ctx).getAsString(Ctx, getType());
}

bool Pointer::isInitialized() const {
  if (!isBlockPointer())
    return true;

  if (isRoot() && BS.Base == sizeof(GlobalInlineDescriptor) &&
      Offset == BS.Base) {
    const GlobalInlineDescriptor &GD =
        *reinterpret_cast<const GlobalInlineDescriptor *>(block()->rawData());
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

bool Pointer::isElementInitialized(unsigned Index) const {
  if (!isBlockPointer())
    return true;

  const Descriptor *Desc = getFieldDesc();
  assert(Desc);

  if (isStatic() && BS.Base == 0)
    return true;

  if (isRoot() && BS.Base == sizeof(GlobalInlineDescriptor) &&
      Offset == BS.Base) {
    const GlobalInlineDescriptor &GD =
        *reinterpret_cast<const GlobalInlineDescriptor *>(block()->rawData());
    return GD.InitState == GlobalInitState::Initialized;
  }

  if (Desc->isPrimitiveArray()) {
    InitMapPtr &IM = getInitMap();
    if (!IM)
      return false;

    if (IM->first)
      return true;

    return IM->second->isElementInitialized(Index);
  }
  return isInitialized();
}

void Pointer::initialize() const {
  if (!isBlockPointer())
    return;

  assert(BS.Pointee && "Cannot initialize null pointer");

  if (isRoot() && BS.Base == sizeof(GlobalInlineDescriptor) &&
      Offset == BS.Base) {
    GlobalInlineDescriptor &GD = *reinterpret_cast<GlobalInlineDescriptor *>(
        asBlockPointer().Pointee->rawData());
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
  assert(BS.Base != 0 && "Only composite fields can be initialised");
  getInlineDesc()->IsInitialized = true;
}

void Pointer::initializeElement(unsigned Index) const {
  // Primitive global arrays don't have an initmap.
  if (isStatic() && BS.Base == 0)
    return;

  assert(Index < getFieldDesc()->getNumElems());

  InitMapPtr &IM = getInitMap();
  if (!IM) {
    const Descriptor *Desc = getFieldDesc();
    IM = std::make_pair(false, std::make_shared<InitMap>(Desc->getNumElems()));
  }

  assert(IM);

  // All initialized.
  if (IM->first)
    return;

  if (IM->second->initializeElement(Index)) {
    IM->first = true;
    IM->second.reset();
  }
}

void Pointer::initializeAllElements() const {
  assert(getFieldDesc()->isPrimitiveArray());
  assert(isArrayRoot());

  InitMapPtr &IM = getInitMap();
  if (!IM) {
    IM = std::make_pair(true, nullptr);
  } else {
    IM->first = true;
    IM->second.reset();
  }
}

bool Pointer::allElementsInitialized() const {
  assert(getFieldDesc()->isPrimitiveArray());
  assert(isArrayRoot());

  if (isStatic() && BS.Base == 0)
    return true;

  if (isRoot() && BS.Base == sizeof(GlobalInlineDescriptor) &&
      Offset == BS.Base) {
    const GlobalInlineDescriptor &GD =
        *reinterpret_cast<const GlobalInlineDescriptor *>(block()->rawData());
    return GD.InitState == GlobalInitState::Initialized;
  }

  InitMapPtr &IM = getInitMap();
  return IM && IM->first;
}

void Pointer::activate() const {
  // Field has its bit in an inline descriptor.
  assert(BS.Base != 0 && "Only composite fields can be activated");

  if (isRoot() && BS.Base == sizeof(GlobalInlineDescriptor))
    return;
  if (!getInlineDesc()->InUnion)
    return;

  std::function<void(Pointer &)> activate;
  activate = [&activate](Pointer &P) -> void {
    P.getInlineDesc()->IsActive = true;
    if (const Record *R = P.getRecord(); R && !R->isUnion()) {
      for (const Record::Field &F : R->fields()) {
        Pointer FieldPtr = P.atField(F.Offset);
        if (!FieldPtr.getInlineDesc()->IsActive)
          activate(FieldPtr);
      }
      // FIXME: Bases?
    }
  };

  std::function<void(Pointer &)> deactivate;
  deactivate = [&deactivate](Pointer &P) -> void {
    P.getInlineDesc()->IsActive = false;

    if (const Record *R = P.getRecord()) {
      for (const Record::Field &F : R->fields()) {
        Pointer FieldPtr = P.atField(F.Offset);
        if (FieldPtr.getInlineDesc()->IsActive)
          deactivate(FieldPtr);
      }
      // FIXME: Bases?
    }
  };

  Pointer B = *this;
  while (!B.isRoot() && B.inUnion()) {
    activate(B);

    // When walking up the pointer chain, deactivate
    // all union child pointers that aren't on our path.
    Pointer Cur = B;
    B = B.getBase();
    if (const Record *BR = B.getRecord(); BR && BR->isUnion()) {
      for (const Record::Field &F : BR->fields()) {
        Pointer FieldPtr = B.atField(F.Offset);
        if (FieldPtr != Cur)
          deactivate(FieldPtr);
      }
    }
  }
}

void Pointer::deactivate() const {
  // TODO: this only appears in constructors, so nothing to deactivate.
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
    return true;

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

std::optional<std::pair<Pointer, Pointer>>
Pointer::computeSplitPoint(const Pointer &A, const Pointer &B) {
  if (!A.isBlockPointer() || !B.isBlockPointer())
    return std::nullopt;

  if (A.asBlockPointer().Pointee != B.asBlockPointer().Pointee)
    return std::nullopt;
  if (A.isRoot() && B.isRoot())
    return std::nullopt;

  if (A == B)
    return std::make_pair(A, B);

  auto getBase = [](const Pointer &P) -> Pointer {
    if (P.isArrayElement())
      return P.expand().getArray();
    return P.getBase();
  };

  Pointer IterA = A;
  Pointer IterB = B;
  Pointer CurA = IterA;
  Pointer CurB = IterB;
  for (;;) {
    if (IterA.asBlockPointer().Base > IterB.asBlockPointer().Base) {
      CurA = IterA;
      IterA = getBase(IterA);
    } else {
      CurB = IterB;
      IterB = getBase(IterB);
    }

    if (IterA == IterB)
      return std::make_pair(CurA, CurB);

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
  std::function<bool(QualType, const Pointer &, APValue &)> Composite;
  Composite = [&Composite, &Ctx, &ASTCtx](QualType Ty, const Pointer &Ptr,
                                          APValue &R) {
    if (const auto *AT = Ty->getAs<AtomicType>())
      Ty = AT->getValueType();

    // Invalid pointers.
    if (Ptr.isDummy() || !Ptr.isLive() || !Ptr.isBlockPointer() ||
        Ptr.isPastEnd())
      return false;

    // Primitive values.
    if (OptPrimType T = Ctx.classify(Ty)) {
      TYPE_SWITCH(*T, R = Ptr.deref<T>().toAPValue(ASTCtx));
      return true;
    }

    if (const auto *RT = Ty->getAsCanonical<RecordType>()) {
      const auto *Record = Ptr.getRecord();
      assert(Record && "Missing record descriptor");

      bool Ok = true;
      if (RT->getDecl()->isUnion()) {
        const FieldDecl *ActiveField = nullptr;
        APValue Value;
        for (const auto &F : Record->fields()) {
          const Pointer &FP = Ptr.atField(F.Offset);
          QualType FieldTy = F.Decl->getType();
          if (FP.isActive()) {
            if (OptPrimType T = Ctx.classify(FieldTy)) {
              TYPE_SWITCH(*T, Value = FP.deref<T>().toAPValue(ASTCtx));
            } else {
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

        for (unsigned I = 0; I < NF; ++I) {
          const Record::Field *FD = Record->getField(I);
          QualType FieldTy = FD->Decl->getType();
          const Pointer &FP = Ptr.atField(FD->Offset);
          APValue &Value = R.getStructField(I);

          if (OptPrimType T = Ctx.classify(FieldTy)) {
            TYPE_SWITCH(*T, Value = FP.deref<T>().toAPValue(ASTCtx));
          } else {
            Ok &= Composite(FieldTy, FP, Value);
          }
        }

        for (unsigned I = 0; I < NB; ++I) {
          const Record::Base *BD = Record->getBase(I);
          QualType BaseTy = Ctx.getASTContext().getCanonicalTagType(BD->Decl);
          const Pointer &BP = Ptr.atField(BD->Offset);
          Ok &= Composite(BaseTy, BP, R.getStructBase(I));
        }

        for (unsigned I = 0; I < NV; ++I) {
          const Record::Base *VD = Record->getVirtualBase(I);
          QualType VirtBaseTy =
              Ctx.getASTContext().getCanonicalTagType(VD->Decl);
          const Pointer &VP = Ptr.atField(VD->Offset);
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
    if (const auto *CT = Ty->getAs<ComplexType>()) {
      // Can happen via C casts.
      if (!Ptr.getFieldDesc()->isPrimitiveArray())
        return false;

      QualType ElemTy = CT->getElementType();
      if (ElemTy->isIntegerType()) {
        OptPrimType ElemT = Ctx.classify(ElemTy);
        assert(ElemT);
        INT_TYPE_SWITCH(*ElemT, {
          auto V1 = Ptr.elem<T>(0);
          auto V2 = Ptr.elem<T>(1);
          R = APValue(V1.toAPSInt(), V2.toAPSInt());
          return true;
        });
      } else if (ElemTy->isFloatingType()) {
        R = APValue(Ptr.elem<Floating>(0).getAPFloat(),
                    Ptr.elem<Floating>(1).getAPFloat());
        return true;
      }
      return false;
    }

    // Vector types.
    if (const auto *VT = Ty->getAs<VectorType>()) {
      assert(Ptr.getFieldDesc()->isPrimitiveArray());
      QualType ElemTy = VT->getElementType();
      PrimType ElemT = *Ctx.classify(ElemTy);

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

    llvm_unreachable("invalid value to return");
  };

  // Invalid to read from.
  if (isDummy() || !isLive() || isPastEnd())
    return std::nullopt;

  // We can return these as rvalues, but we can't deref() them.
  if (isZero() || isIntegralPointer())
    return toAPValue(ASTCtx);

  // Just load primitive types.
  if (OptPrimType T = Ctx.classify(ResultType)) {
    TYPE_SWITCH(*T, return this->deref<T>().toAPValue(ASTCtx));
  }

  // Return the composite type.
  APValue Result;
  if (!Composite(ResultType, *this, Result))
    return std::nullopt;
  return Result;
}

IntPointer IntPointer::atOffset(const ASTContext &ASTCtx,
                                unsigned Offset) const {
  if (!this->Desc)
    return *this;
  const Record *R = this->Desc->ElemRecord;
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
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(FD->getParent());
  unsigned FieldIndex = FD->getFieldIndex();
  uint64_t FieldOffset =
      ASTCtx.toCharUnitsFromBits(Layout.getFieldOffset(FieldIndex))
          .getQuantity();
  return IntPointer{F->Desc, this->Value + FieldOffset};
}

IntPointer IntPointer::baseCast(const ASTContext &ASTCtx,
                                unsigned BaseOffset) const {
  if (!Desc) {
    assert(Value == 0);
    return *this;
  }
  const Record *R = Desc->ElemRecord;
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
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(R->getDecl());
  CharUnits BaseLayoutOffset =
      Layout.getBaseClassOffset(cast<CXXRecordDecl>(BaseDesc->asDecl()));

  return {BaseDesc, Value + BaseLayoutOffset.getQuantity()};
}
