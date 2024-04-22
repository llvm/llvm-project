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
#include "PrimType.h"
#include "Record.h"

using namespace clang;
using namespace clang::interp;

Pointer::Pointer(Block *Pointee)
    : Pointer(Pointee, Pointee->getDescriptor()->getMetadataSize(),
              Pointee->getDescriptor()->getMetadataSize()) {}

Pointer::Pointer(Block *Pointee, uint64_t BaseAndOffset)
    : Pointer(Pointee, BaseAndOffset, BaseAndOffset) {}

Pointer::Pointer(const Pointer &P)
    : Offset(P.Offset), PointeeStorage(P.PointeeStorage),
      StorageKind(P.StorageKind) {

  if (isBlockPointer() && PointeeStorage.BS.Pointee)
    PointeeStorage.BS.Pointee->addPointer(this);
}

Pointer::Pointer(Block *Pointee, unsigned Base, uint64_t Offset)
    : Offset(Offset), StorageKind(Storage::Block) {
  assert((Base == RootPtrMark || Base % alignof(void *) == 0) && "wrong base");

  PointeeStorage.BS = {Pointee, Base};

  if (Pointee)
    Pointee->addPointer(this);
}

Pointer::Pointer(Pointer &&P)
    : Offset(P.Offset), PointeeStorage(P.PointeeStorage),
      StorageKind(P.StorageKind) {

  if (StorageKind == Storage::Block && PointeeStorage.BS.Pointee)
    PointeeStorage.BS.Pointee->replacePointer(&P, this);
}

Pointer::~Pointer() {
  if (isIntegralPointer())
    return;

  if (PointeeStorage.BS.Pointee) {
    PointeeStorage.BS.Pointee->removePointer(this);
    PointeeStorage.BS.Pointee->cleanup();
  }
}

void Pointer::operator=(const Pointer &P) {

  if (!this->isIntegralPointer() || !P.isBlockPointer())
    assert(P.StorageKind == StorageKind);

  bool WasBlockPointer = isBlockPointer();
  StorageKind = P.StorageKind;
  if (StorageKind == Storage::Block) {
    Block *Old = PointeeStorage.BS.Pointee;
    if (WasBlockPointer && PointeeStorage.BS.Pointee)
      PointeeStorage.BS.Pointee->removePointer(this);

    Offset = P.Offset;
    PointeeStorage.BS = P.PointeeStorage.BS;

    if (PointeeStorage.BS.Pointee)
      PointeeStorage.BS.Pointee->addPointer(this);

    if (WasBlockPointer && Old)
      Old->cleanup();

  } else if (StorageKind == Storage::Int) {
    PointeeStorage.Int = P.PointeeStorage.Int;
  } else {
    assert(false && "Unhandled storage kind");
  }
}

void Pointer::operator=(Pointer &&P) {
  if (!this->isIntegralPointer() || !P.isBlockPointer())
    assert(P.StorageKind == StorageKind);

  bool WasBlockPointer = isBlockPointer();
  StorageKind = P.StorageKind;
  if (StorageKind == Storage::Block) {
    Block *Old = PointeeStorage.BS.Pointee;
    if (WasBlockPointer && PointeeStorage.BS.Pointee)
      PointeeStorage.BS.Pointee->removePointer(this);

    Offset = P.Offset;
    PointeeStorage.BS = P.PointeeStorage.BS;

    if (PointeeStorage.BS.Pointee)
      PointeeStorage.BS.Pointee->addPointer(this);

    if (WasBlockPointer && Old)
      Old->cleanup();

  } else if (StorageKind == Storage::Int) {
    PointeeStorage.Int = P.PointeeStorage.Int;
  } else {
    assert(false && "Unhandled storage kind");
  }
}

APValue Pointer::toAPValue() const {
  llvm::SmallVector<APValue::LValuePathEntry, 5> Path;

  if (isZero())
    return APValue(static_cast<const Expr *>(nullptr), CharUnits::Zero(), Path,
                   /*IsOnePastEnd=*/false, /*IsNullPtr=*/true);
  if (isIntegralPointer())
    return APValue(static_cast<const Expr *>(nullptr),
                   CharUnits::fromQuantity(asIntPointer().Value + this->Offset),
                   Path,
                   /*IsOnePastEnd=*/false, /*IsNullPtr=*/false);

  // Build the lvalue base from the block.
  const Descriptor *Desc = getDeclDesc();
  APValue::LValueBase Base;
  if (const auto *VD = Desc->asValueDecl())
    Base = VD;
  else if (const auto *E = Desc->asExpr())
    Base = E;
  else
    llvm_unreachable("Invalid allocation type");

  if (isDummy() || isUnknownSizeArray() || Desc->asExpr())
    return APValue(Base, CharUnits::Zero(), Path,
                   /*IsOnePastEnd=*/false, /*IsNullPtr=*/false);

  // TODO: compute the offset into the object.
  CharUnits Offset = CharUnits::Zero();
  bool IsOnePastEnd = isOnePastEnd();

  // Build the path into the object.
  Pointer Ptr = *this;
  while (Ptr.isField() || Ptr.isArrayElement()) {
    if (Ptr.isArrayElement()) {
      Path.push_back(APValue::LValuePathEntry::ArrayIndex(Ptr.getIndex()));
      Ptr = Ptr.getArray();
    } else {
      // TODO: figure out if base is virtual
      bool IsVirtual = false;

      // Create a path entry for the field.
      const Descriptor *Desc = Ptr.getFieldDesc();
      if (const auto *BaseOrMember = Desc->asDecl()) {
        Path.push_back(APValue::LValuePathEntry({BaseOrMember, IsVirtual}));
        Ptr = Ptr.getBase();
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

  return APValue(Base, Offset, Path, IsOnePastEnd, /*IsNullPtr=*/false);
}

void Pointer::print(llvm::raw_ostream &OS) const {
  OS << PointeeStorage.BS.Pointee << " (";
  if (isBlockPointer()) {
    OS << "Block) {";

    if (PointeeStorage.BS.Base == RootPtrMark)
      OS << "rootptr, ";
    else
      OS << PointeeStorage.BS.Base << ", ";

    if (Offset == PastEndMark)
      OS << "pastend, ";
    else
      OS << Offset << ", ";

    if (isBlockPointer() && PointeeStorage.BS.Pointee)
      OS << PointeeStorage.BS.Pointee->getSize();
    else
      OS << "nullptr";
  } else {
    OS << "Int) {";
    OS << PointeeStorage.Int.Value << ", " << PointeeStorage.Int.Desc;
  }
  OS << "}";
}

std::string Pointer::toDiagnosticString(const ASTContext &Ctx) const {
  if (isZero())
    return "nullptr";

  if (isIntegralPointer())
    return (Twine("&(") + Twine(asIntPointer().Value + Offset) + ")").str();

  return toAPValue().getAsString(Ctx, getType());
}

bool Pointer::isInitialized() const {
  if (isIntegralPointer())
    return true;

  assert(PointeeStorage.BS.Pointee &&
         "Cannot check if null pointer was initialized");
  const Descriptor *Desc = getFieldDesc();
  assert(Desc);
  if (Desc->isPrimitiveArray()) {
    if (isStatic() && PointeeStorage.BS.Base == 0)
      return true;

    InitMapPtr &IM = getInitMap();

    if (!IM)
      return false;

    if (IM->first)
      return true;

    return IM->second->isElementInitialized(getIndex());
  }

  // Field has its bit in an inline descriptor.
  return PointeeStorage.BS.Base == 0 || getInlineDesc()->IsInitialized;
}

void Pointer::initialize() const {
  if (isIntegralPointer())
    return;

  assert(PointeeStorage.BS.Pointee && "Cannot initialize null pointer");
  const Descriptor *Desc = getFieldDesc();

  assert(Desc);
  if (Desc->isPrimitiveArray()) {
    // Primitive global arrays don't have an initmap.
    if (isStatic() && PointeeStorage.BS.Base == 0)
      return;

    // Nothing to do for these.
    if (Desc->getNumElems() == 0)
      return;

    InitMapPtr &IM = getInitMap();
    if (!IM)
      IM =
          std::make_pair(false, std::make_shared<InitMap>(Desc->getNumElems()));

    assert(IM);

    // All initialized.
    if (IM->first)
      return;

    if (IM->second->initializeElement(getIndex())) {
      IM->first = true;
      IM->second.reset();
    }
    return;
  }

  // Field has its bit in an inline descriptor.
  assert(PointeeStorage.BS.Base != 0 &&
         "Only composite fields can be initialised");
  getInlineDesc()->IsInitialized = true;
}

void Pointer::activate() const {
  // Field has its bit in an inline descriptor.
  assert(PointeeStorage.BS.Base != 0 &&
         "Only composite fields can be initialised");
  getInlineDesc()->IsActive = true;
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

  if (A.isIntegralPointer() || B.isIntegralPointer())
    return A.getSource() == B.getSource();

  return A.asBlockPointer().Pointee == B.asBlockPointer().Pointee;
}

bool Pointer::hasSameArray(const Pointer &A, const Pointer &B) {
  return hasSameBase(A, B) &&
         A.PointeeStorage.BS.Base == B.PointeeStorage.BS.Base &&
         A.getFieldDesc()->IsArray;
}

std::optional<APValue> Pointer::toRValue(const Context &Ctx) const {
  // Method to recursively traverse composites.
  std::function<bool(QualType, const Pointer &, APValue &)> Composite;
  Composite = [&Composite, &Ctx](QualType Ty, const Pointer &Ptr, APValue &R) {
    if (const auto *AT = Ty->getAs<AtomicType>())
      Ty = AT->getValueType();

    // Invalid pointers.
    if (Ptr.isDummy() || !Ptr.isLive() ||
        (!Ptr.isUnknownSizeArray() && Ptr.isOnePastEnd()))
      return false;

    // Primitive values.
    if (std::optional<PrimType> T = Ctx.classify(Ty)) {
      TYPE_SWITCH(*T, R = Ptr.deref<T>().toAPValue());
      return true;
    }

    if (const auto *RT = Ty->getAs<RecordType>()) {
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
            if (std::optional<PrimType> T = Ctx.classify(FieldTy)) {
              TYPE_SWITCH(*T, Value = FP.deref<T>().toAPValue());
            } else {
              Ok &= Composite(FieldTy, FP, Value);
            }
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

          if (std::optional<PrimType> T = Ctx.classify(FieldTy)) {
            TYPE_SWITCH(*T, Value = FP.deref<T>().toAPValue());
          } else {
            Ok &= Composite(FieldTy, FP, Value);
          }
        }

        for (unsigned I = 0; I < NB; ++I) {
          const Record::Base *BD = Record->getBase(I);
          QualType BaseTy = Ctx.getASTContext().getRecordType(BD->Decl);
          const Pointer &BP = Ptr.atField(BD->Offset);
          Ok &= Composite(BaseTy, BP, R.getStructBase(I));
        }

        for (unsigned I = 0; I < NV; ++I) {
          const Record::Base *VD = Record->getVirtualBase(I);
          QualType VirtBaseTy = Ctx.getASTContext().getRecordType(VD->Decl);
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
      for (unsigned I = 0; I < NumElems; ++I) {
        APValue &Slot = R.getArrayInitializedElt(I);
        const Pointer &EP = Ptr.atIndex(I);
        if (std::optional<PrimType> T = Ctx.classify(ElemTy)) {
          TYPE_SWITCH(*T, Slot = EP.deref<T>().toAPValue());
        } else {
          Ok &= Composite(ElemTy, EP.narrow(), Slot);
        }
      }
      return Ok;
    }

    // Complex types.
    if (const auto *CT = Ty->getAs<ComplexType>()) {
      QualType ElemTy = CT->getElementType();

      if (ElemTy->isIntegerType()) {
        std::optional<PrimType> ElemT = Ctx.classify(ElemTy);
        assert(ElemT);
        INT_TYPE_SWITCH(*ElemT, {
          auto V1 = Ptr.atIndex(0).deref<T>();
          auto V2 = Ptr.atIndex(1).deref<T>();
          R = APValue(V1.toAPSInt(), V2.toAPSInt());
          return true;
        });
      } else if (ElemTy->isFloatingType()) {
        R = APValue(Ptr.atIndex(0).deref<Floating>().getAPFloat(),
                    Ptr.atIndex(1).deref<Floating>().getAPFloat());
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
        TYPE_SWITCH(ElemT, {
          Values.push_back(Ptr.atIndex(I).deref<T>().toAPValue());
        });
      }

      assert(Values.size() == VT->getNumElements());
      R = APValue(Values.data(), Values.size());
      return true;
    }

    llvm_unreachable("invalid value to return");
  };

  if (isZero())
    return APValue(static_cast<Expr *>(nullptr), CharUnits::Zero(), {}, false,
                   true);

  if (isDummy() || !isLive())
    return std::nullopt;

  // Return the composite type.
  APValue Result;
  if (!Composite(getType(), *this, Result))
    return std::nullopt;
  return Result;
}
