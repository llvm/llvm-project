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

Pointer::Pointer(Block *Pointee) : Pointer(Pointee, 0, 0) {}

Pointer::Pointer(Block *Pointee, unsigned BaseAndOffset)
    : Pointer(Pointee, BaseAndOffset, BaseAndOffset) {}

Pointer::Pointer(const Pointer &P) : Pointer(P.Pointee, P.Base, P.Offset) {}

Pointer::Pointer(Pointer &&P)
    : Pointee(P.Pointee), Base(P.Base), Offset(P.Offset) {
  if (Pointee)
    Pointee->replacePointer(&P, this);
}

Pointer::Pointer(Block *Pointee, unsigned Base, unsigned Offset)
    : Pointee(Pointee), Base(Base), Offset(Offset) {
  assert((Base == RootPtrMark || Base % alignof(void *) == 0) && "wrong base");
  if (Pointee)
    Pointee->addPointer(this);
}

Pointer::~Pointer() {
  if (Pointee) {
    Pointee->removePointer(this);
    Pointee->cleanup();
  }
}

void Pointer::operator=(const Pointer &P) {
  Block *Old = Pointee;

  if (Pointee)
    Pointee->removePointer(this);

  Offset = P.Offset;
  Base = P.Base;

  Pointee = P.Pointee;
  if (Pointee)
    Pointee->addPointer(this);

  if (Old)
    Old->cleanup();
}

void Pointer::operator=(Pointer &&P) {
  Block *Old = Pointee;

  if (Pointee)
    Pointee->removePointer(this);

  Offset = P.Offset;
  Base = P.Base;

  Pointee = P.Pointee;
  if (Pointee)
    Pointee->replacePointer(&P, this);

  if (Old)
    Old->cleanup();
}

APValue Pointer::toAPValue() const {
  APValue::LValueBase Base;
  llvm::SmallVector<APValue::LValuePathEntry, 5> Path;
  CharUnits Offset;
  bool IsNullPtr;
  bool IsOnePastEnd;

  if (isZero()) {
    Base = static_cast<const Expr *>(nullptr);
    IsNullPtr = true;
    IsOnePastEnd = false;
    Offset = CharUnits::Zero();
  } else {
    // Build the lvalue base from the block.
    Descriptor *Desc = getDeclDesc();
    if (auto *VD = Desc->asValueDecl())
      Base = VD;
    else if (auto *E = Desc->asExpr())
      Base = E;
    else
      llvm_unreachable("Invalid allocation type");

    // Not a null pointer.
    IsNullPtr = false;

    if (isUnknownSizeArray()) {
      IsOnePastEnd = false;
      Offset = CharUnits::Zero();
    } else if (Desc->asExpr()) {
      // Pointer pointing to a an expression.
      IsOnePastEnd = false;
      Offset = CharUnits::Zero();
    } else {
      // TODO: compute the offset into the object.
      Offset = CharUnits::Zero();

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

      IsOnePastEnd = isOnePastEnd();
    }
  }

  // We assemble the LValuePath starting from the innermost pointer to the
  // outermost one. SO in a.b.c, the first element in Path will refer to
  // the field 'c', while later code expects it to refer to 'a'.
  // Just invert the order of the elements.
  std::reverse(Path.begin(), Path.end());

  return APValue(Base, Offset, Path, IsOnePastEnd, IsNullPtr);
}

std::string Pointer::toDiagnosticString(const ASTContext &Ctx) const {
  if (!Pointee)
    return "nullptr";

  return toAPValue().getAsString(Ctx, getType());
}

bool Pointer::isInitialized() const {
  assert(Pointee && "Cannot check if null pointer was initialized");
  const Descriptor *Desc = getFieldDesc();
  assert(Desc);
  if (Desc->isPrimitiveArray()) {
    if (isStatic() && Base == 0)
      return true;
    // Primitive array field are stored in a bitset.
    InitMap *Map = getInitMap();
    if (!Map)
      return false;
    if (Map == (InitMap *)-1)
      return true;
    return Map->isInitialized(getIndex());
  }

  // Field has its bit in an inline descriptor.
  return Base == 0 || getInlineDesc()->IsInitialized;
}

void Pointer::initialize() const {
  assert(Pointee && "Cannot initialize null pointer");
  const Descriptor *Desc = getFieldDesc();

  assert(Desc);
  if (Desc->isPrimitiveArray()) {
    // Primitive global arrays don't have an initmap.
    if (isStatic() && Base == 0)
      return;

    // Primitive array initializer.
    InitMap *&Map = getInitMap();
    if (Map == (InitMap *)-1)
      return;
    if (Map == nullptr)
      Map = InitMap::allocate(Desc->getNumElems());
    if (Map->initialize(getIndex())) {
      free(Map);
      Map = (InitMap *)-1;
    }
    return;
  }

  // Field has its bit in an inline descriptor.
  assert(Base != 0 && "Only composite fields can be initialised");
  getInlineDesc()->IsInitialized = true;
}

void Pointer::activate() const {
  // Field has its bit in an inline descriptor.
  assert(Base != 0 && "Only composite fields can be initialised");
  getInlineDesc()->IsActive = true;
}

void Pointer::deactivate() const {
  // TODO: this only appears in constructors, so nothing to deactivate.
}

bool Pointer::hasSameBase(const Pointer &A, const Pointer &B) {
  return A.Pointee == B.Pointee;
}

bool Pointer::hasSameArray(const Pointer &A, const Pointer &B) {
  return hasSameBase(A, B) && A.Base == B.Base && A.getFieldDesc()->IsArray;
}

APValue Pointer::toRValue(const Context &Ctx) const {
  // Primitives.
  if (getFieldDesc()->isPrimitive()) {
    PrimType PT = *Ctx.classify(getType());
    TYPE_SWITCH(PT, return deref<T>().toAPValue());
    llvm_unreachable("Unhandled PrimType?");
  }

  APValue Result;
  // Records.
  if (getFieldDesc()->isRecord()) {
    const Record *R = getRecord();
    Result =
        APValue(APValue::UninitStruct(), R->getNumBases(), R->getNumFields());

    for (unsigned I = 0; I != R->getNumFields(); ++I) {
      const Pointer &FieldPtr = this->atField(R->getField(I)->Offset);
      Result.getStructField(I) = FieldPtr.toRValue(Ctx);
    }

    for (unsigned I = 0; I != R->getNumBases(); ++I) {
      const Pointer &BasePtr = this->atField(R->getBase(I)->Offset);
      Result.getStructBase(I) = BasePtr.toRValue(Ctx);
    }
  }

  // TODO: Arrays

  return Result;
}
