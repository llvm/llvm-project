//===----- EvaluationResult.cpp - Result class  for the VM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EvaluationResult.h"
#include "InterpState.h"
#include "Pointer.h"
#include "Record.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include <iterator>

namespace clang {
namespace interp {

static void DiagnoseUninitializedSubobject(InterpState &S, SourceLocation Loc,
                                           const FieldDecl *SubObjDecl) {
  assert(SubObjDecl && "Subobject declaration does not exist");
  S.FFDiag(Loc, diag::note_constexpr_uninitialized)
      << /*(name)*/ 1 << SubObjDecl;
  S.Note(SubObjDecl->getLocation(),
         diag::note_constexpr_subobject_declared_here);
}

static bool CheckFieldsInitialized(InterpState &S, SourceLocation Loc,
                                   const Pointer &BasePtr, const Record *R);

static bool CheckArrayInitialized(InterpState &S, SourceLocation Loc,
                                  const Pointer &BasePtr) {
  const Descriptor *BaseDesc = BasePtr.getFieldDesc();
  assert(BaseDesc->isArray());
  size_t NumElems = BaseDesc->getNumElems();

  if (NumElems == 0)
    return true;

  bool Result = true;

  if (BaseDesc->isPrimitiveArray()) {
    if (BasePtr.allElementsInitialized())
      return true;
    DiagnoseUninitializedSubobject(S, Loc, BasePtr.getField());
    return false;
  }
  const Descriptor *ElemDesc = BaseDesc->ElemDesc;

  if (ElemDesc->isRecord()) {
    const Record *R = ElemDesc->ElemRecord;
    for (size_t I = 0; I != NumElems; ++I) {
      Pointer ElemPtr = BasePtr.atIndex(I).narrow();
      Result &= CheckFieldsInitialized(S, Loc, ElemPtr, R);
    }
  } else {
    assert(ElemDesc->isArray());
    for (size_t I = 0; I != NumElems; ++I) {
      Pointer ElemPtr = BasePtr.atIndex(I).narrow();
      Result &= CheckArrayInitialized(S, Loc, ElemPtr);
    }
  }

  return Result;
}

static bool CheckFieldsInitialized(InterpState &S, SourceLocation Loc,
                                   const Pointer &BasePtr, const Record *R) {
  assert(R);
  bool Result = true;
  // Check all fields of this record are initialized.
  for (const Record::Field &F : R->fields()) {
    Pointer FieldPtr = BasePtr.atField(F.Offset);

    // Don't check inactive union members.
    if (R->isUnion() && !FieldPtr.isActive())
      continue;

    QualType FieldType = F.Decl->getType();
    const Descriptor *FieldDesc = FieldPtr.getFieldDesc();

    if (FieldDesc->isRecord()) {
      Result &= CheckFieldsInitialized(S, Loc, FieldPtr, FieldPtr.getRecord());
    } else if (FieldType->isIncompleteArrayType()) {
      // Nothing to do here.
    } else if (F.Decl->isUnnamedBitField()) {
      // Nothing do do here.
    } else if (FieldDesc->isArray()) {
      Result &= CheckArrayInitialized(S, Loc, FieldPtr);
    } else if (!FieldPtr.isInitialized()) {
      DiagnoseUninitializedSubobject(S, Loc, F.Decl);
      Result = false;
    }
  }

  // Check Fields in all bases
  for (auto [I, B] : llvm::enumerate(R->bases())) {
    Pointer P = BasePtr.atField(B.Offset);
    if (!P.isInitialized()) {
      const Descriptor *Desc = BasePtr.getDeclDesc();
      if (const auto *CD = dyn_cast_if_present<CXXRecordDecl>(R->getDecl())) {
        const auto &BS = *std::next(CD->bases_begin(), I);
        SourceLocation TypeBeginLoc = BS.getBaseTypeLoc();
        S.FFDiag(TypeBeginLoc, diag::note_constexpr_uninitialized_base)
            << B.Desc->getType() << SourceRange(TypeBeginLoc, BS.getEndLoc());
      } else {
        S.FFDiag(Desc->getLocation(), diag::note_constexpr_uninitialized_base)
            << B.Desc->getType();
      }
      return false;
    }
    Result &= CheckFieldsInitialized(S, Loc, P, B.R);
  }

  // TODO: Virtual bases

  return Result;
}

bool EvaluationResult::checkFullyInitialized(InterpState &S,
                                             const Pointer &Ptr) const {
  assert(Source);
  assert(empty());

  if (Ptr.isZero())
    return true;
  if (!Ptr.isBlockPointer())
    return true;

  // We can't inspect dead pointers at all. Return true here so we can
  // diagnose them later.
  if (!Ptr.isLive())
    return true;

  SourceLocation InitLoc;
  if (const auto *D = dyn_cast<const Decl *>(Source))
    InitLoc = cast<VarDecl>(D)->getAnyInitializer()->getExprLoc();
  else if (const auto *E = dyn_cast<const Expr *>(Source))
    InitLoc = E->getExprLoc();

  if (const Record *R = Ptr.getRecord())
    return CheckFieldsInitialized(S, InitLoc, Ptr, R);

  if (isa_and_nonnull<ConstantArrayType>(Ptr.getType()->getAsArrayTypeUnsafe()))
    return CheckArrayInitialized(S, InitLoc, Ptr);

  return true;
}

static bool isOrHasPtr(const Descriptor *D) {
  if ((D->isPrimitive() || D->isPrimitiveArray()) && D->getPrimType() == PT_Ptr)
    return true;

  if (D->ElemRecord)
    return D->ElemRecord->hasPtrField();
  return false;
}

static void collectBlocks(const Pointer &Ptr,
                          llvm::SetVector<const Block *> &Blocks) {
  auto isUsefulPtr = [](const Pointer &P) -> bool {
    return P.isLive() && P.isBlockPointer() && !P.isZero() && !P.isDummy() &&
           P.isDereferencable() && !P.isUnknownSizeArray() && !P.isOnePastEnd();
  };

  if (!isUsefulPtr(Ptr))
    return;

  Blocks.insert(Ptr.block());

  const Descriptor *Desc = Ptr.getFieldDesc();
  if (!Desc)
    return;

  if (const Record *R = Desc->ElemRecord; R && R->hasPtrField()) {

    for (const Record::Field &F : R->fields()) {
      if (!isOrHasPtr(F.Desc))
        continue;
      Pointer FieldPtr = Ptr.atField(F.Offset);
      assert(FieldPtr.block() == Ptr.block());
      collectBlocks(FieldPtr, Blocks);
    }
  } else if (Desc->isPrimitive() && Desc->getPrimType() == PT_Ptr) {
    Pointer Pointee = Ptr.deref<Pointer>();
    if (isUsefulPtr(Pointee) && !Blocks.contains(Pointee.block()))
      collectBlocks(Pointee, Blocks);

  } else if (Desc->isPrimitiveArray() && Desc->getPrimType() == PT_Ptr) {
    for (unsigned I = 0; I != Desc->getNumElems(); ++I) {
      Pointer ElemPointee = Ptr.elem<Pointer>(I);
      if (isUsefulPtr(ElemPointee) && !Blocks.contains(ElemPointee.block()))
        collectBlocks(ElemPointee, Blocks);
    }
  } else if (Desc->isCompositeArray() && isOrHasPtr(Desc->ElemDesc)) {
    for (unsigned I = 0; I != Desc->getNumElems(); ++I) {
      Pointer ElemPtr = Ptr.atIndex(I).narrow();
      collectBlocks(ElemPtr, Blocks);
    }
  }
}

bool EvaluationResult::checkReturnValue(InterpState &S, const Context &Ctx,
                                        const Pointer &Ptr,
                                        const SourceInfo &Info) {
  // Collect all blocks that this pointer (transitively) points to and
  // return false if any of them is a dynamic block.
  llvm::SetVector<const Block *> Blocks;

  collectBlocks(Ptr, Blocks);

  for (const Block *B : Blocks) {
    if (B->isDynamic()) {
      assert(B->getDescriptor());
      assert(B->getDescriptor()->asExpr());

      bool IsSubobj = !Ptr.isRoot() || Ptr.isArrayElement();
      S.FFDiag(Info, diag::note_constexpr_dynamic_alloc)
          << Ptr.getType()->isReferenceType() << IsSubobj;
      S.Note(B->getDescriptor()->asExpr()->getExprLoc(),
             diag::note_constexpr_dynamic_alloc_here);
      return false;
    }
  }

  return true;
}

} // namespace interp
} // namespace clang
