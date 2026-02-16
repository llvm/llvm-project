//===----- EvaluationResult.cpp - Result class  for the VM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EvaluationResult.h"
#include "../ExprConstShared.h"
#include "InterpState.h"
#include "Pointer.h"
#include "Record.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <iterator>

namespace clang {
namespace interp {

QualType EvaluationResult::getStorageType() const {
  if (const auto *E = Source.dyn_cast<const Expr *>()) {
    if (E->isPRValue())
      return E->getType();

    return Ctx.getASTContext().getLValueReferenceType(E->getType());
  }

  if (const auto *D =
          dyn_cast_if_present<ValueDecl>(Source.dyn_cast<const Decl *>()))
    return D->getType();
  return QualType();
}

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
                                  const Pointer &BasePtr,
                                  const ConstantArrayType *CAT) {
  size_t NumElems = CAT->getZExtSize();

  if (NumElems == 0)
    return true;

  bool Result = true;
  QualType ElemType = CAT->getElementType();

  if (ElemType->isRecordType()) {
    const Record *R = BasePtr.getElemRecord();
    for (size_t I = 0; I != NumElems; ++I) {
      Pointer ElemPtr = BasePtr.atIndex(I).narrow();
      Result &= CheckFieldsInitialized(S, Loc, ElemPtr, R);
    }
  } else if (const auto *ElemCAT = dyn_cast<ConstantArrayType>(ElemType)) {
    for (size_t I = 0; I != NumElems; ++I) {
      Pointer ElemPtr = BasePtr.atIndex(I).narrow();
      Result &= CheckArrayInitialized(S, Loc, ElemPtr, ElemCAT);
    }
  } else {
    // Primitive arrays.
    if (S.getContext().canClassify(ElemType)) {
      if (BasePtr.allElementsInitialized()) {
        return true;
      } else {
        DiagnoseUninitializedSubobject(S, Loc, BasePtr.getField());
        return false;
      }
    }

    for (size_t I = 0; I != NumElems; ++I) {
      if (!BasePtr.isElementInitialized(I)) {
        DiagnoseUninitializedSubobject(S, Loc, BasePtr.getField());
        Result = false;
      }
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
    QualType FieldType = F.Decl->getType();

    // Don't check inactive union members.
    if (R->isUnion() && !FieldPtr.isActive())
      continue;

    if (FieldType->isRecordType()) {
      Result &= CheckFieldsInitialized(S, Loc, FieldPtr, FieldPtr.getRecord());
    } else if (FieldType->isIncompleteArrayType()) {
      // Nothing to do here.
    } else if (F.Decl->isUnnamedBitField()) {
      // Nothing do do here.
    } else if (FieldType->isArrayType()) {
      const auto *CAT =
          cast<ConstantArrayType>(FieldType->getAsArrayTypeUnsafe());
      Result &= CheckArrayInitialized(S, Loc, FieldPtr, CAT);
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

  if (const auto *CAT = dyn_cast_if_present<ConstantArrayType>(
          Ptr.getType()->getAsArrayTypeUnsafe()))
    return CheckArrayInitialized(S, InitLoc, Ptr, CAT);

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

bool EvaluationResult::checkDynamicAllocations(InterpState &S,
                                               const Context &Ctx,
                                               const Pointer &Ptr,
                                               SourceInfo Info) {
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

static bool isGlobalLValue(const Pointer &Ptr) {
  if (Ptr.isBlockPointer() && Ptr.block()->isDynamic())
    return true;
  if (Ptr.isTypeidPointer())
    return true;

  const Descriptor *Desc = Ptr.getDeclDesc();
  return ::isGlobalLValue(Desc->asValueDecl(), Desc->asExpr());
}

/// Check if the given function pointer can be returned from an evaluation.
static bool checkFunctionPtr(InterpState &S, const Pointer &Ptr,
                             QualType PtrType, SourceInfo Info,
                             ConstantExprKind ConstexprKind) {
  assert(Ptr.isFunctionPointer());
  const FunctionPointer &FuncPtr = Ptr.asFunctionPointer();
  const FunctionDecl *FD = FuncPtr.getFunction()->getDecl();
  // E.g. ObjC block pointers.
  if (!FD)
    return true;
  if (FD->isImmediateFunction()) {
    S.FFDiag(Info, diag::note_consteval_address_accessible)
        << !PtrType->isAnyPointerType();
    S.Note(FD->getLocation(), diag::note_declared_at);
    return false;
  }

  // __declspec(dllimport) must be handled very carefully:
  // We must never initialize an expression with the thunk in C++.
  // Doing otherwise would allow the same id-expression to yield
  // different addresses for the same function in different translation
  // units.  However, this means that we must dynamically initialize the
  // expression with the contents of the import address table at runtime.
  //
  // The C language has no notion of ODR; furthermore, it has no notion of
  // dynamic initialization.  This means that we are permitted to
  // perform initialization with the address of the thunk.
  if (S.getLangOpts().CPlusPlus && !isForManglingOnly(ConstexprKind) &&
      FD->hasAttr<DLLImportAttr>())
    // FIXME: Diagnostic!
    return false;
  return true;
}

static bool lvalFields(InterpState &S, const ASTContext &Ctx,
                       const Pointer &Ptr, QualType PtrType, SourceInfo Info,
                       ConstantExprKind ConstexprKind,
                       llvm::SmallPtrSet<const Block *, 4> &CheckedBlocks);
static bool lval(InterpState &S, const ASTContext &Ctx, const Pointer &Ptr,
                 QualType PtrType, SourceInfo Info,
                 ConstantExprKind ConstexprKind,
                 llvm::SmallPtrSet<const Block *, 4> &CheckedBlocks) {
  if (Ptr.isFunctionPointer())
    return checkFunctionPtr(S, Ptr, PtrType, Info, ConstexprKind);

  if (!Ptr.isBlockPointer())
    return true;

  const Descriptor *DeclDesc = Ptr.block()->getDescriptor();
  const Expr *BaseE = DeclDesc->asExpr();
  const ValueDecl *BaseVD = DeclDesc->asValueDecl();
  assert(BaseE || BaseVD);
  bool IsReferenceType = PtrType->isReferenceType();
  bool IsSubObj = !Ptr.isRoot() || (Ptr.inArray() && !Ptr.isArrayRoot());

  if (!isGlobalLValue(Ptr)) {
    if (S.getLangOpts().CPlusPlus11) {
      S.FFDiag(Info, diag::note_constexpr_non_global, 1)
          << IsReferenceType << IsSubObj
          << !!DeclDesc->asValueDecl() // DeclDesc->IsTemporary
          << DeclDesc->asValueDecl();
      const VarDecl *VarD = DeclDesc->asVarDecl();
      if (VarD && VarD->isConstexpr()) {
        // Non-static local constexpr variables have unintuitive semantics:
        //   constexpr int a = 1;
        //   constexpr const int *p = &a;
        // ... is invalid because the address of 'a' is not constant. Suggest
        // adding a 'static' in this case.
        S.Note(VarD->getLocation(), diag::note_constexpr_not_static)
            << VarD
            << FixItHint::CreateInsertion(VarD->getBeginLoc(), "static ");
      } else {
        if (const ValueDecl *VD = DeclDesc->asValueDecl())
          S.Note(VD->getLocation(), diag::note_declared_at);
        else if (const Expr *E = DeclDesc->asExpr())
          S.Note(E->getExprLoc(), diag::note_constexpr_temporary_here);
      }
    } else {
      S.FFDiag(Info);
    }
    return false;
  }

  if (const auto *VD = dyn_cast_if_present<VarDecl>(BaseVD)) {
    // Check if this is a thread-local variable.
    if (VD->getTLSKind()) {
      // FIXME: Diagnostic!
      return false;
    }

    // A dllimport variable never acts like a constant, unless we're
    // evaluating a value for use only in name mangling, and unless it's a
    // static local. For the latter case, we'd still need to evaluate the
    // constant expression in case we're inside a (inlined) function.
    if (!isForManglingOnly(ConstexprKind) && VD->hasAttr<DLLImportAttr>() &&
        !VD->isStaticLocal())
      return false;

    // In CUDA/HIP device compilation, only device side variables have
    // constant addresses.
    if (S.getLangOpts().CUDA && S.getLangOpts().CUDAIsDevice &&
        Ctx.CUDAConstantEvalCtx.NoWrongSidedVars) {
      if ((!VD->hasAttr<CUDADeviceAttr>() && !VD->hasAttr<CUDAConstantAttr>() &&
           !VD->getType()->isCUDADeviceBuiltinSurfaceType() &&
           !VD->getType()->isCUDADeviceBuiltinTextureType()) ||
          VD->hasAttr<HIPManagedAttr>())
        return false;
    }

    return true;
  }

  if (const auto *MTE = dyn_cast_if_present<MaterializeTemporaryExpr>(BaseE)) {
    QualType TempType = Ptr.getType();

    if (TempType.isDestructedType()) {
      S.FFDiag(MTE->getExprLoc(),
               diag::note_constexpr_unsupported_temporary_nontrivial_dtor)
          << TempType;
      return false;
    }

    if (Ptr.getFieldDesc()->isPrimitive() &&
        Ptr.getFieldDesc()->getPrimType() == PT_Ptr) {
      // Recurse!
      Pointer Pointee = Ptr.deref<Pointer>();
      if (CheckedBlocks.insert(Pointee.block()).second) {
        if (!lval(S, Ctx, Pointee, Pointee.getType(),
                  Ptr.getDeclDesc()->getLoc(), ConstexprKind, CheckedBlocks))
          return false;
      }
    } else if (Ptr.getRecord()) {
      return lvalFields(S, Ctx, Ptr, Ptr.getType(), Info,
                        ConstantExprKind::Normal, CheckedBlocks);
    }
  }

  return true;
}

static bool lvalFields(InterpState &S, const ASTContext &Ctx,
                       const Pointer &Ptr, QualType PtrType, SourceInfo Info,
                       ConstantExprKind ConstexprKind,
                       llvm::SmallPtrSet<const Block *, 4> &CheckedBlocks) {
  if (!Ptr.isBlockPointer())
    return true;

  const Descriptor *FieldDesc = Ptr.getFieldDesc();
  if (const Record *R = Ptr.getRecord()) {
    if (!R->hasPtrField())
      return true;
    for (const Record::Field &F : R->fields()) {
      if (F.Desc->isPrimitive() && F.Desc->getPrimType() == PT_Ptr) {
        QualType FieldType = F.Decl->getType();
        if (!Ptr.atField(F.Offset).isLive())
          return false;

        Pointer Pointee = Ptr.atField(F.Offset).deref<Pointer>();
        if (CheckedBlocks.insert(Pointee.block()).second) {
          if (!lval(S, Ctx, Pointee, FieldType, Info, ConstexprKind,
                    CheckedBlocks))
            return false;
        }
      } else {
        Pointer FieldPtr = Ptr.atField(F.Offset);
        if (!lvalFields(S, Ctx, FieldPtr, F.Decl->getType(), Info,
                        ConstexprKind, CheckedBlocks))
          return false;
      }
    }

    for (const Record::Base &B : R->bases()) {
      Pointer BasePtr = Ptr.atField(B.Offset);
      if (!lvalFields(S, Ctx, BasePtr, B.Desc->getType(), Info, ConstexprKind,
                      CheckedBlocks))
        return false;
    }
    for (const Record::Base &B : R->virtual_bases()) {
      Pointer BasePtr = Ptr.atField(B.Offset);
      if (!lvalFields(S, Ctx, BasePtr, B.Desc->getType(), Info, ConstexprKind,
                      CheckedBlocks))
        return false;
    }

    return true;
  }
  if (FieldDesc->isPrimitiveArray()) {
    if (FieldDesc->getPrimType() == PT_Ptr) {
      for (unsigned I = 0; I != FieldDesc->getNumElems(); ++I) {
        if (!Ptr.isLive())
          return false;
        Pointer Pointee = Ptr.elem<Pointer>(I);
        if (CheckedBlocks.insert(Pointee.block()).second) {
          if (!lval(S, Ctx, Pointee, FieldDesc->getElemQualType(), Info,
                    ConstexprKind, CheckedBlocks))
            return false;
        }
      }
    }
    return true;
  }
  if (FieldDesc->isCompositeArray()) {
    if (FieldDesc->ElemRecord && !FieldDesc->ElemRecord->hasPtrField())
      return true;

    for (unsigned I = 0; I != FieldDesc->getNumElems(); ++I) {
      Pointer Elem = Ptr.atIndex(I).narrow();
      if (!lvalFields(S, Ctx, Elem, FieldDesc->getElemQualType(), Info,
                      ConstexprKind, CheckedBlocks))
        return false;
    }
    return true;
  }
  if (FieldDesc->isPrimitive() && FieldDesc->getPrimType() == PT_MemberPtr) {
    MemberPointer MP = Ptr.deref<MemberPointer>();
    if (!EvaluationResult::checkMemberPointer(S, MP, Info, ConstexprKind))
      return false;
  }

  return true;
}

/// Toplevel accessor to check all lvalue fields.
bool EvaluationResult::checkLValueFields(InterpState &S, const Pointer &Ptr,
                                         SourceInfo Info,
                                         ConstantExprKind ConstexprKind) {
  QualType SourceType = getStorageType();
  llvm::SmallPtrSet<const Block *, 4> CheckedBlocks;

  return lvalFields(S, Ctx.getASTContext(), Ptr, SourceType, Info,
                    ConstexprKind, CheckedBlocks);
}

bool EvaluationResult::checkLValue(InterpState &S, const Pointer &Ptr,
                                   SourceInfo Info,
                                   ConstantExprKind ConstexprKind) {
  if (Ptr.isZero())
    return true;

  QualType SourceType = getStorageType();
  if (Ptr.isFunctionPointer())
    return checkFunctionPtr(S, Ptr, SourceType, Info, ConstexprKind);

  bool IsReferenceType = SourceType->isReferenceType();
  if (Ptr.isTypeidPointer()) {
    if (isTemplateArgument(ConstexprKind)) {
      S.FFDiag(Info, diag::note_constexpr_invalid_template_arg)
          << IsReferenceType << /*IsSubObj=*/false << /*InvalidBaseKind=*/0;
      return false;
    }
    return true;
  }

  if (!Ptr.isBlockPointer())
    return true;

  const Descriptor *DeclDesc = Ptr.getDeclDesc();
  const Expr *BaseE = DeclDesc->asExpr();
  const ValueDecl *BaseVD = DeclDesc->asValueDecl();
  assert(BaseE || BaseVD);
  bool IsSubObj = !Ptr.isRoot() || (Ptr.inArray() && !Ptr.isArrayRoot());

  // Additional restrictions apply in a template argument. We only enforce the
  // C++20 restrictions here; additional syntactic and semantic restrictions
  // are applied elsewhere.
  if (isTemplateArgument(ConstexprKind)) {
    int InvalidBaseKind = -1;
    StringRef Ident;
    if (isa_and_nonnull<StringLiteral>(BaseE))
      InvalidBaseKind = 1;
    else if (isa_and_nonnull<MaterializeTemporaryExpr>(BaseE) ||
             isa_and_nonnull<LifetimeExtendedTemporaryDecl>(BaseVD))
      InvalidBaseKind = 2;
    else if (const auto *PE = dyn_cast_if_present<PredefinedExpr>(BaseE)) {
      InvalidBaseKind = 3;
      Ident = PE->getIdentKindName();
      IsSubObj = true;
    }

    if (InvalidBaseKind != -1) {
      S.FFDiag(Info, diag::note_constexpr_invalid_template_arg)
          << IsReferenceType << IsSubObj << InvalidBaseKind << Ident;
      return false;
    }
  }

  llvm::SmallPtrSet<const Block *, 4> CheckedBlocks;
  if (!lval(S, Ctx.getASTContext(), Ptr, SourceType, Info, ConstexprKind,
            CheckedBlocks)) {
    return false;
  }

  return true;
}

bool EvaluationResult::checkMemberPointer(InterpState &S,
                                          const MemberPointer &MemberPtr,
                                          SourceInfo Info,
                                          ConstantExprKind ConstexprKind) {
  const CXXMethodDecl *MD = MemberPtr.getMemberFunction();
  if (!MD)
    return true;

  if (MD->isImmediateFunction()) {
    S.FFDiag(Info, diag::note_consteval_address_accessible)
        << /*pointer=*/false;
    S.Note(MD->getLocation(), diag::note_declared_at);
    return false;
  }

  if (isForManglingOnly(ConstexprKind) || MD->isVirtual() ||
      !MD->hasAttr<DLLImportAttr>()) {
    return true;
  }
  return false;
}

bool EvaluationResult::checkFunctionPointer(InterpState &S, const Pointer &Ptr,
                                            SourceInfo Info,
                                            ConstantExprKind ConstexprKind) {
  return checkFunctionPtr(S, Ptr, getStorageType(), Info, ConstexprKind);
}

} // namespace interp
} // namespace clang
