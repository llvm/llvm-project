//===----- EvaluationResult.cpp - Result class  for the VM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EvaluationResult.h"
#include "Context.h"
#include "InterpState.h"
#include "Record.h"
#include "clang/AST/ExprCXX.h"

namespace clang {
namespace interp {

APValue EvaluationResult::toAPValue() const {
  assert(!empty());
  switch (Kind) {
  case LValue:
    // Either a pointer or a function pointer.
    if (const auto *P = std::get_if<Pointer>(&Value))
      return P->toAPValue();
    else if (const auto *FP = std::get_if<FunctionPointer>(&Value))
      return FP->toAPValue();
    else
      llvm_unreachable("Unhandled LValue type");
    break;
  case RValue:
    return std::get<APValue>(Value);
  case Valid:
    return APValue();
  default:
    llvm_unreachable("Unhandled result kind?");
  }
}

std::optional<APValue> EvaluationResult::toRValue() const {
  if (Kind == RValue)
    return toAPValue();

  assert(Kind == LValue);

  // We have a pointer and want an RValue.
  if (const auto *P = std::get_if<Pointer>(&Value))
    return P->toRValue(*Ctx);
  else if (const auto *FP = std::get_if<FunctionPointer>(&Value)) // Nope
    return FP->toAPValue();
  llvm_unreachable("Unhandled lvalue kind");
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
  bool Result = true;
  size_t NumElems = CAT->getSize().getZExtValue();
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
    for (size_t I = 0; I != NumElems; ++I) {
      if (!BasePtr.atIndex(I).isInitialized()) {
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

    if (FieldType->isRecordType()) {
      Result &= CheckFieldsInitialized(S, Loc, FieldPtr, FieldPtr.getRecord());
    } else if (FieldType->isIncompleteArrayType()) {
      // Nothing to do here.
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
  for (const Record::Base &B : R->bases()) {
    Pointer P = BasePtr.atField(B.Offset);
    if (!P.isInitialized()) {
      S.FFDiag(BasePtr.getDeclDesc()->asDecl()->getLocation(),
               diag::note_constexpr_uninitialized_base)
          << B.Desc->getType();
      return false;
    }
    Result &= CheckFieldsInitialized(S, Loc, P, B.R);
  }

  // TODO: Virtual bases

  return Result;
}

bool EvaluationResult::checkFullyInitialized(InterpState &S) const {
  assert(Source);
  assert(isLValue());

  // Our Source must be a VarDecl.
  const Decl *SourceDecl = Source.dyn_cast<const Decl *>();
  assert(SourceDecl);
  const auto *VD = cast<VarDecl>(SourceDecl);
  assert(VD->getType()->isRecordType() || VD->getType()->isArrayType());
  SourceLocation InitLoc = VD->getAnyInitializer()->getExprLoc();

  const Pointer &Ptr = *std::get_if<Pointer>(&Value);
  assert(!Ptr.isZero());

  if (const Record *R = Ptr.getRecord())
    return CheckFieldsInitialized(S, InitLoc, Ptr, R);
  const auto *CAT =
      cast<ConstantArrayType>(Ptr.getType()->getAsArrayTypeUnsafe());
  return CheckArrayInitialized(S, InitLoc, Ptr, CAT);

  return true;
}

void EvaluationResult::dump() const {
  assert(Ctx);
  auto &OS = llvm::errs();
  const ASTContext &ASTCtx = Ctx->getASTContext();

  switch (Kind) {
  case Empty:
    OS << "Empty\n";
    break;
  case RValue:
    OS << "RValue: ";
    std::get<APValue>(Value).dump(OS, ASTCtx);
    break;
  case LValue: {
    assert(Source);
    QualType SourceType;
    if (const auto *D = Source.dyn_cast<const Decl *>()) {
      if (const auto *VD = dyn_cast<ValueDecl>(D))
        SourceType = VD->getType();
    } else if (const auto *E = Source.dyn_cast<const Expr *>()) {
      SourceType = E->getType();
    }

    OS << "LValue: ";
    if (const auto *P = std::get_if<Pointer>(&Value))
      P->toAPValue().printPretty(OS, ASTCtx, SourceType);
    else if (const auto *FP = std::get_if<FunctionPointer>(&Value)) // Nope
      FP->toAPValue().printPretty(OS, ASTCtx, SourceType);
    OS << "\n";
    break;
  }
  case Invalid:
    OS << "Invalid\n";
  break;
  case Valid:
    OS << "Valid\n";
  break;
  }
}

} // namespace interp
} // namespace clang
