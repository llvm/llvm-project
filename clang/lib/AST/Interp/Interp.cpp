//===------- Interp.cpp - Interpreter for the constexpr VM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interp.h"
#include <limits>
#include <vector>
#include "Function.h"
#include "InterpFrame.h"
#include "InterpStack.h"
#include "Opcode.h"
#include "PrimType.h"
#include "Program.h"
#include "State.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/ADT/APSInt.h"

using namespace clang;
using namespace clang::interp;

static bool RetValue(InterpState &S, CodePtr &Pt, APValue &Result) {
  llvm::report_fatal_error("Interpreter cannot return values");
}

//===----------------------------------------------------------------------===//
// Jmp, Jt, Jf
//===----------------------------------------------------------------------===//

static bool Jmp(InterpState &S, CodePtr &PC, int32_t Offset) {
  PC += Offset;
  return true;
}

static bool Jt(InterpState &S, CodePtr &PC, int32_t Offset) {
  if (S.Stk.pop<bool>()) {
    PC += Offset;
  }
  return true;
}

static bool Jf(InterpState &S, CodePtr &PC, int32_t Offset) {
  if (!S.Stk.pop<bool>()) {
    PC += Offset;
  }
  return true;
}

static bool CheckActive(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                        AccessKinds AK) {
  if (Ptr.isActive())
    return true;

  // Get the inactive field descriptor.
  const FieldDecl *InactiveField = Ptr.getField();

  // Walk up the pointer chain to find the union which is not active.
  Pointer U = Ptr.getBase();
  while (!U.isActive()) {
    U = U.getBase();
  }

  // Find the active field of the union.
  const Record *R = U.getRecord();
  assert(R && R->isUnion() && "Not a union");
  const FieldDecl *ActiveField = nullptr;
  for (unsigned I = 0, N = R->getNumFields(); I < N; ++I) {
    const Pointer &Field = U.atField(R->getField(I)->Offset);
    if (Field.isActive()) {
      ActiveField = Field.getField();
      break;
    }
  }

  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_access_inactive_union_member)
      << AK << InactiveField << !ActiveField << ActiveField;
  return false;
}

static bool CheckTemporary(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                           AccessKinds AK) {
  if (auto ID = Ptr.getDeclID()) {
    if (!Ptr.isStaticTemporary())
      return true;

    if (Ptr.getDeclDesc()->getType().isConstQualified())
      return true;

    if (S.P.getCurrentDecl() == ID)
      return true;

    const SourceInfo &E = S.Current->getSource(OpPC);
    S.FFDiag(E, diag::note_constexpr_access_static_temporary, 1) << AK;
    S.Note(Ptr.getDeclLoc(), diag::note_constexpr_temporary_here);
    return false;
  }
  return true;
}

static bool CheckGlobal(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (auto ID = Ptr.getDeclID()) {
    if (!Ptr.isStatic())
      return true;

    if (S.P.getCurrentDecl() == ID)
      return true;

    S.FFDiag(S.Current->getLocation(OpPC), diag::note_constexpr_modify_global);
    return false;
  }
  return true;
}

namespace clang {
namespace interp {

bool popBuiltinArgs(InterpState &S, CodePtr OpPC) {
  assert(S.Current && S.Current->getFunction()->needsRuntimeArgPop(S.getCtx()));
  const Expr *E = S.Current->getExpr(OpPC);
  assert(isa<CallExpr>(E));
  const CallExpr *CE = cast<CallExpr>(E);
  for (int32_t I = CE->getNumArgs() - 1; I >= 0; --I) {
    const Expr *A = CE->getArg(I);
    PrimType Ty = S.getContext().classify(A->getType()).value_or(PT_Ptr);
    TYPE_SWITCH(Ty, S.Stk.discard<T>());
  }
  return true;
}

bool CheckExtern(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!Ptr.isExtern())
    return true;

  if (!S.checkingPotentialConstantExpression() && S.getLangOpts().CPlusPlus) {
    const auto *VD = Ptr.getDeclDesc()->asValueDecl();
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_constexpr_ltor_non_constexpr, 1) << VD;
    S.Note(VD->getLocation(), diag::note_declared_at);
  }
  return false;
}

bool CheckArray(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!Ptr.isUnknownSizeArray())
    return true;
  const SourceInfo &E = S.Current->getSource(OpPC);
  S.FFDiag(E, diag::note_constexpr_unsized_array_indexed);
  return false;
}

bool CheckLive(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               AccessKinds AK) {
  if (Ptr.isZero()) {
    const auto &Src = S.Current->getSource(OpPC);

    if (Ptr.isField())
      S.FFDiag(Src, diag::note_constexpr_null_subobject) << CSK_Field;
    else
      S.FFDiag(Src, diag::note_constexpr_access_null) << AK;

    return false;
  }

  if (!Ptr.isLive()) {
    const auto &Src = S.Current->getSource(OpPC);
    bool IsTemp = Ptr.isTemporary();

    S.FFDiag(Src, diag::note_constexpr_lifetime_ended, 1) << AK << !IsTemp;

    if (IsTemp)
      S.Note(Ptr.getDeclLoc(), diag::note_constexpr_temporary_here);
    else
      S.Note(Ptr.getDeclLoc(), diag::note_declared_at);

    return false;
  }

  return true;
}

bool CheckNull(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               CheckSubobjectKind CSK) {
  if (!Ptr.isZero())
    return true;
  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_null_subobject) << CSK;
  return false;
}

bool CheckRange(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                AccessKinds AK) {
  if (!Ptr.isOnePastEnd())
    return true;
  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_access_past_end) << AK;
  return false;
}

bool CheckRange(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                CheckSubobjectKind CSK) {
  if (!Ptr.isElementPastEnd())
    return true;
  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_past_end_subobject) << CSK;
  return false;
}

bool CheckSubobject(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                    CheckSubobjectKind CSK) {
  if (!Ptr.isOnePastEnd())
    return true;

  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_past_end_subobject) << CSK;
  return false;
}

bool CheckConst(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  assert(Ptr.isLive() && "Pointer is not live");
  if (!Ptr.isConst())
    return true;

  // The This pointer is writable in constructors and destructors,
  // even if isConst() returns true.
  if (const Function *Func = S.Current->getFunction();
      Func && (Func->isConstructor() || Func->isDestructor()) &&
      Ptr.block() == S.Current->getThis().block()) {
    return true;
  }

  const QualType Ty = Ptr.getType();
  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_modify_const_type) << Ty;
  return false;
}

bool CheckMutable(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  assert(Ptr.isLive() && "Pointer is not live");
  if (!Ptr.isMutable()) {
    return true;
  }

  const SourceInfo &Loc = S.Current->getSource(OpPC);
  const FieldDecl *Field = Ptr.getField();
  S.FFDiag(Loc, diag::note_constexpr_access_mutable, 1) << AK_Read << Field;
  S.Note(Field->getLocation(), diag::note_declared_at);
  return false;
}

bool CheckInitialized(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                      AccessKinds AK) {
  if (Ptr.isInitialized())
    return true;

  if (!S.checkingPotentialConstantExpression()) {
    S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_access_uninit)
        << AK << /*uninitialized=*/true << S.Current->getRange(OpPC);
  }
  return false;
}

bool CheckLoad(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!CheckLive(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckExtern(S, OpPC, Ptr))
    return false;
  if (!CheckRange(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckInitialized(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckActive(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckTemporary(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckMutable(S, OpPC, Ptr))
    return false;
  return true;
}

bool CheckStore(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!CheckLive(S, OpPC, Ptr, AK_Assign))
    return false;
  if (!CheckExtern(S, OpPC, Ptr))
    return false;
  if (!CheckRange(S, OpPC, Ptr, AK_Assign))
    return false;
  if (!CheckGlobal(S, OpPC, Ptr))
    return false;
  if (!CheckConst(S, OpPC, Ptr))
    return false;
  return true;
}

bool CheckInvoke(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!CheckLive(S, OpPC, Ptr, AK_MemberCall))
    return false;
  if (!CheckExtern(S, OpPC, Ptr))
    return false;
  if (!CheckRange(S, OpPC, Ptr, AK_MemberCall))
    return false;
  return true;
}

bool CheckInit(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!CheckLive(S, OpPC, Ptr, AK_Assign))
    return false;
  if (!CheckRange(S, OpPC, Ptr, AK_Assign))
    return false;
  return true;
}

bool CheckCallable(InterpState &S, CodePtr OpPC, const Function *F) {

  if (F->isVirtual() && !S.getLangOpts().CPlusPlus20) {
    const SourceLocation &Loc = S.Current->getLocation(OpPC);
    S.CCEDiag(Loc, diag::note_constexpr_virtual_call);
    return false;
  }

  if (!F->isConstexpr()) {
    // Don't emit anything if we're checking for a potential constant
    // expression. That will happen later when actually executing.
    if (S.checkingPotentialConstantExpression())
      return false;

    const SourceLocation &Loc = S.Current->getLocation(OpPC);
    if (S.getLangOpts().CPlusPlus11) {
      const FunctionDecl *DiagDecl = F->getDecl();

      // If this function is not constexpr because it is an inherited
      // non-constexpr constructor, diagnose that directly.
      const auto *CD = dyn_cast<CXXConstructorDecl>(DiagDecl);
      if (CD && CD->isInheritingConstructor()) {
        const auto *Inherited = CD->getInheritedConstructor().getConstructor();
        if (!Inherited->isConstexpr())
          DiagDecl = CD = Inherited;
      }

      // FIXME: If DiagDecl is an implicitly-declared special member function
      // or an inheriting constructor, we should be much more explicit about why
      // it's not constexpr.
      if (CD && CD->isInheritingConstructor())
        S.FFDiag(Loc, diag::note_constexpr_invalid_inhctor, 1)
          << CD->getInheritedConstructor().getConstructor()->getParent();
      else
        S.FFDiag(Loc, diag::note_constexpr_invalid_function, 1)
          << DiagDecl->isConstexpr() << (bool)CD << DiagDecl;
      S.Note(DiagDecl->getLocation(), diag::note_declared_at);
    } else {
      S.FFDiag(Loc, diag::note_invalid_subexpr_in_const_expr);
    }
    return false;
  }

  return true;
}

bool CheckCallDepth(InterpState &S, CodePtr OpPC) {
  if ((S.Current->getDepth() + 1) > S.getLangOpts().ConstexprCallDepth) {
    S.FFDiag(S.Current->getSource(OpPC),
             diag::note_constexpr_depth_limit_exceeded)
        << S.getLangOpts().ConstexprCallDepth;
    return false;
  }

  return true;
}

bool CheckThis(InterpState &S, CodePtr OpPC, const Pointer &This) {
  if (!This.isZero())
    return true;

  const SourceInfo &Loc = S.Current->getSource(OpPC);

  bool IsImplicit = false;
  if (const auto *E = dyn_cast_if_present<CXXThisExpr>(Loc.asExpr()))
    IsImplicit = E->isImplicit();

  if (S.getLangOpts().CPlusPlus11)
    S.FFDiag(Loc, diag::note_constexpr_this) << IsImplicit;
  else
    S.FFDiag(Loc);

  return false;
}

bool CheckPure(InterpState &S, CodePtr OpPC, const CXXMethodDecl *MD) {
  if (!MD->isPure())
    return true;
  const SourceInfo &E = S.Current->getSource(OpPC);
  S.FFDiag(E, diag::note_constexpr_pure_virtual_call, 1) << MD;
  S.Note(MD->getLocation(), diag::note_declared_at);
  return false;
}

static void DiagnoseUninitializedSubobject(InterpState &S, const SourceInfo &SI,
                                           const FieldDecl *SubObjDecl) {
  assert(SubObjDecl && "Subobject declaration does not exist");
  S.FFDiag(SI, diag::note_constexpr_uninitialized) << SubObjDecl;
  S.Note(SubObjDecl->getLocation(),
         diag::note_constexpr_subobject_declared_here);
}

static bool CheckFieldsInitialized(InterpState &S, CodePtr OpPC,
                                   const Pointer &BasePtr, const Record *R);

static bool CheckArrayInitialized(InterpState &S, CodePtr OpPC,
                                  const Pointer &BasePtr,
                                  const ConstantArrayType *CAT) {
  bool Result = true;
  size_t NumElems = CAT->getSize().getZExtValue();
  QualType ElemType = CAT->getElementType();

  if (ElemType->isRecordType()) {
    const Record *R = BasePtr.getElemRecord();
    for (size_t I = 0; I != NumElems; ++I) {
      Pointer ElemPtr = BasePtr.atIndex(I).narrow();
      Result &= CheckFieldsInitialized(S, OpPC, ElemPtr, R);
    }
  } else if (const auto *ElemCAT = dyn_cast<ConstantArrayType>(ElemType)) {
    for (size_t I = 0; I != NumElems; ++I) {
      Pointer ElemPtr = BasePtr.atIndex(I).narrow();
      Result &= CheckArrayInitialized(S, OpPC, ElemPtr, ElemCAT);
    }
  } else {
    for (size_t I = 0; I != NumElems; ++I) {
      if (!BasePtr.atIndex(I).isInitialized()) {
        DiagnoseUninitializedSubobject(S, S.Current->getSource(OpPC),
                                       BasePtr.getField());
        Result = false;
      }
    }
  }

  return Result;
}

static bool CheckFieldsInitialized(InterpState &S, CodePtr OpPC,
                                   const Pointer &BasePtr, const Record *R) {
  assert(R);
  bool Result = true;
  // Check all fields of this record are initialized.
  for (const Record::Field &F : R->fields()) {
    Pointer FieldPtr = BasePtr.atField(F.Offset);
    QualType FieldType = F.Decl->getType();

    if (FieldType->isRecordType()) {
      Result &= CheckFieldsInitialized(S, OpPC, FieldPtr, FieldPtr.getRecord());
    } else if (FieldType->isArrayType()) {
      const auto *CAT =
          cast<ConstantArrayType>(FieldType->getAsArrayTypeUnsafe());
      Result &= CheckArrayInitialized(S, OpPC, FieldPtr, CAT);
    } else if (!FieldPtr.isInitialized()) {
      DiagnoseUninitializedSubobject(S, S.Current->getSource(OpPC), F.Decl);
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
    Result &= CheckFieldsInitialized(S, OpPC, P, B.R);
  }

  // TODO: Virtual bases

  return Result;
}

bool CheckCtorCall(InterpState &S, CodePtr OpPC, const Pointer &This) {
  assert(!This.isZero());
  if (const Record *R = This.getRecord())
    return CheckFieldsInitialized(S, OpPC, This, R);
  const auto *CAT =
      cast<ConstantArrayType>(This.getType()->getAsArrayTypeUnsafe());
  return CheckArrayInitialized(S, OpPC, This, CAT);
}

bool CheckPotentialReinterpretCast(InterpState &S, CodePtr OpPC,
                                   const Pointer &Ptr) {
  if (!S.inConstantContext())
    return true;

  const SourceInfo &E = S.Current->getSource(OpPC);
  S.CCEDiag(E, diag::note_constexpr_invalid_cast)
      << 2 << S.getLangOpts().CPlusPlus << S.Current->getRange(OpPC);
  return false;
}

bool CheckFloatResult(InterpState &S, CodePtr OpPC, const Floating &Result,
                      APFloat::opStatus Status) {
  const SourceInfo &E = S.Current->getSource(OpPC);

  // [expr.pre]p4:
  //   If during the evaluation of an expression, the result is not
  //   mathematically defined [...], the behavior is undefined.
  // FIXME: C++ rules require us to not conform to IEEE 754 here.
  if (Result.isNan()) {
    S.CCEDiag(E, diag::note_constexpr_float_arithmetic)
        << /*NaN=*/true << S.Current->getRange(OpPC);
    return S.noteUndefinedBehavior();
  }

  // In a constant context, assume that any dynamic rounding mode or FP
  // exception state matches the default floating-point environment.
  if (S.inConstantContext())
    return true;

  FPOptions FPO = E.asExpr()->getFPFeaturesInEffect(S.Ctx.getLangOpts());

  if ((Status & APFloat::opInexact) &&
      FPO.getRoundingMode() == llvm::RoundingMode::Dynamic) {
    // Inexact result means that it depends on rounding mode. If the requested
    // mode is dynamic, the evaluation cannot be made in compile time.
    S.FFDiag(E, diag::note_constexpr_dynamic_rounding);
    return false;
  }

  if ((Status != APFloat::opOK) &&
      (FPO.getRoundingMode() == llvm::RoundingMode::Dynamic ||
       FPO.getExceptionMode() != LangOptions::FPE_Ignore ||
       FPO.getAllowFEnvAccess())) {
    S.FFDiag(E, diag::note_constexpr_float_arithmetic_strict);
    return false;
  }

  if ((Status & APFloat::opStatus::opInvalidOp) &&
      FPO.getExceptionMode() != LangOptions::FPE_Ignore) {
    // There is no usefully definable result.
    S.FFDiag(E);
    return false;
  }

  return true;
}

/// We aleady know the given DeclRefExpr is invalid for some reason,
/// now figure out why and print appropriate diagnostics.
bool CheckDeclRef(InterpState &S, CodePtr OpPC, const DeclRefExpr *DR) {
  const ValueDecl *D = DR->getDecl();
  const SourceInfo &E = S.Current->getSource(OpPC);

  if (isa<ParmVarDecl>(D)) {
    S.FFDiag(E, diag::note_constexpr_function_param_value_unknown) << D;
    S.Note(D->getLocation(), diag::note_declared_at) << D->getSourceRange();
  } else if (const auto *VD = dyn_cast<VarDecl>(D)) {
    if (!VD->getType().isConstQualified()) {
      S.FFDiag(E,
               VD->getType()->isIntegralOrEnumerationType()
                   ? diag::note_constexpr_ltor_non_const_int
                   : diag::note_constexpr_ltor_non_constexpr,
               1)
          << VD;
      S.Note(VD->getLocation(), diag::note_declared_at) << VD->getSourceRange();
      return false;
    }

    // const, but no initializer.
    if (!VD->getAnyInitializer()) {
      S.FFDiag(E, diag::note_constexpr_var_init_unknown, 1) << VD;
      S.Note(VD->getLocation(), diag::note_declared_at) << VD->getSourceRange();
      return false;
    }
  }

  return false;
}

bool Interpret(InterpState &S, APValue &Result) {
  // The current stack frame when we started Interpret().
  // This is being used by the ops to determine wheter
  // to return from this function and thus terminate
  // interpretation.
  const InterpFrame *StartFrame = S.Current;
  assert(!S.Current->isRoot());
  CodePtr PC = S.Current->getPC();

  // Empty program.
  if (!PC)
    return true;

  for (;;) {
    auto Op = PC.read<Opcode>();
    CodePtr OpPC = PC;

    switch (Op) {
#define GET_INTERP
#include "Opcodes.inc"
#undef GET_INTERP
    }
  }
}

} // namespace interp
} // namespace clang
