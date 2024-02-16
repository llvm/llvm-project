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

static void diagnoseNonConstVariable(InterpState &S, CodePtr OpPC,
                                     const ValueDecl *VD) {
  if (!S.getLangOpts().CPlusPlus)
    return;

  const SourceInfo &Loc = S.Current->getSource(OpPC);

  if (VD->getType()->isIntegralOrEnumerationType())
    S.FFDiag(Loc, diag::note_constexpr_ltor_non_const_int, 1) << VD;
  else
    S.FFDiag(Loc,
             S.getLangOpts().CPlusPlus11
                 ? diag::note_constexpr_ltor_non_constexpr
                 : diag::note_constexpr_ltor_non_integral,
             1)
        << VD << VD->getType();
  S.Note(VD->getLocation(), diag::note_declared_at);
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
static void popArg(InterpState &S, const Expr *Arg) {
  PrimType Ty = S.getContext().classify(Arg).value_or(PT_Ptr);
  TYPE_SWITCH(Ty, S.Stk.discard<T>());
}

void cleanupAfterFunctionCall(InterpState &S, CodePtr OpPC) {
  assert(S.Current);
  const Function *CurFunc = S.Current->getFunction();
  assert(CurFunc);

  if (CurFunc->isUnevaluatedBuiltin())
    return;

  // Some builtin functions require us to only look at the call site, since
  // the classified parameter types do not match.
  if (CurFunc->isBuiltin()) {
    const auto *CE =
        cast<CallExpr>(S.Current->Caller->getExpr(S.Current->getRetPC()));
    for (int32_t I = CE->getNumArgs() - 1; I >= 0; --I) {
      const Expr *A = CE->getArg(I);
      popArg(S, A);
    }
    return;
  }

  if (S.Current->Caller && CurFunc->isVariadic()) {
    // CallExpr we're look for is at the return PC of the current function, i.e.
    // in the caller.
    // This code path should be executed very rarely.
    unsigned NumVarArgs;
    const Expr *const *Args = nullptr;
    unsigned NumArgs = 0;
    const Expr *CallSite = S.Current->Caller->getExpr(S.Current->getRetPC());
    if (const auto *CE = dyn_cast<CallExpr>(CallSite)) {
      Args = CE->getArgs();
      NumArgs = CE->getNumArgs();
    } else if (const auto *CE = dyn_cast<CXXConstructExpr>(CallSite)) {
      Args = CE->getArgs();
      NumArgs = CE->getNumArgs();
    } else
      assert(false && "Can't get arguments from that expression type");

    assert(NumArgs >= CurFunc->getNumWrittenParams());
    NumVarArgs = NumArgs - CurFunc->getNumWrittenParams();
    for (unsigned I = 0; I != NumVarArgs; ++I) {
      const Expr *A = Args[NumArgs - 1 - I];
      popArg(S, A);
    }
  }

  // And in any case, remove the fixed parameters (the non-variadic ones)
  // at the end.
  S.Current->popArgs();
}

bool CheckExtern(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!Ptr.isExtern())
    return true;

  if (!S.checkingPotentialConstantExpression() && S.getLangOpts().CPlusPlus) {
    const auto *VD = Ptr.getDeclDesc()->asValueDecl();
    diagnoseNonConstVariable(S, OpPC, VD);
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

bool CheckConstant(InterpState &S, CodePtr OpPC, const Descriptor *Desc) {
  assert(Desc);

  auto IsConstType = [&S](const VarDecl *VD) -> bool {
    if (VD->isConstexpr())
      return true;

    if (S.getLangOpts().CPlusPlus && !S.getLangOpts().CPlusPlus11)
      return false;

    QualType T = VD->getType();
    if (T.isConstQualified())
      return true;

    if (const auto *RT = T->getAs<ReferenceType>())
      return RT->getPointeeType().isConstQualified();

    if (const auto *PT = T->getAs<PointerType>())
      return PT->getPointeeType().isConstQualified();

    return false;
  };

  if (const auto *D = Desc->asValueDecl()) {
    if (const auto *VD = dyn_cast<VarDecl>(D);
        VD && VD->hasGlobalStorage() && !IsConstType(VD)) {
      diagnoseNonConstVariable(S, OpPC, VD);
      return S.inConstantContext();
    }
  }

  return true;
}

static bool CheckConstant(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  return CheckConstant(S, OpPC, Ptr.getDeclDesc());
}

bool CheckDummy(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  return !Ptr.isDummy();
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

bool CheckGlobalInitialized(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (Ptr.isInitialized())
    return true;

  assert(S.getLangOpts().CPlusPlus);
  const auto *VD = cast<VarDecl>(Ptr.getDeclDesc()->asValueDecl());
  if ((!VD->hasConstantInitialization() &&
       VD->mightBeUsableInConstantExpressions(S.getCtx())) ||
      (S.getLangOpts().OpenCL && !S.getLangOpts().CPlusPlus11 &&
       !VD->hasICEInitializer(S.getCtx()))) {
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_constexpr_var_init_non_constant, 1) << VD;
    S.Note(VD->getLocation(), diag::note_declared_at);
  }
  return false;
}

bool CheckLoad(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!CheckLive(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckConstant(S, OpPC, Ptr))
    return false;

  if (!CheckDummy(S, OpPC, Ptr))
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
  if (!CheckDummy(S, OpPC, Ptr))
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
      if (CD && CD->isInheritingConstructor()) {
        S.FFDiag(Loc, diag::note_constexpr_invalid_inhctor, 1)
          << CD->getInheritedConstructor().getConstructor()->getParent();
        S.Note(DiagDecl->getLocation(), diag::note_declared_at);
      } else {
        // Don't emit anything if the function isn't defined and we're checking
        // for a constant expression. It might be defined at the point we're
        // actually calling it.
        if (!DiagDecl->isDefined() && S.checkingPotentialConstantExpression())
          return false;

        // If the declaration is defined _and_ declared 'constexpr', the below
        // diagnostic doesn't add anything useful.
        if (DiagDecl->isDefined() && DiagDecl->isConstexpr())
          return false;

        S.FFDiag(Loc, diag::note_constexpr_invalid_function, 1)
          << DiagDecl->isConstexpr() << (bool)CD << DiagDecl;
        S.Note(DiagDecl->getLocation(), diag::note_declared_at);
      }
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
  if (!MD->isPureVirtual())
    return true;
  const SourceInfo &E = S.Current->getSource(OpPC);
  S.FFDiag(E, diag::note_constexpr_pure_virtual_call, 1) << MD;
  S.Note(MD->getLocation(), diag::note_declared_at);
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
    if (S.getLangOpts().CPlusPlus11) {
      S.FFDiag(E, diag::note_constexpr_function_param_value_unknown) << D;
      S.Note(D->getLocation(), diag::note_declared_at) << D->getSourceRange();
    } else {
      S.FFDiag(E);
    }
  } else if (const auto *VD = dyn_cast<VarDecl>(D)) {
    if (!VD->getType().isConstQualified()) {
      diagnoseNonConstVariable(S, OpPC, VD);
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
