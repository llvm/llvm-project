//===------- Interp.cpp - Interpreter for the constexpr VM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interp.h"
#include "Compiler.h"
#include "Function.h"
#include "InterpFrame.h"
#include "InterpShared.h"
#include "InterpStack.h"
#include "Opcode.h"
#include "PrimType.h"
#include "Program.h"
#include "State.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace clang::interp;

static bool RetValue(InterpState &S, CodePtr &Pt) {
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

// https://github.com/llvm/llvm-project/issues/102513
#if defined(_MSC_VER) && !defined(__clang__) && !defined(NDEBUG)
#pragma optimize("", off)
#endif
// FIXME: We have the large switch over all opcodes here again, and in
// Interpret().
static bool BCP(InterpState &S, CodePtr &RealPC, int32_t Offset, PrimType PT) {
  [[maybe_unused]] CodePtr PCBefore = RealPC;
  size_t StackSizeBefore = S.Stk.size();

  auto SpeculativeInterp = [&S, RealPC]() -> bool {
    const InterpFrame *StartFrame = S.Current;
    CodePtr PC = RealPC;

    for (;;) {
      auto Op = PC.read<Opcode>();
      if (Op == OP_EndSpeculation)
        return true;
      CodePtr OpPC = PC;

      switch (Op) {
#define GET_INTERP
#include "Opcodes.inc"
#undef GET_INTERP
      }
    }
    llvm_unreachable("We didn't see an EndSpeculation op?");
  };

  if (SpeculativeInterp()) {
    if (PT == PT_Ptr) {
      const auto &Ptr = S.Stk.pop<Pointer>();
      assert(S.Stk.size() == StackSizeBefore);
      S.Stk.push<Integral<32, true>>(
          Integral<32, true>::from(CheckBCPResult(S, Ptr)));
    } else {
      // Pop the result from the stack and return success.
      TYPE_SWITCH(PT, S.Stk.pop<T>(););
      assert(S.Stk.size() == StackSizeBefore);
      S.Stk.push<Integral<32, true>>(Integral<32, true>::from(1));
    }
  } else {
    if (!S.inConstantContext())
      return Invalid(S, RealPC);

    S.Stk.clearTo(StackSizeBefore);
    S.Stk.push<Integral<32, true>>(Integral<32, true>::from(0));
  }

  // RealPC should not have been modified.
  assert(*RealPC == *PCBefore);

  // Jump to end label. This is a little tricker than just RealPC += Offset
  // because our usual jump instructions don't have any arguments, to the offset
  // we get is a little too much and we need to subtract the size of the
  // bool and PrimType arguments again.
  int32_t ParamSize = align(sizeof(PrimType));
  assert(Offset >= ParamSize);
  RealPC += Offset - ParamSize;

  [[maybe_unused]] CodePtr PCCopy = RealPC;
  assert(PCCopy.read<Opcode>() == OP_EndSpeculation);

  return true;
}
// https://github.com/llvm/llvm-project/issues/102513
#if defined(_MSC_VER) && !defined(__clang__) && !defined(NDEBUG)
#pragma optimize("", on)
#endif

static void diagnoseMissingInitializer(InterpState &S, CodePtr OpPC,
                                       const ValueDecl *VD) {
  const SourceInfo &E = S.Current->getSource(OpPC);
  S.FFDiag(E, diag::note_constexpr_var_init_unknown, 1) << VD;
  S.Note(VD->getLocation(), diag::note_declared_at) << VD->getSourceRange();
}

static void diagnoseNonConstVariable(InterpState &S, CodePtr OpPC,
                                     const ValueDecl *VD);
static bool diagnoseUnknownDecl(InterpState &S, CodePtr OpPC,
                                const ValueDecl *D) {

  if (isa<ParmVarDecl>(D)) {
    if (D->getType()->isReferenceType())
      return false;

    const SourceInfo &Loc = S.Current->getSource(OpPC);
    if (S.getLangOpts().CPlusPlus11) {
      S.FFDiag(Loc, diag::note_constexpr_function_param_value_unknown) << D;
      S.Note(D->getLocation(), diag::note_declared_at) << D->getSourceRange();
    } else {
      S.FFDiag(Loc);
    }
    return false;
  }

  if (!D->getType().isConstQualified()) {
    diagnoseNonConstVariable(S, OpPC, D);
  } else if (const auto *VD = dyn_cast<VarDecl>(D)) {
    if (!VD->getAnyInitializer()) {
      diagnoseMissingInitializer(S, OpPC, VD);
    } else {
      const SourceInfo &Loc = S.Current->getSource(OpPC);
      S.FFDiag(Loc, diag::note_constexpr_var_init_non_constant, 1) << VD;
      S.Note(VD->getLocation(), diag::note_declared_at);
    }
  }

  return false;
}

static void diagnoseNonConstVariable(InterpState &S, CodePtr OpPC,
                                     const ValueDecl *VD) {
  const SourceInfo &Loc = S.Current->getSource(OpPC);
  if (!S.getLangOpts().CPlusPlus) {
    S.FFDiag(Loc);
    return;
  }

  if (const auto *VarD = dyn_cast<VarDecl>(VD);
      VarD && VarD->getType().isConstQualified() &&
      !VarD->getAnyInitializer()) {
    diagnoseMissingInitializer(S, OpPC, VD);
    return;
  }

  // Rather random, but this is to match the diagnostic output of the current
  // interpreter.
  if (isa<ObjCIvarDecl>(VD))
    return;

  if (VD->getType()->isIntegralOrEnumerationType()) {
    S.FFDiag(Loc, diag::note_constexpr_ltor_non_const_int, 1) << VD;
    S.Note(VD->getLocation(), diag::note_declared_at);
    return;
  }

  S.FFDiag(Loc,
           S.getLangOpts().CPlusPlus11 ? diag::note_constexpr_ltor_non_constexpr
                                       : diag::note_constexpr_ltor_non_integral,
           1)
      << VD << VD->getType();
  S.Note(VD->getLocation(), diag::note_declared_at);
}

static bool CheckTemporary(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                           AccessKinds AK) {
  if (auto ID = Ptr.getDeclID()) {
    if (!Ptr.isStaticTemporary())
      return true;

    const auto *MTE = dyn_cast_if_present<MaterializeTemporaryExpr>(
        Ptr.getDeclDesc()->asExpr());
    if (!MTE)
      return true;

    // FIXME(perf): Since we do this check on every Load from a static
    // temporary, it might make sense to cache the value of the
    // isUsableInConstantExpressions call.
    if (!MTE->isUsableInConstantExpressions(S.getASTContext()) &&
        Ptr.block()->getEvalID() != S.Ctx.getEvalID()) {
      const SourceInfo &E = S.Current->getSource(OpPC);
      S.FFDiag(E, diag::note_constexpr_access_static_temporary, 1) << AK;
      S.Note(Ptr.getDeclLoc(), diag::note_constexpr_temporary_here);
      return false;
    }
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

void cleanupAfterFunctionCall(InterpState &S, CodePtr OpPC,
                              const Function *Func) {
  assert(S.Current);
  assert(Func);

  if (S.Current->Caller && Func->isVariadic()) {
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

    assert(NumArgs >= Func->getNumWrittenParams());
    NumVarArgs = NumArgs - (Func->getNumWrittenParams() +
                            isa<CXXOperatorCallExpr>(CallSite));
    for (unsigned I = 0; I != NumVarArgs; ++I) {
      const Expr *A = Args[NumArgs - 1 - I];
      popArg(S, A);
    }
  }

  // And in any case, remove the fixed parameters (the non-variadic ones)
  // at the end.
  for (PrimType Ty : Func->args_reverse())
    TYPE_SWITCH(Ty, S.Stk.discard<T>());
}

bool isConstexprUnknown(const Pointer &P) {
  if (!P.isBlockPointer())
    return false;

  if (P.isDummy())
    return isa_and_nonnull<ParmVarDecl>(P.getDeclDesc()->asValueDecl());

  return P.getDeclDesc()->IsConstexprUnknown;
}

bool CheckBCPResult(InterpState &S, const Pointer &Ptr) {
  if (Ptr.isDummy())
    return false;
  if (Ptr.isZero())
    return true;
  if (Ptr.isFunctionPointer())
    return false;
  if (Ptr.isIntegralPointer())
    return true;
  if (Ptr.isTypeidPointer())
    return true;

  if (Ptr.getType()->isAnyComplexType())
    return true;

  if (const Expr *Base = Ptr.getDeclDesc()->asExpr())
    return isa<StringLiteral>(Base) && Ptr.getIndex() == 0;
  return false;
}

bool CheckActive(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                 AccessKinds AK) {
  if (Ptr.isActive())
    return true;

  assert(Ptr.inUnion());
  assert(Ptr.isField() && Ptr.getField());

  Pointer U = Ptr.getBase();
  Pointer C = Ptr;
  while (!U.isRoot() && !U.isActive()) {
    // A little arbitrary, but this is what the current interpreter does.
    // See the AnonymousUnion test in test/AST/ByteCode/unions.cpp.
    // GCC's output is more similar to what we would get without
    // this condition.
    if (U.getRecord() && U.getRecord()->isAnonymousUnion())
      break;

    C = U;
    U = U.getBase();
  }
  assert(C.isField());

  // Consider:
  // union U {
  //   struct {
  //     int x;
  //     int y;
  //   } a;
  // }
  //
  // When activating x, we will also activate a. If we now try to read
  // from y, we will get to CheckActive, because y is not active. In that
  // case, our U will be a (not a union). We return here and let later code
  // handle this.
  if (!U.getFieldDesc()->isUnion())
    return true;

  // Get the inactive field descriptor.
  assert(!C.isActive());
  const FieldDecl *InactiveField = C.getField();
  assert(InactiveField);

  // Find the active field of the union.
  const Record *R = U.getRecord();
  assert(R && R->isUnion() && "Not a union");

  const FieldDecl *ActiveField = nullptr;
  for (const Record::Field &F : R->fields()) {
    const Pointer &Field = U.atField(F.Offset);
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

bool CheckExtern(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!Ptr.isExtern())
    return true;

  if (Ptr.isInitialized() ||
      (Ptr.getDeclDesc()->asVarDecl() == S.EvaluatingDecl))
    return true;

  if (S.checkingPotentialConstantExpression() && S.getLangOpts().CPlusPlus &&
      Ptr.isConst())
    return false;

  const auto *VD = Ptr.getDeclDesc()->asValueDecl();
  diagnoseNonConstVariable(S, OpPC, VD);
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

    if (Ptr.isDynamic()) {
      S.FFDiag(Src, diag::note_constexpr_access_deleted_object) << AK;
    } else if (!S.checkingPotentialConstantExpression()) {
      bool IsTemp = Ptr.isTemporary();
      S.FFDiag(Src, diag::note_constexpr_lifetime_ended, 1) << AK << !IsTemp;

      if (IsTemp)
        S.Note(Ptr.getDeclLoc(), diag::note_constexpr_temporary_here);
      else
        S.Note(Ptr.getDeclLoc(), diag::note_declared_at);
    }

    return false;
  }

  return true;
}

bool CheckConstant(InterpState &S, CodePtr OpPC, const Descriptor *Desc) {
  assert(Desc);

  const auto *D = Desc->asVarDecl();
  if (!D || !D->hasGlobalStorage())
    return true;

  if (D == S.EvaluatingDecl)
    return true;

  if (D->isConstexpr())
    return true;

  // If we're evaluating the initializer for a constexpr variable in C23, we may
  // only read other contexpr variables. Abort here since this one isn't
  // constexpr.
  if (const auto *VD = dyn_cast_if_present<VarDecl>(S.EvaluatingDecl);
      VD && VD->isConstexpr() && S.getLangOpts().C23)
    return Invalid(S, OpPC);

  QualType T = D->getType();
  bool IsConstant = T.isConstant(S.getASTContext());
  if (T->isIntegralOrEnumerationType()) {
    if (!IsConstant) {
      diagnoseNonConstVariable(S, OpPC, D);
      return false;
    }
    return true;
  }

  if (IsConstant) {
    if (S.getLangOpts().CPlusPlus) {
      S.CCEDiag(S.Current->getLocation(OpPC),
                S.getLangOpts().CPlusPlus11
                    ? diag::note_constexpr_ltor_non_constexpr
                    : diag::note_constexpr_ltor_non_integral,
                1)
          << D << T;
      S.Note(D->getLocation(), diag::note_declared_at);
    } else {
      S.CCEDiag(S.Current->getLocation(OpPC));
    }
    return true;
  }

  if (T->isPointerOrReferenceType()) {
    if (!T->getPointeeType().isConstant(S.getASTContext()) ||
        !S.getLangOpts().CPlusPlus11) {
      diagnoseNonConstVariable(S, OpPC, D);
      return false;
    }
    return true;
  }

  diagnoseNonConstVariable(S, OpPC, D);
  return false;
}

static bool CheckConstant(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!Ptr.isStatic() || !Ptr.isBlockPointer())
    return true;
  if (!Ptr.getDeclID())
    return true;
  return CheckConstant(S, OpPC, Ptr.getDeclDesc());
}

bool CheckNull(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               CheckSubobjectKind CSK) {
  if (!Ptr.isZero())
    return true;
  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_null_subobject)
      << CSK << S.Current->getRange(OpPC);

  return false;
}

bool CheckRange(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                AccessKinds AK) {
  if (!Ptr.isOnePastEnd())
    return true;
  if (S.getLangOpts().CPlusPlus) {
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_constexpr_access_past_end)
        << AK << S.Current->getRange(OpPC);
  }
  return false;
}

bool CheckRange(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                CheckSubobjectKind CSK) {
  if (!Ptr.isElementPastEnd())
    return true;
  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_past_end_subobject)
      << CSK << S.Current->getRange(OpPC);
  return false;
}

bool CheckSubobject(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                    CheckSubobjectKind CSK) {
  if (!Ptr.isOnePastEnd())
    return true;

  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_past_end_subobject)
      << CSK << S.Current->getRange(OpPC);
  return false;
}

bool CheckDowncast(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                   uint32_t Offset) {
  uint32_t MinOffset = Ptr.getDeclDesc()->getMetadataSize();
  uint32_t PtrOffset = Ptr.getByteOffset();

  // We subtract Offset from PtrOffset. The result must be at least
  // MinOffset.
  if (Offset < PtrOffset && (PtrOffset - Offset) >= MinOffset)
    return true;

  const auto *E = cast<CastExpr>(S.Current->getExpr(OpPC));
  QualType TargetQT = E->getType()->getPointeeType();
  QualType MostDerivedQT = Ptr.getDeclPtr().getType();

  S.CCEDiag(E, diag::note_constexpr_invalid_downcast)
      << MostDerivedQT << TargetQT;

  return false;
}

bool CheckConst(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  assert(Ptr.isLive() && "Pointer is not live");
  if (!Ptr.isConst() || Ptr.isMutable())
    return true;

  // The This pointer is writable in constructors and destructors,
  // even if isConst() returns true.
  // TODO(perf): We could be hitting this code path quite a lot in complex
  // constructors. Is there a better way to do this?
  if (S.Current->getFunction()) {
    for (const InterpFrame *Frame = S.Current; Frame; Frame = Frame->Caller) {
      if (const Function *Func = Frame->getFunction();
          Func && (Func->isConstructor() || Func->isDestructor()) &&
          Ptr.block() == Frame->getThis().block()) {
        return true;
      }
    }
  }

  if (!Ptr.isBlockPointer())
    return false;

  const QualType Ty = Ptr.getType();
  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_modify_const_type) << Ty;
  return false;
}

bool CheckMutable(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  assert(Ptr.isLive() && "Pointer is not live");
  if (!Ptr.isMutable())
    return true;

  // In C++14 onwards, it is permitted to read a mutable member whose
  // lifetime began within the evaluation.
  if (S.getLangOpts().CPlusPlus14 &&
      Ptr.block()->getEvalID() == S.Ctx.getEvalID()) {
    // FIXME: This check is necessary because (of the way) we revisit
    // variables in Compiler.cpp:visitDeclRef. Revisiting a so far
    // unknown variable will get the same EvalID and we end up allowing
    // reads from mutable members of it.
    if (!S.inConstantContext() && isConstexprUnknown(Ptr))
      return false;
    return true;
  }

  const SourceInfo &Loc = S.Current->getSource(OpPC);
  const FieldDecl *Field = Ptr.getField();
  S.FFDiag(Loc, diag::note_constexpr_access_mutable, 1) << AK_Read << Field;
  S.Note(Field->getLocation(), diag::note_declared_at);
  return false;
}

static bool CheckVolatile(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                          AccessKinds AK) {
  assert(Ptr.isLive());

  if (!Ptr.isVolatile())
    return true;

  if (!S.getLangOpts().CPlusPlus)
    return Invalid(S, OpPC);

  // The reason why Ptr is volatile might be further up the hierarchy.
  // Find that pointer.
  Pointer P = Ptr;
  while (!P.isRoot()) {
    if (P.getType().isVolatileQualified())
      break;
    P = P.getBase();
  }

  const NamedDecl *ND = nullptr;
  int DiagKind;
  SourceLocation Loc;
  if (const auto *F = P.getField()) {
    DiagKind = 2;
    Loc = F->getLocation();
    ND = F;
  } else if (auto *VD = P.getFieldDesc()->asValueDecl()) {
    DiagKind = 1;
    Loc = VD->getLocation();
    ND = VD;
  } else {
    DiagKind = 0;
    if (const auto *E = P.getFieldDesc()->asExpr())
      Loc = E->getExprLoc();
  }

  S.FFDiag(S.Current->getLocation(OpPC),
           diag::note_constexpr_access_volatile_obj, 1)
      << AK << DiagKind << ND;
  S.Note(Loc, diag::note_constexpr_volatile_here) << DiagKind;
  return false;
}

bool CheckInitialized(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                      AccessKinds AK) {
  assert(Ptr.isLive());

  if (Ptr.isInitialized())
    return true;

  if (const auto *VD = Ptr.getDeclDesc()->asVarDecl();
      VD && (VD->isConstexpr() || VD->hasGlobalStorage())) {

    if (!S.getLangOpts().CPlusPlus23 && VD == S.EvaluatingDecl) {
      if (!S.getLangOpts().CPlusPlus14 &&
          !VD->getType().isConstant(S.getASTContext())) {
        // Diagnose as non-const read.
        diagnoseNonConstVariable(S, OpPC, VD);
      } else {
        const SourceInfo &Loc = S.Current->getSource(OpPC);
        // Diagnose as "read of object outside its lifetime".
        S.FFDiag(Loc, diag::note_constexpr_access_uninit)
            << AK << /*IsIndeterminate=*/false;
      }
      return false;
    }

    if (VD->getAnyInitializer()) {
      const SourceInfo &Loc = S.Current->getSource(OpPC);
      S.FFDiag(Loc, diag::note_constexpr_var_init_non_constant, 1) << VD;
      S.Note(VD->getLocation(), diag::note_declared_at);
    } else {
      diagnoseMissingInitializer(S, OpPC, VD);
    }
    return false;
  }

  if (!S.checkingPotentialConstantExpression()) {
    S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_access_uninit)
        << AK << /*uninitialized=*/true << S.Current->getRange(OpPC);
  }
  return false;
}

static bool CheckLifetime(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                          AccessKinds AK) {
  if (Ptr.getLifetime() == Lifetime::Started)
    return true;

  if (!S.checkingPotentialConstantExpression()) {
    S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_access_uninit)
        << AK << /*uninitialized=*/false << S.Current->getRange(OpPC);
  }
  return false;
}

bool CheckGlobalInitialized(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (Ptr.isInitialized())
    return true;

  assert(S.getLangOpts().CPlusPlus);
  const auto *VD = cast<VarDecl>(Ptr.getDeclDesc()->asValueDecl());
  if ((!VD->hasConstantInitialization() &&
       VD->mightBeUsableInConstantExpressions(S.getASTContext())) ||
      (S.getLangOpts().OpenCL && !S.getLangOpts().CPlusPlus11 &&
       !VD->hasICEInitializer(S.getASTContext()))) {
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_constexpr_var_init_non_constant, 1) << VD;
    S.Note(VD->getLocation(), diag::note_declared_at);
  }
  return false;
}

static bool CheckWeak(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!Ptr.isWeak())
    return true;

  const auto *VD = Ptr.getDeclDesc()->asVarDecl();
  assert(VD);
  S.FFDiag(S.Current->getLocation(OpPC), diag::note_constexpr_var_init_weak)
      << VD;
  S.Note(VD->getLocation(), diag::note_declared_at);

  return false;
}

bool CheckLoad(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               AccessKinds AK) {
  if (!CheckLive(S, OpPC, Ptr, AK))
    return false;
  if (!CheckExtern(S, OpPC, Ptr))
    return false;
  if (!CheckConstant(S, OpPC, Ptr))
    return false;
  if (!CheckDummy(S, OpPC, Ptr, AK))
    return false;
  if (!CheckRange(S, OpPC, Ptr, AK))
    return false;
  if (!CheckActive(S, OpPC, Ptr, AK))
    return false;
  if (!CheckLifetime(S, OpPC, Ptr, AK))
    return false;
  if (!CheckInitialized(S, OpPC, Ptr, AK))
    return false;
  if (!CheckTemporary(S, OpPC, Ptr, AK))
    return false;
  if (!CheckWeak(S, OpPC, Ptr))
    return false;
  if (!CheckMutable(S, OpPC, Ptr))
    return false;
  if (!CheckVolatile(S, OpPC, Ptr, AK))
    return false;
  return true;
}

/// This is not used by any of the opcodes directly. It's used by
/// EvalEmitter to do the final lvalue-to-rvalue conversion.
bool CheckFinalLoad(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!CheckLive(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckConstant(S, OpPC, Ptr))
    return false;

  if (!CheckDummy(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckExtern(S, OpPC, Ptr))
    return false;
  if (!CheckRange(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckActive(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckLifetime(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckInitialized(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckTemporary(S, OpPC, Ptr, AK_Read))
    return false;
  if (!CheckWeak(S, OpPC, Ptr))
    return false;
  if (!CheckMutable(S, OpPC, Ptr))
    return false;
  return true;
}

bool CheckStore(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!CheckLive(S, OpPC, Ptr, AK_Assign))
    return false;
  if (!CheckDummy(S, OpPC, Ptr, AK_Assign))
    return false;
  if (!CheckLifetime(S, OpPC, Ptr, AK_Assign))
    return false;
  if (!CheckExtern(S, OpPC, Ptr))
    return false;
  if (!CheckRange(S, OpPC, Ptr, AK_Assign))
    return false;
  if (!CheckGlobal(S, OpPC, Ptr))
    return false;
  if (!CheckConst(S, OpPC, Ptr))
    return false;
  if (!S.inConstantContext() && isConstexprUnknown(Ptr))
    return false;
  return true;
}

bool CheckInvoke(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!CheckLive(S, OpPC, Ptr, AK_MemberCall))
    return false;
  if (!Ptr.isDummy()) {
    if (!CheckExtern(S, OpPC, Ptr))
      return false;
    if (!CheckRange(S, OpPC, Ptr, AK_MemberCall))
      return false;
  }
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

  // Bail out if the function declaration itself is invalid.  We will
  // have produced a relevant diagnostic while parsing it, so just
  // note the problematic sub-expression.
  if (F->getDecl()->isInvalidDecl())
    return Invalid(S, OpPC);

  if (S.checkingPotentialConstantExpression() && S.Current->getDepth() != 0)
    return false;

  if (F->isConstexpr() && F->hasBody() &&
      (F->getDecl()->isConstexpr() || F->getDecl()->hasAttr<MSConstexprAttr>()))
    return true;

  // Implicitly constexpr.
  if (F->isLambdaStaticInvoker())
    return true;

  // Diagnose failed assertions specially.
  if (S.Current->getLocation(OpPC).isMacroID() &&
      F->getDecl()->getIdentifier()) {
    // FIXME: Instead of checking for an implementation-defined function,
    // check and evaluate the assert() macro.
    StringRef Name = F->getDecl()->getName();
    bool AssertFailed =
        Name == "__assert_rtn" || Name == "__assert_fail" || Name == "_wassert";
    if (AssertFailed) {
      S.FFDiag(S.Current->getLocation(OpPC),
               diag::note_constexpr_assert_failed);
      return false;
    }
  }

  if (S.getLangOpts().CPlusPlus11) {
    const FunctionDecl *DiagDecl = F->getDecl();

    // Invalid decls have been diagnosed before.
    if (DiagDecl->isInvalidDecl())
      return false;

    // If this function is not constexpr because it is an inherited
    // non-constexpr constructor, diagnose that directly.
    const auto *CD = dyn_cast<CXXConstructorDecl>(DiagDecl);
    if (CD && CD->isInheritingConstructor()) {
      const auto *Inherited = CD->getInheritedConstructor().getConstructor();
      if (!Inherited->isConstexpr())
        DiagDecl = CD = Inherited;
    }

    // Silently reject constructors of invalid classes. The invalid class
    // has been rejected elsewhere before.
    if (CD && CD->getParent()->isInvalidDecl())
      return false;

    // FIXME: If DiagDecl is an implicitly-declared special member function
    // or an inheriting constructor, we should be much more explicit about why
    // it's not constexpr.
    if (CD && CD->isInheritingConstructor()) {
      S.FFDiag(S.Current->getLocation(OpPC),
               diag::note_constexpr_invalid_inhctor, 1)
          << CD->getInheritedConstructor().getConstructor()->getParent();
      S.Note(DiagDecl->getLocation(), diag::note_declared_at);
    } else {
      // Don't emit anything if the function isn't defined and we're checking
      // for a constant expression. It might be defined at the point we're
      // actually calling it.
      bool IsExtern = DiagDecl->getStorageClass() == SC_Extern;
      if (!DiagDecl->isDefined() && !IsExtern && DiagDecl->isConstexpr() &&
          S.checkingPotentialConstantExpression())
        return false;

      // If the declaration is defined, declared 'constexpr' _and_ has a body,
      // the below diagnostic doesn't add anything useful.
      if (DiagDecl->isDefined() && DiagDecl->isConstexpr() &&
          DiagDecl->hasBody())
        return false;

      S.FFDiag(S.Current->getLocation(OpPC),
               diag::note_constexpr_invalid_function, 1)
          << DiagDecl->isConstexpr() << (bool)CD << DiagDecl;

      if (DiagDecl->getDefinition())
        S.Note(DiagDecl->getDefinition()->getLocation(),
               diag::note_declared_at);
      else
        S.Note(DiagDecl->getLocation(), diag::note_declared_at);
    }
  } else {
    S.FFDiag(S.Current->getLocation(OpPC),
             diag::note_invalid_subexpr_in_const_expr);
  }

  return false;
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

bool CheckFloatResult(InterpState &S, CodePtr OpPC, const Floating &Result,
                      APFloat::opStatus Status, FPOptions FPO) {
  // [expr.pre]p4:
  //   If during the evaluation of an expression, the result is not
  //   mathematically defined [...], the behavior is undefined.
  // FIXME: C++ rules require us to not conform to IEEE 754 here.
  if (Result.isNan()) {
    const SourceInfo &E = S.Current->getSource(OpPC);
    S.CCEDiag(E, diag::note_constexpr_float_arithmetic)
        << /*NaN=*/true << S.Current->getRange(OpPC);
    return S.noteUndefinedBehavior();
  }

  // In a constant context, assume that any dynamic rounding mode or FP
  // exception state matches the default floating-point environment.
  if (S.inConstantContext())
    return true;

  if ((Status & APFloat::opInexact) &&
      FPO.getRoundingMode() == llvm::RoundingMode::Dynamic) {
    // Inexact result means that it depends on rounding mode. If the requested
    // mode is dynamic, the evaluation cannot be made in compile time.
    const SourceInfo &E = S.Current->getSource(OpPC);
    S.FFDiag(E, diag::note_constexpr_dynamic_rounding);
    return false;
  }

  if ((Status != APFloat::opOK) &&
      (FPO.getRoundingMode() == llvm::RoundingMode::Dynamic ||
       FPO.getExceptionMode() != LangOptions::FPE_Ignore ||
       FPO.getAllowFEnvAccess())) {
    const SourceInfo &E = S.Current->getSource(OpPC);
    S.FFDiag(E, diag::note_constexpr_float_arithmetic_strict);
    return false;
  }

  if ((Status & APFloat::opStatus::opInvalidOp) &&
      FPO.getExceptionMode() != LangOptions::FPE_Ignore) {
    const SourceInfo &E = S.Current->getSource(OpPC);
    // There is no usefully definable result.
    S.FFDiag(E);
    return false;
  }

  return true;
}

bool CheckDynamicMemoryAllocation(InterpState &S, CodePtr OpPC) {
  if (S.getLangOpts().CPlusPlus20)
    return true;

  const SourceInfo &E = S.Current->getSource(OpPC);
  S.CCEDiag(E, diag::note_constexpr_new);
  return true;
}

bool CheckNewDeleteForms(InterpState &S, CodePtr OpPC,
                         DynamicAllocator::Form AllocForm,
                         DynamicAllocator::Form DeleteForm, const Descriptor *D,
                         const Expr *NewExpr) {
  if (AllocForm == DeleteForm)
    return true;

  QualType TypeToDiagnose = D->getDataType(S.getASTContext());

  const SourceInfo &E = S.Current->getSource(OpPC);
  S.FFDiag(E, diag::note_constexpr_new_delete_mismatch)
      << static_cast<int>(DeleteForm) << static_cast<int>(AllocForm)
      << TypeToDiagnose;
  S.Note(NewExpr->getExprLoc(), diag::note_constexpr_dynamic_alloc_here)
      << NewExpr->getSourceRange();
  return false;
}

bool CheckDeleteSource(InterpState &S, CodePtr OpPC, const Expr *Source,
                       const Pointer &Ptr) {
  // Regular new type(...) call.
  if (isa_and_nonnull<CXXNewExpr>(Source))
    return true;
  // operator new.
  if (const auto *CE = dyn_cast_if_present<CallExpr>(Source);
      CE && CE->getBuiltinCallee() == Builtin::BI__builtin_operator_new)
    return true;
  // std::allocator.allocate() call
  if (const auto *MCE = dyn_cast_if_present<CXXMemberCallExpr>(Source);
      MCE && MCE->getMethodDecl()->getIdentifier()->isStr("allocate"))
    return true;

  // Whatever this is, we didn't heap allocate it.
  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_delete_not_heap_alloc)
      << Ptr.toDiagnosticString(S.getASTContext());

  if (Ptr.isTemporary())
    S.Note(Ptr.getDeclLoc(), diag::note_constexpr_temporary_here);
  else
    S.Note(Ptr.getDeclLoc(), diag::note_declared_at);
  return false;
}

/// We aleady know the given DeclRefExpr is invalid for some reason,
/// now figure out why and print appropriate diagnostics.
bool CheckDeclRef(InterpState &S, CodePtr OpPC, const DeclRefExpr *DR) {
  const ValueDecl *D = DR->getDecl();
  return diagnoseUnknownDecl(S, OpPC, D);
}

bool CheckDummy(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                AccessKinds AK) {
  if (!Ptr.isDummy())
    return true;

  const Descriptor *Desc = Ptr.getDeclDesc();
  const ValueDecl *D = Desc->asValueDecl();
  if (!D)
    return false;

  if (AK == AK_Read || AK == AK_Increment || AK == AK_Decrement)
    return diagnoseUnknownDecl(S, OpPC, D);

  if (AK == AK_Destroy || S.getLangOpts().CPlusPlus14) {
    const SourceInfo &E = S.Current->getSource(OpPC);
    S.FFDiag(E, diag::note_constexpr_modify_global);
  }
  return false;
}

bool CheckNonNullArgs(InterpState &S, CodePtr OpPC, const Function *F,
                      const CallExpr *CE, unsigned ArgSize) {
  auto Args = llvm::ArrayRef(CE->getArgs(), CE->getNumArgs());
  auto NonNullArgs = collectNonNullArgs(F->getDecl(), Args);
  unsigned Offset = 0;
  unsigned Index = 0;
  for (const Expr *Arg : Args) {
    if (NonNullArgs[Index] && Arg->getType()->isPointerType()) {
      const Pointer &ArgPtr = S.Stk.peek<Pointer>(ArgSize - Offset);
      if (ArgPtr.isZero()) {
        const SourceLocation &Loc = S.Current->getLocation(OpPC);
        S.CCEDiag(Loc, diag::note_non_null_attribute_failed);
        return false;
      }
    }

    Offset += align(primSize(S.Ctx.classify(Arg).value_or(PT_Ptr)));
    ++Index;
  }
  return true;
}

static bool runRecordDestructor(InterpState &S, CodePtr OpPC,
                                const Pointer &BasePtr,
                                const Descriptor *Desc) {
  assert(Desc->isRecord());
  const Record *R = Desc->ElemRecord;
  assert(R);

  if (Pointer::pointToSameBlock(BasePtr, S.Current->getThis()) &&
      S.Current->getFunction()->isDestructor()) {
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_constexpr_double_destroy);
    return false;
  }

  // Destructor of this record.
  if (const CXXDestructorDecl *Dtor = R->getDestructor();
      Dtor && !Dtor->isTrivial()) {
    const Function *DtorFunc = S.getContext().getOrCreateFunction(Dtor);
    if (!DtorFunc)
      return false;

    S.Stk.push<Pointer>(BasePtr);
    if (!Call(S, OpPC, DtorFunc, 0))
      return false;
  }
  return true;
}

static bool RunDestructors(InterpState &S, CodePtr OpPC, const Block *B) {
  assert(B);
  const Descriptor *Desc = B->getDescriptor();

  if (Desc->isPrimitive() || Desc->isPrimitiveArray())
    return true;

  assert(Desc->isRecord() || Desc->isCompositeArray());

  if (Desc->isCompositeArray()) {
    unsigned N = Desc->getNumElems();
    if (N == 0)
      return true;
    const Descriptor *ElemDesc = Desc->ElemDesc;
    assert(ElemDesc->isRecord());

    Pointer RP(const_cast<Block *>(B));
    for (int I = static_cast<int>(N) - 1; I >= 0; --I) {
      if (!runRecordDestructor(S, OpPC, RP.atIndex(I).narrow(), ElemDesc))
        return false;
    }
    return true;
  }

  assert(Desc->isRecord());
  return runRecordDestructor(S, OpPC, Pointer(const_cast<Block *>(B)), Desc);
}

static bool hasVirtualDestructor(QualType T) {
  if (const CXXRecordDecl *RD = T->getAsCXXRecordDecl())
    if (const CXXDestructorDecl *DD = RD->getDestructor())
      return DD->isVirtual();
  return false;
}

bool Free(InterpState &S, CodePtr OpPC, bool DeleteIsArrayForm,
          bool IsGlobalDelete) {
  if (!CheckDynamicMemoryAllocation(S, OpPC))
    return false;

  const Expr *Source = nullptr;
  const Block *BlockToDelete = nullptr;
  {
    // Extra scope for this so the block doesn't have this pointer
    // pointing to it when we destroy it.
    Pointer Ptr = S.Stk.pop<Pointer>();

    // Deleteing nullptr is always fine.
    if (Ptr.isZero())
      return true;

    // Remove base casts.
    QualType InitialType = Ptr.getType();
    while (Ptr.isBaseClass())
      Ptr = Ptr.getBase();

    // For the non-array case, the types must match if the static type
    // does not have a virtual destructor.
    if (!DeleteIsArrayForm && Ptr.getType() != InitialType &&
        !hasVirtualDestructor(InitialType)) {
      S.FFDiag(S.Current->getSource(OpPC),
               diag::note_constexpr_delete_base_nonvirt_dtor)
          << InitialType << Ptr.getType();
      return false;
    }

    if (!Ptr.isRoot() || Ptr.isOnePastEnd() ||
        (Ptr.isArrayElement() && Ptr.getIndex() != 0)) {
      const SourceInfo &Loc = S.Current->getSource(OpPC);
      S.FFDiag(Loc, diag::note_constexpr_delete_subobject)
          << Ptr.toDiagnosticString(S.getASTContext()) << Ptr.isOnePastEnd();
      return false;
    }

    Source = Ptr.getDeclDesc()->asExpr();
    BlockToDelete = Ptr.block();

    if (!CheckDeleteSource(S, OpPC, Source, Ptr))
      return false;

    // For a class type with a virtual destructor, the selected operator delete
    // is the one looked up when building the destructor.
    if (!DeleteIsArrayForm && !IsGlobalDelete) {
      QualType AllocType = Ptr.getType();
      auto getVirtualOperatorDelete = [](QualType T) -> const FunctionDecl * {
        if (const CXXRecordDecl *RD = T->getAsCXXRecordDecl())
          if (const CXXDestructorDecl *DD = RD->getDestructor())
            return DD->isVirtual() ? DD->getOperatorDelete() : nullptr;
        return nullptr;
      };

      if (const FunctionDecl *VirtualDelete =
              getVirtualOperatorDelete(AllocType);
          VirtualDelete &&
          !VirtualDelete
               ->isUsableAsGlobalAllocationFunctionInConstantEvaluation()) {
        S.FFDiag(S.Current->getSource(OpPC),
                 diag::note_constexpr_new_non_replaceable)
            << isa<CXXMethodDecl>(VirtualDelete) << VirtualDelete;
        return false;
      }
    }
  }
  assert(Source);
  assert(BlockToDelete);

  // Invoke destructors before deallocating the memory.
  if (!RunDestructors(S, OpPC, BlockToDelete))
    return false;

  DynamicAllocator &Allocator = S.getAllocator();
  const Descriptor *BlockDesc = BlockToDelete->getDescriptor();
  std::optional<DynamicAllocator::Form> AllocForm =
      Allocator.getAllocationForm(Source);

  if (!Allocator.deallocate(Source, BlockToDelete, S)) {
    // Nothing has been deallocated, this must be a double-delete.
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_constexpr_double_delete);
    return false;
  }

  assert(AllocForm);
  DynamicAllocator::Form DeleteForm = DeleteIsArrayForm
                                          ? DynamicAllocator::Form::Array
                                          : DynamicAllocator::Form::NonArray;
  return CheckNewDeleteForms(S, OpPC, *AllocForm, DeleteForm, BlockDesc,
                             Source);
}

void diagnoseEnumValue(InterpState &S, CodePtr OpPC, const EnumDecl *ED,
                       const APSInt &Value) {
  if (S.EvaluatingDecl && !S.EvaluatingDecl->isConstexpr())
    return;

  llvm::APInt Min;
  llvm::APInt Max;
  ED->getValueRange(Max, Min);
  --Max;

  if (ED->getNumNegativeBits() &&
      (Max.slt(Value.getSExtValue()) || Min.sgt(Value.getSExtValue()))) {
    const SourceLocation &Loc = S.Current->getLocation(OpPC);
    S.CCEDiag(Loc, diag::note_constexpr_unscoped_enum_out_of_range)
        << llvm::toString(Value, 10) << Min.getSExtValue() << Max.getSExtValue()
        << ED;
  } else if (!ED->getNumNegativeBits() && Max.ult(Value.getZExtValue())) {
    const SourceLocation &Loc = S.Current->getLocation(OpPC);
    S.CCEDiag(Loc, diag::note_constexpr_unscoped_enum_out_of_range)
        << llvm::toString(Value, 10) << Min.getZExtValue() << Max.getZExtValue()
        << ED;
  }
}

bool CheckLiteralType(InterpState &S, CodePtr OpPC, const Type *T) {
  assert(T);
  assert(!S.getLangOpts().CPlusPlus23);

  // C++1y: A constant initializer for an object o [...] may also invoke
  // constexpr constructors for o and its subobjects even if those objects
  // are of non-literal class types.
  //
  // C++11 missed this detail for aggregates, so classes like this:
  //   struct foo_t { union { int i; volatile int j; } u; };
  // are not (obviously) initializable like so:
  //   __attribute__((__require_constant_initialization__))
  //   static const foo_t x = {{0}};
  // because "i" is a subobject with non-literal initialization (due to the
  // volatile member of the union). See:
  //   http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#1677
  // Therefore, we use the C++1y behavior.

  if (S.Current->getFunction() && S.Current->getFunction()->isConstructor() &&
      S.Current->getThis().getDeclDesc()->asDecl() == S.EvaluatingDecl) {
    return true;
  }

  const Expr *E = S.Current->getExpr(OpPC);
  if (S.getLangOpts().CPlusPlus11)
    S.FFDiag(E, diag::note_constexpr_nonliteral) << E->getType();
  else
    S.FFDiag(E, diag::note_invalid_subexpr_in_const_expr);
  return false;
}

static bool getField(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                     uint32_t Off) {
  if (S.getLangOpts().CPlusPlus && S.inConstantContext() &&
      !CheckNull(S, OpPC, Ptr, CSK_Field))
    return false;

  if (!CheckRange(S, OpPC, Ptr, CSK_Field))
    return false;
  if (!CheckArray(S, OpPC, Ptr))
    return false;
  if (!CheckSubobject(S, OpPC, Ptr, CSK_Field))
    return false;

  if (Ptr.isIntegralPointer()) {
    S.Stk.push<Pointer>(Ptr.asIntPointer().atOffset(S.getASTContext(), Off));
    return true;
  }

  if (!Ptr.isBlockPointer()) {
    // FIXME: The only time we (seem to) get here is when trying to access a
    // field of a typeid pointer. In that case, we're supposed to diagnose e.g.
    // `typeid(int).name`, but we currently diagnose `&typeid(int)`.
    S.FFDiag(S.Current->getSource(OpPC),
             diag::note_constexpr_access_unreadable_object)
        << AK_Read << Ptr.toDiagnosticString(S.getASTContext());
    return false;
  }

  if ((Ptr.getByteOffset() + Off) >= Ptr.block()->getSize())
    return false;

  S.Stk.push<Pointer>(Ptr.atField(Off));
  return true;
}

bool GetPtrField(InterpState &S, CodePtr OpPC, uint32_t Off) {
  const auto &Ptr = S.Stk.peek<Pointer>();
  return getField(S, OpPC, Ptr, Off);
}

bool GetPtrFieldPop(InterpState &S, CodePtr OpPC, uint32_t Off) {
  const auto &Ptr = S.Stk.pop<Pointer>();
  return getField(S, OpPC, Ptr, Off);
}

static bool checkConstructor(InterpState &S, CodePtr OpPC, const Function *Func,
                             const Pointer &ThisPtr) {
  assert(Func->isConstructor());

  if (Func->getParentDecl()->isInvalidDecl())
    return false;

  const Descriptor *D = ThisPtr.getFieldDesc();
  // FIXME: I think this case is not 100% correct. E.g. a pointer into a
  // subobject of a composite array.
  if (!D->ElemRecord)
    return true;

  if (D->ElemRecord->getNumVirtualBases() == 0)
    return true;

  S.FFDiag(S.Current->getLocation(OpPC), diag::note_constexpr_virtual_base)
      << Func->getParentDecl();
  return false;
}

bool CheckDestructor(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!CheckLive(S, OpPC, Ptr, AK_Destroy))
    return false;
  if (!CheckTemporary(S, OpPC, Ptr, AK_Destroy))
    return false;
  if (!CheckRange(S, OpPC, Ptr, AK_Destroy))
    return false;

  // Can't call a dtor on a global variable.
  if (Ptr.block()->isStatic()) {
    const SourceInfo &E = S.Current->getSource(OpPC);
    S.FFDiag(E, diag::note_constexpr_modify_global);
    return false;
  }
  return CheckActive(S, OpPC, Ptr, AK_Destroy);
}

static void compileFunction(InterpState &S, const Function *Func) {
  Compiler<ByteCodeEmitter>(S.getContext(), S.P)
      .compileFunc(Func->getDecl()->getMostRecentDecl(),
                   const_cast<Function *>(Func));
}

bool CallVar(InterpState &S, CodePtr OpPC, const Function *Func,
             uint32_t VarArgSize) {
  if (Func->hasThisPointer()) {
    size_t ArgSize = Func->getArgSize() + VarArgSize;
    size_t ThisOffset = ArgSize - (Func->hasRVO() ? primSize(PT_Ptr) : 0);
    const Pointer &ThisPtr = S.Stk.peek<Pointer>(ThisOffset);

    // If the current function is a lambda static invoker and
    // the function we're about to call is a lambda call operator,
    // skip the CheckInvoke, since the ThisPtr is a null pointer
    // anyway.
    if (!(S.Current->getFunction() &&
          S.Current->getFunction()->isLambdaStaticInvoker() &&
          Func->isLambdaCallOperator())) {
      if (!CheckInvoke(S, OpPC, ThisPtr))
        return false;
    }

    if (S.checkingPotentialConstantExpression())
      return false;
  }

  if (!Func->isFullyCompiled())
    compileFunction(S, Func);

  if (!CheckCallable(S, OpPC, Func))
    return false;

  if (!CheckCallDepth(S, OpPC))
    return false;

  auto NewFrame = std::make_unique<InterpFrame>(S, Func, OpPC, VarArgSize);
  InterpFrame *FrameBefore = S.Current;
  S.Current = NewFrame.get();

  // Note that we cannot assert(CallResult.hasValue()) here since
  // Ret() above only sets the APValue if the curent frame doesn't
  // have a caller set.
  if (Interpret(S)) {
    NewFrame.release(); // Frame was delete'd already.
    assert(S.Current == FrameBefore);
    return true;
  }

  // Interpreting the function failed somehow. Reset to
  // previous state.
  S.Current = FrameBefore;
  return false;
}
bool Call(InterpState &S, CodePtr OpPC, const Function *Func,
          uint32_t VarArgSize) {
  assert(Func);
  auto cleanup = [&]() -> bool {
    cleanupAfterFunctionCall(S, OpPC, Func);
    return false;
  };

  if (Func->hasThisPointer()) {
    size_t ArgSize = Func->getArgSize() + VarArgSize;
    size_t ThisOffset = ArgSize - (Func->hasRVO() ? primSize(PT_Ptr) : 0);

    const Pointer &ThisPtr = S.Stk.peek<Pointer>(ThisOffset);

    // C++23 [expr.const]p5.6
    // an invocation of a virtual function ([class.virtual]) for an object whose
    // dynamic type is constexpr-unknown;
    if (ThisPtr.isDummy() && Func->isVirtual())
      return false;

    // If the current function is a lambda static invoker and
    // the function we're about to call is a lambda call operator,
    // skip the CheckInvoke, since the ThisPtr is a null pointer
    // anyway.
    if (S.Current->getFunction() &&
        S.Current->getFunction()->isLambdaStaticInvoker() &&
        Func->isLambdaCallOperator()) {
      assert(ThisPtr.isZero());
    } else {
      if (!CheckInvoke(S, OpPC, ThisPtr))
        return cleanup();
      if (!Func->isConstructor() && !Func->isDestructor() &&
          !Func->isCopyOrMoveOperator() &&
          !CheckActive(S, OpPC, ThisPtr, AK_MemberCall))
        return false;
    }

    if (Func->isConstructor() && !checkConstructor(S, OpPC, Func, ThisPtr))
      return false;
    if (Func->isDestructor() && !CheckDestructor(S, OpPC, ThisPtr))
      return false;
  }

  if (!Func->isFullyCompiled())
    compileFunction(S, Func);

  if (!CheckCallable(S, OpPC, Func))
    return cleanup();

  // FIXME: The isConstructor() check here is not always right. The current
  // constant evaluator is somewhat inconsistent in when it allows a function
  // call when checking for a constant expression.
  if (Func->hasThisPointer() && S.checkingPotentialConstantExpression() &&
      !Func->isConstructor())
    return cleanup();

  if (!CheckCallDepth(S, OpPC))
    return cleanup();

  auto NewFrame = std::make_unique<InterpFrame>(S, Func, OpPC, VarArgSize);
  InterpFrame *FrameBefore = S.Current;
  S.Current = NewFrame.get();

  InterpStateCCOverride CCOverride(S, Func->isImmediate());
  // Note that we cannot assert(CallResult.hasValue()) here since
  // Ret() above only sets the APValue if the curent frame doesn't
  // have a caller set.
  if (Interpret(S)) {
    NewFrame.release(); // Frame was delete'd already.
    assert(S.Current == FrameBefore);
    return true;
  }

  // Interpreting the function failed somehow. Reset to
  // previous state.
  S.Current = FrameBefore;
  return false;
}

bool CallVirt(InterpState &S, CodePtr OpPC, const Function *Func,
              uint32_t VarArgSize) {
  assert(Func->hasThisPointer());
  assert(Func->isVirtual());
  size_t ArgSize = Func->getArgSize() + VarArgSize;
  size_t ThisOffset = ArgSize - (Func->hasRVO() ? primSize(PT_Ptr) : 0);
  Pointer &ThisPtr = S.Stk.peek<Pointer>(ThisOffset);
  const FunctionDecl *Callee = Func->getDecl();

  if (!Func->isFullyCompiled())
    compileFunction(S, Func);

  // C++2a [class.abstract]p6:
  //   the effect of making a virtual call to a pure virtual function [...] is
  //   undefined
  if (Callee->isPureVirtual()) {
    S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_pure_virtual_call,
             1)
        << Callee;
    S.Note(Callee->getLocation(), diag::note_declared_at);
    return false;
  }

  const CXXRecordDecl *DynamicDecl = nullptr;
  {
    Pointer TypePtr = ThisPtr;
    while (TypePtr.isBaseClass())
      TypePtr = TypePtr.getBase();

    QualType DynamicType = TypePtr.getType();
    if (DynamicType->isPointerType() || DynamicType->isReferenceType())
      DynamicDecl = DynamicType->getPointeeCXXRecordDecl();
    else
      DynamicDecl = DynamicType->getAsCXXRecordDecl();
  }
  assert(DynamicDecl);

  const auto *StaticDecl = cast<CXXRecordDecl>(Func->getParentDecl());
  const auto *InitialFunction = cast<CXXMethodDecl>(Callee);
  const CXXMethodDecl *Overrider = S.getContext().getOverridingFunction(
      DynamicDecl, StaticDecl, InitialFunction);

  if (Overrider != InitialFunction) {
    // DR1872: An instantiated virtual constexpr function can't be called in a
    // constant expression (prior to C++20). We can still constant-fold such a
    // call.
    if (!S.getLangOpts().CPlusPlus20 && Overrider->isVirtual()) {
      const Expr *E = S.Current->getExpr(OpPC);
      S.CCEDiag(E, diag::note_constexpr_virtual_call) << E->getSourceRange();
    }

    Func = S.getContext().getOrCreateFunction(Overrider);

    const CXXRecordDecl *ThisFieldDecl =
        ThisPtr.getFieldDesc()->getType()->getAsCXXRecordDecl();
    if (Func->getParentDecl()->isDerivedFrom(ThisFieldDecl)) {
      // If the function we call is further DOWN the hierarchy than the
      // FieldDesc of our pointer, just go up the hierarchy of this field
      // the furthest we can go.
      while (ThisPtr.isBaseClass())
        ThisPtr = ThisPtr.getBase();
    }
  }

  if (!Call(S, OpPC, Func, VarArgSize))
    return false;

  // Covariant return types. The return type of Overrider is a pointer
  // or reference to a class type.
  if (Overrider != InitialFunction &&
      Overrider->getReturnType()->isPointerOrReferenceType() &&
      InitialFunction->getReturnType()->isPointerOrReferenceType()) {
    QualType OverriderPointeeType =
        Overrider->getReturnType()->getPointeeType();
    QualType InitialPointeeType =
        InitialFunction->getReturnType()->getPointeeType();
    // We've called Overrider above, but calling code expects us to return what
    // InitialFunction returned. According to the rules for covariant return
    // types, what InitialFunction returns needs to be a base class of what
    // Overrider returns. So, we need to do an upcast here.
    unsigned Offset = S.getContext().collectBaseOffset(
        InitialPointeeType->getAsRecordDecl(),
        OverriderPointeeType->getAsRecordDecl());
    return GetPtrBasePop(S, OpPC, Offset, /*IsNullOK=*/true);
  }

  return true;
}

bool CallBI(InterpState &S, CodePtr OpPC, const CallExpr *CE,
            uint32_t BuiltinID) {
  // A little arbitrary, but the current interpreter allows evaluation
  // of builtin functions in this mode, with some exceptions.
  if (BuiltinID == Builtin::BI__builtin_operator_new &&
      S.checkingPotentialConstantExpression())
    return false;

  return InterpretBuiltin(S, OpPC, CE, BuiltinID);
}

bool CallPtr(InterpState &S, CodePtr OpPC, uint32_t ArgSize,
             const CallExpr *CE) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (Ptr.isZero()) {
    const auto *E = cast<CallExpr>(S.Current->getExpr(OpPC));
    S.FFDiag(E, diag::note_constexpr_null_callee)
        << const_cast<Expr *>(E->getCallee()) << E->getSourceRange();
    return false;
  }

  if (!Ptr.isFunctionPointer())
    return Invalid(S, OpPC);

  const FunctionPointer &FuncPtr = Ptr.asFunctionPointer();
  const Function *F = FuncPtr.getFunction();
  assert(F);
  // Don't allow calling block pointers.
  if (!F->getDecl())
    return Invalid(S, OpPC);

  // This happens when the call expression has been cast to
  // something else, but we don't support that.
  if (S.Ctx.classify(F->getDecl()->getReturnType()) !=
      S.Ctx.classify(CE->getCallReturnType(S.getASTContext())))
    return false;

  // Check argument nullability state.
  if (F->hasNonNullAttr()) {
    if (!CheckNonNullArgs(S, OpPC, F, CE, ArgSize))
      return false;
  }

  assert(ArgSize >= F->getWrittenArgSize());
  uint32_t VarArgSize = ArgSize - F->getWrittenArgSize();

  // We need to do this explicitly here since we don't have the necessary
  // information to do it automatically.
  if (F->isThisPointerExplicit())
    VarArgSize -= align(primSize(PT_Ptr));

  if (F->isVirtual())
    return CallVirt(S, OpPC, F, VarArgSize);

  return Call(S, OpPC, F, VarArgSize);
}

bool CheckNewTypeMismatch(InterpState &S, CodePtr OpPC, const Expr *E,
                          std::optional<uint64_t> ArraySize) {
  const Pointer &Ptr = S.Stk.peek<Pointer>();

  if (!CheckStore(S, OpPC, Ptr))
    return false;

  if (!InvalidNewDeleteExpr(S, OpPC, E))
    return false;

  const auto *NewExpr = cast<CXXNewExpr>(E);
  QualType StorageType = Ptr.getFieldDesc()->getDataType(S.getASTContext());
  const ASTContext &ASTCtx = S.getASTContext();
  QualType AllocType;
  if (ArraySize) {
    AllocType = ASTCtx.getConstantArrayType(
        NewExpr->getAllocatedType(),
        APInt(64, static_cast<uint64_t>(*ArraySize), false), nullptr,
        ArraySizeModifier::Normal, 0);
  } else {
    AllocType = NewExpr->getAllocatedType();
  }

  unsigned StorageSize = 1;
  unsigned AllocSize = 1;
  if (const auto *CAT = dyn_cast<ConstantArrayType>(AllocType))
    AllocSize = CAT->getZExtSize();
  if (const auto *CAT = dyn_cast<ConstantArrayType>(StorageType))
    StorageSize = CAT->getZExtSize();

  if (AllocSize > StorageSize ||
      !ASTCtx.hasSimilarType(ASTCtx.getBaseElementType(AllocType),
                             ASTCtx.getBaseElementType(StorageType))) {
    S.FFDiag(S.Current->getLocation(OpPC),
             diag::note_constexpr_placement_new_wrong_type)
        << StorageType << AllocType;
    return false;
  }

  // Can't activate fields in a union, unless the direct base is the union.
  if (Ptr.inUnion() && !Ptr.isActive() && !Ptr.getBase().getRecord()->isUnion())
    return CheckActive(S, OpPC, Ptr, AK_Construct);

  return true;
}

bool InvalidNewDeleteExpr(InterpState &S, CodePtr OpPC, const Expr *E) {
  assert(E);

  if (const auto *NewExpr = dyn_cast<CXXNewExpr>(E)) {
    const FunctionDecl *OperatorNew = NewExpr->getOperatorNew();

    if (NewExpr->getNumPlacementArgs() > 0) {
      // This is allowed pre-C++26, but only an std function.
      if (S.getLangOpts().CPlusPlus26 || S.Current->isStdFunction())
        return true;
      S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_new_placement)
          << /*C++26 feature*/ 1 << E->getSourceRange();
    } else if (
        !OperatorNew
             ->isUsableAsGlobalAllocationFunctionInConstantEvaluation()) {
      S.FFDiag(S.Current->getSource(OpPC),
               diag::note_constexpr_new_non_replaceable)
          << isa<CXXMethodDecl>(OperatorNew) << OperatorNew;
      return false;
    } else if (!S.getLangOpts().CPlusPlus26 &&
               NewExpr->getNumPlacementArgs() == 1 &&
               !OperatorNew->isReservedGlobalPlacementOperator()) {
      if (!S.getLangOpts().CPlusPlus26) {
        S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_new_placement)
            << /*Unsupported*/ 0 << E->getSourceRange();
        return false;
      }
      return true;
    }
  } else {
    const auto *DeleteExpr = cast<CXXDeleteExpr>(E);
    const FunctionDecl *OperatorDelete = DeleteExpr->getOperatorDelete();
    if (!OperatorDelete
             ->isUsableAsGlobalAllocationFunctionInConstantEvaluation()) {
      S.FFDiag(S.Current->getSource(OpPC),
               diag::note_constexpr_new_non_replaceable)
          << isa<CXXMethodDecl>(OperatorDelete) << OperatorDelete;
      return false;
    }
  }

  return false;
}

bool handleFixedPointOverflow(InterpState &S, CodePtr OpPC,
                              const FixedPoint &FP) {
  const Expr *E = S.Current->getExpr(OpPC);
  if (S.checkingForUndefinedBehavior()) {
    S.getASTContext().getDiagnostics().Report(
        E->getExprLoc(), diag::warn_fixedpoint_constant_overflow)
        << FP.toDiagnosticString(S.getASTContext()) << E->getType();
  }
  S.CCEDiag(E, diag::note_constexpr_overflow)
      << FP.toDiagnosticString(S.getASTContext()) << E->getType();
  return S.noteUndefinedBehavior();
}

bool InvalidShuffleVectorIndex(InterpState &S, CodePtr OpPC, uint32_t Index) {
  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc,
           diag::err_shufflevector_minus_one_is_undefined_behavior_constexpr)
      << Index;
  return false;
}

bool CheckPointerToIntegralCast(InterpState &S, CodePtr OpPC,
                                const Pointer &Ptr, unsigned BitWidth) {
  if (Ptr.isDummy())
    return false;
  if (Ptr.isFunctionPointer())
    return true;

  const SourceInfo &E = S.Current->getSource(OpPC);
  S.CCEDiag(E, diag::note_constexpr_invalid_cast)
      << 2 << S.getLangOpts().CPlusPlus << S.Current->getRange(OpPC);

  if (Ptr.isBlockPointer() && !Ptr.isZero()) {
    // Only allow based lvalue casts if they are lossless.
    if (S.getASTContext().getTargetInfo().getPointerWidth(LangAS::Default) !=
        BitWidth)
      return Invalid(S, OpPC);
  }
  return true;
}

bool CastPointerIntegralAP(InterpState &S, CodePtr OpPC, uint32_t BitWidth) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckPointerToIntegralCast(S, OpPC, Ptr, BitWidth))
    return false;

  S.Stk.push<IntegralAP<false>>(
      IntegralAP<false>::from(Ptr.getIntegerRepresentation(), BitWidth));
  return true;
}

bool CastPointerIntegralAPS(InterpState &S, CodePtr OpPC, uint32_t BitWidth) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckPointerToIntegralCast(S, OpPC, Ptr, BitWidth))
    return false;

  S.Stk.push<IntegralAP<true>>(
      IntegralAP<true>::from(Ptr.getIntegerRepresentation(), BitWidth));
  return true;
}

bool CheckBitCast(InterpState &S, CodePtr OpPC, bool HasIndeterminateBits,
                  bool TargetIsUCharOrByte) {
  // This is always fine.
  if (!HasIndeterminateBits)
    return true;

  // Indeterminate bits can only be bitcast to unsigned char or std::byte.
  if (TargetIsUCharOrByte)
    return true;

  const Expr *E = S.Current->getExpr(OpPC);
  QualType ExprType = E->getType();
  S.FFDiag(E, diag::note_constexpr_bit_cast_indet_dest)
      << ExprType << S.getLangOpts().CharIsSigned << E->getSourceRange();
  return false;
}

bool GetTypeid(InterpState &S, CodePtr OpPC, const Type *TypePtr,
               const Type *TypeInfoType) {
  S.Stk.push<Pointer>(TypePtr, TypeInfoType);
  return true;
}

bool GetTypeidPtr(InterpState &S, CodePtr OpPC, const Type *TypeInfoType) {
  const auto &P = S.Stk.pop<Pointer>();

  if (!P.isBlockPointer())
    return false;

  // Pick the most-derived type.
  const Type *T = P.getDeclPtr().getType().getTypePtr();
  // ... unless we're currently constructing this object.
  // FIXME: We have a similar check to this in more places.
  if (S.Current->getFunction()) {
    for (const InterpFrame *Frame = S.Current; Frame; Frame = Frame->Caller) {
      if (const Function *Func = Frame->getFunction();
          Func && (Func->isConstructor() || Func->isDestructor()) &&
          P.block() == Frame->getThis().block()) {
        T = Func->getParentDecl()->getTypeForDecl();
        break;
      }
    }
  }

  S.Stk.push<Pointer>(T->getCanonicalTypeUnqualified().getTypePtr(),
                      TypeInfoType);
  return true;
}

bool DiagTypeid(InterpState &S, CodePtr OpPC) {
  const auto *E = cast<CXXTypeidExpr>(S.Current->getExpr(OpPC));
  S.CCEDiag(E, diag::note_constexpr_typeid_polymorphic)
      << E->getExprOperand()->getType()
      << E->getExprOperand()->getSourceRange();
  return false;
}

// https://github.com/llvm/llvm-project/issues/102513
#if defined(_MSC_VER) && !defined(__clang__) && !defined(NDEBUG)
#pragma optimize("", off)
#endif
bool Interpret(InterpState &S) {
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
// https://github.com/llvm/llvm-project/issues/102513
#if defined(_MSC_VER) && !defined(__clang__) && !defined(NDEBUG)
#pragma optimize("", on)
#endif

} // namespace interp
} // namespace clang
