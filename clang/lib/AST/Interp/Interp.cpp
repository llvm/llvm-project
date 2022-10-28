//===------- Interpcpp - Interpreter for the constexpr VM ------*- C++ -*-===//
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

//===----------------------------------------------------------------------===//
// Ret
//===----------------------------------------------------------------------===//

template <PrimType Name, class T = typename PrimConv<Name>::T>
static bool Ret(InterpState &S, CodePtr &PC, APValue &Result) {
  S.CallStackDepth--;
  const T &Ret = S.Stk.pop<T>();

  assert(S.Current->getFrameOffset() == S.Stk.size() && "Invalid frame");
  if (!S.checkingPotentialConstantExpression())
    S.Current->popArgs();

  if (InterpFrame *Caller = S.Current->Caller) {
    PC = S.Current->getRetPC();
    delete S.Current;
    S.Current = Caller;
    S.Stk.push<T>(Ret);
  } else {
    delete S.Current;
    S.Current = nullptr;
    if (!ReturnValue<T>(Ret, Result))
      return false;
  }
  return true;
}

static bool RetVoid(InterpState &S, CodePtr &PC, APValue &Result) {
  S.CallStackDepth--;

  assert(S.Current->getFrameOffset() == S.Stk.size() && "Invalid frame");
  if (!S.checkingPotentialConstantExpression())
    S.Current->popArgs();

  if (InterpFrame *Caller = S.Current->Caller) {
    PC = S.Current->getRetPC();
    delete S.Current;
    S.Current = Caller;
  } else {
    delete S.Current;
    S.Current = nullptr;
  }
  return true;
}

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

static bool CheckInitialized(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                             AccessKinds AK) {
  if (Ptr.isInitialized())
    return true;
  if (!S.checkingPotentialConstantExpression()) {
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_constexpr_access_uninit) << AK << false;
  }
  return false;
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
  Record *R = U.getRecord();
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

bool CheckExtern(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (!Ptr.isExtern())
    return true;

  if (!S.checkingPotentialConstantExpression()) {
    auto *VD = Ptr.getDeclDesc()->asValueDecl();
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
  const auto &Src = S.Current->getSource(OpPC);
  if (Ptr.isZero()) {

    if (Ptr.isField())
      S.FFDiag(Src, diag::note_constexpr_null_subobject) << CSK_Field;
    else
      S.FFDiag(Src, diag::note_constexpr_access_null) << AK;

    return false;
  }

  if (!Ptr.isLive()) {
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

bool CheckConst(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  assert(Ptr.isLive() && "Pointer is not live");
  if (!Ptr.isConst()) {
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

bool CheckCallable(InterpState &S, CodePtr OpPC, Function *F) {
  const SourceLocation &Loc = S.Current->getLocation(OpPC);

  if (F->isVirtual()) {
    if (!S.getLangOpts().CPlusPlus20) {
      S.CCEDiag(Loc, diag::note_constexpr_virtual_call);
      return false;
    }
  }

  if (!F->isConstexpr()) {
    if (S.getLangOpts().CPlusPlus11) {
      const FunctionDecl *DiagDecl = F->getDecl();

      // If this function is not constexpr because it is an inherited
      // non-constexpr constructor, diagnose that directly.
      auto *CD = dyn_cast<CXXConstructorDecl>(DiagDecl);
      if (CD && CD->isInheritingConstructor()) {
        auto *Inherited = CD->getInheritedConstructor().getConstructor();
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

bool CheckThis(InterpState &S, CodePtr OpPC, const Pointer &This) {
  if (!This.isZero())
    return true;

  const SourceInfo &Loc = S.Current->getSource(OpPC);

  bool IsImplicit = false;
  if (auto *E = dyn_cast_or_null<CXXThisExpr>(Loc.asExpr()))
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
