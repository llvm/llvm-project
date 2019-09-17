//===--- Context.cpp - Context for the constexpr VM -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "ByteCodeEmitter.h"
#include "ByteCodeExprGen.h"
#include "ByteCodeStmtGen.h"
#include "EvalEmitter.h"
#include "Interp.h"
#include "InterpFrame.h"
#include "InterpStack.h"
#include "PrimType.h"
#include "Program.h"
#include "clang/AST/Expr.h"

using namespace clang;
using namespace clang::interp;

Context::Context(ASTContext &Ctx)
    : Ctx(Ctx), ForceInterp(getLangOpts().ForceNewConstInterp),
      P(new Program(*this)) {}

Context::~Context() {}

InterpResult Context::isPotentialConstantExpr(State &Parent,
                                              const FunctionDecl *FD) {
  Function *Func = P->getFunction(FD);
  if (!Func) {
    if (auto R = ByteCodeStmtGen<ByteCodeEmitter>(*this, *P).compileFunc(FD)) {
      Func = *R;
    } else if (ForceInterp) {
      handleAllErrors(R.takeError(), [&Parent](ByteCodeGenError &Err) {
        Parent.FFDiag(Err.getLoc(), diag::err_experimental_clang_interp_failed);
      });
      return InterpResult::Fail;
    } else {
      consumeError(R.takeError());
      return InterpResult::Bail;
    }
  }

  if (!Func->isConstexpr())
    return InterpResult::Fail;

  APValue Dummy;
  return Run(Parent, Func, Dummy);
}

InterpResult Context::evaluateAsRValue(State &Parent, const Expr *E,
                                       APValue &Result) {
  ByteCodeExprGen<EvalEmitter> C(*this, *P, Parent, Stk, Result);
  return Check(Parent, C.interpretExpr(E));
}

InterpResult Context::evaluateAsInitializer(State &Parent, const VarDecl *VD,
                                            APValue &Result) {
  ByteCodeExprGen<EvalEmitter> C(*this, *P, Parent, Stk, Result);
  return Check(Parent, C.interpretDecl(VD));
}

const LangOptions &Context::getLangOpts() const { return Ctx.getLangOpts(); }

llvm::Optional<PrimType> Context::classify(QualType T) {
  if (T->isReferenceType() || T->isPointerType()) {
    return PT_Ptr;
  }

  if (T->isBooleanType())
    return PT_Bool;

  if (T->isSignedIntegerOrEnumerationType()) {
    switch (Ctx.getIntWidth(T)) {
    case 64:
      return PT_Sint64;
    case 32:
      return PT_Sint32;
    case 16:
      return PT_Sint16;
    case 8:
      return PT_Sint8;
    default:
      return {};
    }
  }

  if (T->isUnsignedIntegerOrEnumerationType()) {
    switch (Ctx.getIntWidth(T)) {
    case 64:
      return PT_Uint64;
    case 32:
      return PT_Uint32;
    case 16:
      return PT_Uint16;
    case 8:
      return PT_Uint8;
    default:
      return {};
    }
  }

  if (T->isNullPtrType())
    return PT_Ptr;

  if (auto *AT = dyn_cast<AtomicType>(T))
    return classify(AT->getValueType());

  return {};
}

unsigned Context::getCharBit() const {
  return Ctx.getTargetInfo().getCharWidth();
}

InterpResult Context::Run(State &Parent, Function *Func, APValue &Result) {
  InterpResult Flag;
  {
    InterpState State(Parent, *P, Stk, *this);
    State.Current = new InterpFrame(State, Func, nullptr, {}, {});
    if (Interpret(State, Result)) {
      Flag = InterpResult::Success;
    } else {
      Flag = InterpResult::Fail;
    }
  }

  if (Flag != InterpResult::Success)
    Stk.clear();
  return Flag;
}

InterpResult Context::Check(State &Parent, llvm::Expected<bool> &&R) {
  if (R) {
    return *R ? InterpResult::Success : InterpResult::Fail;
  } else if (ForceInterp) {
    handleAllErrors(R.takeError(), [&Parent](ByteCodeGenError &Err) {
      Parent.FFDiag(Err.getLoc(), diag::err_experimental_clang_interp_failed);
    });
    return InterpResult::Fail;
  } else {
    consumeError(R.takeError());
    return InterpResult::Bail;
  }
}
