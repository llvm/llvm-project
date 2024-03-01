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
#include "ByteCodeGenError.h"
#include "ByteCodeStmtGen.h"
#include "EvalEmitter.h"
#include "Interp.h"
#include "InterpFrame.h"
#include "InterpStack.h"
#include "PrimType.h"
#include "Program.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"

using namespace clang;
using namespace clang::interp;

Context::Context(ASTContext &Ctx) : Ctx(Ctx), P(new Program(*this)) {}

Context::~Context() {}

bool Context::isPotentialConstantExpr(State &Parent, const FunctionDecl *FD) {
  assert(Stk.empty());
  Function *Func = P->getFunction(FD);
  if (!Func || !Func->hasBody())
    Func = ByteCodeStmtGen<ByteCodeEmitter>(*this, *P).compileFunc(FD);

  APValue DummyResult;
  if (!Run(Parent, Func, DummyResult))
    return false;

  return Func->isConstexpr();
}

bool Context::evaluateAsRValue(State &Parent, const Expr *E, APValue &Result) {
  bool Recursing = !Stk.empty();
  ByteCodeExprGen<EvalEmitter> C(*this, *P, Parent, Stk);

  auto Res = C.interpretExpr(E, /*ConvertResultToRValue=*/E->isGLValue());

  if (Res.isInvalid()) {
    Stk.clear();
    return false;
  }

  if (!Recursing) {
    assert(Stk.empty());
#ifndef NDEBUG
    // Make sure we don't rely on some value being still alive in
    // InterpStack memory.
    Stk.clear();
#endif
  }

  Result = Res.toAPValue();

  return true;
}

bool Context::evaluate(State &Parent, const Expr *E, APValue &Result) {
  bool Recursing = !Stk.empty();
  ByteCodeExprGen<EvalEmitter> C(*this, *P, Parent, Stk);

  auto Res = C.interpretExpr(E);
  if (Res.isInvalid()) {
    Stk.clear();
    return false;
  }

  if (!Recursing) {
    assert(Stk.empty());
#ifndef NDEBUG
    // Make sure we don't rely on some value being still alive in
    // InterpStack memory.
    Stk.clear();
#endif
  }

  Result = Res.toAPValue();
  return true;
}

bool Context::evaluateAsInitializer(State &Parent, const VarDecl *VD,
                                    APValue &Result) {
  bool Recursing = !Stk.empty();
  ByteCodeExprGen<EvalEmitter> C(*this, *P, Parent, Stk);

  bool CheckGlobalInitialized =
      shouldBeGloballyIndexed(VD) &&
      (VD->getType()->isRecordType() || VD->getType()->isArrayType());
  auto Res = C.interpretDecl(VD, CheckGlobalInitialized);
  if (Res.isInvalid()) {
    Stk.clear();
    return false;
  }

  if (!Recursing) {
    assert(Stk.empty());
#ifndef NDEBUG
    // Make sure we don't rely on some value being still alive in
    // InterpStack memory.
    Stk.clear();
#endif
  }

  Result = Res.toAPValue();
  return true;
}

const LangOptions &Context::getLangOpts() const { return Ctx.getLangOpts(); }

std::optional<PrimType> Context::classify(QualType T) const {
  if (T->isBooleanType())
    return PT_Bool;

  if (T->isAnyComplexType())
    return std::nullopt;

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
      return PT_IntAPS;
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
      return PT_IntAP;
    }
  }

  if (T->isNullPtrType())
    return PT_Ptr;

  if (T->isFloatingType())
    return PT_Float;

  if (T->isFunctionPointerType() || T->isFunctionReferenceType() ||
      T->isFunctionType() || T->isSpecificBuiltinType(BuiltinType::BoundMember))
    return PT_FnPtr;

  if (T->isReferenceType() || T->isPointerType())
    return PT_Ptr;

  if (const auto *AT = T->getAs<AtomicType>())
    return classify(AT->getValueType());

  if (const auto *DT = dyn_cast<DecltypeType>(T))
    return classify(DT->getUnderlyingType());

  if (const auto *DT = dyn_cast<MemberPointerType>(T))
    return classify(DT->getPointeeType());

  return std::nullopt;
}

unsigned Context::getCharBit() const {
  return Ctx.getTargetInfo().getCharWidth();
}

/// Simple wrapper around getFloatTypeSemantics() to make code a
/// little shorter.
const llvm::fltSemantics &Context::getFloatSemantics(QualType T) const {
  return Ctx.getFloatTypeSemantics(T);
}

bool Context::Run(State &Parent, const Function *Func, APValue &Result) {

  {
    InterpState State(Parent, *P, Stk, *this);
    State.Current = new InterpFrame(State, Func, /*Caller=*/nullptr, CodePtr(),
                                    Func->getArgSize());
    if (Interpret(State, Result)) {
      assert(Stk.empty());
      return true;
    }

    // State gets destroyed here, so the Stk.clear() below doesn't accidentally
    // remove values the State's destructor might access.
  }

  Stk.clear();
  return false;
}

bool Context::Check(State &Parent, llvm::Expected<bool> &&Flag) {
  if (Flag)
    return *Flag;
  handleAllErrors(Flag.takeError(), [&Parent](ByteCodeGenError &Err) {
    Parent.FFDiag(Err.getRange().getBegin(),
                  diag::err_experimental_clang_interp_failed)
        << Err.getRange();
  });
  return false;
}

// TODO: Virtual bases?
const CXXMethodDecl *
Context::getOverridingFunction(const CXXRecordDecl *DynamicDecl,
                               const CXXRecordDecl *StaticDecl,
                               const CXXMethodDecl *InitialFunction) const {
  assert(DynamicDecl);
  assert(StaticDecl);
  assert(InitialFunction);

  const CXXRecordDecl *CurRecord = DynamicDecl;
  const CXXMethodDecl *FoundFunction = InitialFunction;
  for (;;) {
    const CXXMethodDecl *Overrider =
        FoundFunction->getCorrespondingMethodDeclaredInClass(CurRecord, false);
    if (Overrider)
      return Overrider;

    // Common case of only one base class.
    if (CurRecord->getNumBases() == 1) {
      CurRecord = CurRecord->bases_begin()->getType()->getAsCXXRecordDecl();
      continue;
    }

    // Otherwise, go to the base class that will lead to the StaticDecl.
    for (const CXXBaseSpecifier &Spec : CurRecord->bases()) {
      const CXXRecordDecl *Base = Spec.getType()->getAsCXXRecordDecl();
      if (Base == StaticDecl || Base->isDerivedFrom(StaticDecl)) {
        CurRecord = Base;
        break;
      }
    }
  }

  llvm_unreachable(
      "Couldn't find an overriding function in the class hierarchy?");
  return nullptr;
}

const Function *Context::getOrCreateFunction(const FunctionDecl *FD) {
  assert(FD);
  const Function *Func = P->getFunction(FD);
  bool IsBeingCompiled = Func && Func->isDefined() && !Func->isFullyCompiled();
  bool WasNotDefined = Func && !Func->isConstexpr() && !Func->isDefined();

  if (IsBeingCompiled)
    return Func;

  if (!Func || WasNotDefined) {
    if (auto F = ByteCodeStmtGen<ByteCodeEmitter>(*this, *P).compileFunc(FD))
      Func = F;
  }

  return Func;
}
