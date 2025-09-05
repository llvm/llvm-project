//===--- EvalEmitter.cpp - Instruction emitter for the VM -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EvalEmitter.h"
#include "Context.h"
#include "IntegralAP.h"
#include "Interp.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace clang::interp;

EvalEmitter::EvalEmitter(Context &Ctx, Program &P, State &Parent,
                         InterpStack &Stk)
    : Ctx(Ctx), P(P), S(Parent, P, Stk, Ctx, this), EvalResult(&Ctx) {}

EvalEmitter::~EvalEmitter() {
  for (auto &V : Locals) {
    Block *B = reinterpret_cast<Block *>(V.get());
    if (B->isInitialized())
      B->invokeDtor();
  }
}

/// Clean up all our resources. This needs to done in failed evaluations before
/// we call InterpStack::clear(), because there might be a Pointer on the stack
/// pointing into a Block in the EvalEmitter.
void EvalEmitter::cleanup() { S.cleanup(); }

EvaluationResult EvalEmitter::interpretExpr(const Expr *E,
                                            bool ConvertResultToRValue,
                                            bool DestroyToplevelScope) {
  S.setEvalLocation(E->getExprLoc());
  this->ConvertResultToRValue = ConvertResultToRValue && !isa<ConstantExpr>(E);
  this->CheckFullyInitialized = isa<ConstantExpr>(E);
  EvalResult.setSource(E);

  if (!this->visitExpr(E, DestroyToplevelScope)) {
    // EvalResult may already have a result set, but something failed
    // after that (e.g. evaluating destructors).
    EvalResult.setInvalid();
  }

  return std::move(this->EvalResult);
}

EvaluationResult EvalEmitter::interpretDecl(const VarDecl *VD,
                                            bool CheckFullyInitialized) {
  this->CheckFullyInitialized = CheckFullyInitialized;
  S.EvaluatingDecl = VD;
  S.setEvalLocation(VD->getLocation());
  EvalResult.setSource(VD);

  if (const Expr *Init = VD->getAnyInitializer()) {
    QualType T = VD->getType();
    this->ConvertResultToRValue = !Init->isGLValue() && !T->isPointerType() &&
                                  !T->isObjCObjectPointerType();
  } else
    this->ConvertResultToRValue = false;

  EvalResult.setSource(VD);

  if (!this->visitDeclAndReturn(VD, S.inConstantContext()))
    EvalResult.setInvalid();

  S.EvaluatingDecl = nullptr;
  updateGlobalTemporaries();
  return std::move(this->EvalResult);
}

EvaluationResult EvalEmitter::interpretAsPointer(const Expr *E,
                                                 PtrCallback PtrCB) {

  S.setEvalLocation(E->getExprLoc());
  this->ConvertResultToRValue = false;
  this->CheckFullyInitialized = false;
  this->PtrCB = PtrCB;
  EvalResult.setSource(E);

  if (!this->visitExpr(E, /*DestroyToplevelScope=*/true)) {
    // EvalResult may already have a result set, but something failed
    // after that (e.g. evaluating destructors).
    EvalResult.setInvalid();
  }

  return std::move(this->EvalResult);
}

bool EvalEmitter::interpretCall(const FunctionDecl *FD, const Expr *E) {
  // Add parameters to the parameter map. The values in the ParamOffset don't
  // matter in this case as reading from them can't ever work.
  for (const ParmVarDecl *PD : FD->parameters()) {
    this->Params.insert({PD, {0, false}});
  }

  return this->visitExpr(E, /*DestroyToplevelScope=*/false);
}

void EvalEmitter::emitLabel(LabelTy Label) { CurrentLabel = Label; }

EvalEmitter::LabelTy EvalEmitter::getLabel() { return NextLabel++; }

Scope::Local EvalEmitter::createLocal(Descriptor *D) {
  // Allocate memory for a local.
  auto Memory = std::make_unique<char[]>(sizeof(Block) + D->getAllocSize());
  auto *B = new (Memory.get()) Block(Ctx.getEvalID(), D, /*isStatic=*/false);
  B->invokeCtor();

  // Initialize local variable inline descriptor.
  InlineDescriptor &Desc = *reinterpret_cast<InlineDescriptor *>(B->rawData());
  Desc.Desc = D;
  Desc.Offset = sizeof(InlineDescriptor);
  Desc.IsActive = true;
  Desc.IsBase = false;
  Desc.IsFieldMutable = false;
  Desc.IsConst = false;
  Desc.IsInitialized = false;

  // Register the local.
  unsigned Off = Locals.size();
  Locals.push_back(std::move(Memory));
  return {Off, D};
}

bool EvalEmitter::jumpTrue(const LabelTy &Label) {
  if (isActive()) {
    if (S.Stk.pop<bool>())
      ActiveLabel = Label;
  }
  return true;
}

bool EvalEmitter::jumpFalse(const LabelTy &Label) {
  if (isActive()) {
    if (!S.Stk.pop<bool>())
      ActiveLabel = Label;
  }
  return true;
}

bool EvalEmitter::jump(const LabelTy &Label) {
  if (isActive())
    CurrentLabel = ActiveLabel = Label;
  return true;
}

bool EvalEmitter::fallthrough(const LabelTy &Label) {
  if (isActive())
    ActiveLabel = Label;
  CurrentLabel = Label;
  return true;
}

bool EvalEmitter::speculate(const CallExpr *E, const LabelTy &EndLabel) {
  size_t StackSizeBefore = S.Stk.size();
  const Expr *Arg = E->getArg(0);
  if (!this->visit(Arg)) {
    S.Stk.clearTo(StackSizeBefore);

    if (S.inConstantContext() || Arg->HasSideEffects(S.getASTContext()))
      return this->emitBool(false, E);
    return Invalid(S, OpPC);
  }

  PrimType T = Ctx.classify(Arg->getType()).value_or(PT_Ptr);
  if (T == PT_Ptr) {
    const auto &Ptr = S.Stk.pop<Pointer>();
    return this->emitBool(CheckBCPResult(S, Ptr), E);
  }

  // Otherwise, this is fine!
  if (!this->emitPop(T, E))
    return false;
  return this->emitBool(true, E);
}

template <PrimType OpType> bool EvalEmitter::emitRet(const SourceInfo &Info) {
  if (!isActive())
    return true;

  using T = typename PrimConv<OpType>::T;
  EvalResult.takeValue(S.Stk.pop<T>().toAPValue(Ctx.getASTContext()));
  return true;
}

template <> bool EvalEmitter::emitRet<PT_Ptr>(const SourceInfo &Info) {
  if (!isActive())
    return true;

  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (Ptr.isFunctionPointer()) {
    EvalResult.takeValue(Ptr.toAPValue(Ctx.getASTContext()));
    return true;
  }

  // If we're returning a raw pointer, call our callback.
  if (this->PtrCB)
    return (*this->PtrCB)(Ptr);

  if (!EvalResult.checkReturnValue(S, Ctx, Ptr, Info))
    return false;
  if (CheckFullyInitialized && !EvalResult.checkFullyInitialized(S, Ptr))
    return false;

  // Implicitly convert lvalue to rvalue, if requested.
  if (ConvertResultToRValue) {
    if (!Ptr.isZero() && !Ptr.isDereferencable())
      return false;

    if (Ptr.pointsToStringLiteral() && Ptr.isArrayRoot())
      return false;

    if (!Ptr.isZero() && !CheckFinalLoad(S, OpPC, Ptr))
      return false;

    // Never allow reading from a non-const pointer, unless the memory
    // has been created in this evaluation.
    if (!Ptr.isZero() && !Ptr.isConst() && Ptr.isBlockPointer() &&
        Ptr.block()->getEvalID() != Ctx.getEvalID())
      return false;

    if (std::optional<APValue> V =
            Ptr.toRValue(Ctx, EvalResult.getSourceType())) {
      EvalResult.takeValue(std::move(*V));
    } else {
      return false;
    }
  } else {
    // If this is pointing to a local variable, just return
    // the result, even if the pointer is dead.
    // This will later be diagnosed by CheckLValueConstantExpression.
    if (Ptr.isBlockPointer() && !Ptr.block()->isStatic()) {
      EvalResult.takeValue(Ptr.toAPValue(Ctx.getASTContext()));
      return true;
    }

    if (!Ptr.isLive() && !Ptr.isTemporary())
      return false;

    EvalResult.takeValue(Ptr.toAPValue(Ctx.getASTContext()));
  }

  return true;
}

bool EvalEmitter::emitRetVoid(const SourceInfo &Info) {
  EvalResult.setValid();
  return true;
}

bool EvalEmitter::emitRetValue(const SourceInfo &Info) {
  const auto &Ptr = S.Stk.pop<Pointer>();

  if (!EvalResult.checkReturnValue(S, Ctx, Ptr, Info))
    return false;
  if (CheckFullyInitialized && !EvalResult.checkFullyInitialized(S, Ptr))
    return false;

  if (std::optional<APValue> APV =
          Ptr.toRValue(S.getASTContext(), EvalResult.getSourceType())) {
    EvalResult.takeValue(std::move(*APV));
    return true;
  }

  EvalResult.setInvalid();
  return false;
}

bool EvalEmitter::emitGetPtrLocal(uint32_t I, const SourceInfo &Info) {
  if (!isActive())
    return true;

  Block *B = getLocal(I);
  S.Stk.push<Pointer>(B, sizeof(InlineDescriptor));
  return true;
}

template <PrimType OpType>
bool EvalEmitter::emitGetLocal(uint32_t I, const SourceInfo &Info) {
  if (!isActive())
    return true;

  using T = typename PrimConv<OpType>::T;

  Block *B = getLocal(I);

  if (!CheckLocalLoad(S, OpPC, B))
    return false;

  S.Stk.push<T>(*reinterpret_cast<T *>(B->data()));
  return true;
}

template <PrimType OpType>
bool EvalEmitter::emitSetLocal(uint32_t I, const SourceInfo &Info) {
  if (!isActive())
    return true;

  using T = typename PrimConv<OpType>::T;

  Block *B = getLocal(I);
  *reinterpret_cast<T *>(B->data()) = S.Stk.pop<T>();
  InlineDescriptor &Desc = *reinterpret_cast<InlineDescriptor *>(B->rawData());
  Desc.IsInitialized = true;

  return true;
}

bool EvalEmitter::emitDestroy(uint32_t I, const SourceInfo &Info) {
  if (!isActive())
    return true;

  for (auto &Local : Descriptors[I]) {
    Block *B = getLocal(Local.Offset);
    S.deallocate(B);
  }

  return true;
}

/// Global temporaries (LifetimeExtendedTemporary) carry their value
/// around as an APValue, which codegen accesses.
/// We set their value once when creating them, but we don't update it
/// afterwards when code changes it later.
/// This is what we do here.
void EvalEmitter::updateGlobalTemporaries() {
  for (const auto &[E, Temp] : S.SeenGlobalTemporaries) {
    if (std::optional<unsigned> GlobalIndex = P.getGlobal(E)) {
      const Pointer &Ptr = P.getPtrGlobal(*GlobalIndex);
      APValue *Cached = Temp->getOrCreateValue(true);

      if (OptPrimType T = Ctx.classify(E->getType())) {
        TYPE_SWITCH(
            *T, { *Cached = Ptr.deref<T>().toAPValue(Ctx.getASTContext()); });
      } else {
        if (std::optional<APValue> APV =
                Ptr.toRValue(Ctx, Temp->getTemporaryExpr()->getType()))
          *Cached = *APV;
      }
    }
  }
  S.SeenGlobalTemporaries.clear();
}

//===----------------------------------------------------------------------===//
// Opcode evaluators
//===----------------------------------------------------------------------===//

#define GET_EVAL_IMPL
#include "Opcodes.inc"
#undef GET_EVAL_IMPL
