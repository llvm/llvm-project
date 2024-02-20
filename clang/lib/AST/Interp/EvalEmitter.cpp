//===--- EvalEmitter.cpp - Instruction emitter for the VM -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EvalEmitter.h"
#include "ByteCodeGenError.h"
#include "Context.h"
#include "IntegralAP.h"
#include "Interp.h"
#include "Opcode.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace clang::interp;

EvalEmitter::EvalEmitter(Context &Ctx, Program &P, State &Parent,
                         InterpStack &Stk, APValue &Result)
    : Ctx(Ctx), P(P), S(Parent, P, Stk, Ctx, this), EvalResult(&Ctx) {
  // Create a dummy frame for the interpreter which does not have locals.
  S.Current =
      new InterpFrame(S, /*Func=*/nullptr, /*Caller=*/nullptr, CodePtr(), 0);
}

EvalEmitter::~EvalEmitter() {
  for (auto &[K, V] : Locals) {
    Block *B = reinterpret_cast<Block *>(V.get());
    if (B->isInitialized())
      B->invokeDtor();
  }
}

EvaluationResult EvalEmitter::interpretExpr(const Expr *E,
                                            bool ConvertResultToRValue) {
  this->ConvertResultToRValue = ConvertResultToRValue;
  EvalResult.setSource(E);

  if (!this->visitExpr(E) && EvalResult.empty())
    EvalResult.setInvalid();

  return std::move(this->EvalResult);
}

EvaluationResult EvalEmitter::interpretDecl(const VarDecl *VD,
                                            bool CheckFullyInitialized) {
  this->CheckFullyInitialized = CheckFullyInitialized;
  EvalResult.setSource(VD);

  if (!this->visitDecl(VD) && EvalResult.empty())
    EvalResult.setInvalid();

  return std::move(this->EvalResult);
}

void EvalEmitter::emitLabel(LabelTy Label) {
  CurrentLabel = Label;
}

EvalEmitter::LabelTy EvalEmitter::getLabel() { return NextLabel++; }

Scope::Local EvalEmitter::createLocal(Descriptor *D) {
  // Allocate memory for a local.
  auto Memory = std::make_unique<char[]>(sizeof(Block) + D->getAllocSize());
  auto *B = new (Memory.get()) Block(D, /*isStatic=*/false);
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
  Locals.insert({Off, std::move(Memory)});
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

template <PrimType OpType> bool EvalEmitter::emitRet(const SourceInfo &Info) {
  if (!isActive())
    return true;
  using T = typename PrimConv<OpType>::T;
  EvalResult.setValue(S.Stk.pop<T>().toAPValue());
  return true;
}

template <> bool EvalEmitter::emitRet<PT_Ptr>(const SourceInfo &Info) {
  if (!isActive())
    return true;

  const Pointer &Ptr = S.Stk.pop<Pointer>();
  // Implicitly convert lvalue to rvalue, if requested.
  if (ConvertResultToRValue) {
    if (std::optional<APValue> V = Ptr.toRValue(Ctx)) {
      EvalResult.setValue(*V);
    } else {
      return false;
    }
  } else {
    if (CheckFullyInitialized) {
      if (!EvalResult.checkFullyInitialized(S, Ptr))
        return false;

      std::optional<APValue> RValueResult = Ptr.toRValue(Ctx);
      if (!RValueResult)
        return false;
      EvalResult.setValue(*RValueResult);
    } else {
      EvalResult.setValue(Ptr.toAPValue());
    }
  }

  return true;
}
template <> bool EvalEmitter::emitRet<PT_FnPtr>(const SourceInfo &Info) {
  if (!isActive())
    return true;
  // Function pointers cannot be converted to rvalues.
  EvalResult.setFunctionPointer(S.Stk.pop<FunctionPointer>());
  return true;
}

bool EvalEmitter::emitRetVoid(const SourceInfo &Info) {
  EvalResult.setValid();
  return true;
}

bool EvalEmitter::emitRetValue(const SourceInfo &Info) {
  const auto &Ptr = S.Stk.pop<Pointer>();
  if (std::optional<APValue> APV = Ptr.toRValue(S.getCtx())) {
    EvalResult.setValue(*APV);
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

//===----------------------------------------------------------------------===//
// Opcode evaluators
//===----------------------------------------------------------------------===//

#define GET_EVAL_IMPL
#include "Opcodes.inc"
#undef GET_EVAL_IMPL
