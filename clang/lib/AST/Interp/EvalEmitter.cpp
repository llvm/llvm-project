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
#include "Interp.h"
#include "Opcode.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace clang::interp;

EvalEmitter::EvalEmitter(Context &Ctx, Program &P, State &Parent,
                         InterpStack &Stk, APValue &Result)
    : Ctx(Ctx), P(P), S(Parent, P, Stk, Ctx, this), Result(Result) {
  // Create a dummy frame for the interpreter which does not have locals.
  S.Current =
      new InterpFrame(S, /*Func=*/nullptr, /*Caller=*/nullptr, CodePtr());
}

llvm::Expected<bool> EvalEmitter::interpretExpr(const Expr *E) {
  if (this->visitExpr(E))
    return true;
  if (BailLocation)
    return llvm::make_error<ByteCodeGenError>(*BailLocation);
  return false;
}

llvm::Expected<bool> EvalEmitter::interpretDecl(const VarDecl *VD) {
  if (this->visitDecl(VD))
    return true;
  if (BailLocation)
    return llvm::make_error<ByteCodeGenError>(*BailLocation);
  return false;
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

bool EvalEmitter::bail(const SourceLocation &Loc) {
  if (!BailLocation)
    BailLocation = Loc;
  return false;
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
  return ReturnValue<T>(S.Stk.pop<T>(), Result);
}

bool EvalEmitter::emitRetVoid(const SourceInfo &Info) { return true; }

bool EvalEmitter::emitRetValue(const SourceInfo &Info) {
  // Method to recursively traverse composites.
  std::function<bool(QualType, const Pointer &, APValue &)> Composite;
  Composite = [this, &Composite](QualType Ty, const Pointer &Ptr, APValue &R) {
    if (const auto *AT = Ty->getAs<AtomicType>())
      Ty = AT->getValueType();

    if (const auto *RT = Ty->getAs<RecordType>()) {
      const auto *Record = Ptr.getRecord();
      assert(Record && "Missing record descriptor");

      bool Ok = true;
      if (RT->getDecl()->isUnion()) {
        const FieldDecl *ActiveField = nullptr;
        APValue Value;
        for (const auto &F : Record->fields()) {
          const Pointer &FP = Ptr.atField(F.Offset);
          QualType FieldTy = F.Decl->getType();
          if (FP.isActive()) {
            if (std::optional<PrimType> T = Ctx.classify(FieldTy)) {
              TYPE_SWITCH(*T, Ok &= ReturnValue<T>(FP.deref<T>(), Value));
            } else {
              Ok &= Composite(FieldTy, FP, Value);
            }
            break;
          }
        }
        R = APValue(ActiveField, Value);
      } else {
        unsigned NF = Record->getNumFields();
        unsigned NB = Record->getNumBases();
        unsigned NV = Ptr.isBaseClass() ? 0 : Record->getNumVirtualBases();

        R = APValue(APValue::UninitStruct(), NB, NF);

        for (unsigned I = 0; I < NF; ++I) {
          const Record::Field *FD = Record->getField(I);
          QualType FieldTy = FD->Decl->getType();
          const Pointer &FP = Ptr.atField(FD->Offset);
          APValue &Value = R.getStructField(I);

          if (std::optional<PrimType> T = Ctx.classify(FieldTy)) {
            TYPE_SWITCH(*T, Ok &= ReturnValue<T>(FP.deref<T>(), Value));
          } else {
            Ok &= Composite(FieldTy, FP, Value);
          }
        }

        for (unsigned I = 0; I < NB; ++I) {
          const Record::Base *BD = Record->getBase(I);
          QualType BaseTy = Ctx.getASTContext().getRecordType(BD->Decl);
          const Pointer &BP = Ptr.atField(BD->Offset);
          Ok &= Composite(BaseTy, BP, R.getStructBase(I));
        }

        for (unsigned I = 0; I < NV; ++I) {
          const Record::Base *VD = Record->getVirtualBase(I);
          QualType VirtBaseTy = Ctx.getASTContext().getRecordType(VD->Decl);
          const Pointer &VP = Ptr.atField(VD->Offset);
          Ok &= Composite(VirtBaseTy, VP, R.getStructBase(NB + I));
        }
      }
      return Ok;
    }
    if (const auto *AT = Ty->getAsArrayTypeUnsafe()) {
      const size_t NumElems = Ptr.getNumElems();
      QualType ElemTy = AT->getElementType();
      R = APValue(APValue::UninitArray{}, NumElems, NumElems);

      bool Ok = true;
      for (unsigned I = 0; I < NumElems; ++I) {
        APValue &Slot = R.getArrayInitializedElt(I);
        const Pointer &EP = Ptr.atIndex(I);
        if (std::optional<PrimType> T = Ctx.classify(ElemTy)) {
          TYPE_SWITCH(*T, Ok &= ReturnValue<T>(EP.deref<T>(), Slot));
        } else {
          Ok &= Composite(ElemTy, EP.narrow(), Slot);
        }
      }
      return Ok;
    }
    llvm_unreachable("invalid value to return");
  };

  // Return the composite type.
  const auto &Ptr = S.Stk.pop<Pointer>();
  return Composite(Ptr.getType(), Ptr, Result);
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
