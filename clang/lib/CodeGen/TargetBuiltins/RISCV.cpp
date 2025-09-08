//===-------- RISCV.cpp - Emit LLVM Code for builtins ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/TargetParser/RISCVTargetParser.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

// The 0th bit simulates the `vta` of RVV
// The 1st bit simulates the `vma` of RVV
static constexpr unsigned RVV_VTA = 0x1;
static constexpr unsigned RVV_VMA = 0x2;

// RISC-V Vector builtin helper functions are marked NOINLINE to prevent
// excessive inlining in CodeGenFunction::EmitRISCVBuiltinExpr's large switch
// statement, which would significantly increase compilation time.
static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVVLEFFBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                    ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                    Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 3> IntrinsicTypes;
  if (IsMasked) {
    // Move mask to right before vl.
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
    if ((PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA))
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
    IntrinsicTypes = {ResultType, Ops[4]->getType(), Ops[2]->getType()};
  } else {
    if (PolicyAttrs & RVV_VTA)
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
    IntrinsicTypes = {ResultType, Ops[3]->getType(), Ops[1]->getType()};
  }
  Value *NewVL = Ops[2];
  Ops.erase(Ops.begin() + 2);
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  llvm::Value *LoadValue = Builder.CreateCall(F, Ops, "");
  llvm::Value *V = Builder.CreateExtractValue(LoadValue, {0});
  // Store new_vl.
  clang::CharUnits Align;
  if (IsMasked)
    Align = CGM.getNaturalPointeeTypeAlignment(
        E->getArg(E->getNumArgs() - 2)->getType());
  else
    Align = CGM.getNaturalPointeeTypeAlignment(E->getArg(1)->getType());
  llvm::Value *Val = Builder.CreateExtractValue(LoadValue, {1});
  Builder.CreateStore(Val, Address(NewVL, Val->getType(), Align));
  return V;
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVVSSEBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                   ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                   Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                   int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 3> IntrinsicTypes;
  if (IsMasked) {
    // Builtin: (mask, ptr, stride, value, vl). Intrinsic: (value, ptr, stride,
    // mask, vl)
    std::swap(Ops[0], Ops[3]);
  } else {
    // Builtin: (ptr, stride, value, vl). Intrinsic: (value, ptr, stride, vl)
    std::rotate(Ops.begin(), Ops.begin() + 2, Ops.begin() + 3);
  }
  if (IsMasked)
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[4]->getType()};
  else
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[3]->getType()};
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVIndexedStoreBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 4> IntrinsicTypes;
  if (IsMasked) {
    // Builtin: (mask, ptr, index, value, vl).
    // Intrinsic: (value, ptr, index, mask, vl)
    std::swap(Ops[0], Ops[3]);
  } else {
    // Builtin: (ptr, index, value, vl).
    // Intrinsic: (value, ptr, index, vl)
    std::rotate(Ops.begin(), Ops.begin() + 2, Ops.begin() + 3);
  }
  if (IsMasked)
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(),
                      Ops[4]->getType()};
  else
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(),
                      Ops[3]->getType()};
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVPseudoUnaryBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                          ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                          Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                          int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 3> IntrinsicTypes;
  if (IsMasked) {
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
    if ((PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA))
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
  } else {
    if (PolicyAttrs & RVV_VTA)
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
  }
  auto ElemTy = cast<llvm::VectorType>(ResultType)->getElementType();
  Ops.insert(Ops.begin() + 2, llvm::Constant::getNullValue(ElemTy));
  if (IsMasked) {
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
    // maskedoff, op1, op2, mask, vl, policy
    IntrinsicTypes = {ResultType, ElemTy, Ops[4]->getType()};
  } else {
    // passthru, op1, op2, vl
    IntrinsicTypes = {ResultType, ElemTy, Ops[3]->getType()};
  }
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVPseudoVNotBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                         ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                         Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                         int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 3> IntrinsicTypes;
  if (IsMasked) {
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
    if ((PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA))
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
  } else {
    if (PolicyAttrs & RVV_VTA)
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
  }
  auto ElemTy = cast<llvm::VectorType>(ResultType)->getElementType();
  Ops.insert(Ops.begin() + 2, llvm::Constant::getAllOnesValue(ElemTy));
  if (IsMasked) {
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
    // maskedoff, op1, po2, mask, vl, policy
    IntrinsicTypes = {ResultType, ElemTy, Ops[4]->getType()};
  } else {
    // passthru, op1, op2, vl
    IntrinsicTypes = {ResultType, ElemTy, Ops[3]->getType()};
  }
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVPseudoMaskBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                         ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                         Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                         int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 3> IntrinsicTypes;
  // op1, vl
  IntrinsicTypes = {ResultType, Ops[1]->getType()};
  Ops.insert(Ops.begin() + 1, Ops[0]);
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVPseudoVFUnaryBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 3> IntrinsicTypes;
  if (IsMasked) {
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
    if ((PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA))
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
    Ops.insert(Ops.begin() + 2, Ops[1]);
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
    // maskedoff, op1, op2, mask, vl
    IntrinsicTypes = {ResultType, Ops[2]->getType(), Ops.back()->getType()};
  } else {
    if (PolicyAttrs & RVV_VTA)
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
    // op1, po2, vl
    IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType()};
    Ops.insert(Ops.begin() + 2, Ops[1]);
  }
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVPseudoVWCVTBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                          ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                          Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                          int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 4> IntrinsicTypes;
  if (IsMasked) {
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
    if ((PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA))
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
  } else {
    if (PolicyAttrs & RVV_VTA)
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
  }
  auto ElemTy = cast<llvm::VectorType>(Ops[1]->getType())->getElementType();
  Ops.insert(Ops.begin() + 2, llvm::Constant::getNullValue(ElemTy));
  if (IsMasked) {
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
    // maskedoff, op1, op2, mask, vl, policy
    IntrinsicTypes = {ResultType, Ops[1]->getType(), ElemTy, Ops[4]->getType()};
  } else {
    // passtru, op1, op2, vl
    IntrinsicTypes = {ResultType, Ops[1]->getType(), ElemTy, Ops[3]->getType()};
  }
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVPseudoVNCVTBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                          ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                          Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                          int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 4> IntrinsicTypes;
  if (IsMasked) {
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
    if ((PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA))
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
  } else {
    if (PolicyAttrs & RVV_VTA)
      Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
  }
  Ops.insert(Ops.begin() + 2,
             llvm::Constant::getNullValue(Ops.back()->getType()));
  if (IsMasked) {
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
    // maskedoff, op1, xlen, mask, vl
    IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[4]->getType(),
                      Ops[4]->getType()};
  } else {
    // passthru, op1, xlen, vl
    IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[3]->getType(),
                      Ops[3]->getType()};
  }
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVVlenbBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                    ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                    Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  LLVMContext &Context = CGM.getLLVMContext();
  llvm::MDBuilder MDHelper(Context);
  llvm::Metadata *OpsMD[] = {llvm::MDString::get(Context, "vlenb")};
  llvm::MDNode *RegName = llvm::MDNode::get(Context, OpsMD);
  llvm::Value *Metadata = llvm::MetadataAsValue::get(Context, RegName);
  llvm::Function *F =
      CGM.getIntrinsic(llvm::Intrinsic::read_register, {CGF->SizeTy});
  return Builder.CreateCall(F, Metadata);
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVVsetvliBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                      ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                      Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                      int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::Function *F = CGM.getIntrinsic(ID, {ResultType});
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVVSEMaskBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                      ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                      Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                      int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 3> IntrinsicTypes;
  if (IsMasked) {
    // Builtin: (mask, ptr, value, vl).
    // Intrinsic: (value, ptr, mask, vl)
    std::swap(Ops[0], Ops[2]);
  } else {
    // Builtin: (ptr, value, vl).
    // Intrinsic: (value, ptr, vl)
    std::swap(Ops[0], Ops[1]);
  }
  if (IsMasked)
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[3]->getType()};
  else
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType()};
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVUnitStridedSegLoadTupleBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 4> IntrinsicTypes;
  bool NoPassthru =
      (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) |
      (!IsMasked && (PolicyAttrs & RVV_VTA));
  unsigned Offset = IsMasked ? NoPassthru ? 1 : 2 : NoPassthru ? 0 : 1;
  if (IsMasked)
    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Ops[0]->getType(),
                      Ops.back()->getType()};
  else
    IntrinsicTypes = {ResultType, Ops[Offset]->getType(),
                      Ops.back()->getType()};
  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
  if (NoPassthru)
    Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
  if (IsMasked)
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), SegInstSEW));
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  llvm::Value *LoadValue = Builder.CreateCall(F, Ops, "");
  if (ReturnValue.isNull())
    return LoadValue;
  return Builder.CreateStore(LoadValue, ReturnValue.getValue());
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVUnitStridedSegStoreTupleBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 4> IntrinsicTypes;
  // Masked
  // Builtin: (mask, ptr, v_tuple, vl)
  // Intrinsic: (tuple, ptr, mask, vl, SegInstSEW)
  // Unmasked
  // Builtin: (ptr, v_tuple, vl)
  // Intrinsic: (tuple, ptr, vl, SegInstSEW)
  if (IsMasked)
    std::swap(Ops[0], Ops[2]);
  else
    std::swap(Ops[0], Ops[1]);
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), SegInstSEW));
  if (IsMasked)
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(),
                      Ops[3]->getType()};
  else
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType()};
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVUnitStridedSegLoadFFTupleBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 4> IntrinsicTypes;
  bool NoPassthru =
      (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) |
      (!IsMasked && (PolicyAttrs & RVV_VTA));
  unsigned Offset = IsMasked ? NoPassthru ? 1 : 2 : NoPassthru ? 0 : 1;
  if (IsMasked)
    IntrinsicTypes = {ResultType, Ops.back()->getType(), Ops[Offset]->getType(),
                      Ops[0]->getType()};
  else
    IntrinsicTypes = {ResultType, Ops.back()->getType(),
                      Ops[Offset]->getType()};
  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
  if (NoPassthru)
    Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
  if (IsMasked)
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), SegInstSEW));
  Value *NewVL = Ops[2];
  Ops.erase(Ops.begin() + 2);
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  llvm::Value *LoadValue = Builder.CreateCall(F, Ops, "");
  // Get alignment from the new vl operand
  clang::CharUnits Align =
      CGM.getNaturalPointeeTypeAlignment(E->getArg(Offset + 1)->getType());
  llvm::Value *ReturnTuple = Builder.CreateExtractValue(LoadValue, 0);
  // Store new_vl
  llvm::Value *V = Builder.CreateExtractValue(LoadValue, 1);
  Builder.CreateStore(V, Address(NewVL, V->getType(), Align));
  if (ReturnValue.isNull())
    return ReturnTuple;
  return Builder.CreateStore(ReturnTuple, ReturnValue.getValue());
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVStridedSegLoadTupleBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 4> IntrinsicTypes;
  bool NoPassthru =
      (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) |
      (!IsMasked && (PolicyAttrs & RVV_VTA));
  unsigned Offset = IsMasked ? NoPassthru ? 1 : 2 : NoPassthru ? 0 : 1;
  if (IsMasked)
    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Ops.back()->getType(),
                      Ops[0]->getType()};
  else
    IntrinsicTypes = {ResultType, Ops[Offset]->getType(),
                      Ops.back()->getType()};
  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
  if (NoPassthru)
    Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
  if (IsMasked)
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), SegInstSEW));
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  llvm::Value *LoadValue = Builder.CreateCall(F, Ops, "");
  if (ReturnValue.isNull())
    return LoadValue;
  return Builder.CreateStore(LoadValue, ReturnValue.getValue());
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVStridedSegStoreTupleBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 4> IntrinsicTypes;
  // Masked
  // Builtin: (mask, ptr, stride, v_tuple, vl)
  // Intrinsic: (tuple, ptr, stride, mask, vl, SegInstSEW)
  // Unmasked
  // Builtin: (ptr, stride, v_tuple, vl)
  // Intrinsic: (tuple, ptr, stride, vl, SegInstSEW)
  if (IsMasked)
    std::swap(Ops[0], Ops[3]);
  else
    std::rotate(Ops.begin(), Ops.begin() + 2, Ops.begin() + 3);
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), SegInstSEW));
  if (IsMasked)
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[4]->getType(),
                      Ops[3]->getType()};
  else
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[3]->getType()};
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVAveragingBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                        ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                        Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                        int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  // LLVM intrinsic
  // Unmasked: (passthru, op0, op1, round_mode, vl)
  // Masked:   (passthru, vector_in, vector_in/scalar_in, mask, vxrm, vl,
  // policy)

  bool HasMaskedOff =
      !((IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));

  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 2);

  if (!HasMaskedOff)
    Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));

  if (IsMasked)
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

  llvm::Function *F = CGM.getIntrinsic(
      ID, {ResultType, Ops[2]->getType(), Ops.back()->getType()});
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVNarrowingClipBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  // LLVM intrinsic
  // Unmasked: (passthru, op0, op1, round_mode, vl)
  // Masked:   (passthru, vector_in, vector_in/scalar_in, mask, vxrm, vl,
  // policy)

  bool HasMaskedOff =
      !((IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));

  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 2);

  if (!HasMaskedOff)
    Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));

  if (IsMasked)
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

  llvm::Function *F =
      CGM.getIntrinsic(ID, {ResultType, Ops[1]->getType(), Ops[2]->getType(),
                            Ops.back()->getType()});
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVFloatingPointBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  // LLVM intrinsic
  // Unmasked: (passthru, op0, op1, round_mode, vl)
  // Masked:   (passthru, vector_in, vector_in/scalar_in, mask, frm, vl, policy)

  bool HasMaskedOff =
      !((IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
  bool HasRoundModeOp =
      IsMasked ? (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5)
               : (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

  if (!HasRoundModeOp)
    Ops.insert(Ops.end() - 1,
               ConstantInt::get(Ops.back()->getType(), 7)); // frm

  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 2);

  if (!HasMaskedOff)
    Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));

  if (IsMasked)
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

  llvm::Function *F = CGM.getIntrinsic(
      ID, {ResultType, Ops[2]->getType(), Ops.back()->getType()});
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVWideningFloatingPointBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  // LLVM intrinsic
  // Unmasked: (passthru, op0, op1, round_mode, vl)
  // Masked:   (passthru, vector_in, vector_in/scalar_in, mask, frm, vl, policy)

  bool HasMaskedOff =
      !((IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
  bool HasRoundModeOp =
      IsMasked ? (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5)
               : (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

  if (!HasRoundModeOp)
    Ops.insert(Ops.end() - 1,
               ConstantInt::get(Ops.back()->getType(), 7)); // frm

  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 2);

  if (!HasMaskedOff)
    Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));

  if (IsMasked)
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

  llvm::Function *F =
      CGM.getIntrinsic(ID, {ResultType, Ops[1]->getType(), Ops[2]->getType(),
                            Ops.back()->getType()});
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVIndexedSegLoadTupleBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 5> IntrinsicTypes;

  bool NoPassthru =
      (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) |
      (!IsMasked && (PolicyAttrs & RVV_VTA));

  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
  if (NoPassthru)
    Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));

  if (IsMasked)
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), SegInstSEW));

  if (IsMasked)
    IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(),
                      Ops[3]->getType(), Ops[4]->getType()};
  else
    IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(),
                      Ops[3]->getType()};
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  llvm::Value *LoadValue = Builder.CreateCall(F, Ops, "");

  if (ReturnValue.isNull())
    return LoadValue;
  return Builder.CreateStore(LoadValue, ReturnValue.getValue());
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVIndexedSegStoreTupleBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 5> IntrinsicTypes;
  // Masked
  // Builtin: (mask, ptr, index, v_tuple, vl)
  // Intrinsic: (tuple, ptr, index, mask, vl, SegInstSEW)
  // Unmasked
  // Builtin: (ptr, index, v_tuple, vl)
  // Intrinsic: (tuple, ptr, index, vl, SegInstSEW)

  if (IsMasked)
    std::swap(Ops[0], Ops[3]);
  else
    std::rotate(Ops.begin(), Ops.begin() + 2, Ops.begin() + 3);

  Ops.push_back(ConstantInt::get(Ops.back()->getType(), SegInstSEW));

  if (IsMasked)
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(),
                      Ops[3]->getType(), Ops[4]->getType()};
  else
    IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(),
                      Ops[3]->getType()};
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVFMABuiltin(CodeGenFunction *CGF, const CallExpr *E,
                  ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                  Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                  int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  // LLVM intrinsic
  // Unmasked: (vector_in, vector_in/scalar_in, vector_in, round_mode,
  //            vl, policy)
  // Masked:   (vector_in, vector_in/scalar_in, vector_in, mask, frm,
  //            vl, policy)

  bool HasRoundModeOp = IsMasked ? Ops.size() == 6 : Ops.size() == 5;

  if (!HasRoundModeOp)
    Ops.insert(Ops.end() - 1,
               ConstantInt::get(Ops.back()->getType(), 7)); // frm

  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 2);

  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

  llvm::Function *F = CGM.getIntrinsic(
      ID, {ResultType, Ops[1]->getType(), Ops.back()->getType()});
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVWideningFMABuiltin(CodeGenFunction *CGF, const CallExpr *E,
                          ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                          Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                          int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  // LLVM intrinsic
  // Unmasked: (vector_in, vector_in/scalar_in, vector_in, round_mode, vl,
  // policy) Masked:   (vector_in, vector_in/scalar_in, vector_in, mask, frm,
  // vl, policy)

  bool HasRoundModeOp = IsMasked ? Ops.size() == 6 : Ops.size() == 5;

  if (!HasRoundModeOp)
    Ops.insert(Ops.end() - 1,
               ConstantInt::get(Ops.back()->getType(), 7)); // frm

  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.begin() + 4);

  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

  llvm::Function *F =
      CGM.getIntrinsic(ID, {ResultType, Ops[1]->getType(), Ops[2]->getType(),
                            Ops.back()->getType()});
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVFloatingUnaryBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  llvm::SmallVector<llvm::Type *, 3> IntrinsicTypes;
  // LLVM intrinsic
  // Unmasked: (passthru, op0, round_mode, vl)
  // Masked:   (passthru, op0, mask, frm, vl, policy)

  bool HasMaskedOff =
      !((IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
  bool HasRoundModeOp =
      IsMasked ? (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4)
               : (HasMaskedOff ? Ops.size() == 4 : Ops.size() == 3);

  if (!HasRoundModeOp)
    Ops.insert(Ops.end() - 1,
               ConstantInt::get(Ops.back()->getType(), 7)); // frm

  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 2);

  if (!HasMaskedOff)
    Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));

  if (IsMasked)
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

  IntrinsicTypes = {ResultType, Ops.back()->getType()};
  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVFloatingConvBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  // LLVM intrinsic
  // Unmasked: (passthru, op0, frm, vl)
  // Masked:   (passthru, op0, mask, frm, vl, policy)
  bool HasMaskedOff =
      !((IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
  bool HasRoundModeOp =
      IsMasked ? (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4)
               : (HasMaskedOff ? Ops.size() == 4 : Ops.size() == 3);

  if (!HasRoundModeOp)
    Ops.insert(Ops.end() - 1,
               ConstantInt::get(Ops.back()->getType(), 7)); // frm

  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 2);

  if (!HasMaskedOff)
    Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));

  if (IsMasked)
    Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

  llvm::Function *F = CGM.getIntrinsic(
      ID, {ResultType, Ops[1]->getType(), Ops.back()->getType()});
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *emitRVVFloatingReductionBuiltin(
    CodeGenFunction *CGF, const CallExpr *E, ReturnValueSlot ReturnValue,
    llvm::Type *ResultType, Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
    int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;
  // LLVM intrinsic
  // Unmasked: (passthru, op0, op1, round_mode, vl)
  // Masked:   (passthru, vector_in, vector_in/scalar_in, mask, frm, vl, policy)

  bool HasMaskedOff =
      !((IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
  bool HasRoundModeOp =
      IsMasked ? (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5)
               : (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

  if (!HasRoundModeOp)
    Ops.insert(Ops.end() - 1,
               ConstantInt::get(Ops.back()->getType(), 7)); // frm

  if (IsMasked)
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 2);

  if (!HasMaskedOff)
    Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));

  llvm::Function *F = CGM.getIntrinsic(
      ID, {ResultType, Ops[1]->getType(), Ops.back()->getType()});
  return Builder.CreateCall(F, Ops, "");
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVReinterpretBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                          ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                          Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                          int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto &CGM = CGF->CGM;

  if (ResultType->isIntOrIntVectorTy(1) ||
      Ops[0]->getType()->isIntOrIntVectorTy(1)) {
    assert(isa<ScalableVectorType>(ResultType) &&
           isa<ScalableVectorType>(Ops[0]->getType()));

    LLVMContext &Context = CGM.getLLVMContext();
    ScalableVectorType *Boolean64Ty =
        ScalableVectorType::get(llvm::Type::getInt1Ty(Context), 64);

    if (ResultType->isIntOrIntVectorTy(1)) {
      // Casting from m1 vector integer -> vector boolean
      // Ex: <vscale x 8 x i8>
      //     --(bitcast)--------> <vscale x 64 x i1>
      //     --(vector_extract)-> <vscale x  8 x i1>
      llvm::Value *BitCast = Builder.CreateBitCast(Ops[0], Boolean64Ty);
      return Builder.CreateExtractVector(ResultType, BitCast,
                                         ConstantInt::get(CGF->Int64Ty, 0));
    } else {
      // Casting from vector boolean -> m1 vector integer
      // Ex: <vscale x  1 x i1>
      //       --(vector_insert)-> <vscale x 64 x i1>
      //       --(bitcast)-------> <vscale x  8 x i8>
      llvm::Value *Boolean64Val = Builder.CreateInsertVector(
          Boolean64Ty, llvm::PoisonValue::get(Boolean64Ty), Ops[0],
          ConstantInt::get(CGF->Int64Ty, 0));
      return Builder.CreateBitCast(Boolean64Val, ResultType);
    }
  }
  return Builder.CreateBitCast(Ops[0], ResultType);
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVGetBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                  ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                  Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                  int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  auto *VecTy = cast<ScalableVectorType>(ResultType);
  if (auto *OpVecTy = dyn_cast<ScalableVectorType>(Ops[0]->getType())) {
    unsigned MaxIndex =
        OpVecTy->getMinNumElements() / VecTy->getMinNumElements();
    assert(isPowerOf2_32(MaxIndex));
    // Mask to only valid indices.
    Ops[1] = Builder.CreateZExt(Ops[1], Builder.getInt64Ty());
    Ops[1] = Builder.CreateAnd(Ops[1], MaxIndex - 1);
    Ops[1] =
        Builder.CreateMul(Ops[1], ConstantInt::get(Ops[1]->getType(),
                                                   VecTy->getMinNumElements()));
    return Builder.CreateExtractVector(ResultType, Ops[0], Ops[1]);
  }

  return Builder.CreateIntrinsic(
      Intrinsic::riscv_tuple_extract, {ResultType, Ops[0]->getType()},
      {Ops[0], Builder.CreateTrunc(Ops[1], Builder.getInt32Ty())});
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVSetBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                  ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                  Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                  int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  if (auto *ResVecTy = dyn_cast<ScalableVectorType>(ResultType)) {
    auto *VecTy = cast<ScalableVectorType>(Ops[2]->getType());
    unsigned MaxIndex =
        ResVecTy->getMinNumElements() / VecTy->getMinNumElements();
    assert(isPowerOf2_32(MaxIndex));
    // Mask to only valid indices.
    Ops[1] = Builder.CreateZExt(Ops[1], Builder.getInt64Ty());
    Ops[1] = Builder.CreateAnd(Ops[1], MaxIndex - 1);
    Ops[1] =
        Builder.CreateMul(Ops[1], ConstantInt::get(Ops[1]->getType(),
                                                   VecTy->getMinNumElements()));
    return Builder.CreateInsertVector(ResultType, Ops[0], Ops[2], Ops[1]);
  }

  return Builder.CreateIntrinsic(
      Intrinsic::riscv_tuple_insert, {ResultType, Ops[2]->getType()},
      {Ops[0], Ops[2], Builder.CreateTrunc(Ops[1], Builder.getInt32Ty())});
}

static LLVM_ATTRIBUTE_NOINLINE Value *
emitRVVCreateBuiltin(CodeGenFunction *CGF, const CallExpr *E,
                     ReturnValueSlot ReturnValue, llvm::Type *ResultType,
                     Intrinsic::ID ID, SmallVectorImpl<Value *> &Ops,
                     int PolicyAttrs, bool IsMasked, unsigned SegInstSEW) {
  auto &Builder = CGF->Builder;
  llvm::Value *ReturnVector = llvm::PoisonValue::get(ResultType);
  auto *VecTy = cast<ScalableVectorType>(Ops[0]->getType());
  for (unsigned I = 0, N = Ops.size(); I < N; ++I) {
    if (isa<ScalableVectorType>(ResultType)) {
      llvm::Value *Idx = ConstantInt::get(Builder.getInt64Ty(),
                                          VecTy->getMinNumElements() * I);
      ReturnVector =
          Builder.CreateInsertVector(ResultType, ReturnVector, Ops[I], Idx);
    } else {
      llvm::Value *Idx = ConstantInt::get(Builder.getInt32Ty(), I);
      ReturnVector = Builder.CreateIntrinsic(Intrinsic::riscv_tuple_insert,
                                             {ResultType, Ops[I]->getType()},
                                             {ReturnVector, Ops[I], Idx});
    }
  }
  return ReturnVector;
}

Value *CodeGenFunction::EmitRISCVCpuInit() {
  llvm::FunctionType *FTy = llvm::FunctionType::get(VoidTy, {VoidPtrTy}, false);
  llvm::FunctionCallee Func =
      CGM.CreateRuntimeFunction(FTy, "__init_riscv_feature_bits");
  auto *CalleeGV = cast<llvm::GlobalValue>(Func.getCallee());
  CalleeGV->setDSOLocal(true);
  CalleeGV->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
  return Builder.CreateCall(Func, {llvm::ConstantPointerNull::get(VoidPtrTy)});
}

Value *CodeGenFunction::EmitRISCVCpuSupports(const CallExpr *E) {

  const Expr *FeatureExpr = E->getArg(0)->IgnoreParenCasts();
  StringRef FeatureStr = cast<StringLiteral>(FeatureExpr)->getString();
  if (!getContext().getTargetInfo().validateCpuSupports(FeatureStr))
    return Builder.getFalse();

  return EmitRISCVCpuSupports(ArrayRef<StringRef>(FeatureStr));
}

static Value *loadRISCVFeatureBits(unsigned Index, CGBuilderTy &Builder,
                                   CodeGenModule &CGM) {
  llvm::Type *Int32Ty = Builder.getInt32Ty();
  llvm::Type *Int64Ty = Builder.getInt64Ty();
  llvm::ArrayType *ArrayOfInt64Ty =
      llvm::ArrayType::get(Int64Ty, llvm::RISCVISAInfo::FeatureBitSize);
  llvm::Type *StructTy = llvm::StructType::get(Int32Ty, ArrayOfInt64Ty);
  llvm::Constant *RISCVFeaturesBits =
      CGM.CreateRuntimeVariable(StructTy, "__riscv_feature_bits");
  cast<llvm::GlobalValue>(RISCVFeaturesBits)->setDSOLocal(true);
  Value *IndexVal = llvm::ConstantInt::get(Int32Ty, Index);
  llvm::Value *GEPIndices[] = {Builder.getInt32(0), Builder.getInt32(1),
                               IndexVal};
  Value *Ptr =
      Builder.CreateInBoundsGEP(StructTy, RISCVFeaturesBits, GEPIndices);
  Value *FeaturesBit =
      Builder.CreateAlignedLoad(Int64Ty, Ptr, CharUnits::fromQuantity(8));
  return FeaturesBit;
}

Value *CodeGenFunction::EmitRISCVCpuSupports(ArrayRef<StringRef> FeaturesStrs) {
  const unsigned RISCVFeatureLength = llvm::RISCVISAInfo::FeatureBitSize;
  uint64_t RequireBitMasks[RISCVFeatureLength] = {0};

  for (auto Feat : FeaturesStrs) {
    auto [GroupID, BitPos] = RISCVISAInfo::getRISCVFeaturesBitsInfo(Feat);

    // If there isn't BitPos for this feature, skip this version.
    // It also report the warning to user during compilation.
    if (BitPos == -1)
      return Builder.getFalse();

    RequireBitMasks[GroupID] |= (1ULL << BitPos);
  }

  Value *Result = nullptr;
  for (unsigned Idx = 0; Idx < RISCVFeatureLength; Idx++) {
    if (RequireBitMasks[Idx] == 0)
      continue;

    Value *Mask = Builder.getInt64(RequireBitMasks[Idx]);
    Value *Bitset =
        Builder.CreateAnd(loadRISCVFeatureBits(Idx, Builder, CGM), Mask);
    Value *CmpV = Builder.CreateICmpEQ(Bitset, Mask);
    Result = (!Result) ? CmpV : Builder.CreateAnd(Result, CmpV);
  }

  assert(Result && "Should have value here.");

  return Result;
}

Value *CodeGenFunction::EmitRISCVCpuIs(const CallExpr *E) {
  const Expr *CPUExpr = E->getArg(0)->IgnoreParenCasts();
  StringRef CPUStr = cast<clang::StringLiteral>(CPUExpr)->getString();
  return EmitRISCVCpuIs(CPUStr);
}

Value *CodeGenFunction::EmitRISCVCpuIs(StringRef CPUStr) {
  llvm::Type *Int32Ty = Builder.getInt32Ty();
  llvm::Type *Int64Ty = Builder.getInt64Ty();
  llvm::StructType *StructTy = llvm::StructType::get(Int32Ty, Int64Ty, Int64Ty);
  llvm::Constant *RISCVCPUModel =
      CGM.CreateRuntimeVariable(StructTy, "__riscv_cpu_model");
  cast<llvm::GlobalValue>(RISCVCPUModel)->setDSOLocal(true);

  auto loadRISCVCPUID = [&](unsigned Index) {
    Value *Ptr = Builder.CreateStructGEP(StructTy, RISCVCPUModel, Index);
    Value *CPUID = Builder.CreateAlignedLoad(StructTy->getTypeAtIndex(Index),
                                             Ptr, llvm::MaybeAlign());
    return CPUID;
  };

  const llvm::RISCV::CPUModel Model = llvm::RISCV::getCPUModel(CPUStr);

  // Compare mvendorid.
  Value *VendorID = loadRISCVCPUID(0);
  Value *Result =
      Builder.CreateICmpEQ(VendorID, Builder.getInt32(Model.MVendorID));

  // Compare marchid.
  Value *ArchID = loadRISCVCPUID(1);
  Result = Builder.CreateAnd(
      Result, Builder.CreateICmpEQ(ArchID, Builder.getInt64(Model.MArchID)));

  // Compare mimpid.
  Value *ImpID = loadRISCVCPUID(2);
  Result = Builder.CreateAnd(
      Result, Builder.CreateICmpEQ(ImpID, Builder.getInt64(Model.MImpID)));

  return Result;
}

Value *CodeGenFunction::EmitRISCVBuiltinExpr(unsigned BuiltinID,
                                             const CallExpr *E,
                                             ReturnValueSlot ReturnValue) {

  if (BuiltinID == Builtin::BI__builtin_cpu_supports)
    return EmitRISCVCpuSupports(E);
  if (BuiltinID == Builtin::BI__builtin_cpu_init)
    return EmitRISCVCpuInit();
  if (BuiltinID == Builtin::BI__builtin_cpu_is)
    return EmitRISCVCpuIs(E);

  SmallVector<Value *, 4> Ops;
  llvm::Type *ResultType = ConvertType(E->getType());

  // Find out if any arguments are required to be integer constant expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
  if (Error == ASTContext::GE_Missing_type) {
    // Vector intrinsics don't have a type string.
    assert(BuiltinID >= clang::RISCV::FirstRVVBuiltin &&
           BuiltinID <= clang::RISCV::LastRVVBuiltin);
    ICEArguments = 0;
    if (BuiltinID == RISCVVector::BI__builtin_rvv_vget_v ||
        BuiltinID == RISCVVector::BI__builtin_rvv_vset_v)
      ICEArguments = 1 << 1;
  } else {
    assert(Error == ASTContext::GE_None && "Unexpected error");
  }

  if (BuiltinID == RISCV::BI__builtin_riscv_ntl_load)
    ICEArguments |= (1 << 1);
  if (BuiltinID == RISCV::BI__builtin_riscv_ntl_store)
    ICEArguments |= (1 << 2);

  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++) {
    // Handle aggregate argument, namely RVV tuple types in segment load/store
    if (hasAggregateEvaluationKind(E->getArg(i)->getType())) {
      LValue L = EmitAggExprToLValue(E->getArg(i));
      llvm::Value *AggValue = Builder.CreateLoad(L.getAddress());
      Ops.push_back(AggValue);
      continue;
    }
    Ops.push_back(EmitScalarOrConstFoldImmArg(ICEArguments, i, E));
  }

  Intrinsic::ID ID = Intrinsic::not_intrinsic;
  int PolicyAttrs = 0;
  bool IsMasked = false;
  // This is used by segment load/store to determine it's llvm type.
  unsigned SegInstSEW = 8;

  // Required for overloaded intrinsics.
  llvm::SmallVector<llvm::Type *, 2> IntrinsicTypes;
  switch (BuiltinID) {
  default: llvm_unreachable("unexpected builtin ID");
  case RISCV::BI__builtin_riscv_orc_b_32:
  case RISCV::BI__builtin_riscv_orc_b_64:
  case RISCV::BI__builtin_riscv_clmul_32:
  case RISCV::BI__builtin_riscv_clmul_64:
  case RISCV::BI__builtin_riscv_clmulh_32:
  case RISCV::BI__builtin_riscv_clmulh_64:
  case RISCV::BI__builtin_riscv_clmulr_32:
  case RISCV::BI__builtin_riscv_clmulr_64:
  case RISCV::BI__builtin_riscv_xperm4_32:
  case RISCV::BI__builtin_riscv_xperm4_64:
  case RISCV::BI__builtin_riscv_xperm8_32:
  case RISCV::BI__builtin_riscv_xperm8_64:
  case RISCV::BI__builtin_riscv_brev8_32:
  case RISCV::BI__builtin_riscv_brev8_64:
  case RISCV::BI__builtin_riscv_zip_32:
  case RISCV::BI__builtin_riscv_unzip_32: {
    switch (BuiltinID) {
    default: llvm_unreachable("unexpected builtin ID");
    // Zbb
    case RISCV::BI__builtin_riscv_orc_b_32:
    case RISCV::BI__builtin_riscv_orc_b_64:
      ID = Intrinsic::riscv_orc_b;
      break;

    // Zbc
    case RISCV::BI__builtin_riscv_clmul_32:
    case RISCV::BI__builtin_riscv_clmul_64:
      ID = Intrinsic::riscv_clmul;
      break;
    case RISCV::BI__builtin_riscv_clmulh_32:
    case RISCV::BI__builtin_riscv_clmulh_64:
      ID = Intrinsic::riscv_clmulh;
      break;
    case RISCV::BI__builtin_riscv_clmulr_32:
    case RISCV::BI__builtin_riscv_clmulr_64:
      ID = Intrinsic::riscv_clmulr;
      break;

    // Zbkx
    case RISCV::BI__builtin_riscv_xperm8_32:
    case RISCV::BI__builtin_riscv_xperm8_64:
      ID = Intrinsic::riscv_xperm8;
      break;
    case RISCV::BI__builtin_riscv_xperm4_32:
    case RISCV::BI__builtin_riscv_xperm4_64:
      ID = Intrinsic::riscv_xperm4;
      break;

    // Zbkb
    case RISCV::BI__builtin_riscv_brev8_32:
    case RISCV::BI__builtin_riscv_brev8_64:
      ID = Intrinsic::riscv_brev8;
      break;
    case RISCV::BI__builtin_riscv_zip_32:
      ID = Intrinsic::riscv_zip;
      break;
    case RISCV::BI__builtin_riscv_unzip_32:
      ID = Intrinsic::riscv_unzip;
      break;
    }

    IntrinsicTypes = {ResultType};
    break;
  }

  // RISCV Packed SIMD extension

  // Intrinsic type is obtained from Ops[0].
  case RISCV::BI__builtin_riscv_psll_bs_32:
  case RISCV::BI__builtin_riscv_psll_bs_64:
  case RISCV::BI__builtin_riscv_psll_hs_32:
  case RISCV::BI__builtin_riscv_psll_hs_64:
  case RISCV::BI__builtin_riscv_psll_ws:
  case RISCV::BI__builtin_riscv_padd_bs_32:
  case RISCV::BI__builtin_riscv_padd_bs_64:
  case RISCV::BI__builtin_riscv_padd_hs_32:
  case RISCV::BI__builtin_riscv_padd_hs_64:
  case RISCV::BI__builtin_riscv_padd_ws:
  case RISCV::BI__builtin_riscv_psrl_bs_32:
  case RISCV::BI__builtin_riscv_psrl_bs_64:
  case RISCV::BI__builtin_riscv_psrl_hs_32:
  case RISCV::BI__builtin_riscv_psrl_hs_64:
  case RISCV::BI__builtin_riscv_psrl_ws:
  case RISCV::BI__builtin_riscv_predsum_bs_32:
  case RISCV::BI__builtin_riscv_predsum_bs_64:
  case RISCV::BI__builtin_riscv_predsum_hs_32:
  case RISCV::BI__builtin_riscv_predsum_hs_64:
  case RISCV::BI__builtin_riscv_predsum_ws:
  case RISCV::BI__builtin_riscv_predsumu_bs_32:
  case RISCV::BI__builtin_riscv_predsumu_bs_64:
  case RISCV::BI__builtin_riscv_predsumu_hs_32:
  case RISCV::BI__builtin_riscv_predsumu_hs_64:
  case RISCV::BI__builtin_riscv_predsumu_ws:
  case RISCV::BI__builtin_riscv_psra_bs_32:
  case RISCV::BI__builtin_riscv_psra_bs_64:
  case RISCV::BI__builtin_riscv_psra_hs_32:
  case RISCV::BI__builtin_riscv_psra_hs_64:
  case RISCV::BI__builtin_riscv_psra_ws:
  case RISCV::BI__builtin_riscv_padd_b_32:
  case RISCV::BI__builtin_riscv_padd_b_64:
  case RISCV::BI__builtin_riscv_padd_h_32:
  case RISCV::BI__builtin_riscv_padd_h_64:
  case RISCV::BI__builtin_riscv_padd_w:
  case RISCV::BI__builtin_riscv_sadd:
  case RISCV::BI__builtin_riscv_psadd_b_32:
  case RISCV::BI__builtin_riscv_psadd_b_64:
  case RISCV::BI__builtin_riscv_psadd_h_32:
  case RISCV::BI__builtin_riscv_psadd_h_64:
  case RISCV::BI__builtin_riscv_psadd_w:
  case RISCV::BI__builtin_riscv_aadd:
  case RISCV::BI__builtin_riscv_paadd_b_32:
  case RISCV::BI__builtin_riscv_paadd_b_64:
  case RISCV::BI__builtin_riscv_paadd_h_32:
  case RISCV::BI__builtin_riscv_paadd_h_64:
  case RISCV::BI__builtin_riscv_paadd_w:
  case RISCV::BI__builtin_riscv_saddu:
  case RISCV::BI__builtin_riscv_psaddu_b_32:
  case RISCV::BI__builtin_riscv_psaddu_b_64:
  case RISCV::BI__builtin_riscv_psaddu_h_32:
  case RISCV::BI__builtin_riscv_psaddu_h_64:
  case RISCV::BI__builtin_riscv_psaddu_w:
  case RISCV::BI__builtin_riscv_aaddu:
  case RISCV::BI__builtin_riscv_paaddu_b_32:
  case RISCV::BI__builtin_riscv_paaddu_b_64:
  case RISCV::BI__builtin_riscv_paaddu_h_32:
  case RISCV::BI__builtin_riscv_paaddu_h_64:
  case RISCV::BI__builtin_riscv_paaddu_w:
  case RISCV::BI__builtin_riscv_psub_b_32:
  case RISCV::BI__builtin_riscv_psub_b_64:
  case RISCV::BI__builtin_riscv_psub_h_32:
  case RISCV::BI__builtin_riscv_psub_h_64:
  case RISCV::BI__builtin_riscv_psub_w:
  case RISCV::BI__builtin_riscv_ssub:
  case RISCV::BI__builtin_riscv_pssub_b_32:
  case RISCV::BI__builtin_riscv_pssub_b_64:
  case RISCV::BI__builtin_riscv_pssub_h_32:
  case RISCV::BI__builtin_riscv_pssub_h_64:
  case RISCV::BI__builtin_riscv_pssub_w:
  case RISCV::BI__builtin_riscv_asub:
  case RISCV::BI__builtin_riscv_pasub_b_32:
  case RISCV::BI__builtin_riscv_pasub_b_64:
  case RISCV::BI__builtin_riscv_pasub_h_32:
  case RISCV::BI__builtin_riscv_pasub_h_64:
  case RISCV::BI__builtin_riscv_pasub_w:
  case RISCV::BI__builtin_riscv_ssubu:
  case RISCV::BI__builtin_riscv_pssubu_b_32:
  case RISCV::BI__builtin_riscv_pssubu_b_64:
  case RISCV::BI__builtin_riscv_pssubu_h_32:
  case RISCV::BI__builtin_riscv_pssubu_h_64:
  case RISCV::BI__builtin_riscv_pssubu_w:
  case RISCV::BI__builtin_riscv_asubu:
  case RISCV::BI__builtin_riscv_pasubu_b_32:
  case RISCV::BI__builtin_riscv_pasubu_b_64:
  case RISCV::BI__builtin_riscv_pasubu_h_32:
  case RISCV::BI__builtin_riscv_pasubu_h_64:
  case RISCV::BI__builtin_riscv_pasubu_w:
  case RISCV::BI__builtin_riscv_pdif_b_32:
  case RISCV::BI__builtin_riscv_pdif_b_64:
  case RISCV::BI__builtin_riscv_pdif_h_32:
  case RISCV::BI__builtin_riscv_pdif_h_64:
  case RISCV::BI__builtin_riscv_pdifu_b_32:
  case RISCV::BI__builtin_riscv_pdifu_b_64:
  case RISCV::BI__builtin_riscv_pdifu_h_32:
  case RISCV::BI__builtin_riscv_pdifu_h_64:
  case RISCV::BI__builtin_riscv_mul_h01:
  case RISCV::BI__builtin_riscv_mul_w01:
  case RISCV::BI__builtin_riscv_mulu_h01:
  case RISCV::BI__builtin_riscv_mulu_w01:
  case RISCV::BI__builtin_riscv_slx_32:
  case RISCV::BI__builtin_riscv_slx_64:
  case RISCV::BI__builtin_riscv_psh1add_h_32:
  case RISCV::BI__builtin_riscv_psh1add_h_64:
  case RISCV::BI__builtin_riscv_psh1add_w:
  case RISCV::BI__builtin_riscv_ssh1sadd:
  case RISCV::BI__builtin_riscv_pssh1sadd_h_32:
  case RISCV::BI__builtin_riscv_pssh1sadd_h_64:
  case RISCV::BI__builtin_riscv_pssh1sadd_w:
  case RISCV::BI__builtin_riscv_unzip8p:
  case RISCV::BI__builtin_riscv_unzip16p:
  case RISCV::BI__builtin_riscv_unzip8hp:
  case RISCV::BI__builtin_riscv_unzip16hp:
  case RISCV::BI__builtin_riscv_zip8p:
  case RISCV::BI__builtin_riscv_zip16p:
  case RISCV::BI__builtin_riscv_zip8hp:
  case RISCV::BI__builtin_riscv_zip16hp:
  case RISCV::BI__builtin_riscv_ppack_h_32:
  case RISCV::BI__builtin_riscv_ppack_h_64:
  case RISCV::BI__builtin_riscv_ppack_w:
  case RISCV::BI__builtin_riscv_ppackbt_h_32:
  case RISCV::BI__builtin_riscv_ppackbt_h_64:
  case RISCV::BI__builtin_riscv_ppackbt_w:
  case RISCV::BI__builtin_riscv_packbt_32:
  case RISCV::BI__builtin_riscv_packbt_64:
  case RISCV::BI__builtin_riscv_ppacktb_h_32:
  case RISCV::BI__builtin_riscv_ppacktb_h_64:
  case RISCV::BI__builtin_riscv_ppacktb_w:
  case RISCV::BI__builtin_riscv_packtb_32:
  case RISCV::BI__builtin_riscv_packtb_64:
  case RISCV::BI__builtin_riscv_ppackt_h_32:
  case RISCV::BI__builtin_riscv_ppackt_h_64:
  case RISCV::BI__builtin_riscv_ppackt_w:
  case RISCV::BI__builtin_riscv_packt_32:
  case RISCV::BI__builtin_riscv_packt_64:
  case RISCV::BI__builtin_riscv_pas_hx_32:
  case RISCV::BI__builtin_riscv_pas_hx_64:
  case RISCV::BI__builtin_riscv_pas_wx:
  case RISCV::BI__builtin_riscv_psa_hx_32:
  case RISCV::BI__builtin_riscv_psa_hx_64:
  case RISCV::BI__builtin_riscv_psa_wx:
  case RISCV::BI__builtin_riscv_psas_hx_32:
  case RISCV::BI__builtin_riscv_psas_hx_64:
  case RISCV::BI__builtin_riscv_psas_wx:
  case RISCV::BI__builtin_riscv_pssa_hx_32:
  case RISCV::BI__builtin_riscv_pssa_hx_64:
  case RISCV::BI__builtin_riscv_pssa_wx:
  case RISCV::BI__builtin_riscv_paas_hx_32:
  case RISCV::BI__builtin_riscv_paas_hx_64:
  case RISCV::BI__builtin_riscv_paas_wx:
  case RISCV::BI__builtin_riscv_pasa_hx_32:
  case RISCV::BI__builtin_riscv_pasa_hx_64:
  case RISCV::BI__builtin_riscv_pasa_wx:
  case RISCV::BI__builtin_riscv_mseq:
  case RISCV::BI__builtin_riscv_pmseq_b_32:
  case RISCV::BI__builtin_riscv_pmseq_b_64:
  case RISCV::BI__builtin_riscv_pmseq_h_32:
  case RISCV::BI__builtin_riscv_pmseq_h_64:
  case RISCV::BI__builtin_riscv_pmseq_w:
  case RISCV::BI__builtin_riscv_mslt:
  case RISCV::BI__builtin_riscv_pmslt_b_32:
  case RISCV::BI__builtin_riscv_pmslt_b_64:
  case RISCV::BI__builtin_riscv_pmslt_h_32:
  case RISCV::BI__builtin_riscv_pmslt_h_64:
  case RISCV::BI__builtin_riscv_pmslt_w:
  case RISCV::BI__builtin_riscv_msltu:
  case RISCV::BI__builtin_riscv_pmsltu_b_32:
  case RISCV::BI__builtin_riscv_pmsltu_b_64:
  case RISCV::BI__builtin_riscv_pmsltu_h_32:
  case RISCV::BI__builtin_riscv_pmsltu_h_64:
  case RISCV::BI__builtin_riscv_pmsltu_w:
  case RISCV::BI__builtin_riscv_pmin_b_32:
  case RISCV::BI__builtin_riscv_pmin_b_64:
  case RISCV::BI__builtin_riscv_pmin_h_32:
  case RISCV::BI__builtin_riscv_pmin_h_64:
  case RISCV::BI__builtin_riscv_pmin_w:
  case RISCV::BI__builtin_riscv_pminu_b_32:
  case RISCV::BI__builtin_riscv_pminu_b_64:
  case RISCV::BI__builtin_riscv_pminu_h_32:
  case RISCV::BI__builtin_riscv_pminu_h_64:
  case RISCV::BI__builtin_riscv_pminu_w:
  case RISCV::BI__builtin_riscv_pmax_b_32:
  case RISCV::BI__builtin_riscv_pmax_b_64:
  case RISCV::BI__builtin_riscv_pmax_h_32:
  case RISCV::BI__builtin_riscv_pmax_h_64:
  case RISCV::BI__builtin_riscv_pmax_w:
  case RISCV::BI__builtin_riscv_pmaxu_b_32:
  case RISCV::BI__builtin_riscv_pmaxu_b_64:
  case RISCV::BI__builtin_riscv_pmaxu_h_32:
  case RISCV::BI__builtin_riscv_pmaxu_h_64:
  case RISCV::BI__builtin_riscv_pmaxu_w:
  case RISCV::BI__builtin_riscv_pmulh_h_32:
  case RISCV::BI__builtin_riscv_pmulh_h_64:
  case RISCV::BI__builtin_riscv_pmulh_w:
  case RISCV::BI__builtin_riscv_pmulhu_h_32:
  case RISCV::BI__builtin_riscv_pmulhu_h_64:
  case RISCV::BI__builtin_riscv_pmulhu_w:
  case RISCV::BI__builtin_riscv_pmulhr_h_32:
  case RISCV::BI__builtin_riscv_pmulhr_h_64:
  case RISCV::BI__builtin_riscv_pmulhr_w:
  case RISCV::BI__builtin_riscv_pmulhru_h_32:
  case RISCV::BI__builtin_riscv_pmulhru_h_64:
  case RISCV::BI__builtin_riscv_pmulhru_w:
  case RISCV::BI__builtin_riscv_mulh_h1:
  case RISCV::BI__builtin_riscv_mulhr:
  case RISCV::BI__builtin_riscv_mulhru:
  case RISCV::BI__builtin_riscv_mulh_h0: {
    switch (BuiltinID) {
    default:
      llvm_unreachable("unexpected builtin ID");
    case RISCV::BI__builtin_riscv_psll_bs_32:
    case RISCV::BI__builtin_riscv_psll_bs_64:
      ID = Intrinsic::riscv_psll_bs;
      break;
    case RISCV::BI__builtin_riscv_psll_hs_32:
    case RISCV::BI__builtin_riscv_psll_hs_64:
      ID = Intrinsic::riscv_psll_hs;
      break;
    case RISCV::BI__builtin_riscv_psll_ws:
      ID = Intrinsic::riscv_psll_ws;
      break;
    case RISCV::BI__builtin_riscv_padd_bs_32:
    case RISCV::BI__builtin_riscv_padd_bs_64:
      ID = Intrinsic::riscv_padd_bs;
      break;
    case RISCV::BI__builtin_riscv_padd_hs_32:
    case RISCV::BI__builtin_riscv_padd_hs_64:
      ID = Intrinsic::riscv_padd_hs;
      break;
    case RISCV::BI__builtin_riscv_padd_ws:
      ID = Intrinsic::riscv_padd_ws;
      break;
    case RISCV::BI__builtin_riscv_sadd:
      ID = Intrinsic::riscv_sadd;
      break;
    case RISCV::BI__builtin_riscv_psrl_bs_32:
    case RISCV::BI__builtin_riscv_psrl_bs_64:
      ID = Intrinsic::riscv_psrl_bs;
      break;
    case RISCV::BI__builtin_riscv_psrl_hs_32:
    case RISCV::BI__builtin_riscv_psrl_hs_64:
      ID = Intrinsic::riscv_psrl_hs;
      break;
    case RISCV::BI__builtin_riscv_psrl_ws:
      ID = Intrinsic::riscv_psrl_ws;
      break;
    case RISCV::BI__builtin_riscv_predsum_bs_32:
    case RISCV::BI__builtin_riscv_predsum_bs_64:
      ID = Intrinsic::riscv_predsum_bs;
      break;
    case RISCV::BI__builtin_riscv_predsum_hs_32:
    case RISCV::BI__builtin_riscv_predsum_hs_64:
      ID = Intrinsic::riscv_predsum_hs;
      break;
    case RISCV::BI__builtin_riscv_predsum_ws:
      ID = Intrinsic::riscv_predsum_ws;
      break;
    case RISCV::BI__builtin_riscv_predsumu_bs_32:
    case RISCV::BI__builtin_riscv_predsumu_bs_64:
      ID = Intrinsic::riscv_predsumu_bs;
      break;
    case RISCV::BI__builtin_riscv_predsumu_hs_32:
    case RISCV::BI__builtin_riscv_predsumu_hs_64:
      ID = Intrinsic::riscv_predsumu_hs;
      break;
    case RISCV::BI__builtin_riscv_predsumu_ws:
      ID = Intrinsic::riscv_predsumu_ws;
      break;
    case RISCV::BI__builtin_riscv_psra_bs_32:
    case RISCV::BI__builtin_riscv_psra_bs_64:
      ID = Intrinsic::riscv_psra_bs;
      break;
    case RISCV::BI__builtin_riscv_psra_hs_32:
    case RISCV::BI__builtin_riscv_psra_hs_64:
      ID = Intrinsic::riscv_psra_hs;
      break;
    case RISCV::BI__builtin_riscv_psra_ws:
      ID = Intrinsic::riscv_psra_ws;
      break;
    case RISCV::BI__builtin_riscv_padd_b_32:
    case RISCV::BI__builtin_riscv_padd_b_64:
      ID = Intrinsic::riscv_padd_b;
      break;
    case RISCV::BI__builtin_riscv_padd_h_32:
    case RISCV::BI__builtin_riscv_padd_h_64:
      ID = Intrinsic::riscv_padd_h;
      break;
    case RISCV::BI__builtin_riscv_padd_w:
      ID = Intrinsic::riscv_padd_w;
      break;
    case RISCV::BI__builtin_riscv_psadd_b_32:
    case RISCV::BI__builtin_riscv_psadd_b_64:
      ID = Intrinsic::riscv_psadd_b;
      break;
    case RISCV::BI__builtin_riscv_psadd_h_32:
    case RISCV::BI__builtin_riscv_psadd_h_64:
      ID = Intrinsic::riscv_psadd_h;
      break;
    case RISCV::BI__builtin_riscv_psadd_w:
      ID = Intrinsic::riscv_psadd_w;
      break;
    case RISCV::BI__builtin_riscv_aadd:
      ID = Intrinsic::riscv_aadd;
      break;
    case RISCV::BI__builtin_riscv_paadd_b_32:
    case RISCV::BI__builtin_riscv_paadd_b_64:
      ID = Intrinsic::riscv_paadd_b;
      break;
    case RISCV::BI__builtin_riscv_paadd_h_32:
    case RISCV::BI__builtin_riscv_paadd_h_64:
      ID = Intrinsic::riscv_paadd_h;
      break;
    case RISCV::BI__builtin_riscv_paadd_w:
      ID = Intrinsic::riscv_paadd_w;
      break;
    case RISCV::BI__builtin_riscv_saddu:
      ID = Intrinsic::riscv_saddu;
      break;
    case RISCV::BI__builtin_riscv_psaddu_b_32:
    case RISCV::BI__builtin_riscv_psaddu_b_64:
      ID = Intrinsic::riscv_psaddu_b;
      break;
    case RISCV::BI__builtin_riscv_psaddu_h_32:
    case RISCV::BI__builtin_riscv_psaddu_h_64:
      ID = Intrinsic::riscv_psaddu_h;
      break;
    case RISCV::BI__builtin_riscv_psaddu_w:
      ID = Intrinsic::riscv_psaddu_w;
      break;
    case RISCV::BI__builtin_riscv_aaddu:
      ID = Intrinsic::riscv_aaddu;
      break;
    case RISCV::BI__builtin_riscv_paaddu_b_32:
    case RISCV::BI__builtin_riscv_paaddu_b_64:
      ID = Intrinsic::riscv_paaddu_b;
      break;
    case RISCV::BI__builtin_riscv_paaddu_h_32:
    case RISCV::BI__builtin_riscv_paaddu_h_64:
      ID = Intrinsic::riscv_paaddu_h;
      break;
    case RISCV::BI__builtin_riscv_paaddu_w:
      ID = Intrinsic::riscv_paaddu_w;
      break;
    case RISCV::BI__builtin_riscv_psub_b_32:
    case RISCV::BI__builtin_riscv_psub_b_64:
      ID = Intrinsic::riscv_psub_b;
      break;
    case RISCV::BI__builtin_riscv_psub_h_32:
    case RISCV::BI__builtin_riscv_psub_h_64:
      ID = Intrinsic::riscv_psub_h;
      break;
    case RISCV::BI__builtin_riscv_psub_w:
      ID = Intrinsic::riscv_psub_w;
      break;
    case RISCV::BI__builtin_riscv_ssub:
      ID = Intrinsic::riscv_ssub;
      break;
    case RISCV::BI__builtin_riscv_pssub_b_32:
    case RISCV::BI__builtin_riscv_pssub_b_64:
      ID = Intrinsic::riscv_pssub_b;
      break;
    case RISCV::BI__builtin_riscv_pssub_h_32:
    case RISCV::BI__builtin_riscv_pssub_h_64:
      ID = Intrinsic::riscv_pssub_h;
      break;
    case RISCV::BI__builtin_riscv_pssub_w:
      ID = Intrinsic::riscv_pssub_w;
      break;
    case RISCV::BI__builtin_riscv_asub:
      ID = Intrinsic::riscv_asub;
      break;
    case RISCV::BI__builtin_riscv_pasub_b_32:
    case RISCV::BI__builtin_riscv_pasub_b_64:
      ID = Intrinsic::riscv_pasub_b;
      break;
    case RISCV::BI__builtin_riscv_pasub_h_32:
    case RISCV::BI__builtin_riscv_pasub_h_64:
      ID = Intrinsic::riscv_pasub_h;
      break;
    case RISCV::BI__builtin_riscv_pasub_w:
      ID = Intrinsic::riscv_pasub_w;
      break;
    case RISCV::BI__builtin_riscv_ssubu:
      ID = Intrinsic::riscv_ssubu;
      break;
    case RISCV::BI__builtin_riscv_pssubu_b_32:
    case RISCV::BI__builtin_riscv_pssubu_b_64:
      ID = Intrinsic::riscv_pssubu_b;
      break;
    case RISCV::BI__builtin_riscv_pssubu_h_32:
    case RISCV::BI__builtin_riscv_pssubu_h_64:
      ID = Intrinsic::riscv_pssubu_h;
      break;
    case RISCV::BI__builtin_riscv_pssubu_w:
      ID = Intrinsic::riscv_pssubu_w;
      break;
    case RISCV::BI__builtin_riscv_asubu:
      ID = Intrinsic::riscv_asubu;
      break;
    case RISCV::BI__builtin_riscv_pasubu_b_32:
    case RISCV::BI__builtin_riscv_pasubu_b_64:
      ID = Intrinsic::riscv_pasubu_b;
      break;
    case RISCV::BI__builtin_riscv_pasubu_h_32:
    case RISCV::BI__builtin_riscv_pasubu_h_64:
      ID = Intrinsic::riscv_pasubu_h;
      break;
    case RISCV::BI__builtin_riscv_pasubu_w:
      ID = Intrinsic::riscv_pasubu_w;
      break;
    case RISCV::BI__builtin_riscv_pdif_b_32:
    case RISCV::BI__builtin_riscv_pdif_b_64:
      ID = Intrinsic::riscv_pdif_b;
      break;
    case RISCV::BI__builtin_riscv_pdif_h_32:
    case RISCV::BI__builtin_riscv_pdif_h_64:
      ID = Intrinsic::riscv_pdif_h;
      break;
    case RISCV::BI__builtin_riscv_pdifu_b_32:
    case RISCV::BI__builtin_riscv_pdifu_b_64:
      ID = Intrinsic::riscv_pdifu_b;
      break;
    case RISCV::BI__builtin_riscv_pdifu_h_32:
    case RISCV::BI__builtin_riscv_pdifu_h_64:
      ID = Intrinsic::riscv_pdifu_h;
      break;
    case RISCV::BI__builtin_riscv_mul_h01:
      ID = Intrinsic::riscv_mul_h01;
      break;
    case RISCV::BI__builtin_riscv_mul_w01:
      ID = Intrinsic::riscv_mul_w01;
      break;
    case RISCV::BI__builtin_riscv_mulu_h01:
      ID = Intrinsic::riscv_mulu_h01;
      break;
    case RISCV::BI__builtin_riscv_mulu_w01:
      ID = Intrinsic::riscv_mulu_w01;
      break;
    case RISCV::BI__builtin_riscv_slx_32:
    case RISCV::BI__builtin_riscv_slx_64:
      ID = Intrinsic::riscv_slx;
      break;
    case RISCV::BI__builtin_riscv_psh1add_h_32:
    case RISCV::BI__builtin_riscv_psh1add_h_64:
      ID = Intrinsic::riscv_psh1add_h;
      break;
    case RISCV::BI__builtin_riscv_psh1add_w:
      ID = Intrinsic::riscv_psh1add_w;
      break;
    case RISCV::BI__builtin_riscv_ssh1sadd:
      ID = Intrinsic::riscv_ssh1sadd;
      break;
    case RISCV::BI__builtin_riscv_pssh1sadd_h_32:
    case RISCV::BI__builtin_riscv_pssh1sadd_h_64:
      ID = Intrinsic::riscv_pssh1sadd_h;
      break;
    case RISCV::BI__builtin_riscv_pssh1sadd_w:
      ID = Intrinsic::riscv_pssh1sadd_w;
      break;
    case RISCV::BI__builtin_riscv_unzip8p:
      ID = Intrinsic::riscv_unzip8p;
      break;
    case RISCV::BI__builtin_riscv_unzip16p:
      ID = Intrinsic::riscv_unzip16p;
      break;
    case RISCV::BI__builtin_riscv_unzip8hp:
      ID = Intrinsic::riscv_unzip8hp;
      break;
    case RISCV::BI__builtin_riscv_unzip16hp:
      ID = Intrinsic::riscv_unzip16hp;
      break;
    case RISCV::BI__builtin_riscv_zip8p:
      ID = Intrinsic::riscv_zip8p;
      break;
    case RISCV::BI__builtin_riscv_zip16p:
      ID = Intrinsic::riscv_zip16p;
      break;
    case RISCV::BI__builtin_riscv_zip8hp:
      ID = Intrinsic::riscv_zip8hp;
      break;
    case RISCV::BI__builtin_riscv_zip16hp:
      ID = Intrinsic::riscv_zip16hp;
      break;
    case RISCV::BI__builtin_riscv_ppack_h_32:
    case RISCV::BI__builtin_riscv_ppack_h_64:
      ID = Intrinsic::riscv_ppack_h;
      break;
    case RISCV::BI__builtin_riscv_ppack_w:
      ID = Intrinsic::riscv_ppack_w;
      break;
    case RISCV::BI__builtin_riscv_ppackbt_h_32:
    case RISCV::BI__builtin_riscv_ppackbt_h_64:
      ID = Intrinsic::riscv_ppackbt_h;
      break;
    case RISCV::BI__builtin_riscv_ppackbt_w:
      ID = Intrinsic::riscv_ppackbt_w;
      break;
    case RISCV::BI__builtin_riscv_packbt_32:
    case RISCV::BI__builtin_riscv_packbt_64:
      ID = Intrinsic::riscv_packbt;
      break;
    case RISCV::BI__builtin_riscv_ppacktb_h_32:
    case RISCV::BI__builtin_riscv_ppacktb_h_64:
      ID = Intrinsic::riscv_ppacktb_h;
      break;
    case RISCV::BI__builtin_riscv_ppacktb_w:
      ID = Intrinsic::riscv_ppacktb_w;
      break;
    case RISCV::BI__builtin_riscv_packtb_32:
    case RISCV::BI__builtin_riscv_packtb_64:
      ID = Intrinsic::riscv_packtb;
      break;
    case RISCV::BI__builtin_riscv_ppackt_h_32:
    case RISCV::BI__builtin_riscv_ppackt_h_64:
      ID = Intrinsic::riscv_ppackt_h;
      break;
    case RISCV::BI__builtin_riscv_ppackt_w:
      ID = Intrinsic::riscv_ppackt_w;
      break;
    case RISCV::BI__builtin_riscv_packt_32:
    case RISCV::BI__builtin_riscv_packt_64:
      ID = Intrinsic::riscv_packt;
      break;
    case RISCV::BI__builtin_riscv_pas_hx_32:
    case RISCV::BI__builtin_riscv_pas_hx_64:
      ID = Intrinsic::riscv_pas_hx;
      break;
    case RISCV::BI__builtin_riscv_pas_wx:
      ID = Intrinsic::riscv_pas_wx;
      break;
    case RISCV::BI__builtin_riscv_psa_hx_32:
    case RISCV::BI__builtin_riscv_psa_hx_64:
      ID = Intrinsic::riscv_psa_hx;
      break;
    case RISCV::BI__builtin_riscv_psa_wx:
      ID = Intrinsic::riscv_psa_wx;
      break;
    case RISCV::BI__builtin_riscv_psas_hx_32:
    case RISCV::BI__builtin_riscv_psas_hx_64:
      ID = Intrinsic::riscv_psas_hx;
      break;
    case RISCV::BI__builtin_riscv_psas_wx:
      ID = Intrinsic::riscv_psas_wx;
      break;
    case RISCV::BI__builtin_riscv_pssa_hx_32:
    case RISCV::BI__builtin_riscv_pssa_hx_64:
      ID = Intrinsic::riscv_pssa_hx;
      break;
    case RISCV::BI__builtin_riscv_pssa_wx:
      ID = Intrinsic::riscv_pssa_wx;
      break;
    case RISCV::BI__builtin_riscv_paas_hx_32:
    case RISCV::BI__builtin_riscv_paas_hx_64:
      ID = Intrinsic::riscv_paas_hx;
      break;
    case RISCV::BI__builtin_riscv_paas_wx:
      ID = Intrinsic::riscv_paas_wx;
      break;
    case RISCV::BI__builtin_riscv_pasa_hx_32:
    case RISCV::BI__builtin_riscv_pasa_hx_64:
      ID = Intrinsic::riscv_pasa_hx;
      break;
    case RISCV::BI__builtin_riscv_pasa_wx:
      ID = Intrinsic::riscv_pasa_wx;
      break;
    case RISCV::BI__builtin_riscv_mseq:
      ID = Intrinsic::riscv_mseq;
      break;
    case RISCV::BI__builtin_riscv_pmseq_b_32:
    case RISCV::BI__builtin_riscv_pmseq_b_64:
      ID = Intrinsic::riscv_pmseq_b;
      break;
    case RISCV::BI__builtin_riscv_pmseq_h_32:
    case RISCV::BI__builtin_riscv_pmseq_h_64:
      ID = Intrinsic::riscv_pmseq_h;
      break;
    case RISCV::BI__builtin_riscv_pmseq_w:
      ID = Intrinsic::riscv_pmseq_w;
      break;
    case RISCV::BI__builtin_riscv_mslt:
      ID = Intrinsic::riscv_mslt;
      break;
    case RISCV::BI__builtin_riscv_pmslt_b_32:
    case RISCV::BI__builtin_riscv_pmslt_b_64:
      ID = Intrinsic::riscv_pmslt_b;
      break;
    case RISCV::BI__builtin_riscv_pmslt_h_32:
    case RISCV::BI__builtin_riscv_pmslt_h_64:
      ID = Intrinsic::riscv_pmslt_h;
      break;
    case RISCV::BI__builtin_riscv_pmslt_w:
      ID = Intrinsic::riscv_pmslt_w;
      break;
    case RISCV::BI__builtin_riscv_msltu:
      ID = Intrinsic::riscv_msltu;
      break;
    case RISCV::BI__builtin_riscv_pmsltu_b_32:
    case RISCV::BI__builtin_riscv_pmsltu_b_64:
      ID = Intrinsic::riscv_pmsltu_b;
      break;
    case RISCV::BI__builtin_riscv_pmsltu_h_32:
    case RISCV::BI__builtin_riscv_pmsltu_h_64:
      ID = Intrinsic::riscv_pmsltu_h;
      break;
    case RISCV::BI__builtin_riscv_pmsltu_w:
      ID = Intrinsic::riscv_pmsltu_w;
      break;
    case RISCV::BI__builtin_riscv_pmin_b_32:
    case RISCV::BI__builtin_riscv_pmin_b_64:
      ID = Intrinsic::riscv_pmin_b;
      break;
    case RISCV::BI__builtin_riscv_pmin_h_32:
    case RISCV::BI__builtin_riscv_pmin_h_64:
      ID = Intrinsic::riscv_pmin_h;
      break;
    case RISCV::BI__builtin_riscv_pmin_w:
      ID = Intrinsic::riscv_pmin_w;
      break;
    case RISCV::BI__builtin_riscv_pminu_b_32:
    case RISCV::BI__builtin_riscv_pminu_b_64:
      ID = Intrinsic::riscv_pminu_b;
      break;
    case RISCV::BI__builtin_riscv_pminu_h_32:
    case RISCV::BI__builtin_riscv_pminu_h_64:
      ID = Intrinsic::riscv_pminu_h;
      break;
    case RISCV::BI__builtin_riscv_pminu_w:
      ID = Intrinsic::riscv_pminu_w;
      break;
    case RISCV::BI__builtin_riscv_pmax_b_32:
    case RISCV::BI__builtin_riscv_pmax_b_64:
      ID = Intrinsic::riscv_pmax_b;
      break;
    case RISCV::BI__builtin_riscv_pmax_h_32:
    case RISCV::BI__builtin_riscv_pmax_h_64:
      ID = Intrinsic::riscv_pmax_h;
      break;
    case RISCV::BI__builtin_riscv_pmax_w:
      ID = Intrinsic::riscv_pmax_w;
      break;
    case RISCV::BI__builtin_riscv_pmaxu_b_32:
    case RISCV::BI__builtin_riscv_pmaxu_b_64:
      ID = Intrinsic::riscv_pmaxu_b;
      break;
    case RISCV::BI__builtin_riscv_pmaxu_h_32:
    case RISCV::BI__builtin_riscv_pmaxu_h_64:
      ID = Intrinsic::riscv_pmaxu_h;
      break;
    case RISCV::BI__builtin_riscv_pmaxu_w:
      ID = Intrinsic::riscv_pmaxu_w;
      break;
    case RISCV::BI__builtin_riscv_pmulh_h_32:
    case RISCV::BI__builtin_riscv_pmulh_h_64:
      ID = Intrinsic::riscv_pmulh_h;
      break;
    case RISCV::BI__builtin_riscv_pmulh_w:
      ID = Intrinsic::riscv_pmulh_w;
      break;
    case RISCV::BI__builtin_riscv_pmulhu_h_32:
    case RISCV::BI__builtin_riscv_pmulhu_h_64:
      ID = Intrinsic::riscv_pmulhu_h;
      break;
    case RISCV::BI__builtin_riscv_pmulhu_w:
      ID = Intrinsic::riscv_pmulhu_w;
      break;
    case RISCV::BI__builtin_riscv_pmulhr_h_32:
    case RISCV::BI__builtin_riscv_pmulhr_h_64:
      ID = Intrinsic::riscv_pmulhr_h;
      break;
    case RISCV::BI__builtin_riscv_pmulhr_w:
      ID = Intrinsic::riscv_pmulhr_w;
      break;
    case RISCV::BI__builtin_riscv_pmulhru_h_32:
    case RISCV::BI__builtin_riscv_pmulhru_h_64:
      ID = Intrinsic::riscv_pmulhru_h;
      break;
    case RISCV::BI__builtin_riscv_pmulhru_w:
      ID = Intrinsic::riscv_pmulhru_w;
      break;
    case RISCV::BI__builtin_riscv_mulh_h1:
      ID = Intrinsic::riscv_mulh_h1;
      break;
    case RISCV::BI__builtin_riscv_mulhr:
      ID = Intrinsic::riscv_mulhr;
      break;
    case RISCV::BI__builtin_riscv_mulhru:
      ID = Intrinsic::riscv_mulhru;
      break;
    case RISCV::BI__builtin_riscv_mulh_h0:
      ID = Intrinsic::riscv_mulh_h0;
      break;
    }

    IntrinsicTypes = {Ops[0]->getType()};
    break;
  }

  // Intrinsic type is obtained from Result and Ops[1].
  case RISCV::BI__builtin_riscv_pslli_b_32:
  case RISCV::BI__builtin_riscv_pslli_b_64:
  case RISCV::BI__builtin_riscv_pslli_h_32:
  case RISCV::BI__builtin_riscv_pslli_h_64:
  case RISCV::BI__builtin_riscv_pslli_w:
  case RISCV::BI__builtin_riscv_psslai_h_32:
  case RISCV::BI__builtin_riscv_psslai_h_64:
  case RISCV::BI__builtin_riscv_psslai_w:
  case RISCV::BI__builtin_riscv_sslai:
  case RISCV::BI__builtin_riscv_pusati_h_32:
  case RISCV::BI__builtin_riscv_pusati_h_64:
  case RISCV::BI__builtin_riscv_pusati_w:
  case RISCV::BI__builtin_riscv_usati_32:
  case RISCV::BI__builtin_riscv_usati_64:
  case RISCV::BI__builtin_riscv_psrai_b_32:
  case RISCV::BI__builtin_riscv_psrai_b_64:
  case RISCV::BI__builtin_riscv_psrai_h_32:
  case RISCV::BI__builtin_riscv_psrai_h_64:
  case RISCV::BI__builtin_riscv_psrai_w:
  case RISCV::BI__builtin_riscv_psrari_h_32:
  case RISCV::BI__builtin_riscv_psrari_h_64:
  case RISCV::BI__builtin_riscv_psrari_w:
  case RISCV::BI__builtin_riscv_srari_32:
  case RISCV::BI__builtin_riscv_srari_64:
  case RISCV::BI__builtin_riscv_psati_h_32:
  case RISCV::BI__builtin_riscv_psati_h_64:
  case RISCV::BI__builtin_riscv_psati_w:
  case RISCV::BI__builtin_riscv_sati_32:
  case RISCV::BI__builtin_riscv_sati_64:
  case RISCV::BI__builtin_riscv_pmulhrsu_h_32:
  case RISCV::BI__builtin_riscv_pmulhrsu_h_64:
  case RISCV::BI__builtin_riscv_pmulhrsu_w:
  case RISCV::BI__builtin_riscv_mulhsu_h0:
  case RISCV::BI__builtin_riscv_mulhsu_h1:
  case RISCV::BI__builtin_riscv_mulhrsu: {
    switch (BuiltinID) {
    default:
      llvm_unreachable("unexpected builtin ID");
    case RISCV::BI__builtin_riscv_pslli_b_32:
    case RISCV::BI__builtin_riscv_pslli_b_64:
      ID = Intrinsic::riscv_pslli_b;
      break;
    case RISCV::BI__builtin_riscv_pslli_h_32:
    case RISCV::BI__builtin_riscv_pslli_h_64:
      ID = Intrinsic::riscv_pslli_h;
      break;
    case RISCV::BI__builtin_riscv_pslli_w:
      ID = Intrinsic::riscv_pslli_w;
      break;
    case RISCV::BI__builtin_riscv_psslai_h_32:
    case RISCV::BI__builtin_riscv_psslai_h_64:
      ID = Intrinsic::riscv_psslai_h;
      break;
    case RISCV::BI__builtin_riscv_psslai_w:
      ID = Intrinsic::riscv_psslai_w;
      break;
    case RISCV::BI__builtin_riscv_sslai:
      ID = Intrinsic::riscv_sslai;
      break;
    case RISCV::BI__builtin_riscv_pusati_h_32:
    case RISCV::BI__builtin_riscv_pusati_h_64:
      ID = Intrinsic::riscv_pusati_h;
      break;
    case RISCV::BI__builtin_riscv_pusati_w:
      ID = Intrinsic::riscv_pusati_w;
      break;
    case RISCV::BI__builtin_riscv_usati_32:
    case RISCV::BI__builtin_riscv_usati_64:
      ID = Intrinsic::riscv_usati;
      break;
    case RISCV::BI__builtin_riscv_psrai_b_32:
    case RISCV::BI__builtin_riscv_psrai_b_64:
      ID = Intrinsic::riscv_psrai_b;
      break;
    case RISCV::BI__builtin_riscv_psrai_h_32:
    case RISCV::BI__builtin_riscv_psrai_h_64:
      ID = Intrinsic::riscv_psrai_h;
      break;
    case RISCV::BI__builtin_riscv_psrai_w:
      ID = Intrinsic::riscv_psrai_w;
      break;
    case RISCV::BI__builtin_riscv_psrari_h_32:
    case RISCV::BI__builtin_riscv_psrari_h_64:
      ID = Intrinsic::riscv_psrari_h;
      break;
    case RISCV::BI__builtin_riscv_psrari_w:
      ID = Intrinsic::riscv_psrari_w;
      break;
    case RISCV::BI__builtin_riscv_srari_32:
    case RISCV::BI__builtin_riscv_srari_64:
      ID = Intrinsic::riscv_srari;
      break;
    case RISCV::BI__builtin_riscv_psati_h_32:
    case RISCV::BI__builtin_riscv_psati_h_64:
      ID = Intrinsic::riscv_psati_h;
      break;
    case RISCV::BI__builtin_riscv_psati_w:
      ID = Intrinsic::riscv_psati_w;
      break;
    case RISCV::BI__builtin_riscv_sati_32:
    case RISCV::BI__builtin_riscv_sati_64:
      ID = Intrinsic::riscv_sati;
      break;
    case RISCV::BI__builtin_riscv_pmulhrsu_h_32:
    case RISCV::BI__builtin_riscv_pmulhrsu_h_64:
      ID = Intrinsic::riscv_pmulhrsu_h;
      break;
    case RISCV::BI__builtin_riscv_pmulhrsu_w:
      ID = Intrinsic::riscv_pmulhrsu_w;
      break;
    case RISCV::BI__builtin_riscv_mulhsu_h0:
      ID = Intrinsic::riscv_mulhsu_h0;
      break;
    case RISCV::BI__builtin_riscv_mulhsu_h1:
      ID = Intrinsic::riscv_mulhsu_h1;
      break;
    case RISCV::BI__builtin_riscv_mulhrsu:
      ID = Intrinsic::riscv_mulhrsu;
      break;
    }

    IntrinsicTypes = {ResultType, Ops[1]->getType()};
    break;
  }

  // Intrinsic type is obtained from Result and Ops[0].
  case RISCV::BI__builtin_riscv_pmul_h_b01_32:
  case RISCV::BI__builtin_riscv_pmul_h_b01_64:
  case RISCV::BI__builtin_riscv_pmul_w_h01:
  case RISCV::BI__builtin_riscv_pmulu_h_b01_32:
  case RISCV::BI__builtin_riscv_pmulu_h_b01_64:
  case RISCV::BI__builtin_riscv_pmulu_w_h01:
  case RISCV::BI__builtin_riscv_pmul_h_b00_32:
  case RISCV::BI__builtin_riscv_pmul_h_b00_64:
  case RISCV::BI__builtin_riscv_pmul_w_h00:
  case RISCV::BI__builtin_riscv_pmul_h_b11_32:
  case RISCV::BI__builtin_riscv_pmul_h_b11_64:
  case RISCV::BI__builtin_riscv_pmul_w_h11:
  case RISCV::BI__builtin_riscv_pmulu_h_b00_32:
  case RISCV::BI__builtin_riscv_pmulu_h_b00_64:
  case RISCV::BI__builtin_riscv_pmulu_w_h00:
  case RISCV::BI__builtin_riscv_pmulu_h_b11_32:
  case RISCV::BI__builtin_riscv_pmulu_h_b11_64:
  case RISCV::BI__builtin_riscv_pmulu_w_h11:
  case RISCV::BI__builtin_riscv_mul_h00:
  case RISCV::BI__builtin_riscv_mul_w00:
  case RISCV::BI__builtin_riscv_mul_h11:
  case RISCV::BI__builtin_riscv_mul_w11:
  case RISCV::BI__builtin_riscv_mulu_h00:
  case RISCV::BI__builtin_riscv_mulu_w00:
  case RISCV::BI__builtin_riscv_mulu_h11:
  case RISCV::BI__builtin_riscv_mulu_w11:
  case RISCV::BI__builtin_riscv_pmulh_h_b0_32:
  case RISCV::BI__builtin_riscv_pmulh_h_b0_64:
  case RISCV::BI__builtin_riscv_pmulh_w_h0:
  case RISCV::BI__builtin_riscv_pmulh_h_b1_32:
  case RISCV::BI__builtin_riscv_pmulh_h_b1_64:
  case RISCV::BI__builtin_riscv_pmulh_w_h1: {
    switch (BuiltinID) {
    default:
      llvm_unreachable("unexpected builtin ID");
    case RISCV::BI__builtin_riscv_pmul_h_b01_32:
    case RISCV::BI__builtin_riscv_pmul_h_b01_64:
      ID = Intrinsic::riscv_pmul_h_b01;
      break;
    case RISCV::BI__builtin_riscv_pmul_w_h01:
      ID = Intrinsic::riscv_pmul_w_h01;
      break;
    case RISCV::BI__builtin_riscv_pmulu_h_b01_32:
    case RISCV::BI__builtin_riscv_pmulu_h_b01_64:
      ID = Intrinsic::riscv_pmulu_h_b01;
      break;
    case RISCV::BI__builtin_riscv_pmulu_w_h01:
      ID = Intrinsic::riscv_pmulu_w_h01;
      break;
    case RISCV::BI__builtin_riscv_pmul_h_b00_32:
    case RISCV::BI__builtin_riscv_pmul_h_b00_64:
      ID = Intrinsic::riscv_pmul_h_b00;
      break;
    case RISCV::BI__builtin_riscv_pmul_w_h00:
      ID = Intrinsic::riscv_pmul_w_h00;
      break;
    case RISCV::BI__builtin_riscv_pmul_h_b11_32:
    case RISCV::BI__builtin_riscv_pmul_h_b11_64:
      ID = Intrinsic::riscv_pmul_h_b11;
      break;
    case RISCV::BI__builtin_riscv_pmul_w_h11:
      ID = Intrinsic::riscv_pmul_w_h11;
      break;
    case RISCV::BI__builtin_riscv_pmulu_h_b00_32:
    case RISCV::BI__builtin_riscv_pmulu_h_b00_64:
      ID = Intrinsic::riscv_pmulu_h_b00;
      break;
    case RISCV::BI__builtin_riscv_pmulu_w_h00:
      ID = Intrinsic::riscv_pmulu_w_h00;
      break;
    case RISCV::BI__builtin_riscv_pmulu_h_b11_32:
    case RISCV::BI__builtin_riscv_pmulu_h_b11_64:
      ID = Intrinsic::riscv_pmulu_h_b11;
      break;
    case RISCV::BI__builtin_riscv_pmulu_w_h11:
      ID = Intrinsic::riscv_pmulu_w_h11;
      break;
    case RISCV::BI__builtin_riscv_mul_h00:
      ID = Intrinsic::riscv_mul_h00;
      break;
    case RISCV::BI__builtin_riscv_mul_w00:
      ID = Intrinsic::riscv_mul_w00;
      break;
    case RISCV::BI__builtin_riscv_mul_h11:
      ID = Intrinsic::riscv_mul_h11;
      break;
    case RISCV::BI__builtin_riscv_mul_w11:
      ID = Intrinsic::riscv_mul_w11;
      break;
    case RISCV::BI__builtin_riscv_mulu_h00:
      ID = Intrinsic::riscv_mulu_h00;
      break;
    case RISCV::BI__builtin_riscv_mulu_w00:
      ID = Intrinsic::riscv_mulu_w00;
      break;
    case RISCV::BI__builtin_riscv_mulu_h11:
      ID = Intrinsic::riscv_mulu_h11;
      break;
    case RISCV::BI__builtin_riscv_mulu_w11:
      ID = Intrinsic::riscv_mulu_w11;
      break;
    case RISCV::BI__builtin_riscv_pmulh_h_b0_32:
    case RISCV::BI__builtin_riscv_pmulh_h_b0_64:
      ID = Intrinsic::riscv_pmulh_h_b0;
      break;
    case RISCV::BI__builtin_riscv_pmulh_w_h0:
      ID = Intrinsic::riscv_pmulh_w_h0;
      break;
    case RISCV::BI__builtin_riscv_pmulh_h_b1_32:
    case RISCV::BI__builtin_riscv_pmulh_h_b1_64:
      ID = Intrinsic::riscv_pmulh_h_b1;
      break;
    case RISCV::BI__builtin_riscv_pmulh_w_h1:
      ID = Intrinsic::riscv_pmulh_w_h1;
      break;
    }

    IntrinsicTypes = {ResultType, Ops[0]->getType()};
    break;
  }

  case RISCV::BI__builtin_riscv_pmulsu_h_b00_32:
  case RISCV::BI__builtin_riscv_pmulsu_h_b00_64:
  case RISCV::BI__builtin_riscv_pmulsu_w_h00:
  case RISCV::BI__builtin_riscv_pmulsu_h_b11_32:
  case RISCV::BI__builtin_riscv_pmulsu_h_b11_64:
  case RISCV::BI__builtin_riscv_pmulsu_w_h11:
  case RISCV::BI__builtin_riscv_mulsu_h00:
  case RISCV::BI__builtin_riscv_mulsu_w00:
  case RISCV::BI__builtin_riscv_mulsu_h11:
  case RISCV::BI__builtin_riscv_mulsu_w11:
  case RISCV::BI__builtin_riscv_pmulhsu_h_32:
  case RISCV::BI__builtin_riscv_pmulhsu_h_64:
  case RISCV::BI__builtin_riscv_pmulhsu_w:
  case RISCV::BI__builtin_riscv_pmulhsu_h_b0_32:
  case RISCV::BI__builtin_riscv_pmulhsu_h_b0_64:
  case RISCV::BI__builtin_riscv_pmulhsu_w_h0:
  case RISCV::BI__builtin_riscv_pmulhsu_h_b1_32:
  case RISCV::BI__builtin_riscv_pmulhsu_h_b1_64:
  case RISCV::BI__builtin_riscv_pmulhsu_w_h1: {
    switch (BuiltinID) {
    default:
      llvm_unreachable("unexpected builtin ID");
    case RISCV::BI__builtin_riscv_pmulsu_h_b00_32:
    case RISCV::BI__builtin_riscv_pmulsu_h_b00_64:
      ID = Intrinsic::riscv_pmulsu_h_b00;
      break;
    case RISCV::BI__builtin_riscv_pmulsu_w_h00:
      ID = Intrinsic::riscv_pmulsu_w_h00;
      break;
    case RISCV::BI__builtin_riscv_pmulsu_h_b11_32:
    case RISCV::BI__builtin_riscv_pmulsu_h_b11_64:
      ID = Intrinsic::riscv_pmulsu_h_b11;
      break;
    case RISCV::BI__builtin_riscv_pmulsu_w_h11:
      ID = Intrinsic::riscv_pmulsu_w_h11;
      break;
    case RISCV::BI__builtin_riscv_mulsu_h00:
      ID = Intrinsic::riscv_mulsu_h00;
      break;
    case RISCV::BI__builtin_riscv_mulsu_w00:
      ID = Intrinsic::riscv_mulsu_w00;
      break;
    case RISCV::BI__builtin_riscv_mulsu_h11:
      ID = Intrinsic::riscv_mulsu_h11;
      break;
    case RISCV::BI__builtin_riscv_mulsu_w11:
      ID = Intrinsic::riscv_mulsu_w11;
      break;
    case RISCV::BI__builtin_riscv_pmulhsu_h_32:
    case RISCV::BI__builtin_riscv_pmulhsu_h_64:
      ID = Intrinsic::riscv_pmulhsu_h;
      break;
    case RISCV::BI__builtin_riscv_pmulhsu_w:
      ID = Intrinsic::riscv_pmulhsu_w;
      break;
    case RISCV::BI__builtin_riscv_pmulhsu_h_b0_32:
    case RISCV::BI__builtin_riscv_pmulhsu_h_b0_64:
      ID = Intrinsic::riscv_pmulhsu_h_b0;
      break;
    case RISCV::BI__builtin_riscv_pmulhsu_w_h0:
      ID = Intrinsic::riscv_pmulhsu_w_h0;
      break;
    case RISCV::BI__builtin_riscv_pmulhsu_h_b1_32:
    case RISCV::BI__builtin_riscv_pmulhsu_h_b1_64:
      ID = Intrinsic::riscv_pmulhsu_h_b1;
      break;
    case RISCV::BI__builtin_riscv_pmulhsu_w_h1:
      ID = Intrinsic::riscv_pmulhsu_w_h1;
      break;
    }

    IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType()};
    break;
  }

  // Zk builtins

  // Zknh
  case RISCV::BI__builtin_riscv_sha256sig0:
    ID = Intrinsic::riscv_sha256sig0;
    break;
  case RISCV::BI__builtin_riscv_sha256sig1:
    ID = Intrinsic::riscv_sha256sig1;
    break;
  case RISCV::BI__builtin_riscv_sha256sum0:
    ID = Intrinsic::riscv_sha256sum0;
    break;
  case RISCV::BI__builtin_riscv_sha256sum1:
    ID = Intrinsic::riscv_sha256sum1;
    break;

  // Zksed
  case RISCV::BI__builtin_riscv_sm4ks:
    ID = Intrinsic::riscv_sm4ks;
    break;
  case RISCV::BI__builtin_riscv_sm4ed:
    ID = Intrinsic::riscv_sm4ed;
    break;

  // Zksh
  case RISCV::BI__builtin_riscv_sm3p0:
    ID = Intrinsic::riscv_sm3p0;
    break;
  case RISCV::BI__builtin_riscv_sm3p1:
    ID = Intrinsic::riscv_sm3p1;
    break;

  case RISCV::BI__builtin_riscv_clz_32:
  case RISCV::BI__builtin_riscv_clz_64: {
    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, Ops[0]->getType());
    Value *Result = Builder.CreateCall(F, {Ops[0], Builder.getInt1(false)});
    if (Result->getType() != ResultType)
      Result =
          Builder.CreateIntCast(Result, ResultType, /*isSigned*/ false, "cast");
    return Result;
  }
  case RISCV::BI__builtin_riscv_ctz_32:
  case RISCV::BI__builtin_riscv_ctz_64: {
    Function *F = CGM.getIntrinsic(Intrinsic::cttz, Ops[0]->getType());
    Value *Result = Builder.CreateCall(F, {Ops[0], Builder.getInt1(false)});
    if (Result->getType() != ResultType)
      Result =
          Builder.CreateIntCast(Result, ResultType, /*isSigned*/ false, "cast");
    return Result;
  }

  // Zihintntl
  case RISCV::BI__builtin_riscv_ntl_load: {
    llvm::Type *ResTy = ConvertType(E->getType());
    unsigned DomainVal = 5; // Default __RISCV_NTLH_ALL
    if (Ops.size() == 2)
      DomainVal = cast<ConstantInt>(Ops[1])->getZExtValue();

    llvm::MDNode *RISCVDomainNode = llvm::MDNode::get(
        getLLVMContext(),
        llvm::ConstantAsMetadata::get(Builder.getInt32(DomainVal)));
    llvm::MDNode *NontemporalNode = llvm::MDNode::get(
        getLLVMContext(), llvm::ConstantAsMetadata::get(Builder.getInt32(1)));

    int Width;
    if(ResTy->isScalableTy()) {
      const ScalableVectorType *SVTy = cast<ScalableVectorType>(ResTy);
      llvm::Type *ScalarTy = ResTy->getScalarType();
      Width = ScalarTy->getPrimitiveSizeInBits() *
              SVTy->getElementCount().getKnownMinValue();
    } else
      Width = ResTy->getPrimitiveSizeInBits();
    LoadInst *Load = Builder.CreateLoad(
        Address(Ops[0], ResTy, CharUnits::fromQuantity(Width / 8)));

    Load->setMetadata(llvm::LLVMContext::MD_nontemporal, NontemporalNode);
    Load->setMetadata(CGM.getModule().getMDKindID("riscv-nontemporal-domain"),
                      RISCVDomainNode);

    return Load;
  }
  case RISCV::BI__builtin_riscv_ntl_store: {
    unsigned DomainVal = 5; // Default __RISCV_NTLH_ALL
    if (Ops.size() == 3)
      DomainVal = cast<ConstantInt>(Ops[2])->getZExtValue();

    llvm::MDNode *RISCVDomainNode = llvm::MDNode::get(
        getLLVMContext(),
        llvm::ConstantAsMetadata::get(Builder.getInt32(DomainVal)));
    llvm::MDNode *NontemporalNode = llvm::MDNode::get(
        getLLVMContext(), llvm::ConstantAsMetadata::get(Builder.getInt32(1)));

    StoreInst *Store = Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
    Store->setMetadata(llvm::LLVMContext::MD_nontemporal, NontemporalNode);
    Store->setMetadata(CGM.getModule().getMDKindID("riscv-nontemporal-domain"),
                       RISCVDomainNode);

    return Store;
  }
  // Zihintpause
  case RISCV::BI__builtin_riscv_pause: {
    llvm::Function *Fn = CGM.getIntrinsic(llvm::Intrinsic::riscv_pause);
    return Builder.CreateCall(Fn, {});
  }

  // XCValu
  case RISCV::BI__builtin_riscv_cv_alu_addN:
    ID = Intrinsic::riscv_cv_alu_addN;
    break;
  case RISCV::BI__builtin_riscv_cv_alu_addRN:
    ID = Intrinsic::riscv_cv_alu_addRN;
    break;
  case RISCV::BI__builtin_riscv_cv_alu_adduN:
    ID = Intrinsic::riscv_cv_alu_adduN;
    break;
  case RISCV::BI__builtin_riscv_cv_alu_adduRN:
    ID = Intrinsic::riscv_cv_alu_adduRN;
    break;
  case RISCV::BI__builtin_riscv_cv_alu_clip:
    ID = Intrinsic::riscv_cv_alu_clip;
    break;
  case RISCV::BI__builtin_riscv_cv_alu_clipu:
    ID = Intrinsic::riscv_cv_alu_clipu;
    break;
  case RISCV::BI__builtin_riscv_cv_alu_extbs:
    return Builder.CreateSExt(Builder.CreateTrunc(Ops[0], Int8Ty), Int32Ty,
                              "extbs");
  case RISCV::BI__builtin_riscv_cv_alu_extbz:
    return Builder.CreateZExt(Builder.CreateTrunc(Ops[0], Int8Ty), Int32Ty,
                              "extbz");
  case RISCV::BI__builtin_riscv_cv_alu_exths:
    return Builder.CreateSExt(Builder.CreateTrunc(Ops[0], Int16Ty), Int32Ty,
                              "exths");
  case RISCV::BI__builtin_riscv_cv_alu_exthz:
    return Builder.CreateZExt(Builder.CreateTrunc(Ops[0], Int16Ty), Int32Ty,
                              "exthz");
  case RISCV::BI__builtin_riscv_cv_alu_sle:
    return Builder.CreateZExt(Builder.CreateICmpSLE(Ops[0], Ops[1]), Int32Ty,
                              "sle");
  case RISCV::BI__builtin_riscv_cv_alu_sleu:
    return Builder.CreateZExt(Builder.CreateICmpULE(Ops[0], Ops[1]), Int32Ty,
                              "sleu");
  case RISCV::BI__builtin_riscv_cv_alu_subN:
    ID = Intrinsic::riscv_cv_alu_subN;
    break;
  case RISCV::BI__builtin_riscv_cv_alu_subRN:
    ID = Intrinsic::riscv_cv_alu_subRN;
    break;
  case RISCV::BI__builtin_riscv_cv_alu_subuN:
    ID = Intrinsic::riscv_cv_alu_subuN;
    break;
  case RISCV::BI__builtin_riscv_cv_alu_subuRN:
    ID = Intrinsic::riscv_cv_alu_subuRN;
    break;

  // XAndesBFHCvt
  case RISCV::BI__builtin_riscv_nds_fcvt_s_bf16:
    return Builder.CreateFPExt(Ops[0], FloatTy);
  case RISCV::BI__builtin_riscv_nds_fcvt_bf16_s:
    return Builder.CreateFPTrunc(Ops[0], BFloatTy);

    // Vector builtins are handled from here.
#include "clang/Basic/riscv_vector_builtin_cg.inc"

    // SiFive Vector builtins are handled from here.
#include "clang/Basic/riscv_sifive_vector_builtin_cg.inc"

    // Andes Vector builtins are handled from here.
#include "clang/Basic/riscv_andes_vector_builtin_cg.inc"
  }

  assert(ID != Intrinsic::not_intrinsic);

  llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
  return Builder.CreateCall(F, Ops, "");
}
