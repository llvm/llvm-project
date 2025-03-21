//===------- AMDCPU.cpp - Emit LLVM Code for builtins ---------------------===//
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

#include "ABIInfo.h"
#include "CGHLSLRuntime.h"
#include "CGBuiltin.h"
#include "TargetInfo.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/ConvertUTF.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

namespace {
// If \p E is not null pointer, insert address space cast to match return
// type of \p E if necessary.
Value *EmitAMDGPUDispatchPtr(CodeGenFunction &CGF,
                             const CallExpr *E = nullptr) {
  auto *F = CGF.CGM.getIntrinsic(Intrinsic::amdgcn_dispatch_ptr);
  auto *Call = CGF.Builder.CreateCall(F);
  Call->addRetAttr(
      Attribute::getWithDereferenceableBytes(Call->getContext(), 64));
  Call->addRetAttr(Attribute::getWithAlignment(Call->getContext(), Align(4)));
  if (!E)
    return Call;
  QualType BuiltinRetType = E->getType();
  auto *RetTy = cast<llvm::PointerType>(CGF.ConvertType(BuiltinRetType));
  if (RetTy == Call->getType())
    return Call;
  return CGF.Builder.CreateAddrSpaceCast(Call, RetTy);
}

Value *EmitAMDGPUImplicitArgPtr(CodeGenFunction &CGF) {
  auto *F = CGF.CGM.getIntrinsic(Intrinsic::amdgcn_implicitarg_ptr);
  auto *Call = CGF.Builder.CreateCall(F);
  Call->addRetAttr(
      Attribute::getWithDereferenceableBytes(Call->getContext(), 256));
  Call->addRetAttr(Attribute::getWithAlignment(Call->getContext(), Align(8)));
  return Call;
}

// \p Index is 0, 1, and 2 for x, y, and z dimension, respectively.
/// Emit code based on Code Object ABI version.
/// COV_4    : Emit code to use dispatch ptr
/// COV_5+   : Emit code to use implicitarg ptr
/// COV_NONE : Emit code to load a global variable "__oclc_ABI_version"
///            and use its value for COV_4 or COV_5+ approach. It is used for
///            compiling device libraries in an ABI-agnostic way.
///
/// Note: "__oclc_ABI_version" is supposed to be emitted and intialized by
///       clang during compilation of user code.
Value *EmitAMDGPUWorkGroupSize(CodeGenFunction &CGF, unsigned Index) {
  llvm::LoadInst *LD;

  auto Cov = CGF.getTarget().getTargetOpts().CodeObjectVersion;

  if (Cov == CodeObjectVersionKind::COV_None) {
    StringRef Name = "__oclc_ABI_version";
    auto *ABIVersionC = CGF.CGM.getModule().getNamedGlobal(Name);
    if (!ABIVersionC)
      ABIVersionC = new llvm::GlobalVariable(
          CGF.CGM.getModule(), CGF.Int32Ty, false,
          llvm::GlobalValue::ExternalLinkage, nullptr, Name, nullptr,
          llvm::GlobalVariable::NotThreadLocal,
          CGF.CGM.getContext().getTargetAddressSpace(LangAS::opencl_constant));

    // This load will be eliminated by the IPSCCP because it is constant
    // weak_odr without externally_initialized. Either changing it to weak or
    // adding externally_initialized will keep the load.
    Value *ABIVersion = CGF.Builder.CreateAlignedLoad(CGF.Int32Ty, ABIVersionC,
                                                      CGF.CGM.getIntAlign());

    Value *IsCOV5 = CGF.Builder.CreateICmpSGE(
        ABIVersion,
        llvm::ConstantInt::get(CGF.Int32Ty, CodeObjectVersionKind::COV_5));

    // Indexing the implicit kernarg segment.
    Value *ImplicitGEP = CGF.Builder.CreateConstGEP1_32(
        CGF.Int8Ty, EmitAMDGPUImplicitArgPtr(CGF), 12 + Index * 2);

    // Indexing the HSA kernel_dispatch_packet struct.
    Value *DispatchGEP = CGF.Builder.CreateConstGEP1_32(
        CGF.Int8Ty, EmitAMDGPUDispatchPtr(CGF), 4 + Index * 2);

    auto Result = CGF.Builder.CreateSelect(IsCOV5, ImplicitGEP, DispatchGEP);
    LD = CGF.Builder.CreateLoad(
        Address(Result, CGF.Int16Ty, CharUnits::fromQuantity(2)));
  } else {
    Value *GEP = nullptr;
    if (Cov >= CodeObjectVersionKind::COV_5) {
      // Indexing the implicit kernarg segment.
      GEP = CGF.Builder.CreateConstGEP1_32(
          CGF.Int8Ty, EmitAMDGPUImplicitArgPtr(CGF), 12 + Index * 2);
    } else {
      // Indexing the HSA kernel_dispatch_packet struct.
      GEP = CGF.Builder.CreateConstGEP1_32(
          CGF.Int8Ty, EmitAMDGPUDispatchPtr(CGF), 4 + Index * 2);
    }
    LD = CGF.Builder.CreateLoad(
        Address(GEP, CGF.Int16Ty, CharUnits::fromQuantity(2)));
  }

  llvm::MDBuilder MDHelper(CGF.getLLVMContext());
  llvm::MDNode *RNode = MDHelper.createRange(APInt(16, 1),
      APInt(16, CGF.getTarget().getMaxOpenCLWorkGroupSize() + 1));
  LD->setMetadata(llvm::LLVMContext::MD_range, RNode);
  LD->setMetadata(llvm::LLVMContext::MD_noundef,
                  llvm::MDNode::get(CGF.getLLVMContext(), {}));
  LD->setMetadata(llvm::LLVMContext::MD_invariant_load,
                  llvm::MDNode::get(CGF.getLLVMContext(), {}));
  return LD;
}

// \p Index is 0, 1, and 2 for x, y, and z dimension, respectively.
Value *EmitAMDGPUGridSize(CodeGenFunction &CGF, unsigned Index) {
  const unsigned XOffset = 12;
  auto *DP = EmitAMDGPUDispatchPtr(CGF);
  // Indexing the HSA kernel_dispatch_packet struct.
  auto *Offset = llvm::ConstantInt::get(CGF.Int32Ty, XOffset + Index * 4);
  auto *GEP = CGF.Builder.CreateGEP(CGF.Int8Ty, DP, Offset);
  auto *LD = CGF.Builder.CreateLoad(
      Address(GEP, CGF.Int32Ty, CharUnits::fromQuantity(4)));

  llvm::MDBuilder MDB(CGF.getLLVMContext());

  // Known non-zero.
  LD->setMetadata(llvm::LLVMContext::MD_range,
                  MDB.createRange(APInt(32, 1), APInt::getZero(32)));
  LD->setMetadata(llvm::LLVMContext::MD_invariant_load,
                  llvm::MDNode::get(CGF.getLLVMContext(), {}));
  return LD;
}
} // namespace

// Generates the IR for __builtin_read_exec_*.
// Lowers the builtin to amdgcn_ballot intrinsic.
static Value *EmitAMDGCNBallotForExec(CodeGenFunction &CGF, const CallExpr *E,
                                      llvm::Type *RegisterType,
                                      llvm::Type *ValueType, bool isExecHi) {
  CodeGen::CGBuilderTy &Builder = CGF.Builder;
  CodeGen::CodeGenModule &CGM = CGF.CGM;

  Function *F = CGM.getIntrinsic(Intrinsic::amdgcn_ballot, {RegisterType});
  llvm::Value *Call = Builder.CreateCall(F, {Builder.getInt1(true)});

  if (isExecHi) {
    Value *Rt2 = Builder.CreateLShr(Call, 32);
    Rt2 = Builder.CreateTrunc(Rt2, CGF.Int32Ty);
    return Rt2;
  }

  return Call;
}

// Emit an intrinsic that has 1 float or double operand, and 1 integer.
static Value *emitFPIntBuiltin(CodeGenFunction &CGF,
                               const CallExpr *E,
                               unsigned IntrinsicID) {
  llvm::Value *Src0 = CGF.EmitScalarExpr(E->getArg(0));
  llvm::Value *Src1 = CGF.EmitScalarExpr(E->getArg(1));

  Function *F = CGF.CGM.getIntrinsic(IntrinsicID, Src0->getType());
  return CGF.Builder.CreateCall(F, {Src0, Src1});
}

static Value *emitRangedBuiltin(CodeGenFunction &CGF, unsigned IntrinsicID,
                                int low, int high) {
  Function *F = CGF.CGM.getIntrinsic(IntrinsicID, {});
  llvm::CallInst *Call = CGF.Builder.CreateCall(F);
  llvm::ConstantRange CR(APInt(32, low), APInt(32, high));
  Call->addRangeRetAttr(CR);
  Call->addRetAttr(llvm::Attribute::AttrKind::NoUndef);
  return Call;
}

static Value *handleAsDoubleBuiltin(CodeGenFunction &CGF, const CallExpr *E) {
  assert((E->getArg(0)->getType()->hasUnsignedIntegerRepresentation() &&
          E->getArg(1)->getType()->hasUnsignedIntegerRepresentation()) &&
         "asdouble operands types mismatch");
  Value *OpLowBits = CGF.EmitScalarExpr(E->getArg(0));
  Value *OpHighBits = CGF.EmitScalarExpr(E->getArg(1));

  llvm::Type *ResultType = CGF.DoubleTy;
  int N = 1;
  if (auto *VTy = E->getArg(0)->getType()->getAs<clang::VectorType>()) {
    N = VTy->getNumElements();
    ResultType = llvm::FixedVectorType::get(CGF.DoubleTy, N);
  }

  if (CGF.CGM.getTarget().getTriple().isDXIL())
    return CGF.Builder.CreateIntrinsic(
        /*ReturnType=*/ResultType, Intrinsic::dx_asdouble,
        {OpLowBits, OpHighBits}, nullptr, "hlsl.asdouble");

  if (!E->getArg(0)->getType()->isVectorType()) {
    OpLowBits = CGF.Builder.CreateVectorSplat(1, OpLowBits);
    OpHighBits = CGF.Builder.CreateVectorSplat(1, OpHighBits);
  }

  llvm::SmallVector<int> Mask;
  for (int i = 0; i < N; i++) {
    Mask.push_back(i);
    Mask.push_back(i + N);
  }

  Value *BitVec = CGF.Builder.CreateShuffleVector(OpLowBits, OpHighBits, Mask);

  return CGF.Builder.CreateBitCast(BitVec, ResultType);
}

static Value *handleHlslClip(const CallExpr *E, CodeGenFunction *CGF) {
  Value *Op0 = CGF->EmitScalarExpr(E->getArg(0));

  Constant *FZeroConst = ConstantFP::getZero(CGF->FloatTy);
  Value *CMP;
  Value *LastInstr;

  if (const auto *VecTy = E->getArg(0)->getType()->getAs<clang::VectorType>()) {
    FZeroConst = ConstantVector::getSplat(
        ElementCount::getFixed(VecTy->getNumElements()), FZeroConst);
    auto *FCompInst = CGF->Builder.CreateFCmpOLT(Op0, FZeroConst);
    CMP = CGF->Builder.CreateIntrinsic(
        CGF->Builder.getInt1Ty(), CGF->CGM.getHLSLRuntime().getAnyIntrinsic(),
        {FCompInst});
  } else
    CMP = CGF->Builder.CreateFCmpOLT(Op0, FZeroConst);

  if (CGF->CGM.getTarget().getTriple().isDXIL())
    LastInstr = CGF->Builder.CreateIntrinsic(
        CGF->VoidTy, Intrinsic::dx_discard, {CMP});
  else if (CGF->CGM.getTarget().getTriple().isSPIRV()) {
    BasicBlock *LT0 = CGF->createBasicBlock("lt0", CGF->CurFn);
    BasicBlock *End = CGF->createBasicBlock("end", CGF->CurFn);

    CGF->Builder.CreateCondBr(CMP, LT0, End);

    CGF->Builder.SetInsertPoint(LT0);

    CGF->Builder.CreateIntrinsic(CGF->VoidTy, Intrinsic::spv_discard, {});

    LastInstr = CGF->Builder.CreateBr(End);
    CGF->Builder.SetInsertPoint(End);
  } else {
    llvm_unreachable("Backend Codegen not supported.");
  }

  return LastInstr;
}

static Value *handleHlslSplitdouble(const CallExpr *E, CodeGenFunction *CGF) {
  Value *Op0 = CGF->EmitScalarExpr(E->getArg(0));
  const auto *OutArg1 = dyn_cast<HLSLOutArgExpr>(E->getArg(1));
  const auto *OutArg2 = dyn_cast<HLSLOutArgExpr>(E->getArg(2));

  CallArgList Args;
  LValue Op1TmpLValue =
      CGF->EmitHLSLOutArgExpr(OutArg1, Args, OutArg1->getType());
  LValue Op2TmpLValue =
      CGF->EmitHLSLOutArgExpr(OutArg2, Args, OutArg2->getType());

  if (CGF->getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee())
    Args.reverseWritebacks();

  Value *LowBits = nullptr;
  Value *HighBits = nullptr;

  if (CGF->CGM.getTarget().getTriple().isDXIL()) {

    llvm::Type *RetElementTy = CGF->Int32Ty;
    if (auto *Op0VecTy = E->getArg(0)->getType()->getAs<clang::VectorType>())
      RetElementTy = llvm::VectorType::get(
          CGF->Int32Ty, ElementCount::getFixed(Op0VecTy->getNumElements()));
    auto *RetTy = llvm::StructType::get(RetElementTy, RetElementTy);

    CallInst *CI = CGF->Builder.CreateIntrinsic(
        RetTy, Intrinsic::dx_splitdouble, {Op0}, nullptr, "hlsl.splitdouble");

    LowBits = CGF->Builder.CreateExtractValue(CI, 0);
    HighBits = CGF->Builder.CreateExtractValue(CI, 1);

  } else {
    // For Non DXIL targets we generate the instructions.

    if (!Op0->getType()->isVectorTy()) {
      FixedVectorType *DestTy = FixedVectorType::get(CGF->Int32Ty, 2);
      Value *Bitcast = CGF->Builder.CreateBitCast(Op0, DestTy);

      LowBits = CGF->Builder.CreateExtractElement(Bitcast, (uint64_t)0);
      HighBits = CGF->Builder.CreateExtractElement(Bitcast, 1);
    } else {
      int NumElements = 1;
      if (const auto *VecTy =
              E->getArg(0)->getType()->getAs<clang::VectorType>())
        NumElements = VecTy->getNumElements();

      FixedVectorType *Uint32VecTy =
          FixedVectorType::get(CGF->Int32Ty, NumElements * 2);
      Value *Uint32Vec = CGF->Builder.CreateBitCast(Op0, Uint32VecTy);
      if (NumElements == 1) {
        LowBits = CGF->Builder.CreateExtractElement(Uint32Vec, (uint64_t)0);
        HighBits = CGF->Builder.CreateExtractElement(Uint32Vec, 1);
      } else {
        SmallVector<int> EvenMask, OddMask;
        for (int I = 0, E = NumElements; I != E; ++I) {
          EvenMask.push_back(I * 2);
          OddMask.push_back(I * 2 + 1);
        }
        LowBits = CGF->Builder.CreateShuffleVector(Uint32Vec, EvenMask);
        HighBits = CGF->Builder.CreateShuffleVector(Uint32Vec, OddMask);
      }
    }
  }
  CGF->Builder.CreateStore(LowBits, Op1TmpLValue.getAddress());
  auto *LastInst =
      CGF->Builder.CreateStore(HighBits, Op2TmpLValue.getAddress());
  CGF->EmitWritebacks(Args);
  return LastInst;
}

// For processing memory ordering and memory scope arguments of various
// amdgcn builtins.
// \p Order takes a C++11 comptabile memory-ordering specifier and converts
// it into LLVM's memory ordering specifier using atomic C ABI, and writes
// to \p AO. \p Scope takes a const char * and converts it into AMDGCN
// specific SyncScopeID and writes it to \p SSID.
void CodeGenFunction::ProcessOrderScopeAMDGCN(Value *Order, Value *Scope,
                                              llvm::AtomicOrdering &AO,
                                              llvm::SyncScope::ID &SSID) {
  int ord = cast<llvm::ConstantInt>(Order)->getZExtValue();

  // Map C11/C++11 memory ordering to LLVM memory ordering
  assert(llvm::isValidAtomicOrderingCABI(ord));
  switch (static_cast<llvm::AtomicOrderingCABI>(ord)) {
  case llvm::AtomicOrderingCABI::acquire:
  case llvm::AtomicOrderingCABI::consume:
    AO = llvm::AtomicOrdering::Acquire;
    break;
  case llvm::AtomicOrderingCABI::release:
    AO = llvm::AtomicOrdering::Release;
    break;
  case llvm::AtomicOrderingCABI::acq_rel:
    AO = llvm::AtomicOrdering::AcquireRelease;
    break;
  case llvm::AtomicOrderingCABI::seq_cst:
    AO = llvm::AtomicOrdering::SequentiallyConsistent;
    break;
  case llvm::AtomicOrderingCABI::relaxed:
    AO = llvm::AtomicOrdering::Monotonic;
    break;
  }

  // Some of the atomic builtins take the scope as a string name.
  StringRef scp;
  if (llvm::getConstantStringInfo(Scope, scp)) {
    SSID = getLLVMContext().getOrInsertSyncScopeID(scp);
    return;
  }

  // Older builtins had an enum argument for the memory scope.
  int scope = cast<llvm::ConstantInt>(Scope)->getZExtValue();
  switch (scope) {
  case 0: // __MEMORY_SCOPE_SYSTEM
    SSID = llvm::SyncScope::System;
    break;
  case 1: // __MEMORY_SCOPE_DEVICE
    SSID = getLLVMContext().getOrInsertSyncScopeID("agent");
    break;
  case 2: // __MEMORY_SCOPE_WRKGRP
    SSID = getLLVMContext().getOrInsertSyncScopeID("workgroup");
    break;
  case 3: // __MEMORY_SCOPE_WVFRNT
    SSID = getLLVMContext().getOrInsertSyncScopeID("wavefront");
    break;
  case 4: // __MEMORY_SCOPE_SINGLE
    SSID = llvm::SyncScope::SingleThread;
    break;
  default:
    SSID = llvm::SyncScope::System;
    break;
  }
}

llvm::Value *CodeGenFunction::EmitScalarOrConstFoldImmArg(unsigned ICEArguments,
                                                          unsigned Idx,
                                                          const CallExpr *E) {
  llvm::Value *Arg = nullptr;
  if ((ICEArguments & (1 << Idx)) == 0) {
    Arg = EmitScalarExpr(E->getArg(Idx));
  } else {
    // If this is required to be a constant, constant fold it so that we
    // know that the generated intrinsic gets a ConstantInt.
    std::optional<llvm::APSInt> Result =
        E->getArg(Idx)->getIntegerConstantExpr(getContext());
    assert(Result && "Expected argument to be a constant");
    Arg = llvm::ConstantInt::get(getLLVMContext(), *Result);
  }
  return Arg;
}

// Return dot product intrinsic that corresponds to the QT scalar type
static Intrinsic::ID getDotProductIntrinsic(CGHLSLRuntime &RT, QualType QT) {
  if (QT->isFloatingType())
    return RT.getFDotIntrinsic();
  if (QT->isSignedIntegerType())
    return RT.getSDotIntrinsic();
  assert(QT->isUnsignedIntegerType());
  return RT.getUDotIntrinsic();
}

static Intrinsic::ID getFirstBitHighIntrinsic(CGHLSLRuntime &RT, QualType QT) {
  if (QT->hasSignedIntegerRepresentation()) {
    return RT.getFirstBitSHighIntrinsic();
  }

  assert(QT->hasUnsignedIntegerRepresentation());
  return RT.getFirstBitUHighIntrinsic();
}

// Return wave active sum that corresponds to the QT scalar type
static Intrinsic::ID getWaveActiveSumIntrinsic(llvm::Triple::ArchType Arch,
                                               CGHLSLRuntime &RT, QualType QT) {
  switch (Arch) {
  case llvm::Triple::spirv:
    return Intrinsic::spv_wave_reduce_sum;
  case llvm::Triple::dxil: {
    if (QT->isUnsignedIntegerType())
      return Intrinsic::dx_wave_reduce_usum;
    return Intrinsic::dx_wave_reduce_sum;
  }
  default:
    llvm_unreachable("Intrinsic WaveActiveSum"
                     " not supported by target architecture");
  }
}

// Return wave active sum that corresponds to the QT scalar type
static Intrinsic::ID getWaveActiveMaxIntrinsic(llvm::Triple::ArchType Arch,
                                               CGHLSLRuntime &RT, QualType QT) {
  switch (Arch) {
  case llvm::Triple::spirv:
    if (QT->isUnsignedIntegerType())
      return Intrinsic::spv_wave_reduce_umax;
    return Intrinsic::spv_wave_reduce_max;
  case llvm::Triple::dxil: {
    if (QT->isUnsignedIntegerType())
      return Intrinsic::dx_wave_reduce_umax;
    return Intrinsic::dx_wave_reduce_max;
  }
  default:
    llvm_unreachable("Intrinsic WaveActiveMax"
                     " not supported by target architecture");
  }
}

Value *CodeGenFunction::EmitHLSLBuiltinExpr(unsigned BuiltinID,
                                            const CallExpr *E,
                                            ReturnValueSlot ReturnValue) {
  if (!getLangOpts().HLSL)
    return nullptr;

  switch (BuiltinID) {
  case Builtin::BI__builtin_hlsl_adduint64: {
    Value *OpA = EmitScalarExpr(E->getArg(0));
    Value *OpB = EmitScalarExpr(E->getArg(1));
    QualType Arg0Ty = E->getArg(0)->getType();
    uint64_t NumElements = Arg0Ty->castAs<VectorType>()->getNumElements();
    assert(Arg0Ty == E->getArg(1)->getType() &&
           "AddUint64 operand types must match");
    assert(Arg0Ty->hasIntegerRepresentation() &&
           "AddUint64 operands must have an integer representation");
    assert((NumElements == 2 || NumElements == 4) &&
           "AddUint64 operands must have 2 or 4 elements");

    llvm::Value *LowA;
    llvm::Value *HighA;
    llvm::Value *LowB;
    llvm::Value *HighB;

    // Obtain low and high words of inputs A and B
    if (NumElements == 2) {
      LowA = Builder.CreateExtractElement(OpA, (uint64_t)0, "LowA");
      HighA = Builder.CreateExtractElement(OpA, (uint64_t)1, "HighA");
      LowB = Builder.CreateExtractElement(OpB, (uint64_t)0, "LowB");
      HighB = Builder.CreateExtractElement(OpB, (uint64_t)1, "HighB");
    } else {
      LowA = Builder.CreateShuffleVector(OpA, {0, 2}, "LowA");
      HighA = Builder.CreateShuffleVector(OpA, {1, 3}, "HighA");
      LowB = Builder.CreateShuffleVector(OpB, {0, 2}, "LowB");
      HighB = Builder.CreateShuffleVector(OpB, {1, 3}, "HighB");
    }

    // Use an uadd_with_overflow to compute the sum of low words and obtain a
    // carry value
    llvm::Value *Carry;
    llvm::Value *LowSum = EmitOverflowIntrinsic(
        *this, Intrinsic::uadd_with_overflow, LowA, LowB, Carry);
    llvm::Value *ZExtCarry =
        Builder.CreateZExt(Carry, HighA->getType(), "CarryZExt");

    // Sum the high words and the carry
    llvm::Value *HighSum = Builder.CreateAdd(HighA, HighB, "HighSum");
    llvm::Value *HighSumPlusCarry =
        Builder.CreateAdd(HighSum, ZExtCarry, "HighSumPlusCarry");

    if (NumElements == 4) {
      return Builder.CreateShuffleVector(LowSum, HighSumPlusCarry,
                                         {0, 2, 1, 3},
                                         "hlsl.AddUint64");
    }

    llvm::Value *Result = PoisonValue::get(OpA->getType());
    Result = Builder.CreateInsertElement(Result, LowSum, (uint64_t)0,
                                         "hlsl.AddUint64.upto0");
    Result = Builder.CreateInsertElement(Result, HighSumPlusCarry, (uint64_t)1,
                                         "hlsl.AddUint64");
    return Result;
  }
  case Builtin::BI__builtin_hlsl_resource_getpointer: {
    Value *HandleOp = EmitScalarExpr(E->getArg(0));
    Value *IndexOp = EmitScalarExpr(E->getArg(1));

    // TODO: Map to an hlsl_device address space.
    llvm::Type *RetTy = llvm::PointerType::getUnqual(getLLVMContext());

    return Builder.CreateIntrinsic(
        RetTy, CGM.getHLSLRuntime().getCreateResourceGetPointerIntrinsic(),
        ArrayRef<Value *>{HandleOp, IndexOp});
  }
  case Builtin::BI__builtin_hlsl_all: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    return Builder.CreateIntrinsic(
        /*ReturnType=*/llvm::Type::getInt1Ty(getLLVMContext()),
        CGM.getHLSLRuntime().getAllIntrinsic(), ArrayRef<Value *>{Op0}, nullptr,
        "hlsl.all");
  }
  case Builtin::BI__builtin_hlsl_and: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    return Builder.CreateAnd(Op0, Op1, "hlsl.and");
  }
  case Builtin::BI__builtin_hlsl_or: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    return Builder.CreateOr(Op0, Op1, "hlsl.or");
  }
  case Builtin::BI__builtin_hlsl_any: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    return Builder.CreateIntrinsic(
        /*ReturnType=*/llvm::Type::getInt1Ty(getLLVMContext()),
        CGM.getHLSLRuntime().getAnyIntrinsic(), ArrayRef<Value *>{Op0}, nullptr,
        "hlsl.any");
  }
  case Builtin::BI__builtin_hlsl_asdouble:
    return handleAsDoubleBuiltin(*this, E);
  case Builtin::BI__builtin_hlsl_elementwise_clamp: {
    Value *OpX = EmitScalarExpr(E->getArg(0));
    Value *OpMin = EmitScalarExpr(E->getArg(1));
    Value *OpMax = EmitScalarExpr(E->getArg(2));

    QualType Ty = E->getArg(0)->getType();
    if (auto *VecTy = Ty->getAs<VectorType>())
      Ty = VecTy->getElementType();

    Intrinsic::ID Intr;
    if (Ty->isFloatingType()) {
      Intr = CGM.getHLSLRuntime().getNClampIntrinsic();
    } else if (Ty->isUnsignedIntegerType()) {
      Intr = CGM.getHLSLRuntime().getUClampIntrinsic();
    } else {
      assert(Ty->isSignedIntegerType());
      Intr = CGM.getHLSLRuntime().getSClampIntrinsic();
    }
    return Builder.CreateIntrinsic(
        /*ReturnType=*/OpX->getType(), Intr,
        ArrayRef<Value *>{OpX, OpMin, OpMax}, nullptr, "hlsl.clamp");
  }
  case Builtin::BI__builtin_hlsl_cross: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           E->getArg(1)->getType()->hasFloatingRepresentation() &&
           "cross operands must have a float representation");
    // make sure each vector has exactly 3 elements
    assert(
        E->getArg(0)->getType()->castAs<VectorType>()->getNumElements() == 3 &&
        E->getArg(1)->getType()->castAs<VectorType>()->getNumElements() == 3 &&
        "input vectors must have 3 elements each");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(), CGM.getHLSLRuntime().getCrossIntrinsic(),
        ArrayRef<Value *>{Op0, Op1}, nullptr, "hlsl.cross");
  }
  case Builtin::BI__builtin_hlsl_dot: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    llvm::Type *T0 = Op0->getType();
    llvm::Type *T1 = Op1->getType();

    // If the arguments are scalars, just emit a multiply
    if (!T0->isVectorTy() && !T1->isVectorTy()) {
      if (T0->isFloatingPointTy())
        return Builder.CreateFMul(Op0, Op1, "hlsl.dot");

      if (T0->isIntegerTy())
        return Builder.CreateMul(Op0, Op1, "hlsl.dot");

      llvm_unreachable(
          "Scalar dot product is only supported on ints and floats.");
    }
    // For vectors, validate types and emit the appropriate intrinsic

    // A VectorSplat should have happened
    assert(T0->isVectorTy() && T1->isVectorTy() &&
           "Dot product of vector and scalar is not supported.");

    auto *VecTy0 = E->getArg(0)->getType()->castAs<VectorType>();
    [[maybe_unused]] auto *VecTy1 =
        E->getArg(1)->getType()->castAs<VectorType>();

    assert(VecTy0->getElementType() == VecTy1->getElementType() &&
           "Dot product of vectors need the same element types.");

    assert(VecTy0->getNumElements() == VecTy1->getNumElements() &&
           "Dot product requires vectors to be of the same size.");

    return Builder.CreateIntrinsic(
        /*ReturnType=*/T0->getScalarType(),
        getDotProductIntrinsic(CGM.getHLSLRuntime(), VecTy0->getElementType()),
        ArrayRef<Value *>{Op0, Op1}, nullptr, "hlsl.dot");
  }
  case Builtin::BI__builtin_hlsl_dot4add_i8packed: {
    Value *A = EmitScalarExpr(E->getArg(0));
    Value *B = EmitScalarExpr(E->getArg(1));
    Value *C = EmitScalarExpr(E->getArg(2));

    Intrinsic::ID ID = CGM.getHLSLRuntime().getDot4AddI8PackedIntrinsic();
    return Builder.CreateIntrinsic(
        /*ReturnType=*/C->getType(), ID, ArrayRef<Value *>{A, B, C}, nullptr,
        "hlsl.dot4add.i8packed");
  }
  case Builtin::BI__builtin_hlsl_dot4add_u8packed: {
    Value *A = EmitScalarExpr(E->getArg(0));
    Value *B = EmitScalarExpr(E->getArg(1));
    Value *C = EmitScalarExpr(E->getArg(2));

    Intrinsic::ID ID = CGM.getHLSLRuntime().getDot4AddU8PackedIntrinsic();
    return Builder.CreateIntrinsic(
        /*ReturnType=*/C->getType(), ID, ArrayRef<Value *>{A, B, C}, nullptr,
        "hlsl.dot4add.u8packed");
  }
  case Builtin::BI__builtin_hlsl_elementwise_firstbithigh: {
    Value *X = EmitScalarExpr(E->getArg(0));

    return Builder.CreateIntrinsic(
        /*ReturnType=*/ConvertType(E->getType()),
        getFirstBitHighIntrinsic(CGM.getHLSLRuntime(), E->getArg(0)->getType()),
        ArrayRef<Value *>{X}, nullptr, "hlsl.firstbithigh");
  }
  case Builtin::BI__builtin_hlsl_elementwise_firstbitlow: {
    Value *X = EmitScalarExpr(E->getArg(0));

    return Builder.CreateIntrinsic(
        /*ReturnType=*/ConvertType(E->getType()),
        CGM.getHLSLRuntime().getFirstBitLowIntrinsic(), ArrayRef<Value *>{X},
        nullptr, "hlsl.firstbitlow");
  }
  case Builtin::BI__builtin_hlsl_lerp: {
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    Value *S = EmitScalarExpr(E->getArg(2));
    if (!E->getArg(0)->getType()->hasFloatingRepresentation())
      llvm_unreachable("lerp operand must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/X->getType(), CGM.getHLSLRuntime().getLerpIntrinsic(),
        ArrayRef<Value *>{X, Y, S}, nullptr, "hlsl.lerp");
  }
  case Builtin::BI__builtin_hlsl_normalize: {
    Value *X = EmitScalarExpr(E->getArg(0));

    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           "normalize operand must have a float representation");

    return Builder.CreateIntrinsic(
        /*ReturnType=*/X->getType(),
        CGM.getHLSLRuntime().getNormalizeIntrinsic(), ArrayRef<Value *>{X},
        nullptr, "hlsl.normalize");
  }
  case Builtin::BI__builtin_hlsl_elementwise_degrees: {
    Value *X = EmitScalarExpr(E->getArg(0));

    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           "degree operand must have a float representation");

    return Builder.CreateIntrinsic(
        /*ReturnType=*/X->getType(), CGM.getHLSLRuntime().getDegreesIntrinsic(),
        ArrayRef<Value *>{X}, nullptr, "hlsl.degrees");
  }
  case Builtin::BI__builtin_hlsl_elementwise_frac: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    if (!E->getArg(0)->getType()->hasFloatingRepresentation())
      llvm_unreachable("frac operand must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(), CGM.getHLSLRuntime().getFracIntrinsic(),
        ArrayRef<Value *>{Op0}, nullptr, "hlsl.frac");
}
case Builtin::BI__builtin_hlsl_elementwise_isinf: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    llvm::Type *Xty = Op0->getType();
    llvm::Type *retType = llvm::Type::getInt1Ty(this->getLLVMContext());
    if (Xty->isVectorTy()) {
      auto *XVecTy = E->getArg(0)->getType()->castAs<VectorType>();
      retType = llvm::VectorType::get(
          retType, ElementCount::getFixed(XVecTy->getNumElements()));
    }
    if (!E->getArg(0)->getType()->hasFloatingRepresentation())
      llvm_unreachable("isinf operand must have a float representation");
    return Builder.CreateIntrinsic(retType, Intrinsic::dx_isinf,
                                   ArrayRef<Value *>{Op0}, nullptr, "dx.isinf");
  }
  case Builtin::BI__builtin_hlsl_mad: {
    Value *M = EmitScalarExpr(E->getArg(0));
    Value *A = EmitScalarExpr(E->getArg(1));
    Value *B = EmitScalarExpr(E->getArg(2));
    if (E->getArg(0)->getType()->hasFloatingRepresentation())
      return Builder.CreateIntrinsic(
          /*ReturnType*/ M->getType(), Intrinsic::fmuladd,
          ArrayRef<Value *>{M, A, B}, nullptr, "hlsl.fmad");

    if (E->getArg(0)->getType()->hasSignedIntegerRepresentation()) {
      if (CGM.getTarget().getTriple().getArch() == llvm::Triple::dxil)
        return Builder.CreateIntrinsic(
            /*ReturnType*/ M->getType(), Intrinsic::dx_imad,
            ArrayRef<Value *>{M, A, B}, nullptr, "dx.imad");

      Value *Mul = Builder.CreateNSWMul(M, A);
      return Builder.CreateNSWAdd(Mul, B);
    }
    assert(E->getArg(0)->getType()->hasUnsignedIntegerRepresentation());
    if (CGM.getTarget().getTriple().getArch() == llvm::Triple::dxil)
      return Builder.CreateIntrinsic(
          /*ReturnType=*/M->getType(), Intrinsic::dx_umad,
          ArrayRef<Value *>{M, A, B}, nullptr, "dx.umad");

    Value *Mul = Builder.CreateNUWMul(M, A);
    return Builder.CreateNUWAdd(Mul, B);
  }
  case Builtin::BI__builtin_hlsl_elementwise_rcp: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    if (!E->getArg(0)->getType()->hasFloatingRepresentation())
      llvm_unreachable("rcp operand must have a float representation");
    llvm::Type *Ty = Op0->getType();
    llvm::Type *EltTy = Ty->getScalarType();
    Constant *One = Ty->isVectorTy()
                        ? ConstantVector::getSplat(
                              ElementCount::getFixed(
                                  cast<FixedVectorType>(Ty)->getNumElements()),
                              ConstantFP::get(EltTy, 1.0))
                        : ConstantFP::get(EltTy, 1.0);
    return Builder.CreateFDiv(One, Op0, "hlsl.rcp");
  }
  case Builtin::BI__builtin_hlsl_elementwise_rsqrt: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    if (!E->getArg(0)->getType()->hasFloatingRepresentation())
      llvm_unreachable("rsqrt operand must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(), CGM.getHLSLRuntime().getRsqrtIntrinsic(),
        ArrayRef<Value *>{Op0}, nullptr, "hlsl.rsqrt");
  }
  case Builtin::BI__builtin_hlsl_elementwise_saturate: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           "saturate operand must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(),
        CGM.getHLSLRuntime().getSaturateIntrinsic(), ArrayRef<Value *>{Op0},
        nullptr, "hlsl.saturate");
  }
  case Builtin::BI__builtin_hlsl_select: {
    Value *OpCond = EmitScalarExpr(E->getArg(0));
    RValue RValTrue = EmitAnyExpr(E->getArg(1));
    Value *OpTrue =
        RValTrue.isScalar()
            ? RValTrue.getScalarVal()
            : RValTrue.getAggregatePointer(E->getArg(1)->getType(), *this);
    RValue RValFalse = EmitAnyExpr(E->getArg(2));
    Value *OpFalse =
        RValFalse.isScalar()
            ? RValFalse.getScalarVal()
            : RValFalse.getAggregatePointer(E->getArg(2)->getType(), *this);
    if (auto *VTy = E->getType()->getAs<VectorType>()) {
      if (!OpTrue->getType()->isVectorTy())
        OpTrue =
            Builder.CreateVectorSplat(VTy->getNumElements(), OpTrue, "splat");
      if (!OpFalse->getType()->isVectorTy())
        OpFalse =
            Builder.CreateVectorSplat(VTy->getNumElements(), OpFalse, "splat");
    }

    Value *SelectVal =
        Builder.CreateSelect(OpCond, OpTrue, OpFalse, "hlsl.select");
    if (!RValTrue.isScalar())
      Builder.CreateStore(SelectVal, ReturnValue.getAddress(),
                          ReturnValue.isVolatile());

    return SelectVal;
  }
  case Builtin::BI__builtin_hlsl_step: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           E->getArg(1)->getType()->hasFloatingRepresentation() &&
           "step operands must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(), CGM.getHLSLRuntime().getStepIntrinsic(),
        ArrayRef<Value *>{Op0, Op1}, nullptr, "hlsl.step");
  }
  case Builtin::BI__builtin_hlsl_wave_active_all_true: {
    Value *Op = EmitScalarExpr(E->getArg(0));
    assert(Op->getType()->isIntegerTy(1) &&
           "Intrinsic WaveActiveAllTrue operand must be a bool");

    Intrinsic::ID ID = CGM.getHLSLRuntime().getWaveActiveAllTrueIntrinsic();
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID), {Op});
  }
  case Builtin::BI__builtin_hlsl_wave_active_any_true: {
    Value *Op = EmitScalarExpr(E->getArg(0));
    assert(Op->getType()->isIntegerTy(1) &&
           "Intrinsic WaveActiveAnyTrue operand must be a bool");

    Intrinsic::ID ID = CGM.getHLSLRuntime().getWaveActiveAnyTrueIntrinsic();
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID), {Op});
  }
  case Builtin::BI__builtin_hlsl_wave_active_count_bits: {
    Value *OpExpr = EmitScalarExpr(E->getArg(0));
    Intrinsic::ID ID = CGM.getHLSLRuntime().getWaveActiveCountBitsIntrinsic();
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID),
        ArrayRef{OpExpr});
  }
  case Builtin::BI__builtin_hlsl_wave_active_sum: {
    // Due to the use of variadic arguments, explicitly retreive argument
    Value *OpExpr = EmitScalarExpr(E->getArg(0));
    llvm::FunctionType *FT = llvm::FunctionType::get(
        OpExpr->getType(), ArrayRef{OpExpr->getType()}, false);
    Intrinsic::ID IID = getWaveActiveSumIntrinsic(
        getTarget().getTriple().getArch(), CGM.getHLSLRuntime(),
        E->getArg(0)->getType());

    // Get overloaded name
    std::string Name =
        Intrinsic::getName(IID, ArrayRef{OpExpr->getType()}, &CGM.getModule());
    return EmitRuntimeCall(CGM.CreateRuntimeFunction(FT, Name, {},
                                                     /*Local=*/false,
                                                     /*AssumeConvergent=*/true),
                           ArrayRef{OpExpr}, "hlsl.wave.active.sum");
  }
  case Builtin::BI__builtin_hlsl_wave_active_max: {
    // Due to the use of variadic arguments, explicitly retreive argument
    Value *OpExpr = EmitScalarExpr(E->getArg(0));
    llvm::FunctionType *FT = llvm::FunctionType::get(
        OpExpr->getType(), ArrayRef{OpExpr->getType()}, false);
    Intrinsic::ID IID = getWaveActiveMaxIntrinsic(
        getTarget().getTriple().getArch(), CGM.getHLSLRuntime(),
        E->getArg(0)->getType());

    // Get overloaded name
    std::string Name =
        Intrinsic::getName(IID, ArrayRef{OpExpr->getType()}, &CGM.getModule());
    return EmitRuntimeCall(CGM.CreateRuntimeFunction(FT, Name, {},
                                                     /*Local=*/false,
                                                     /*AssumeConvergent=*/true),
                           ArrayRef{OpExpr}, "hlsl.wave.active.max");
  }
  case Builtin::BI__builtin_hlsl_wave_get_lane_index: {
    // We don't define a SPIR-V intrinsic, instead it is a SPIR-V built-in
    // defined in SPIRVBuiltins.td. So instead we manually get the matching name
    // for the DirectX intrinsic and the demangled builtin name
    switch (CGM.getTarget().getTriple().getArch()) {
    case llvm::Triple::dxil:
      return EmitRuntimeCall(Intrinsic::getOrInsertDeclaration(
          &CGM.getModule(), Intrinsic::dx_wave_getlaneindex));
    case llvm::Triple::spirv:
      return EmitRuntimeCall(CGM.CreateRuntimeFunction(
          llvm::FunctionType::get(IntTy, {}, false),
          "__hlsl_wave_get_lane_index", {}, false, true));
    default:
      llvm_unreachable(
          "Intrinsic WaveGetLaneIndex not supported by target architecture");
    }
  }
  case Builtin::BI__builtin_hlsl_wave_is_first_lane: {
    Intrinsic::ID ID = CGM.getHLSLRuntime().getWaveIsFirstLaneIntrinsic();
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID));
  }
  case Builtin::BI__builtin_hlsl_wave_read_lane_at: {
    // Due to the use of variadic arguments we must explicitly retreive them and
    // create our function type.
    Value *OpExpr = EmitScalarExpr(E->getArg(0));
    Value *OpIndex = EmitScalarExpr(E->getArg(1));
    llvm::FunctionType *FT = llvm::FunctionType::get(
        OpExpr->getType(), ArrayRef{OpExpr->getType(), OpIndex->getType()},
        false);

    // Get overloaded name
    std::string Name =
        Intrinsic::getName(CGM.getHLSLRuntime().getWaveReadLaneAtIntrinsic(),
                           ArrayRef{OpExpr->getType()}, &CGM.getModule());
    return EmitRuntimeCall(CGM.CreateRuntimeFunction(FT, Name, {},
                                                     /*Local=*/false,
                                                     /*AssumeConvergent=*/true),
                           ArrayRef{OpExpr, OpIndex}, "hlsl.wave.readlane");
  }
  case Builtin::BI__builtin_hlsl_elementwise_sign: {
    auto *Arg0 = E->getArg(0);
    Value *Op0 = EmitScalarExpr(Arg0);
    llvm::Type *Xty = Op0->getType();
    llvm::Type *retType = llvm::Type::getInt32Ty(this->getLLVMContext());
    if (Xty->isVectorTy()) {
      auto *XVecTy = Arg0->getType()->castAs<VectorType>();
      retType = llvm::VectorType::get(
          retType, ElementCount::getFixed(XVecTy->getNumElements()));
    }
    assert((Arg0->getType()->hasFloatingRepresentation() ||
            Arg0->getType()->hasIntegerRepresentation()) &&
           "sign operand must have a float or int representation");

    if (Arg0->getType()->hasUnsignedIntegerRepresentation()) {
      Value *Cmp = Builder.CreateICmpEQ(Op0, ConstantInt::get(Xty, 0));
      return Builder.CreateSelect(Cmp, ConstantInt::get(retType, 0),
                                  ConstantInt::get(retType, 1), "hlsl.sign");
    }

    return Builder.CreateIntrinsic(
        retType, CGM.getHLSLRuntime().getSignIntrinsic(),
        ArrayRef<Value *>{Op0}, nullptr, "hlsl.sign");
  }
  case Builtin::BI__builtin_hlsl_elementwise_radians: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           "radians operand must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(),
        CGM.getHLSLRuntime().getRadiansIntrinsic(), ArrayRef<Value *>{Op0},
        nullptr, "hlsl.radians");
  }
  case Builtin::BI__builtin_hlsl_buffer_update_counter: {
    Value *ResHandle = EmitScalarExpr(E->getArg(0));
    Value *Offset = EmitScalarExpr(E->getArg(1));
    Value *OffsetI8 = Builder.CreateIntCast(Offset, Int8Ty, true);
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Offset->getType(),
        CGM.getHLSLRuntime().getBufferUpdateCounterIntrinsic(),
        ArrayRef<Value *>{ResHandle, OffsetI8}, nullptr);
  }
  case Builtin::BI__builtin_hlsl_elementwise_splitdouble: {

    assert((E->getArg(0)->getType()->hasFloatingRepresentation() &&
            E->getArg(1)->getType()->hasUnsignedIntegerRepresentation() &&
            E->getArg(2)->getType()->hasUnsignedIntegerRepresentation()) &&
           "asuint operands types mismatch");
    return handleHlslSplitdouble(E, this);
  }
  case Builtin::BI__builtin_hlsl_elementwise_clip:
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           "clip operands types mismatch");
    return handleHlslClip(E, this);
  case Builtin::BI__builtin_hlsl_group_memory_barrier_with_group_sync: {
    Intrinsic::ID ID =
        CGM.getHLSLRuntime().getGroupMemoryBarrierWithGroupSyncIntrinsic();
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID));
  }
  }
  return nullptr;
}

void CodeGenFunction::AddAMDGPUFenceAddressSpaceMMRA(llvm::Instruction *Inst,
                                                     const CallExpr *E) {
  constexpr const char *Tag = "amdgpu-as";

  LLVMContext &Ctx = Inst->getContext();
  SmallVector<MMRAMetadata::TagT, 3> MMRAs;
  for (unsigned K = 2; K < E->getNumArgs(); ++K) {
    llvm::Value *V = EmitScalarExpr(E->getArg(K));
    StringRef AS;
    if (llvm::getConstantStringInfo(V, AS)) {
      MMRAs.push_back({Tag, AS});
      // TODO: Delete the resulting unused constant?
      continue;
    }
    CGM.Error(E->getExprLoc(),
              "expected an address space name as a string literal");
  }

  llvm::sort(MMRAs);
  MMRAs.erase(llvm::unique(MMRAs), MMRAs.end());
  Inst->setMetadata(LLVMContext::MD_mmra, MMRAMetadata::getMD(Ctx, MMRAs));
}

Value *CodeGenFunction::EmitAMDGPUBuiltinExpr(unsigned BuiltinID,
                                              const CallExpr *E) {
  llvm::AtomicOrdering AO = llvm::AtomicOrdering::SequentiallyConsistent;
  llvm::SyncScope::ID SSID;
  switch (BuiltinID) {
  case AMDGPU::BI__builtin_amdgcn_div_scale:
  case AMDGPU::BI__builtin_amdgcn_div_scalef: {
    // Translate from the intrinsics's struct return to the builtin's out
    // argument.

    Address FlagOutPtr = EmitPointerWithAlignment(E->getArg(3));

    llvm::Value *X = EmitScalarExpr(E->getArg(0));
    llvm::Value *Y = EmitScalarExpr(E->getArg(1));
    llvm::Value *Z = EmitScalarExpr(E->getArg(2));

    llvm::Function *Callee = CGM.getIntrinsic(Intrinsic::amdgcn_div_scale,
                                           X->getType());

    llvm::Value *Tmp = Builder.CreateCall(Callee, {X, Y, Z});

    llvm::Value *Result = Builder.CreateExtractValue(Tmp, 0);
    llvm::Value *Flag = Builder.CreateExtractValue(Tmp, 1);

    llvm::Type *RealFlagType = FlagOutPtr.getElementType();

    llvm::Value *FlagExt = Builder.CreateZExt(Flag, RealFlagType);
    Builder.CreateStore(FlagExt, FlagOutPtr);
    return Result;
  }
  case AMDGPU::BI__builtin_amdgcn_div_fmas:
  case AMDGPU::BI__builtin_amdgcn_div_fmasf: {
    llvm::Value *Src0 = EmitScalarExpr(E->getArg(0));
    llvm::Value *Src1 = EmitScalarExpr(E->getArg(1));
    llvm::Value *Src2 = EmitScalarExpr(E->getArg(2));
    llvm::Value *Src3 = EmitScalarExpr(E->getArg(3));

    llvm::Function *F = CGM.getIntrinsic(Intrinsic::amdgcn_div_fmas,
                                      Src0->getType());
    llvm::Value *Src3ToBool = Builder.CreateIsNotNull(Src3);
    return Builder.CreateCall(F, {Src0, Src1, Src2, Src3ToBool});
  }

  case AMDGPU::BI__builtin_amdgcn_ds_swizzle:
    return emitBuiltinWithOneOverloadedType<2>(*this, E,
                                               Intrinsic::amdgcn_ds_swizzle);
  case AMDGPU::BI__builtin_amdgcn_mov_dpp8:
  case AMDGPU::BI__builtin_amdgcn_mov_dpp:
  case AMDGPU::BI__builtin_amdgcn_update_dpp: {
    llvm::SmallVector<llvm::Value *, 6> Args;
    // Find out if any arguments are required to be integer constant
    // expressions.
    unsigned ICEArguments = 0;
    ASTContext::GetBuiltinTypeError Error;
    getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
    assert(Error == ASTContext::GE_None && "Should not codegen an error");
    llvm::Type *DataTy = ConvertType(E->getArg(0)->getType());
    unsigned Size = DataTy->getPrimitiveSizeInBits();
    llvm::Type *IntTy =
        llvm::IntegerType::get(Builder.getContext(), std::max(Size, 32u));
    Function *F =
        CGM.getIntrinsic(BuiltinID == AMDGPU::BI__builtin_amdgcn_mov_dpp8
                             ? Intrinsic::amdgcn_mov_dpp8
                             : Intrinsic::amdgcn_update_dpp,
                         IntTy);
    assert(E->getNumArgs() == 5 || E->getNumArgs() == 6 ||
           E->getNumArgs() == 2);
    bool InsertOld = BuiltinID == AMDGPU::BI__builtin_amdgcn_mov_dpp;
    if (InsertOld)
      Args.push_back(llvm::PoisonValue::get(IntTy));
    for (unsigned I = 0; I != E->getNumArgs(); ++I) {
      llvm::Value *V = EmitScalarOrConstFoldImmArg(ICEArguments, I, E);
      if (I < (BuiltinID == AMDGPU::BI__builtin_amdgcn_update_dpp ? 2u : 1u) &&
          Size < 32) {
        if (!DataTy->isIntegerTy())
          V = Builder.CreateBitCast(
              V, llvm::IntegerType::get(Builder.getContext(), Size));
        V = Builder.CreateZExtOrBitCast(V, IntTy);
      }
      llvm::Type *ExpTy =
          F->getFunctionType()->getFunctionParamType(I + InsertOld);
      Args.push_back(Builder.CreateTruncOrBitCast(V, ExpTy));
    }
    Value *V = Builder.CreateCall(F, Args);
    if (Size < 32 && !DataTy->isIntegerTy())
      V = Builder.CreateTrunc(
          V, llvm::IntegerType::get(Builder.getContext(), Size));
    return Builder.CreateTruncOrBitCast(V, DataTy);
  }
  case AMDGPU::BI__builtin_amdgcn_permlane16:
  case AMDGPU::BI__builtin_amdgcn_permlanex16:
    return emitBuiltinWithOneOverloadedType<6>(
        *this, E,
        BuiltinID == AMDGPU::BI__builtin_amdgcn_permlane16
            ? Intrinsic::amdgcn_permlane16
            : Intrinsic::amdgcn_permlanex16);
  case AMDGPU::BI__builtin_amdgcn_permlane64:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::amdgcn_permlane64);
  case AMDGPU::BI__builtin_amdgcn_readlane:
    return emitBuiltinWithOneOverloadedType<2>(*this, E,
                                               Intrinsic::amdgcn_readlane);
  case AMDGPU::BI__builtin_amdgcn_readfirstlane:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::amdgcn_readfirstlane);
  case AMDGPU::BI__builtin_amdgcn_div_fixup:
  case AMDGPU::BI__builtin_amdgcn_div_fixupf:
  case AMDGPU::BI__builtin_amdgcn_div_fixuph:
    return emitBuiltinWithOneOverloadedType<3>(*this, E,
                                               Intrinsic::amdgcn_div_fixup);
  case AMDGPU::BI__builtin_amdgcn_trig_preop:
  case AMDGPU::BI__builtin_amdgcn_trig_preopf:
    return emitFPIntBuiltin(*this, E, Intrinsic::amdgcn_trig_preop);
  case AMDGPU::BI__builtin_amdgcn_rcp:
  case AMDGPU::BI__builtin_amdgcn_rcpf:
  case AMDGPU::BI__builtin_amdgcn_rcph:
    return emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::amdgcn_rcp);
  case AMDGPU::BI__builtin_amdgcn_sqrt:
  case AMDGPU::BI__builtin_amdgcn_sqrtf:
  case AMDGPU::BI__builtin_amdgcn_sqrth:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::amdgcn_sqrt);
  case AMDGPU::BI__builtin_amdgcn_rsq:
  case AMDGPU::BI__builtin_amdgcn_rsqf:
  case AMDGPU::BI__builtin_amdgcn_rsqh:
    return emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::amdgcn_rsq);
  case AMDGPU::BI__builtin_amdgcn_rsq_clamp:
  case AMDGPU::BI__builtin_amdgcn_rsq_clampf:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::amdgcn_rsq_clamp);
  case AMDGPU::BI__builtin_amdgcn_sinf:
  case AMDGPU::BI__builtin_amdgcn_sinh:
    return emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::amdgcn_sin);
  case AMDGPU::BI__builtin_amdgcn_cosf:
  case AMDGPU::BI__builtin_amdgcn_cosh:
    return emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::amdgcn_cos);
  case AMDGPU::BI__builtin_amdgcn_dispatch_ptr:
    return EmitAMDGPUDispatchPtr(*this, E);
  case AMDGPU::BI__builtin_amdgcn_logf:
    return emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::amdgcn_log);
  case AMDGPU::BI__builtin_amdgcn_exp2f:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::amdgcn_exp2);
  case AMDGPU::BI__builtin_amdgcn_log_clampf:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::amdgcn_log_clamp);
  case AMDGPU::BI__builtin_amdgcn_ldexp:
  case AMDGPU::BI__builtin_amdgcn_ldexpf: {
    llvm::Value *Src0 = EmitScalarExpr(E->getArg(0));
    llvm::Value *Src1 = EmitScalarExpr(E->getArg(1));
    llvm::Function *F =
        CGM.getIntrinsic(Intrinsic::ldexp, {Src0->getType(), Src1->getType()});
    return Builder.CreateCall(F, {Src0, Src1});
  }
  case AMDGPU::BI__builtin_amdgcn_ldexph: {
    // The raw instruction has a different behavior for out of bounds exponent
    // values (implicit truncation instead of saturate to short_min/short_max).
    llvm::Value *Src0 = EmitScalarExpr(E->getArg(0));
    llvm::Value *Src1 = EmitScalarExpr(E->getArg(1));
    llvm::Function *F =
        CGM.getIntrinsic(Intrinsic::ldexp, {Src0->getType(), Int16Ty});
    return Builder.CreateCall(F, {Src0, Builder.CreateTrunc(Src1, Int16Ty)});
  }
  case AMDGPU::BI__builtin_amdgcn_frexp_mant:
  case AMDGPU::BI__builtin_amdgcn_frexp_mantf:
  case AMDGPU::BI__builtin_amdgcn_frexp_manth:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::amdgcn_frexp_mant);
  case AMDGPU::BI__builtin_amdgcn_frexp_exp:
  case AMDGPU::BI__builtin_amdgcn_frexp_expf: {
    Value *Src0 = EmitScalarExpr(E->getArg(0));
    Function *F = CGM.getIntrinsic(Intrinsic::amdgcn_frexp_exp,
                                { Builder.getInt32Ty(), Src0->getType() });
    return Builder.CreateCall(F, Src0);
  }
  case AMDGPU::BI__builtin_amdgcn_frexp_exph: {
    Value *Src0 = EmitScalarExpr(E->getArg(0));
    Function *F = CGM.getIntrinsic(Intrinsic::amdgcn_frexp_exp,
                                { Builder.getInt16Ty(), Src0->getType() });
    return Builder.CreateCall(F, Src0);
  }
  case AMDGPU::BI__builtin_amdgcn_fract:
  case AMDGPU::BI__builtin_amdgcn_fractf:
  case AMDGPU::BI__builtin_amdgcn_fracth:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::amdgcn_fract);
  case AMDGPU::BI__builtin_amdgcn_lerp:
    return emitBuiltinWithOneOverloadedType<3>(*this, E,
                                               Intrinsic::amdgcn_lerp);
  case AMDGPU::BI__builtin_amdgcn_ubfe:
    return emitBuiltinWithOneOverloadedType<3>(*this, E,
                                               Intrinsic::amdgcn_ubfe);
  case AMDGPU::BI__builtin_amdgcn_sbfe:
    return emitBuiltinWithOneOverloadedType<3>(*this, E,
                                               Intrinsic::amdgcn_sbfe);
  case AMDGPU::BI__builtin_amdgcn_ballot_w32:
  case AMDGPU::BI__builtin_amdgcn_ballot_w64: {
    llvm::Type *ResultType = ConvertType(E->getType());
    llvm::Value *Src = EmitScalarExpr(E->getArg(0));
    Function *F = CGM.getIntrinsic(Intrinsic::amdgcn_ballot, { ResultType });
    return Builder.CreateCall(F, { Src });
  }
  case AMDGPU::BI__builtin_amdgcn_uicmp:
  case AMDGPU::BI__builtin_amdgcn_uicmpl:
  case AMDGPU::BI__builtin_amdgcn_sicmp:
  case AMDGPU::BI__builtin_amdgcn_sicmpl: {
    llvm::Value *Src0 = EmitScalarExpr(E->getArg(0));
    llvm::Value *Src1 = EmitScalarExpr(E->getArg(1));
    llvm::Value *Src2 = EmitScalarExpr(E->getArg(2));

    // FIXME-GFX10: How should 32 bit mask be handled?
    Function *F = CGM.getIntrinsic(Intrinsic::amdgcn_icmp,
      { Builder.getInt64Ty(), Src0->getType() });
    return Builder.CreateCall(F, { Src0, Src1, Src2 });
  }
  case AMDGPU::BI__builtin_amdgcn_fcmp:
  case AMDGPU::BI__builtin_amdgcn_fcmpf: {
    llvm::Value *Src0 = EmitScalarExpr(E->getArg(0));
    llvm::Value *Src1 = EmitScalarExpr(E->getArg(1));
    llvm::Value *Src2 = EmitScalarExpr(E->getArg(2));

    // FIXME-GFX10: How should 32 bit mask be handled?
    Function *F = CGM.getIntrinsic(Intrinsic::amdgcn_fcmp,
      { Builder.getInt64Ty(), Src0->getType() });
    return Builder.CreateCall(F, { Src0, Src1, Src2 });
  }
  case AMDGPU::BI__builtin_amdgcn_class:
  case AMDGPU::BI__builtin_amdgcn_classf:
  case AMDGPU::BI__builtin_amdgcn_classh:
    return emitFPIntBuiltin(*this, E, Intrinsic::amdgcn_class);
  case AMDGPU::BI__builtin_amdgcn_fmed3f:
  case AMDGPU::BI__builtin_amdgcn_fmed3h:
    return emitBuiltinWithOneOverloadedType<3>(*this, E,
                                               Intrinsic::amdgcn_fmed3);
  case AMDGPU::BI__builtin_amdgcn_ds_append:
  case AMDGPU::BI__builtin_amdgcn_ds_consume: {
    Intrinsic::ID Intrin = BuiltinID == AMDGPU::BI__builtin_amdgcn_ds_append ?
      Intrinsic::amdgcn_ds_append : Intrinsic::amdgcn_ds_consume;
    Value *Src0 = EmitScalarExpr(E->getArg(0));
    Function *F = CGM.getIntrinsic(Intrin, { Src0->getType() });
    return Builder.CreateCall(F, { Src0, Builder.getFalse() });
  }
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b64_i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4i16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4f16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4bf16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8i16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8f16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8bf16:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr4_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr8_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr6_b96_v3i32:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4f16:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4bf16:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4i16: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b64_i32:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b64_v2i32:
      IID = Intrinsic::amdgcn_global_load_tr_b64;
      break;
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4i16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4f16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4bf16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8i16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8f16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8bf16:
      IID = Intrinsic::amdgcn_global_load_tr_b128;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_read_tr4_b64_v2i32:
      IID = Intrinsic::amdgcn_ds_read_tr4_b64;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_read_tr8_b64_v2i32:
      IID = Intrinsic::amdgcn_ds_read_tr8_b64;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_read_tr6_b96_v3i32:
      IID = Intrinsic::amdgcn_ds_read_tr6_b96;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4i16:
    case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4f16:
    case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4bf16:
      IID = Intrinsic::amdgcn_ds_read_tr16_b64;
      break;
    }
    llvm::Type *LoadTy = ConvertType(E->getType());
    llvm::Value *Addr = EmitScalarExpr(E->getArg(0));
    llvm::Function *F = CGM.getIntrinsic(IID, {LoadTy});
    return Builder.CreateCall(F, {Addr});
  }
  case AMDGPU::BI__builtin_amdgcn_get_fpenv: {
    Function *F = CGM.getIntrinsic(Intrinsic::get_fpenv,
                                   {llvm::Type::getInt64Ty(getLLVMContext())});
    return Builder.CreateCall(F);
  }
  case AMDGPU::BI__builtin_amdgcn_set_fpenv: {
    Function *F = CGM.getIntrinsic(Intrinsic::set_fpenv,
                                   {llvm::Type::getInt64Ty(getLLVMContext())});
    llvm::Value *Env = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(F, {Env});
  }
  case AMDGPU::BI__builtin_amdgcn_read_exec:
    return EmitAMDGCNBallotForExec(*this, E, Int64Ty, Int64Ty, false);
  case AMDGPU::BI__builtin_amdgcn_read_exec_lo:
    return EmitAMDGCNBallotForExec(*this, E, Int32Ty, Int32Ty, false);
  case AMDGPU::BI__builtin_amdgcn_read_exec_hi:
    return EmitAMDGCNBallotForExec(*this, E, Int64Ty, Int64Ty, true);
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray_h:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray_l:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray_lh: {
    llvm::Value *NodePtr = EmitScalarExpr(E->getArg(0));
    llvm::Value *RayExtent = EmitScalarExpr(E->getArg(1));
    llvm::Value *RayOrigin = EmitScalarExpr(E->getArg(2));
    llvm::Value *RayDir = EmitScalarExpr(E->getArg(3));
    llvm::Value *RayInverseDir = EmitScalarExpr(E->getArg(4));
    llvm::Value *TextureDescr = EmitScalarExpr(E->getArg(5));

    // The builtins take these arguments as vec4 where the last element is
    // ignored. The intrinsic takes them as vec3.
    RayOrigin = Builder.CreateShuffleVector(RayOrigin, RayOrigin,
                                            {0, 1, 2});
    RayDir =
        Builder.CreateShuffleVector(RayDir, RayDir, {0, 1, 2});
    RayInverseDir = Builder.CreateShuffleVector(RayInverseDir, RayInverseDir,
                                                {0, 1, 2});

    Function *F = CGM.getIntrinsic(Intrinsic::amdgcn_image_bvh_intersect_ray,
                                   {NodePtr->getType(), RayDir->getType()});
    return Builder.CreateCall(F, {NodePtr, RayExtent, RayOrigin, RayDir,
                                  RayInverseDir, TextureDescr});
  }

  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_rtn: {
    SmallVector<Value *, 4> Args;
    for (int i = 0, e = E->getNumArgs(); i != e; ++i)
      Args.push_back(EmitScalarExpr(E->getArg(i)));

    Function *F = CGM.getIntrinsic(Intrinsic::amdgcn_ds_bvh_stack_rtn);
    Value *Call = Builder.CreateCall(F, Args);
    Value *Rtn = Builder.CreateExtractValue(Call, 0);
    Value *A = Builder.CreateExtractValue(Call, 1);
    llvm::Type *RetTy = ConvertType(E->getType());
    Value *I0 = Builder.CreateInsertElement(PoisonValue::get(RetTy), Rtn,
                                            (uint64_t)0);
    return Builder.CreateInsertElement(I0, A, 1);
  }
  case AMDGPU::BI__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4: {
    llvm::FixedVectorType *VT = FixedVectorType::get(Builder.getInt32Ty(), 8);
    Function *F = CGM.getIntrinsic(
        BuiltinID == AMDGPU::BI__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4
            ? Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4
            : Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4,
        {VT, VT});

    SmallVector<Value *, 9> Args;
    for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
      Args.push_back(EmitScalarExpr(E->getArg(I)));
    return Builder.CreateCall(F, Args);
  }
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x32_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x32_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64: {

    // These operations perform a matrix multiplication and accumulation of
    // the form:
    //             D = A * B + C
    // We need to specify one type for matrices AB and one for matrices CD.
    // Sparse matrix operations can have different types for A and B as well as
    // an additional type for sparsity index.
    // Destination type should be put before types used for source operands.
    SmallVector<unsigned, 2> ArgsForMatchingMatrixTypes;
    // On GFX12, the intrinsics with 16-bit accumulator use a packed layout.
    // There is no need for the variable opsel argument, so always set it to
    // "false".
    bool AppendFalseForOpselArg = false;
    unsigned BuiltinWMMAOp;

    switch (BuiltinID) {
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w32:
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w64:
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12:
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12:
      ArgsForMatchingMatrixTypes = {2, 0}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x16_f16;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32:
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64:
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12:
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12:
      ArgsForMatchingMatrixTypes = {2, 0}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x16_bf16;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12:
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w64_gfx12:
      AppendFalseForOpselArg = true;
      [[fallthrough]];
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w32:
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w64:
      ArgsForMatchingMatrixTypes = {2, 0}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f16_16x16x16_f16;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12:
    case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64_gfx12:
      AppendFalseForOpselArg = true;
      [[fallthrough]];
    case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32:
    case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64:
      ArgsForMatchingMatrixTypes = {2, 0}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_bf16_16x16x16_bf16;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32:
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w64:
      ArgsForMatchingMatrixTypes = {2, 0}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f16_16x16x16_f16_tied;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w32:
    case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w64:
      ArgsForMatchingMatrixTypes = {2, 0}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_bf16_16x16x16_bf16_tied;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32:
    case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64:
    case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12:
    case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12:
      ArgsForMatchingMatrixTypes = {4, 1}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_i32_16x16x16_iu8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32:
    case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w64:
    case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12:
    case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w64_gfx12:
      ArgsForMatchingMatrixTypes = {4, 1}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_i32_16x16x16_iu4;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12:
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w64_gfx12:
      ArgsForMatchingMatrixTypes = {2, 0}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x16_fp8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12:
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w64_gfx12:
      ArgsForMatchingMatrixTypes = {2, 0}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x16_fp8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12:
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w64_gfx12:
      ArgsForMatchingMatrixTypes = {2, 0}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x16_bf8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12:
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w64_gfx12:
      ArgsForMatchingMatrixTypes = {2, 0}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x16_bf8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12:
    case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12:
      ArgsForMatchingMatrixTypes = {4, 1}; // CD, AB
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_i32_16x16x32_iu4;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32:
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_f16_w64:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3}; // CD, A, B, Index
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x32_f16;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32:
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w64:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3}; // CD, A, B, Index
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x32_bf16;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x32_f16_w32:
    case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x32_f16_w64:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3}; // CD, A, B, Index
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f16_16x16x32_f16;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32:
    case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w64:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3}; // CD, A, B, Index
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_bf16_16x16x32_bf16;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32:
    case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w64:
      ArgsForMatchingMatrixTypes = {4, 1, 3, 5}; // CD, A, B, Index
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_i32_16x16x32_iu8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32:
    case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w64:
      ArgsForMatchingMatrixTypes = {4, 1, 3, 5}; // CD, A, B, Index
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_i32_16x16x32_iu4;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32:
    case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w64:
      ArgsForMatchingMatrixTypes = {4, 1, 3, 5}; // CD, A, B, Index
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_i32_16x16x64_iu4;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32:
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w64:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3}; // CD, A, B, Index
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x32_fp8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32:
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w64:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3}; // CD, A, B, Index
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x32_fp8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32:
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w64:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3}; // CD, A, B, Index
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x32_bf8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32:
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3}; // CD, A, B, Index
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x32_bf8_bf8;
      break;
    }

    SmallVector<Value *, 6> Args;
    for (int i = 0, e = E->getNumArgs(); i != e; ++i)
      Args.push_back(EmitScalarExpr(E->getArg(i)));
    if (AppendFalseForOpselArg)
      Args.push_back(Builder.getFalse());

    SmallVector<llvm::Type *, 6> ArgTypes;
    for (auto ArgIdx : ArgsForMatchingMatrixTypes)
      ArgTypes.push_back(Args[ArgIdx]->getType());

    Function *F = CGM.getIntrinsic(BuiltinWMMAOp, ArgTypes);
    return Builder.CreateCall(F, Args);
  }

  // amdgcn workitem
  case AMDGPU::BI__builtin_amdgcn_workitem_id_x:
    return emitRangedBuiltin(*this, Intrinsic::amdgcn_workitem_id_x, 0, 1024);
  case AMDGPU::BI__builtin_amdgcn_workitem_id_y:
    return emitRangedBuiltin(*this, Intrinsic::amdgcn_workitem_id_y, 0, 1024);
  case AMDGPU::BI__builtin_amdgcn_workitem_id_z:
    return emitRangedBuiltin(*this, Intrinsic::amdgcn_workitem_id_z, 0, 1024);

  // amdgcn workgroup size
  case AMDGPU::BI__builtin_amdgcn_workgroup_size_x:
    return EmitAMDGPUWorkGroupSize(*this, 0);
  case AMDGPU::BI__builtin_amdgcn_workgroup_size_y:
    return EmitAMDGPUWorkGroupSize(*this, 1);
  case AMDGPU::BI__builtin_amdgcn_workgroup_size_z:
    return EmitAMDGPUWorkGroupSize(*this, 2);

  // amdgcn grid size
  case AMDGPU::BI__builtin_amdgcn_grid_size_x:
    return EmitAMDGPUGridSize(*this, 0);
  case AMDGPU::BI__builtin_amdgcn_grid_size_y:
    return EmitAMDGPUGridSize(*this, 1);
  case AMDGPU::BI__builtin_amdgcn_grid_size_z:
    return EmitAMDGPUGridSize(*this, 2);

  // r600 intrinsics
  case AMDGPU::BI__builtin_r600_recipsqrt_ieee:
  case AMDGPU::BI__builtin_r600_recipsqrt_ieeef:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::r600_recipsqrt_ieee);
  case AMDGPU::BI__builtin_r600_read_tidig_x:
    return emitRangedBuiltin(*this, Intrinsic::r600_read_tidig_x, 0, 1024);
  case AMDGPU::BI__builtin_r600_read_tidig_y:
    return emitRangedBuiltin(*this, Intrinsic::r600_read_tidig_y, 0, 1024);
  case AMDGPU::BI__builtin_r600_read_tidig_z:
    return emitRangedBuiltin(*this, Intrinsic::r600_read_tidig_z, 0, 1024);
  case AMDGPU::BI__builtin_amdgcn_alignbit: {
    llvm::Value *Src0 = EmitScalarExpr(E->getArg(0));
    llvm::Value *Src1 = EmitScalarExpr(E->getArg(1));
    llvm::Value *Src2 = EmitScalarExpr(E->getArg(2));
    Function *F = CGM.getIntrinsic(Intrinsic::fshr, Src0->getType());
    return Builder.CreateCall(F, { Src0, Src1, Src2 });
  }
  case AMDGPU::BI__builtin_amdgcn_fence: {
    ProcessOrderScopeAMDGCN(EmitScalarExpr(E->getArg(0)),
                            EmitScalarExpr(E->getArg(1)), AO, SSID);
    FenceInst *Fence = Builder.CreateFence(AO, SSID);
    if (E->getNumArgs() > 2)
      AddAMDGPUFenceAddressSpaceMMRA(Fence, E);
    return Fence;
  }
  case AMDGPU::BI__builtin_amdgcn_atomic_inc32:
  case AMDGPU::BI__builtin_amdgcn_atomic_inc64:
  case AMDGPU::BI__builtin_amdgcn_atomic_dec32:
  case AMDGPU::BI__builtin_amdgcn_atomic_dec64:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_f64:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_v2f16:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_v2bf16:
  case AMDGPU::BI__builtin_amdgcn_ds_faddf:
  case AMDGPU::BI__builtin_amdgcn_ds_fminf:
  case AMDGPU::BI__builtin_amdgcn_ds_fmaxf:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_f64:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_v2f16:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_v2f16:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_f64:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_v2bf16:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_v2bf16:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fmin_f64:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fmax_f64:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fmin_f64:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fmax_f64: {
    llvm::AtomicRMWInst::BinOp BinOp;
    switch (BuiltinID) {
    case AMDGPU::BI__builtin_amdgcn_atomic_inc32:
    case AMDGPU::BI__builtin_amdgcn_atomic_inc64:
      BinOp = llvm::AtomicRMWInst::UIncWrap;
      break;
    case AMDGPU::BI__builtin_amdgcn_atomic_dec32:
    case AMDGPU::BI__builtin_amdgcn_atomic_dec64:
      BinOp = llvm::AtomicRMWInst::UDecWrap;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_faddf:
    case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_f64:
    case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_f32:
    case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_v2f16:
    case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_v2bf16:
    case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_f32:
    case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_f64:
    case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_v2f16:
    case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_v2f16:
    case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_f32:
    case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_f64:
    case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_v2bf16:
    case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_v2bf16:
      BinOp = llvm::AtomicRMWInst::FAdd;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_fminf:
    case AMDGPU::BI__builtin_amdgcn_global_atomic_fmin_f64:
    case AMDGPU::BI__builtin_amdgcn_flat_atomic_fmin_f64:
      BinOp = llvm::AtomicRMWInst::FMin;
      break;
    case AMDGPU::BI__builtin_amdgcn_global_atomic_fmax_f64:
    case AMDGPU::BI__builtin_amdgcn_flat_atomic_fmax_f64:
    case AMDGPU::BI__builtin_amdgcn_ds_fmaxf:
      BinOp = llvm::AtomicRMWInst::FMax;
      break;
    }

    Address Ptr = CheckAtomicAlignment(*this, E);
    Value *Val = EmitScalarExpr(E->getArg(1));
    llvm::Type *OrigTy = Val->getType();
    QualType PtrTy = E->getArg(0)->IgnoreImpCasts()->getType();

    bool Volatile;

    if (BuiltinID == AMDGPU::BI__builtin_amdgcn_ds_faddf ||
        BuiltinID == AMDGPU::BI__builtin_amdgcn_ds_fminf ||
        BuiltinID == AMDGPU::BI__builtin_amdgcn_ds_fmaxf) {
      // __builtin_amdgcn_ds_faddf/fminf/fmaxf has an explicit volatile argument
      Volatile =
          cast<ConstantInt>(EmitScalarExpr(E->getArg(4)))->getZExtValue();
    } else {
      // Infer volatile from the passed type.
      Volatile =
          PtrTy->castAs<PointerType>()->getPointeeType().isVolatileQualified();
    }

    if (E->getNumArgs() >= 4) {
      // Some of the builtins have explicit ordering and scope arguments.
      ProcessOrderScopeAMDGCN(EmitScalarExpr(E->getArg(2)),
                              EmitScalarExpr(E->getArg(3)), AO, SSID);
    } else {
      // Most of the builtins do not have syncscope/order arguments. For DS
      // atomics the scope doesn't really matter, as they implicitly operate at
      // workgroup scope.
      //
      // The global/flat cases need to use agent scope to consistently produce
      // the native instruction instead of a cmpxchg expansion.
      SSID = getLLVMContext().getOrInsertSyncScopeID("agent");
      AO = AtomicOrdering::Monotonic;

      // The v2bf16 builtin uses i16 instead of a natural bfloat type.
      if (BuiltinID == AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_v2bf16 ||
          BuiltinID == AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_v2bf16 ||
          BuiltinID == AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_v2bf16) {
        llvm::Type *V2BF16Ty = FixedVectorType::get(
            llvm::Type::getBFloatTy(Builder.getContext()), 2);
        Val = Builder.CreateBitCast(Val, V2BF16Ty);
      }
    }

    llvm::AtomicRMWInst *RMW =
        Builder.CreateAtomicRMW(BinOp, Ptr, Val, AO, SSID);
    if (Volatile)
      RMW->setVolatile(true);

    unsigned AddrSpace = Ptr.getType()->getAddressSpace();
    if (AddrSpace != llvm::AMDGPUAS::LOCAL_ADDRESS) {
      // Most targets require "amdgpu.no.fine.grained.memory" to emit the native
      // instruction for flat and global operations.
      llvm::MDTuple *EmptyMD = MDNode::get(getLLVMContext(), {});
      RMW->setMetadata("amdgpu.no.fine.grained.memory", EmptyMD);

      // Most targets require "amdgpu.ignore.denormal.mode" to emit the native
      // instruction, but this only matters for float fadd.
      if (BinOp == llvm::AtomicRMWInst::FAdd && Val->getType()->isFloatTy())
        RMW->setMetadata("amdgpu.ignore.denormal.mode", EmptyMD);
    }

    return Builder.CreateBitCast(RMW, OrigTy);
  }
  case AMDGPU::BI__builtin_amdgcn_s_sendmsg_rtn:
  case AMDGPU::BI__builtin_amdgcn_s_sendmsg_rtnl: {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    llvm::Type *ResultType = ConvertType(E->getType());
    // s_sendmsg_rtn is mangled using return type only.
    Function *F =
        CGM.getIntrinsic(Intrinsic::amdgcn_s_sendmsg_rtn, {ResultType});
    return Builder.CreateCall(F, {Arg});
  }
  case AMDGPU::BI__builtin_amdgcn_permlane16_swap:
  case AMDGPU::BI__builtin_amdgcn_permlane32_swap: {
    // Because builtin types are limited, and the intrinsic uses a struct/pair
    // output, marshal the pair-of-i32 to <2 x i32>.
    Value *VDstOld = EmitScalarExpr(E->getArg(0));
    Value *VSrcOld = EmitScalarExpr(E->getArg(1));
    Value *FI = EmitScalarExpr(E->getArg(2));
    Value *BoundCtrl = EmitScalarExpr(E->getArg(3));
    Function *F =
        CGM.getIntrinsic(BuiltinID == AMDGPU::BI__builtin_amdgcn_permlane16_swap
                             ? Intrinsic::amdgcn_permlane16_swap
                             : Intrinsic::amdgcn_permlane32_swap);
    llvm::CallInst *Call =
        Builder.CreateCall(F, {VDstOld, VSrcOld, FI, BoundCtrl});

    llvm::Value *Elt0 = Builder.CreateExtractValue(Call, 0);
    llvm::Value *Elt1 = Builder.CreateExtractValue(Call, 1);

    llvm::Type *ResultType = ConvertType(E->getType());

    llvm::Value *Insert0 = Builder.CreateInsertElement(
        llvm::PoisonValue::get(ResultType), Elt0, UINT64_C(0));
    llvm::Value *AsVector =
        Builder.CreateInsertElement(Insert0, Elt1, UINT64_C(1));
    return AsVector;
  }
  case AMDGPU::BI__builtin_amdgcn_bitop3_b32:
  case AMDGPU::BI__builtin_amdgcn_bitop3_b16:
    return emitBuiltinWithOneOverloadedType<4>(*this, E,
                                               Intrinsic::amdgcn_bitop3);
  case AMDGPU::BI__builtin_amdgcn_make_buffer_rsrc: {
    // TODO: LLVM has this overloaded to allow for fat pointers, but since
    // those haven't been plumbed through to Clang yet, default to creating the
    // resource type.
    SmallVector<Value *, 4> Args;
    for (unsigned I = 0; I < 4; ++I)
      Args.push_back(EmitScalarExpr(E->getArg(I)));
    llvm::PointerType *RetTy = llvm::PointerType::get(
        Builder.getContext(), llvm::AMDGPUAS::BUFFER_RESOURCE);
    Function *F = CGM.getIntrinsic(Intrinsic::amdgcn_make_buffer_rsrc,
                                   {RetTy, Args[0]->getType()});
    return Builder.CreateCall(F, Args);
  }
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b8:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b16:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b32:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b64:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b96:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b128:
    return emitBuiltinWithOneOverloadedType<5>(
        *this, E, Intrinsic::amdgcn_raw_ptr_buffer_store);
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b8:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b16:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b32:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b64:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b96:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b128: {
    llvm::Type *RetTy = nullptr;
    switch (BuiltinID) {
    case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b8:
      RetTy = Int8Ty;
      break;
    case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b16:
      RetTy = Int16Ty;
      break;
    case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b32:
      RetTy = Int32Ty;
      break;
    case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b64:
      RetTy = llvm::FixedVectorType::get(Int32Ty, /*NumElements=*/2);
      break;
    case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b96:
      RetTy = llvm::FixedVectorType::get(Int32Ty, /*NumElements=*/3);
      break;
    case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b128:
      RetTy = llvm::FixedVectorType::get(Int32Ty, /*NumElements=*/4);
      break;
    }
    Function *F =
        CGM.getIntrinsic(Intrinsic::amdgcn_raw_ptr_buffer_load, RetTy);
    return Builder.CreateCall(
        F, {EmitScalarExpr(E->getArg(0)), EmitScalarExpr(E->getArg(1)),
            EmitScalarExpr(E->getArg(2)), EmitScalarExpr(E->getArg(3))});
  }
  case AMDGPU::BI__builtin_amdgcn_s_prefetch_data:
    return emitBuiltinWithOneOverloadedType<2>(
        *this, E, Intrinsic::amdgcn_s_prefetch_data);
  default:
    return nullptr;
  }
}
