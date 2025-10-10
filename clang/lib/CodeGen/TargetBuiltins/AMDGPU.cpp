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

#include "CGBuiltin.h"
#include "CodeGenFunction.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/Support/AMDGPUAddrSpace.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

namespace {

// Has second type mangled argument.
static Value *
emitBinaryExpMaybeConstrainedFPBuiltin(CodeGenFunction &CGF, const CallExpr *E,
                                       Intrinsic::ID IntrinsicID,
                                       Intrinsic::ID ConstrainedIntrinsicID) {
  llvm::Value *Src0 = CGF.EmitScalarExpr(E->getArg(0));
  llvm::Value *Src1 = CGF.EmitScalarExpr(E->getArg(1));

  CodeGenFunction::CGFPOptionsRAII FPOptsRAII(CGF, E);
  if (CGF.Builder.getIsFPConstrained()) {
    Function *F = CGF.CGM.getIntrinsic(ConstrainedIntrinsicID,
                                       {Src0->getType(), Src1->getType()});
    return CGF.Builder.CreateConstrainedFPCall(F, {Src0, Src1});
  }

  Function *F =
      CGF.CGM.getIntrinsic(IntrinsicID, {Src0->getType(), Src1->getType()});
  return CGF.Builder.CreateCall(F, {Src0, Src1});
}

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

static llvm::Value *loadTextureDescPtorAsVec8I32(CodeGenFunction &CGF,
                                                 llvm::Value *RsrcPtr) {
  auto &B = CGF.Builder;
  auto *VecTy = llvm::FixedVectorType::get(B.getInt32Ty(), 8);

  if (RsrcPtr->getType() == VecTy)
    return RsrcPtr;

  if (RsrcPtr->getType()->isIntegerTy(32)) {
    llvm::PointerType *VecPtrTy =
        llvm::PointerType::get(CGF.getLLVMContext(), 8);
    llvm::Value *Ptr = B.CreateIntToPtr(RsrcPtr, VecPtrTy, "tex.rsrc.from.int");
    return B.CreateAlignedLoad(VecTy, Ptr, llvm::Align(32), "tex.rsrc.val");
  }

  if (RsrcPtr->getType()->isPointerTy()) {
    auto *VecPtrTy = llvm::PointerType::get(
        CGF.getLLVMContext(), RsrcPtr->getType()->getPointerAddressSpace());
    llvm::Value *Typed = B.CreateBitCast(RsrcPtr, VecPtrTy, "tex.rsrc.typed");
    return B.CreateAlignedLoad(VecTy, Typed, llvm::Align(32), "tex.rsrc.val");
  }

  const auto &DL = CGF.CGM.getDataLayout();
  if (DL.getTypeSizeInBits(RsrcPtr->getType()) == 256)
    return B.CreateBitCast(RsrcPtr, VecTy, "tex.rsrc.val");

  llvm::report_fatal_error("Unexpected texture resource argument form");
}

llvm::CallInst *
emitAMDGCNImageOverloadedReturnType(clang::CodeGen::CodeGenFunction &CGF,
                                    const clang::CallExpr *E,
                                    unsigned IntrinsicID, bool IsImageStore) {
  auto findTextureDescIndex = [&CGF](const CallExpr *E) -> unsigned {
    QualType TexQT = CGF.getContext().AMDGPUTextureTy;
    for (unsigned I = 0, N = E->getNumArgs(); I < N; ++I) {
      QualType ArgTy = E->getArg(I)->getType();
      if (ArgTy == TexQT) {
        return I;
      }

      if (ArgTy.getCanonicalType() == TexQT.getCanonicalType()) {
        return I;
      }
    }

    return ~0U;
  };

  clang::SmallVector<llvm::Value *, 10> Args;
  unsigned RsrcIndex = findTextureDescIndex(E);

  if (RsrcIndex == ~0U) {
    llvm::report_fatal_error("Invalid argument count for image builtin");
  }

  for (unsigned I = 0; I < E->getNumArgs(); ++I) {
    llvm::Value *V = CGF.EmitScalarExpr(E->getArg(I));
    if (I == RsrcIndex)
      V = loadTextureDescPtorAsVec8I32(CGF, V);
    Args.push_back(V);
  }

  llvm::Type *RetTy = IsImageStore ? CGF.VoidTy : CGF.ConvertType(E->getType());
  llvm::CallInst *Call = CGF.Builder.CreateIntrinsic(RetTy, IntrinsicID, Args);
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

static inline StringRef mapScopeToSPIRV(StringRef AMDGCNScope) {
  if (AMDGCNScope == "agent")
    return "device";
  if (AMDGCNScope == "wavefront")
    return "subgroup";
  return AMDGCNScope;
}

// For processing memory ordering and memory scope arguments of various
// amdgcn builtins.
// \p Order takes a C++11 compatible memory-ordering specifier and converts
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
    if (getTarget().getTriple().isSPIRV())
      scp = mapScopeToSPIRV(scp);
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
    if (getTarget().getTriple().isSPIRV())
      SSID = getLLVMContext().getOrInsertSyncScopeID("device");
    else
      SSID = getLLVMContext().getOrInsertSyncScopeID("agent");
    break;
  case 2: // __MEMORY_SCOPE_WRKGRP
    SSID = getLLVMContext().getOrInsertSyncScopeID("workgroup");
    break;
  case 3: // __MEMORY_SCOPE_WVFRNT
    if (getTarget().getTriple().isSPIRV())
      SSID = getLLVMContext().getOrInsertSyncScopeID("subgroup");
    else
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

void CodeGenFunction::AddAMDGPUFenceAddressSpaceMMRA(llvm::Instruction *Inst,
                                                     const CallExpr *E) {
  constexpr const char *Tag = "amdgpu-synchronize-as";

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

static Intrinsic::ID getIntrinsicIDforWaveReduction(unsigned BuiltinID) {
  switch (BuiltinID) {
  default:
    llvm_unreachable("Unknown BuiltinID for wave reduction");
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_add_u32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_add_u64:
    return Intrinsic::amdgcn_wave_reduce_add;
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_sub_u32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_sub_u64:
    return Intrinsic::amdgcn_wave_reduce_sub;
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_min_i32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_min_i64:
    return Intrinsic::amdgcn_wave_reduce_min;
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_min_u32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_min_u64:
    return Intrinsic::amdgcn_wave_reduce_umin;
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_max_i32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_max_i64:
    return Intrinsic::amdgcn_wave_reduce_max;
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_max_u32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_max_u64:
    return Intrinsic::amdgcn_wave_reduce_umax;
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_and_b32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_and_b64:
    return Intrinsic::amdgcn_wave_reduce_and;
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_or_b32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_or_b64:
    return Intrinsic::amdgcn_wave_reduce_or;
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_xor_b32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_xor_b64:
    return Intrinsic::amdgcn_wave_reduce_xor;
  }
}

Value *CodeGenFunction::EmitAMDGPUBuiltinExpr(unsigned BuiltinID,
                                              const CallExpr *E) {
  llvm::AtomicOrdering AO = llvm::AtomicOrdering::SequentiallyConsistent;
  llvm::SyncScope::ID SSID;
  switch (BuiltinID) {
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_add_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_sub_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_i32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_i32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_and_b32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_or_b32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_xor_b32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_add_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_sub_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_i64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_i64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_and_b64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_or_b64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_xor_b64: {
    Intrinsic::ID IID = getIntrinsicIDforWaveReduction(BuiltinID);
    llvm::Value *Value = EmitScalarExpr(E->getArg(0));
    llvm::Value *Strategy = EmitScalarExpr(E->getArg(1));
    llvm::Function *F = CGM.getIntrinsic(IID, {Value->getType()});
    return Builder.CreateCall(F, {Value, Strategy});
  }
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
  case AMDGPU::BI__builtin_amdgcn_rcp_bf16:
    return emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::amdgcn_rcp);
  case AMDGPU::BI__builtin_amdgcn_sqrt:
  case AMDGPU::BI__builtin_amdgcn_sqrtf:
  case AMDGPU::BI__builtin_amdgcn_sqrth:
  case AMDGPU::BI__builtin_amdgcn_sqrt_bf16:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::amdgcn_sqrt);
  case AMDGPU::BI__builtin_amdgcn_rsq:
  case AMDGPU::BI__builtin_amdgcn_rsqf:
  case AMDGPU::BI__builtin_amdgcn_rsqh:
  case AMDGPU::BI__builtin_amdgcn_rsq_bf16:
    return emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::amdgcn_rsq);
  case AMDGPU::BI__builtin_amdgcn_rsq_clamp:
  case AMDGPU::BI__builtin_amdgcn_rsq_clampf:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::amdgcn_rsq_clamp);
  case AMDGPU::BI__builtin_amdgcn_sinf:
  case AMDGPU::BI__builtin_amdgcn_sinh:
  case AMDGPU::BI__builtin_amdgcn_sin_bf16:
    return emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::amdgcn_sin);
  case AMDGPU::BI__builtin_amdgcn_cosf:
  case AMDGPU::BI__builtin_amdgcn_cosh:
  case AMDGPU::BI__builtin_amdgcn_cos_bf16:
    return emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::amdgcn_cos);
  case AMDGPU::BI__builtin_amdgcn_dispatch_ptr:
    return EmitAMDGPUDispatchPtr(*this, E);
  case AMDGPU::BI__builtin_amdgcn_logf:
  case AMDGPU::BI__builtin_amdgcn_log_bf16:
    return emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::amdgcn_log);
  case AMDGPU::BI__builtin_amdgcn_exp2f:
  case AMDGPU::BI__builtin_amdgcn_exp2_bf16:
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
  case AMDGPU::BI__builtin_amdgcn_inverse_ballot_w32:
  case AMDGPU::BI__builtin_amdgcn_inverse_ballot_w64: {
    llvm::Value *Src = EmitScalarExpr(E->getArg(0));
    Function *F =
        CGM.getIntrinsic(Intrinsic::amdgcn_inverse_ballot, {Src->getType()});
    return Builder.CreateCall(F, {Src});
  }
  case AMDGPU::BI__builtin_amdgcn_tanhf:
  case AMDGPU::BI__builtin_amdgcn_tanhh:
  case AMDGPU::BI__builtin_amdgcn_tanh_bf16:
    return emitBuiltinWithOneOverloadedType<1>(*this, E,
                                               Intrinsic::amdgcn_tanh);
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
  case AMDGPU::BI__builtin_amdgcn_global_load_tr4_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr8_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr6_b96_v3i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8i16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8f16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8bf16:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr4_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr8_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr6_b96_v3i32:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8i16:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8f16:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8bf16:
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
    case AMDGPU::BI__builtin_amdgcn_global_load_tr8_b64_v2i32:
      IID = Intrinsic::amdgcn_global_load_tr_b64;
      break;
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4i16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4f16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4bf16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8i16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8f16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8bf16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8i16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8f16:
    case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8bf16:
      IID = Intrinsic::amdgcn_global_load_tr_b128;
      break;
    case AMDGPU::BI__builtin_amdgcn_global_load_tr4_b64_v2i32:
      IID = Intrinsic::amdgcn_global_load_tr4_b64;
      break;
    case AMDGPU::BI__builtin_amdgcn_global_load_tr6_b96_v3i32:
      IID = Intrinsic::amdgcn_global_load_tr6_b96;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_load_tr4_b64_v2i32:
      IID = Intrinsic::amdgcn_ds_load_tr4_b64;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_load_tr6_b96_v3i32:
      IID = Intrinsic::amdgcn_ds_load_tr6_b96;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_load_tr8_b64_v2i32:
      IID = Intrinsic::amdgcn_ds_load_tr8_b64;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8i16:
    case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8f16:
    case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8bf16:
      IID = Intrinsic::amdgcn_ds_load_tr16_b128;
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
  case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b32:
  case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b64:
  case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b128:
  case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b32:
  case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b64:
  case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b128: {

    Intrinsic::ID IID;
    switch (BuiltinID) {
    case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b32:
      IID = Intrinsic::amdgcn_global_load_monitor_b32;
      break;
    case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b64:
      IID = Intrinsic::amdgcn_global_load_monitor_b64;
      break;
    case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b128:
      IID = Intrinsic::amdgcn_global_load_monitor_b128;
      break;
    case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b32:
      IID = Intrinsic::amdgcn_flat_load_monitor_b32;
      break;
    case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b64:
      IID = Intrinsic::amdgcn_flat_load_monitor_b64;
      break;
    case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b128:
      IID = Intrinsic::amdgcn_flat_load_monitor_b128;
      break;
    }

    llvm::Type *LoadTy = ConvertType(E->getType());
    llvm::Value *Addr = EmitScalarExpr(E->getArg(0));
    llvm::Value *Val = EmitScalarExpr(E->getArg(1));
    llvm::Function *F = CGM.getIntrinsic(IID, {LoadTy});
    return Builder.CreateCall(F, {Addr, Val});
  }
  case AMDGPU::BI__builtin_amdgcn_cluster_load_b32:
  case AMDGPU::BI__builtin_amdgcn_cluster_load_b64:
  case AMDGPU::BI__builtin_amdgcn_cluster_load_b128: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    case AMDGPU::BI__builtin_amdgcn_cluster_load_b32:
      IID = Intrinsic::amdgcn_cluster_load_b32;
      break;
    case AMDGPU::BI__builtin_amdgcn_cluster_load_b64:
      IID = Intrinsic::amdgcn_cluster_load_b64;
      break;
    case AMDGPU::BI__builtin_amdgcn_cluster_load_b128:
      IID = Intrinsic::amdgcn_cluster_load_b128;
      break;
    }
    SmallVector<Value *, 3> Args;
    for (int i = 0, e = E->getNumArgs(); i != e; ++i)
      Args.push_back(EmitScalarExpr(E->getArg(i)));
    llvm::Function *F = CGM.getIntrinsic(IID, {ConvertType(E->getType())});
    return Builder.CreateCall(F, {Args});
  }
  case AMDGPU::BI__builtin_amdgcn_load_to_lds: {
    // Should this have asan instrumentation?
    return emitBuiltinWithOneOverloadedType<5>(*this, E,
                                               Intrinsic::amdgcn_load_to_lds);
  }
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_32x4B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_32x4B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_16x8B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_16x8B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_8x16B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_8x16B: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_32x4B:
      IID = Intrinsic::amdgcn_cooperative_atomic_load_32x4B;
      break;
    case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_32x4B:
      IID = Intrinsic::amdgcn_cooperative_atomic_store_32x4B;
      break;
    case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_16x8B:
      IID = Intrinsic::amdgcn_cooperative_atomic_load_16x8B;
      break;
    case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_16x8B:
      IID = Intrinsic::amdgcn_cooperative_atomic_store_16x8B;
      break;
    case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_8x16B:
      IID = Intrinsic::amdgcn_cooperative_atomic_load_8x16B;
      break;
    case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_8x16B:
      IID = Intrinsic::amdgcn_cooperative_atomic_store_8x16B;
      break;
    }

    LLVMContext &Ctx = CGM.getLLVMContext();
    SmallVector<Value *, 5> Args;
    // last argument is a MD string
    const unsigned ScopeArg = E->getNumArgs() - 1;
    for (unsigned i = 0; i != ScopeArg; ++i)
      Args.push_back(EmitScalarExpr(E->getArg(i)));
    StringRef Arg = cast<StringLiteral>(E->getArg(ScopeArg)->IgnoreParenCasts())
                        ->getString();
    llvm::MDNode *MD = llvm::MDNode::get(Ctx, {llvm::MDString::get(Ctx, Arg)});
    Args.push_back(llvm::MetadataAsValue::get(Ctx, MD));
    // Intrinsic is typed based on the pointer AS. Pointer is always the first
    // argument.
    llvm::Function *F = CGM.getIntrinsic(IID, {Args[0]->getType()});
    return Builder.CreateCall(F, {Args});
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
  case AMDGPU::BI__builtin_amdgcn_image_bvh8_intersect_ray:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_dual_intersect_ray: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    case AMDGPU::BI__builtin_amdgcn_image_bvh8_intersect_ray:
      IID = Intrinsic::amdgcn_image_bvh8_intersect_ray;
      break;
    case AMDGPU::BI__builtin_amdgcn_image_bvh_dual_intersect_ray:
      IID = Intrinsic::amdgcn_image_bvh_dual_intersect_ray;
      break;
    }
    llvm::Value *NodePtr = EmitScalarExpr(E->getArg(0));
    llvm::Value *RayExtent = EmitScalarExpr(E->getArg(1));
    llvm::Value *InstanceMask = EmitScalarExpr(E->getArg(2));
    llvm::Value *RayOrigin = EmitScalarExpr(E->getArg(3));
    llvm::Value *RayDir = EmitScalarExpr(E->getArg(4));
    llvm::Value *Offset = EmitScalarExpr(E->getArg(5));
    llvm::Value *TextureDescr = EmitScalarExpr(E->getArg(6));

    Address RetRayOriginPtr = EmitPointerWithAlignment(E->getArg(7));
    Address RetRayDirPtr = EmitPointerWithAlignment(E->getArg(8));

    llvm::Function *IntrinsicFunc = CGM.getIntrinsic(IID);

    llvm::CallInst *CI = Builder.CreateCall(
        IntrinsicFunc, {NodePtr, RayExtent, InstanceMask, RayOrigin, RayDir,
                        Offset, TextureDescr});

    llvm::Value *RetVData = Builder.CreateExtractValue(CI, 0);
    llvm::Value *RetRayOrigin = Builder.CreateExtractValue(CI, 1);
    llvm::Value *RetRayDir = Builder.CreateExtractValue(CI, 2);

    Builder.CreateStore(RetRayOrigin, RetRayOriginPtr);
    Builder.CreateStore(RetRayDir, RetRayDirPtr);

    return RetVData;
  }

  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_rtn:
  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push4_pop1_rtn:
  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push8_pop1_rtn:
  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push8_pop2_rtn: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_rtn:
      IID = Intrinsic::amdgcn_ds_bvh_stack_rtn;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push4_pop1_rtn:
      IID = Intrinsic::amdgcn_ds_bvh_stack_push4_pop1_rtn;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push8_pop1_rtn:
      IID = Intrinsic::amdgcn_ds_bvh_stack_push8_pop1_rtn;
      break;
    case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push8_pop2_rtn:
      IID = Intrinsic::amdgcn_ds_bvh_stack_push8_pop2_rtn;
      break;
    }

    SmallVector<Value *, 4> Args;
    for (int i = 0, e = E->getNumArgs(); i != e; ++i)
      Args.push_back(EmitScalarExpr(E->getArg(i)));

    Function *F = CGM.getIntrinsic(IID);
    Value *Call = Builder.CreateCall(F, Args);
    Value *Rtn = Builder.CreateExtractValue(Call, 0);
    Value *A = Builder.CreateExtractValue(Call, 1);
    llvm::Type *RetTy = ConvertType(E->getType());
    Value *I0 = Builder.CreateInsertElement(PoisonValue::get(RetTy), Rtn,
                                            (uint64_t)0);
    // ds_bvh_stack_push8_pop2_rtn returns {i64, i32} but the builtin returns
    // <2 x i64>, zext the second value.
    if (A->getType()->getPrimitiveSizeInBits() <
        RetTy->getScalarType()->getPrimitiveSizeInBits())
      A = Builder.CreateZExt(A, RetTy->getScalarType());

    return Builder.CreateInsertElement(I0, A, 1);
  }
  case AMDGPU::BI__builtin_amdgcn_image_load_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_1d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_1d, false);
  case AMDGPU::BI__builtin_amdgcn_image_load_1darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_1darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_1darray, false);
  case AMDGPU::BI__builtin_amdgcn_image_load_2d_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_2d, false);
  case AMDGPU::BI__builtin_amdgcn_image_load_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_2darray, false);
  case AMDGPU::BI__builtin_amdgcn_image_load_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_3d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_3d, false);
  case AMDGPU::BI__builtin_amdgcn_image_load_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_cube_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_cube, false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_1d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_mip_1d, false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_1darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_1darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_mip_1darray, false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2d_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_mip_2d, false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_mip_2darray, false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_3d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_mip_3d, false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_cube_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_load_mip_cube, false);
  case AMDGPU::BI__builtin_amdgcn_image_store_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_1d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_1d, true);
  case AMDGPU::BI__builtin_amdgcn_image_store_1darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_1darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_1darray, true);
  case AMDGPU::BI__builtin_amdgcn_image_store_2d_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_2d, true);
  case AMDGPU::BI__builtin_amdgcn_image_store_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_2darray, true);
  case AMDGPU::BI__builtin_amdgcn_image_store_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_3d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_3d, true);
  case AMDGPU::BI__builtin_amdgcn_image_store_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_cube_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_cube, true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_mip_1d, true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_mip_1darray, true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2d_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_mip_2d, true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_mip_2darray, true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_3d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_mip_3d, true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_cube_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_store_mip_cube, true);
  case AMDGPU::BI__builtin_amdgcn_image_sample_1d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_1d_v4f16_f32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_sample_1d, false);
  case AMDGPU::BI__builtin_amdgcn_image_sample_1darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_1darray_v4f16_f32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_sample_1darray, false);
  case AMDGPU::BI__builtin_amdgcn_image_sample_2d_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2d_v4f16_f32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_sample_2d, false);
  case AMDGPU::BI__builtin_amdgcn_image_sample_2darray_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2darray_v4f16_f32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_sample_2darray, false);
  case AMDGPU::BI__builtin_amdgcn_image_sample_3d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_3d_v4f16_f32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_sample_3d, false);
  case AMDGPU::BI__builtin_amdgcn_image_sample_cube_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_cube_v4f16_f32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, E, Intrinsic::amdgcn_image_sample_cube, false);
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
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64:
  // GFX1250 WMMA builtins
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x4_f32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x32_bf16:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x32_f16:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x32_f16:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x32_bf16:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x64_iu8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_32x16x128_f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale_f32_32x16x128_f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x64_f16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x64_bf16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x64_f16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x64_bf16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x128_iu8: {

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
    // Need return type when D and C are of different types.
    bool NeedReturnType = false;

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
    // GFX1250 WMMA builtins
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x4_f32:
      ArgsForMatchingMatrixTypes = {5, 1};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x4_f32;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x32_bf16:
      ArgsForMatchingMatrixTypes = {5, 1};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x32_bf16;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x32_f16:
      ArgsForMatchingMatrixTypes = {5, 1};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x32_f16;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x32_f16:
      ArgsForMatchingMatrixTypes = {5, 1};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f16_16x16x32_f16;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x32_bf16:
      ArgsForMatchingMatrixTypes = {5, 1};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_bf16_16x16x32_bf16;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16:
      NeedReturnType = true;
      ArgsForMatchingMatrixTypes = {1, 5};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_bf16f32_16x16x32_bf16;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f16_16x16x64_fp8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f16_16x16x64_fp8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f16_16x16x64_bf8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f16_16x16x64_bf8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f16_16x16x128_fp8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f16_16x16x128_fp8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f16_16x16x128_bf8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f16_16x16x128_bf8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x128_fp8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x128_fp8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x128_bf8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8:
      ArgsForMatchingMatrixTypes = {3, 0};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x128_bf8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x64_iu8:
      ArgsForMatchingMatrixTypes = {4, 1};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_i32_16x16x64_iu8;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4:
      ArgsForMatchingMatrixTypes = {5, 1, 3};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_16x16x128_f8f6f4;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4:
      ArgsForMatchingMatrixTypes = {5, 1, 3};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_scale_f32_16x16x128_f8f6f4;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4:
      ArgsForMatchingMatrixTypes = {5, 1, 3};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_scale16_f32_16x16x128_f8f6f4;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_f32_32x16x128_f4:
      ArgsForMatchingMatrixTypes = {3, 0, 1};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_f32_32x16x128_f4;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_scale_f32_32x16x128_f4:
      ArgsForMatchingMatrixTypes = {3, 0, 1};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_scale_f32_32x16x128_f4;
      break;
    case AMDGPU::BI__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4:
      ArgsForMatchingMatrixTypes = {3, 0, 1};
      BuiltinWMMAOp = Intrinsic::amdgcn_wmma_scale16_f32_32x16x128_f4;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x64_f16:
      ArgsForMatchingMatrixTypes = {4, 1, 3, 5};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x64_f16;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x64_bf16:
      ArgsForMatchingMatrixTypes = {4, 1, 3, 5};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x64_bf16;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x64_f16:
      ArgsForMatchingMatrixTypes = {4, 1, 3, 5};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f16_16x16x64_f16;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x64_bf16:
      ArgsForMatchingMatrixTypes = {4, 1, 3, 5};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_bf16_16x16x64_bf16;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16:
      ArgsForMatchingMatrixTypes = {4, 1, 3, 5};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_bf16f32_16x16x64_bf16;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x128_fp8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_fp8_bf8:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x128_fp8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_bf8_fp8:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x128_bf8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_bf8_bf8:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f32_16x16x128_bf8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_fp8_fp8:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f16_16x16x128_fp8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_fp8_bf8:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f16_16x16x128_fp8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_bf8_fp8:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f16_16x16x128_bf8_fp8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_bf8_bf8:
      ArgsForMatchingMatrixTypes = {2, 0, 1, 3};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_f16_16x16x128_bf8_bf8;
      break;
    case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x128_iu8:
      ArgsForMatchingMatrixTypes = {4, 1, 3, 5};
      BuiltinWMMAOp = Intrinsic::amdgcn_swmmac_i32_16x16x128_iu8;
      break;
    }

    SmallVector<Value *, 6> Args;
    for (int i = 0, e = E->getNumArgs(); i != e; ++i)
      Args.push_back(EmitScalarExpr(E->getArg(i)));
    if (AppendFalseForOpselArg)
      Args.push_back(Builder.getFalse());

    SmallVector<llvm::Type *, 6> ArgTypes;
    if (NeedReturnType)
      ArgTypes.push_back(ConvertType(E->getType()));
    for (auto ArgIdx : ArgsForMatchingMatrixTypes)
      ArgTypes.push_back(Args[ArgIdx]->getType());

    Function *F = CGM.getIntrinsic(BuiltinWMMAOp, ArgTypes);
    return Builder.CreateCall(F, Args);
  }
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
      if (getTarget().getTriple().isSPIRV())
        SSID = getLLVMContext().getOrInsertSyncScopeID("device");
      else
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
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_add_i32:
    return emitBuiltinWithOneOverloadedType<5>(
        *this, E, Intrinsic::amdgcn_raw_ptr_buffer_atomic_add);
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16:
    return emitBuiltinWithOneOverloadedType<5>(
        *this, E, Intrinsic::amdgcn_raw_ptr_buffer_atomic_fadd);
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32:
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64:
    return emitBuiltinWithOneOverloadedType<5>(
        *this, E, Intrinsic::amdgcn_raw_ptr_buffer_atomic_fmin);
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32:
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64:
    return emitBuiltinWithOneOverloadedType<5>(
        *this, E, Intrinsic::amdgcn_raw_ptr_buffer_atomic_fmax);
  case AMDGPU::BI__builtin_amdgcn_s_prefetch_data:
    return emitBuiltinWithOneOverloadedType<2>(
        *this, E, Intrinsic::amdgcn_s_prefetch_data);
  case Builtin::BIlogbf:
  case Builtin::BI__builtin_logbf: {
    Value *Src0 = EmitScalarExpr(E->getArg(0));
    Function *FrExpFunc = CGM.getIntrinsic(
        Intrinsic::frexp, {Src0->getType(), Builder.getInt32Ty()});
    CallInst *FrExp = Builder.CreateCall(FrExpFunc, Src0);
    Value *Exp = Builder.CreateExtractValue(FrExp, 1);
    Value *Add = Builder.CreateAdd(
        Exp, ConstantInt::getSigned(Exp->getType(), -1), "", false, true);
    Value *SIToFP = Builder.CreateSIToFP(Add, Builder.getFloatTy());
    Value *Fabs =
        emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::fabs);
    Value *FCmpONE = Builder.CreateFCmpONE(
        Fabs, ConstantFP::getInfinity(Builder.getFloatTy()));
    Value *Sel1 = Builder.CreateSelect(FCmpONE, SIToFP, Fabs);
    Value *FCmpOEQ =
        Builder.CreateFCmpOEQ(Src0, ConstantFP::getZero(Builder.getFloatTy()));
    Value *Sel2 = Builder.CreateSelect(
        FCmpOEQ,
        ConstantFP::getInfinity(Builder.getFloatTy(), /*Negative=*/true), Sel1);
    return Sel2;
  }
  case Builtin::BIlogb:
  case Builtin::BI__builtin_logb: {
    Value *Src0 = EmitScalarExpr(E->getArg(0));
    Function *FrExpFunc = CGM.getIntrinsic(
        Intrinsic::frexp, {Src0->getType(), Builder.getInt32Ty()});
    CallInst *FrExp = Builder.CreateCall(FrExpFunc, Src0);
    Value *Exp = Builder.CreateExtractValue(FrExp, 1);
    Value *Add = Builder.CreateAdd(
        Exp, ConstantInt::getSigned(Exp->getType(), -1), "", false, true);
    Value *SIToFP = Builder.CreateSIToFP(Add, Builder.getDoubleTy());
    Value *Fabs =
        emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::fabs);
    Value *FCmpONE = Builder.CreateFCmpONE(
        Fabs, ConstantFP::getInfinity(Builder.getDoubleTy()));
    Value *Sel1 = Builder.CreateSelect(FCmpONE, SIToFP, Fabs);
    Value *FCmpOEQ =
        Builder.CreateFCmpOEQ(Src0, ConstantFP::getZero(Builder.getDoubleTy()));
    Value *Sel2 = Builder.CreateSelect(
        FCmpOEQ,
        ConstantFP::getInfinity(Builder.getDoubleTy(), /*Negative=*/true),
        Sel1);
    return Sel2;
  }
  case Builtin::BIscalbnf:
  case Builtin::BI__builtin_scalbnf:
  case Builtin::BIscalbn:
  case Builtin::BI__builtin_scalbn:
    return emitBinaryExpMaybeConstrainedFPBuiltin(
        *this, E, Intrinsic::ldexp, Intrinsic::experimental_constrained_ldexp);
  default:
    return nullptr;
  }
}
