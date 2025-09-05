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
  // The 0th bit simulates the `vta` of RVV
  // The 1st bit simulates the `vma` of RVV
  constexpr unsigned RVV_VTA = 0x1;
  constexpr unsigned RVV_VMA = 0x2;
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
  case RISCV::BI__builtin_riscv_unzip_32:
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
  case RISCV::BI__builtin_riscv_sadd: {
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

    // Packed SIMD
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
    case RISCV::BI__builtin_riscv_sadd:
      ID = Intrinsic::riscv_sadd;
      break;
    }

    IntrinsicTypes = {ResultType};
    break;
  }

  case RISCV::BI__builtin_riscv_pslli_b_32:
  case RISCV::BI__builtin_riscv_pslli_b_64:
  case RISCV::BI__builtin_riscv_pslli_h_32:
  case RISCV::BI__builtin_riscv_pslli_h_64:
  case RISCV::BI__builtin_riscv_pslli_w:
  case RISCV::BI__builtin_riscv_psslai_h_32:
  case RISCV::BI__builtin_riscv_psslai_h_64:
  case RISCV::BI__builtin_riscv_psslai_w:
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
  case RISCV::BI__builtin_riscv_sslai: {
    switch (BuiltinID) {
    default: llvm_unreachable("unexpected builtin ID");
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
    }
    IntrinsicTypes = {ResultType, Ops[1]->getType()};
    break;
  }


  case RISCV::BI__builtin_riscv_pmul_h_b01_32:
  case RISCV::BI__builtin_riscv_pmul_h_b01_64:
  case RISCV::BI__builtin_riscv_pmul_w_h01:
  case RISCV::BI__builtin_riscv_pmulu_h_b01_32:
  case RISCV::BI__builtin_riscv_pmulu_h_b01_64:
  case RISCV::BI__builtin_riscv_pmulu_w_h01: {
    switch (BuiltinID) {
    default: llvm_unreachable("unexpected builtin ID");
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
    }
    IntrinsicTypes = {ResultType, Ops[0]->getType()};
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
