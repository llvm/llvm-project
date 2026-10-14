//===- wmma-lowering.cpp - Transpiler matrix remapping -------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/wmma-lowering.h"

#include "transpiler/raiser/raise-context.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace COMGR::transpiler {
namespace {

Value *emitDSBpermute(IRBuilder<> &B, Module &M, Value *ByteOffset,
                      Value *Source) {
  Function *Fn =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_ds_bpermute);
  return B.CreateCall(Fn, {ByteOffset, Source}, "bperm");
}

Value *packDwords(IRBuilder<> &B, ArrayRef<Value *> Dwords, Type *TargetTy) {
  FixedVectorType *VectorTy =
      FixedVectorType::get(B.getInt32Ty(), Dwords.size());
  Value *Vector = PoisonValue::get(VectorTy);
  for (auto [Index, Dword] : enumerate(Dwords))
    Vector = B.CreateInsertElement(Vector, Dword, Index, "pack");
  return B.CreateBitCast(Vector, TargetTy, "pack.cast");
}

SmallVector<Value *> unpackDwords(IRBuilder<> &B, Value *Vector,
                                  unsigned Count) {
  FixedVectorType *VectorTy = FixedVectorType::get(B.getInt32Ty(), Count);
  Value *AsI32 = B.CreateBitCast(Vector, VectorTy, "dwords");
  SmallVector<Value *> Result;
  Result.reserve(Count);
  for (unsigned Index = 0; Index != Count; ++Index)
    Result.push_back(B.CreateExtractElement(AsI32, Index, "dword"));
  return Result;
}

Value *selectByLaneGroup(IRBuilder<> &B, Value *LaneGroup,
                         ArrayRef<Value *> Values) {
  assert(Values.size() == 4);
  Value *IsGroup2 = B.CreateICmpEQ(LaneGroup, B.getInt32(2), "is.group2");
  Value *Selected = B.CreateSelect(IsGroup2, Values[2], Values[3], "group23");
  Value *IsGroup1 = B.CreateICmpEQ(LaneGroup, B.getInt32(1), "is.group1");
  Selected = B.CreateSelect(IsGroup1, Values[1], Selected, "group123");
  Value *IsGroup0 = B.CreateICmpEQ(LaneGroup, B.getInt32(0), "is.group0");
  return B.CreateSelect(IsGroup0, Values[0], Selected, "group");
}

void redistributeInput(IRBuilder<> &B, Module &M, ArrayRef<Value *> Source,
                       unsigned SourceWave, Value *LaneInGroup,
                       Value *LaneGroup, MutableArrayRef<Value *> Low,
                       MutableArrayRef<Value *> High) {
  unsigned InputDwords = Low.size();
  unsigned KParts = High.empty() ? 1 : 2;
  for (unsigned TargetDword = 0; TargetDword != InputDwords; ++TargetDword) {
    for (unsigned KHalf = 0; KHalf != KParts; ++KHalf) {
      SmallVector<Value *, 4> Candidates;
      for (unsigned Group = 0; Group != 4; ++Group) {
        detail::FragmentDwordLocation Location = detail::getWMMAInputDword(
            SourceWave, Group * 16, TargetDword, KHalf, InputDwords);
        Value *SourceLane =
            B.CreateAdd(LaneInGroup, B.getInt32(Location.Lane), "source.lane");
        Value *Address =
            B.CreateShl(SourceLane, B.getInt32(2), "source.address");
        Candidates.push_back(
            emitDSBpermute(B, M, Address, Source[Location.Dword]));
      }
      (KHalf == 0 ? Low : High)[TargetDword] =
          selectByLaneGroup(B, LaneGroup, Candidates);
    }
  }
}

void redistributeAccumulator(IRBuilder<> &B, Module &M,
                             ArrayRef<Value *> Source, unsigned SourceWave,
                             Value *LaneInGroup, Value *LaneGroup,
                             MutableArrayRef<Value *> Result) {
  for (unsigned TargetDword = 0; TargetDword != 4; ++TargetDword) {
    SmallVector<Value *, 4> Candidates;
    for (unsigned Group = 0; Group != 4; ++Group) {
      detail::FragmentDwordLocation Location =
          detail::getWMMAAccumulatorDword(SourceWave, Group * 16, TargetDword);
      Value *SourceLane =
          B.CreateAdd(LaneInGroup, B.getInt32(Location.Lane), "source.lane");
      Value *Address = B.CreateShl(SourceLane, B.getInt32(2), "source.address");
      Candidates.push_back(
          emitDSBpermute(B, M, Address, Source[Location.Dword]));
    }
    Result[TargetDword] = selectByLaneGroup(B, LaneGroup, Candidates);
  }
}

void collectResult(IRBuilder<> &B, Module &M, ArrayRef<Value *> Source,
                   Value *Wave32Lane, MutableArrayRef<Value *> Result) {
  Value *LaneInHalf = B.CreateAnd(Wave32Lane, B.getInt32(15), "lane.in.half");
  Value *IsUpper = B.CreateICmpUGE(Wave32Lane, B.getInt32(16), "lane.upper");
  for (unsigned Gpr = 0; Gpr != 8; ++Gpr) {
    detail::FragmentDwordLocation Lower = detail::getMFMAResultDword(0, Gpr);
    detail::FragmentDwordLocation Upper = detail::getMFMAResultDword(16, Gpr);
    Value *LaneBase = B.CreateSelect(IsUpper, B.getInt32(Upper.Lane),
                                     B.getInt32(Lower.Lane), "collect.base");
    Value *SourceLane = B.CreateAdd(LaneBase, LaneInHalf, "collect.lane");
    Value *Address = B.CreateShl(SourceLane, B.getInt32(2), "collect.address");
    Result[Gpr] = emitDSBpermute(B, M, Address, Source[Lower.Dword]);
  }
}

void runGroupPass(RaiseContext &Ctx, unsigned GroupBase, Value *LaneId,
                  ArrayRef<Value *> A, ArrayRef<Value *> B, ArrayRef<Value *> C,
                  WMMAInputType InputType, MutableArrayRef<Value *> Result) {
  IRBuilder<> &Builder = Ctx.B;
  Module &M = *Builder.GetInsertBlock()->getModule();
  Value *LaneInGroup =
      Builder.CreateAnd(LaneId, Builder.getInt32(15), "matrix.lane");
  Value *LaneGroup =
      Builder.CreateLShr(LaneId, Builder.getInt32(4), "lane.group");
  unsigned SourceWave = GroupBase / 32;

  unsigned InputDwords = InputType == WMMAInputType::F32 ? 1 : 2;
  unsigned HighDwords = InputType == WMMAInputType::F32 ? 0 : 2;
  SmallVector<Value *, 2> ALow(InputDwords), AHigh(HighDwords),
      BLow(InputDwords), BHigh(HighDwords);
  SmallVector<Value *, 4> Accumulator(4);
  redistributeInput(Builder, M, A, SourceWave, LaneInGroup, LaneGroup, ALow,
                    AHigh);
  redistributeInput(Builder, M, B, SourceWave, LaneInGroup, LaneGroup, BLow,
                    BHigh);
  redistributeAccumulator(Builder, M, C, SourceWave, LaneInGroup, LaneGroup,
                          Accumulator);

  Type *InputTy;
  Type *AccumulatorTy;
  Intrinsic::ID IntrinsicId;
  switch (InputType) {
  case WMMAInputType::F32:
    InputTy = Builder.getFloatTy();
    AccumulatorTy = FixedVectorType::get(Builder.getFloatTy(), 4);
    IntrinsicId = Intrinsic::amdgcn_mfma_f32_16x16x4f32;
    break;
  case WMMAInputType::F16:
    InputTy = FixedVectorType::get(Builder.getHalfTy(), 4);
    AccumulatorTy = FixedVectorType::get(Builder.getFloatTy(), 4);
    IntrinsicId = Intrinsic::amdgcn_mfma_f32_16x16x16f16;
    break;
  case WMMAInputType::BF16:
    InputTy = FixedVectorType::get(Builder.getInt16Ty(), 4);
    AccumulatorTy = FixedVectorType::get(Builder.getFloatTy(), 4);
    IntrinsicId = Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k;
    break;
  case WMMAInputType::IU8:
    InputTy = Builder.getInt64Ty();
    AccumulatorTy = FixedVectorType::get(Builder.getInt32Ty(), 4);
    IntrinsicId = Intrinsic::amdgcn_mfma_i32_16x16x32_i8;
    break;
  }

  Value *ALowPack = packDwords(Builder, ALow, InputTy);
  Value *BLowPack = packDwords(Builder, BLow, InputTy);
  Value *AccumulatorPack = packDwords(Builder, Accumulator, AccumulatorTy);
  Function *Mfma = Intrinsic::getOrInsertDeclaration(&M, IntrinsicId);
  Value *Zero = Builder.getInt32(0);
  Value *FirstCall = Builder.CreateCall(
      Mfma, {ALowPack, BLowPack, AccumulatorPack, Zero, Zero, Zero}, "mfma.0");
  Value *First =
      Ctx.Projection.wrapAsWWMValue(Builder, FirstCall, "mfma.0.wwm");

  Value *ResultValue = First;
  if (!AHigh.empty()) {
    Value *AHighPack = packDwords(Builder, AHigh, InputTy);
    Value *BHighPack = packDwords(Builder, BHigh, InputTy);
    Value *SecondCall = Builder.CreateCall(
        Mfma, {AHighPack, BHighPack, First, Zero, Zero, Zero}, "mfma.1");
    ResultValue =
        Ctx.Projection.wrapAsWWMValue(Builder, SecondCall, "mfma.1.wwm");
  }
  SmallVector<Value *> MfmaResult = unpackDwords(Builder, ResultValue, 4);
  Value *Wave32Lane =
      Builder.CreateAnd(LaneId, Builder.getInt32(31), "wave32.lane");
  collectResult(Builder, M, MfmaResult, Wave32Lane, Result);
  for (Value *&Dword : Result)
    Dword = Ctx.Projection.wrapAsWWMValue(Builder, Dword, "matrix.collect.wwm");
}

} // namespace

Expected<Value *> emitWMMAtoMFMA(RaiseContext &Ctx, Value *A, Value *B,
                                 Value *C, WMMAInputType InputType) {
  if (Ctx.Projection.sourceWaveSize() != 32 ||
      Ctx.Projection.targetWaveSize() != 64)
    return createStringError(
        "WMMA to MFMA remapping requires wave32 to wave64");

  unsigned NumSourceWaves = Ctx.Projection.numSourceWavesPerTarget();
  if (NumSourceWaves != 1 && NumSourceWaves != 2)
    return createStringError(
        "WMMA to MFMA remapping requires one or two source waves per target");

  IRBuilder<> &Builder = Ctx.B;
  unsigned InputDwords = InputType == WMMAInputType::F32 ? 2 : 8;
  SmallVector<Value *> ADwords = unpackDwords(Builder, A, InputDwords);
  SmallVector<Value *> BDwords = unpackDwords(Builder, B, InputDwords);
  SmallVector<Value *> CDwords = unpackDwords(Builder, C, 8);
  Value *LaneId = Ctx.emitLaneIdx();

  SmallVector<Value *, 8> FirstResult(8);
  runGroupPass(Ctx, 0, LaneId, ADwords, BDwords, CDwords, InputType,
               FirstResult);
  SmallVector<Value *, 8> FinalResult = FirstResult;
  if (NumSourceWaves == 2) {
    SmallVector<Value *, 8> SecondResult(8);
    runGroupPass(Ctx, 32, LaneId, ADwords, BDwords, CDwords, InputType,
                 SecondResult);
    Value *UseSecond =
        Builder.CreateICmpUGE(LaneId, Builder.getInt32(32), "second.wave");
    for (unsigned I = 0; I != FinalResult.size(); ++I)
      FinalResult[I] = Builder.CreateSelect(UseSecond, SecondResult[I],
                                            FirstResult[I], "matrix.result");
  }

  Type *ElementTy = InputType == WMMAInputType::IU8 ? Builder.getInt32Ty()
                                                    : Builder.getFloatTy();
  Type *ResultTy = FixedVectorType::get(ElementTy, 8);
  return packDwords(Builder, FinalResult, ResultTy);
}

} // namespace COMGR::transpiler
