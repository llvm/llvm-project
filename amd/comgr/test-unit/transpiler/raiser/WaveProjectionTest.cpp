//===- WaveProjectionTest.cpp - wave projection unit tests ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Unit tests for the per-projection contract on WaveProjection and its
// subclasses. Each projection fixes its policy (EXEC storage width, source
// waves per target wave, replication factor, and the exec/mbcnt
// predicates) in its constructor, and the raiser and its passes branch on
// those values through a WaveProjection reference. These tests pin the value
// each projection reports, so a constructor that forgets to set a flag, or a
// change that flips a default, is caught here rather than downstream.
//
// The IR-emitting side of the interface (wrapAsWWMValue) is exercised on a
// small synthetic function.

#include "transpiler/raiser/wave-projection.h"

#include "transpiler/decoder/mc-state.h"
#include "transpiler/raiser/launch.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "gtest/gtest.h"

using namespace llvm;
using COMGR::transpiler::initMCState;
using COMGR::transpiler::MCState;
using COMGR::transpiler::ReplicatedDispatchProjection;
using COMGR::transpiler::ReplicationProjection;
using COMGR::transpiler::ThreadLoopProjection;
using COMGR::transpiler::WaveNativeProjection;
using COMGR::transpiler::WaveProjection;

namespace {

// Every projection here covers the widening direction: a wave32 source
// (gfx1250) onto a wave64 target (gfx942). The projections consult only the
// wave size from live MCSubtargetInfo objects, so the fixture stands up the
// real AMDGPU MC stack for both ISAs.
class WaveProjectionContract : public ::testing::Test {
protected:
  void SetUp() override {
    Expected<MCState> SrcState = initMCState("gfx1250");
    ASSERT_TRUE(static_cast<bool>(SrcState)) << toString(SrcState.takeError());
    SrcMc = std::move(*SrcState);

    Expected<MCState> TgtState = initMCState("gfx942");
    ASSERT_TRUE(static_cast<bool>(TgtState)) << toString(TgtState.takeError());
    TgtMc = std::move(*TgtState);
  }

  const MCSubtargetInfo &srcSTI() const { return *SrcMc.SubtargetInfo; }
  const MCSubtargetInfo &tgtSTI() const { return *TgtMc.SubtargetInfo; }

  MCState SrcMc;
  MCState TgtMc;
};

} // namespace

// ----------------------------------------------------------------------------
// providesSourceWaveExecInvariant: WaveNative decouples hardware EXEC from
// the modeled source mask (via init_whole_wave), so only it reports true.
// Cross-lane collective lowerings gate on this before running.
// ----------------------------------------------------------------------------
TEST_F(WaveProjectionContract, ReplicationDoesNotProvideFullWaveExec) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  ReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_FALSE(Proj.providesSourceWaveExecInvariant());
  EXPECT_EQ(&Proj.SourceSTI, &Src);
  EXPECT_EQ(&Proj.TargetSTI, &Tgt);
  EXPECT_EQ(Proj.sourceWaveSize(), 32u);
  EXPECT_EQ(Proj.targetWaveSize(), 64u);

  // Read through a base reference too: the value lives in the base subobject
  // the constructor set, so the non-virtual accessor resolves it correctly.
  const WaveProjection &Base = Proj;
  EXPECT_FALSE(Base.providesSourceWaveExecInvariant());
}

TEST_F(WaveProjectionContract, WaveNativeProvidesFullWaveExec) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  // The constructor asserts a wave32 -> wave64 direction.
  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  WaveNativeProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_TRUE(Proj.providesSourceWaveExecInvariant());
  EXPECT_EQ(Proj.execStorageTy(), I32Ty);
  EXPECT_FALSE(Proj.broadcastNarrowExecLoWrite());
  EXPECT_TRUE(Proj.preservesMbcntDerivedExec());

  const WaveProjection &Base = Proj;
  EXPECT_TRUE(Base.providesSourceWaveExecInvariant());
}

// A minimal concrete projection that implements only the pure virtuals, used
// to pin the base-class defaults for the constructor-set configuration.
namespace {
class DefaultTestProjection final : public WaveProjection {
public:
  using WaveProjection::WaveProjection;
  llvm::Value *emitSourceWaveId(llvm::IRBuilder<> &) const override {
    return nullptr;
  }
  llvm::Value *emitLaneActiveBit(llvm::IRBuilder<> &,
                                 llvm::Value *) const override {
    return nullptr;
  }
  llvm::Value *ballotI1ToWidth(llvm::IRBuilder<> &, llvm::Value *, llvm::Type *,
                               const llvm::Twine &) const override {
    return nullptr;
  }
  llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &,
                                          llvm::Value *) const override {
    return nullptr;
  }
};
} // namespace

TEST_F(WaveProjectionContract, BaseDefaults) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  DefaultTestProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_FALSE(Proj.providesSourceWaveExecInvariant());
  EXPECT_FALSE(Proj.usesReplicatedDispatch());
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 1u);
}

TEST_F(WaveProjectionContract, ThreadLoop) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  ThreadLoopProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_FALSE(Proj.providesSourceWaveExecInvariant());
  EXPECT_TRUE(Proj.sourceWaveScopedLaneOps());
  EXPECT_EQ(Proj.execStorageTy(), I64Ty);

  const WaveProjection &Base = Proj;
  EXPECT_FALSE(Base.providesSourceWaveExecInvariant());
}

// ----------------------------------------------------------------------------
// numSourceWavesPerTarget: the per-source-wave pass count. A wrong value makes
// callers synthesise the wrong number of passes (a bogus second-wave pass for
// replication, or a skipped second wave for the widening projections).
// ----------------------------------------------------------------------------
TEST_F(WaveProjectionContract, ReplicationHasOneSourceWavePerTarget) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  ReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 1u);
}

TEST_F(WaveProjectionContract, WaveNativeHasTwoSourceWavesPerTarget) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  WaveNativeProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 2u);
}

TEST_F(WaveProjectionContract, ThreadLoopReportsSourceWavesPerTargetRatio) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  ThreadLoopProjection Proj(Src, Tgt, I32Ty, I64Ty);
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 2u);
}

// ----------------------------------------------------------------------------
// Replicated dispatch scales the physical X extent by the wave-size ratio.
// ----------------------------------------------------------------------------
TEST_F(WaveProjectionContract, ReplicatedDispatch) {
  LLVMContext Ctx;
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();

  COMGR::transpiler::KernelLaunchRequirements Launch;
  ReplicatedDispatchProjection Proj(Src, Tgt, I32Ty, I64Ty, Launch);
  EXPECT_TRUE(Proj.usesReplicatedDispatch());
  EXPECT_TRUE(Proj.providesSourceWaveExecInvariant());
  EXPECT_EQ(Proj.replicationFactor(), 2u);
  EXPECT_EQ(Proj.numSourceWavesPerTarget(), 1u);

  ReplicationProjection Plain(Src, Tgt, I32Ty, I64Ty);
  EXPECT_FALSE(Plain.usesReplicatedDispatch());
}

// ----------------------------------------------------------------------------
// wrapAsWWMValue: an identity no-op on projections that already guarantee
// participating source waves have full EXEC, and a strict.wwm wrapper on
// those that do not (Replication). The WMMA lowering relies on both behaviours.
// ----------------------------------------------------------------------------
namespace {
struct IRScaffold {
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F;
  BasicBlock *BB;
  Argument *Arg;
  IRBuilder<> B;

  IRScaffold() : Ctx(), M(std::make_unique<Module>("t", Ctx)), B(Ctx) {
    auto *I32Ty = Type::getInt32Ty(Ctx);
    auto *FnTy = FunctionType::get(Type::getVoidTy(Ctx), {I32Ty}, false);
    F = Function::Create(FnTy, Function::ExternalLinkage, "f", M.get());
    BB = BasicBlock::Create(Ctx, "entry", F);
    Arg = F->getArg(0);
    B.SetInsertPoint(BB);
  }
};
} // namespace

TEST_F(WaveProjectionContract, WrapAsWWMValueIsNoOpOnWaveNative) {
  IRScaffold S;
  auto *I32Ty = Type::getInt32Ty(S.Ctx);
  auto *I64Ty = Type::getInt64Ty(S.Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();
  WaveNativeProjection Proj(Src, Tgt, I32Ty, I64Ty);

  Value *Result = Proj.wrapAsWWMValue(S.B, S.Arg);
  EXPECT_EQ(Result, S.Arg);
  EXPECT_TRUE(S.BB->empty());
}

TEST_F(WaveProjectionContract, WrapAsWWMValueEmitsStrictWWMOnReplication) {
  IRScaffold S;
  auto *I32Ty = Type::getInt32Ty(S.Ctx);
  auto *I64Ty = Type::getInt64Ty(S.Ctx);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();
  ReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);

  Value *Result = Proj.wrapAsWWMValue(S.B, S.Arg);
  EXPECT_NE(Result, S.Arg);
  auto *Cb = dyn_cast<CallInst>(Result);
  ASSERT_NE(Cb, nullptr);
  Function *Callee = Cb->getCalledFunction();
  ASSERT_NE(Callee, nullptr);
  EXPECT_EQ(Callee->getIntrinsicID(), Intrinsic::amdgcn_strict_wwm);
  ASSERT_EQ(Cb->arg_size(), 1u);
  EXPECT_EQ(Cb->getArgOperand(0), S.Arg);
  EXPECT_EQ(Cb->getType(), I32Ty);
  EXPECT_EQ(S.BB->size(), 1u);
}

// wmma-lowering wraps both i32 (result dwords) and <4 x float> (MFMA outputs),
// so cover the vector overload too.
TEST_F(WaveProjectionContract, WrapAsWWMValueHandlesVectorFloatOverload) {
  IRScaffold S;
  auto *I32Ty = Type::getInt32Ty(S.Ctx);
  auto *I64Ty = Type::getInt64Ty(S.Ctx);
  auto *F32Ty = Type::getFloatTy(S.Ctx);
  auto *V4f32Ty = FixedVectorType::get(F32Ty, 4);

  const MCSubtargetInfo &Src = srcSTI();
  const MCSubtargetInfo &Tgt = tgtSTI();
  ReplicationProjection Proj(Src, Tgt, I32Ty, I64Ty);

  Value *Vec = PoisonValue::get(V4f32Ty);
  Value *Result = Proj.wrapAsWWMValue(S.B, Vec);

  auto *Cb = dyn_cast<CallInst>(Result);
  ASSERT_NE(Cb, nullptr);
  Function *Callee = Cb->getCalledFunction();
  ASSERT_NE(Callee, nullptr);
  EXPECT_EQ(Callee->getIntrinsicID(), Intrinsic::amdgcn_strict_wwm);
  EXPECT_EQ(Cb->getType(), V4f32Ty);
  ASSERT_EQ(Cb->arg_size(), 1u);
  EXPECT_EQ(Cb->getArgOperand(0)->getType(), V4f32Ty);
}

TEST_F(WaveProjectionContract, SourceWaveBallotRecognition) {
  LLVMContext Context;
  Module M("ballot", Context);
  IRBuilder<> B(Context);
  Function *F = Function::Create(
      FunctionType::get(B.getVoidTy(),
                        {B.getInt1Ty(), B.getInt64Ty(), B.getInt32Ty()}, false),
      Function::ExternalLinkage, "test", M);
  B.SetInsertPoint(BasicBlock::Create(Context, "entry", F));
  WaveNativeProjection Native(srcSTI(), tgtSTI(), B.getInt32Ty(),
                              B.getInt64Ty());
  ReplicationProjection Replicated(srcSTI(), tgtSTI(), B.getInt32Ty(),
                                   B.getInt64Ty());
  Value *Predicate = F->getArg(0);
  Value *Mask = Native.ballotI1ToWidth(B, Predicate, B.getInt32Ty());
  EXPECT_EQ(Native.matchBallotPredicate(Mask), Predicate);
  EXPECT_EQ(Replicated.matchBallotPredicate(Mask), nullptr);
  Value *Low = Replicated.ballotI1ToWidth(B, Predicate, B.getInt32Ty());
  EXPECT_EQ(Replicated.matchBallotPredicate(Low), Predicate);
  EXPECT_EQ(Native.matchBallotPredicate(Low), nullptr);

  Value *Lane = Native.emitLaneIdx(B);
  Value *Ballot = B.CreateIntrinsic(Intrinsic::amdgcn_ballot, {B.getInt64Ty()},
                                    {Predicate});
  auto Slice = [&](Value *Bits, Value *Offset, Type *ResultTy) {
    return B.CreateTrunc(B.CreateLShr(Bits, Offset), ResultTy);
  };
  Value *Base = B.CreateAnd(Lane, B.getInt32(-32));
  Value *Offset = B.CreateZExt(Base, B.getInt64Ty());
  EXPECT_EQ(Native.matchBallotPredicate(Slice(Ballot, Offset, B.getInt32Ty())),
            Predicate);
  // A fixed half cannot represent both source waves.
  for (unsigned Shift : {0u, 32u, 64u})
    EXPECT_EQ(Native.matchBallotPredicate(
                  Slice(Ballot, B.getInt64(Shift), B.getInt32Ty())),
              nullptr);
  for (Value *WrongBase : {B.CreateAnd(Lane, B.getInt32(31)),
                           B.CreateAnd(F->getArg(2), B.getInt32(-32))})
    EXPECT_EQ(
        Native.matchBallotPredicate(Slice(
            Ballot, B.CreateZExt(WrongBase, B.getInt64Ty()), B.getInt32Ty())),
        nullptr);
  EXPECT_EQ(Native.matchBallotPredicate(Slice(
                Ballot, B.CreateSExt(Base, B.getInt64Ty()), B.getInt32Ty())),
            nullptr);
  EXPECT_EQ(Native.matchBallotPredicate(Slice(Ballot, Offset, B.getInt16Ty())),
            nullptr);
  EXPECT_EQ(
      Native.matchBallotPredicate(Slice(F->getArg(1), Offset, B.getInt32Ty())),
      nullptr);
  Value *NarrowBallot = B.CreateIntrinsic(Intrinsic::amdgcn_ballot,
                                          {B.getInt32Ty()}, {Predicate});
  EXPECT_EQ(
      Native.matchBallotPredicate(Slice(
          B.CreateZExt(NarrowBallot, B.getInt64Ty()), Offset, B.getInt32Ty())),
      nullptr);
  EXPECT_EQ(Replicated.matchBallotPredicate(
                B.CreateTrunc(NarrowBallot, B.getInt16Ty())),
            nullptr);
  B.CreateRetVoid();
}
