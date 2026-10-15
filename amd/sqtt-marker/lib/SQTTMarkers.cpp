//===- SQTTMarkers.cpp - SQTT marker lowering -----------------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements marker boundaries, scope checks, and target trace lowering.
///
//===----------------------------------------------------------------------===//

#include "SQTTPass.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

static constexpr uint32_t Gfx12ShaderCyclesLo = 29;
static constexpr StringLiteral SqttScopeFilterMetadata = "sqtt.scope.filter";
static constexpr char M0TraceAsmText[] = "s_mov_b32 m0, $1\n"
                                         "s_nop $2\n"
                                         "s_ttracedata";

// Only sched_barrier and IR fences constrain ttracedata scheduling. AMDGPU
// barriers are IntrNoMem, so they still need an explicit sched_barrier(0).
static bool isHardSchedBoundary(Instruction *I) {
  if (!I)
    return false;
  if (isa<FenceInst>(I))
    return true;
  auto *CI = dyn_cast<CallInst>(I);
  Function *F = CI ? CI->getCalledFunction() : nullptr;
  return F && F->getIntrinsicID() == Intrinsic::amdgcn_sched_barrier;
}

static bool isPayloadSequence(CallInst *Header, CallInst *Payload,
                              MDNode *Group) {
  for (Instruction *I = Header->getNextNode(); I != Payload;
       I = I ? I->getNextNode() : nullptr)
    if (!I || (I->getMetadata(SqttPayloadGroupMetadata) != Group &&
               !isa<PseudoProbeInst>(I) && !I->isLifetimeStartOrEnd()))
      return false;
  return true;
}

static void branchToScopedTrace(BasicBlock *Source, BasicBlock *Trace,
                                BasicBlock *Skip, Value *Ok) {
  Source->getTerminator()->eraseFromParent();
  auto *Branch = IRBuilder<>(Source).CreateCondBr(Ok, Trace, Skip);
  // Every scope ID comes from a scalar hardware register and is uniform for
  // the wave.  Preserve that fact for AMDGPU codegen: treating these small
  // diamonds as divergent extends live ranges across every marker and can
  // substantially increase VGPR pressure.
  Branch->setMetadata("amdgpu.uniform", MDNode::get(Source->getContext(), {}));
}

// Emit the configured compiler ordering barrier; the fence is limited to LDS.
static void emitMemBarrier(IRBuilder<> &B, MemBarrierMode Mode) {
  if (Mode == MemBarrierMode::None)
    return;
  LLVMContext &Ctx = B.getContext();
  if (Mode == MemBarrierMode::AsmClobber) {
    InlineAsm *MF =
        InlineAsm::get(FunctionType::get(Type::getVoidTy(Ctx), false), "",
                       "~{memory}", /*hasSideEffects=*/true);
    B.CreateCall(MF);
    return;
  }
  // MemBarrierMode::Fence
  SyncScope::ID WG = Ctx.getOrInsertSyncScopeID("workgroup");
  FenceInst *F = B.CreateFence(AtomicOrdering::AcquireRelease, WG);

  // Keep compiler ordering without global cache invalidation.
  Metadata *LocalSyncAS[] = {MDString::get(Ctx, "amdgpu-synchronize-as"),
                             MDString::get(Ctx, "local")};
  F->setMetadata(LLVMContext::MD_mmra, MDNode::get(Ctx, LocalSyncAS));
}

void SQTTInstrumentPass::emitTraceBoundary(IRBuilder<> &B, bool After,
                                           bool SchedBarrier) {
  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Type *I32 = Type::getInt32Ty(B.getContext());
  Function *SchedBarrierFn = SchedBarrier
                                 ? Intrinsic::getOrInsertDeclaration(
                                       M, Intrinsic::amdgcn_sched_barrier)
                                 : nullptr;
  if (!After && SchedBarrierFn)
    B.CreateCall(SchedBarrierFn, {ConstantInt::get(I32, 0)});
  emitMemBarrier(B, Config.MemBarrier);
  if (After && SchedBarrierFn)
    B.CreateCall(SchedBarrierFn, {ConstantInt::get(I32, 0)});
}

void SQTTInstrumentPass::emitTraceBoundaries(IRBuilder<> &B, Instruction *First,
                                             Instruction *Last,
                                             bool SchedBarrier) {
  Instruction *Next = Last->getNextNode();
  IRBuilder<> Before(First);
  emitTraceBoundary(Before, /*after=*/false,
                    SchedBarrier && !isHardSchedBoundary(First->getPrevNode()));
  // Newly-built scoped traces end after `last`; existing traces have a tail.
  if (Next)
    B.SetInsertPoint(Next);
  emitTraceBoundary(B, /*after=*/true,
                    SchedBarrier && !isHardSchedBoundary(Next));
}

void SQTTInstrumentPass::emitScopedTrace(
    IRBuilder<> &B, Function &F, GfxGen Gen, const char *TraceBlockName,
    const char *SkipBlockName, function_ref<void(IRBuilder<> &)> Emit) {
  if (!Config.needsScopeCheck()) {
    Emit(B);
    return;
  }

  Value *Ok = getOrCreateScopeCheck(F, Gen);
  Ok = B.CreateICmpNE(Ok, ConstantInt::get(Ok->getType(), 0));
  Instruction *SplitPt = &*B.GetInsertPoint();
  BasicBlock *OrigBB = SplitPt->getParent();
  BasicBlock *TailBB = OrigBB->splitBasicBlock(SplitPt, SkipBlockName);
  BasicBlock *TraceBB =
      BasicBlock::Create(F.getContext(), TraceBlockName, &F, TailBB);
  branchToScopedTrace(OrigBB, TraceBB, TailBB, Ok);

  IRBuilder<> Trace(TraceBB);
  Emit(Trace);
  Trace.CreateBr(TailBB);

  B.SetInsertPoint(&*TailBB->begin());
}

void SQTTInstrumentPass::insertTraceMarker(IRBuilder<> &B, uint32_t MarkerId,
                                           Function &F, GfxGen Gen,
                                           Value *Payload) {
  Module *M = F.getParent();
  CallInst *First = emitBareTrace(B, MarkerId, M, Gen);
  CallInst *Last = Payload ? emitRawTracePayload(B, Payload, M, First) : First;
  if (Config.needsScopeCheck()) {
    MDNode *ScopeFilter = MDNode::get(B.getContext(), {});
    First->setMetadata(SqttScopeFilterMetadata, ScopeFilter);
    Last->setMetadata(SqttScopeFilterMetadata, ScopeFilter);
  }
  emitTraceBoundaries(B, First, Last, /*schedBarrier=*/true);
}

Value *SQTTInstrumentPass::buildScopeCheck(IRBuilder<> &B, GfxGen Gen) {
  Module *M = B.GetInsertBlock()->getParent()->getParent();
  LLVMContext &Ctx = M->getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  HwRegEncodings Hw = getHwRegEncodings(Gen);
  Function *SGetReg =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_getreg);

  Value *Ok = ConstantInt::get(I32, 1);

  auto AddCheck = [&](uint32_t Mask, uint32_t FullMask, uint32_t HwReg) {
    if ((Mask & FullMask) != FullMask) {
      Value *ID = B.CreateCall(SGetReg, {ConstantInt::get(I32, HwReg)});
      Value *Bit = B.CreateAnd(B.CreateLShr(ConstantInt::get(I32, Mask), ID),
                               ConstantInt::get(I32, 1));
      Ok = B.CreateAnd(Ok, Bit);
    }
  };
  AddCheck(Config.WaveMask, FullWaveMask, Hw.Wave);
  AddCheck(Config.SimdMask, FullSimdMask, Hw.Simd);
  AddCheck(Config.CuMask, FullCuMask, Hw.Cu);
  AddCheck(Config.WgMask, FullWgMask, Hw.Wg);

  return Ok;
}

Value *SQTTInstrumentPass::getOrCreateScopeCheck(Function &F, GfxGen Gen) {
  if (CurScopeCheck)
    return CurScopeCheck;
  IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
  CurScopeCheck = buildScopeCheck(B, Gen);
  return CurScopeCheck;
}

bool SQTTInstrumentPass::finalizeExistingMarkers(Function &F) {
  struct TraceRange {
    CallInst *First, *Last;
  };
  SmallVector<SmallVector<TraceRange, 4>, 8> ByBlock;
  for (BasicBlock &BB : F) {
    SmallVector<TraceRange, 4> InBB;
    for (Instruction &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!isTraceDataCall(CI))
        continue;
      MDNode *Group = CI->getMetadata(SqttPayloadGroupMetadata);
      if (CI->getMetadata(SqttRawPayloadMetadata) && Group && !InBB.empty() &&
          InBB.back().First == InBB.back().Last &&
          InBB.back().First->getMetadata(SqttMarkerHeaderMetadata) &&
          InBB.back().First->getMetadata(SqttPayloadGroupMetadata) == Group &&
          isPayloadSequence(InBB.back().First, CI, Group))
        InBB.back().Last = CI;
      else
        InBB.push_back({CI, CI});
    }
    if (!InBB.empty())
      ByBlock.push_back(std::move(InBB));
  }
  if (ByBlock.empty())
    return false;

  auto AddBoundaries = [&](const TraceRange &Range) {
    IRBuilder<> B(Range.First);
    emitTraceBoundaries(B, Range.First, Range.Last, /*schedBarrier=*/true);
  };

  for (SmallVector<TraceRange, 4> &Ranges : ByBlock) {
    for (const TraceRange &Range : Ranges) {
      if (Config.needsScopeCheck()) {
        MDNode *ScopeFilter = MDNode::get(F.getContext(), {});
        Range.First->setMetadata(SqttScopeFilterMetadata, ScopeFilter);
        Range.Last->setMetadata(SqttScopeFilterMetadata, ScopeFilter);
      }
      AddBoundaries(Range);
    }
  }
  return true;
}

CallInst *SQTTInstrumentPass::emitBareTrace(IRBuilder<> &B, uint32_t Encoded,
                                            Module *M, GfxGen Gen) {
  LLVMContext &Ctx = M->getContext();
  bool UseImm = canUseImm(Encoded) && supportsImmTrace(Gen);
  Function *TTD = Intrinsic::getOrInsertDeclaration(
      M, UseImm ? Intrinsic::amdgcn_s_ttracedata_imm
                : Intrinsic::amdgcn_s_ttracedata);
  CallInst *CI = B.CreateCall(
      TTD,
      {ConstantInt::get(UseImm ? Type::getInt16Ty(Ctx) : Type::getInt32Ty(Ctx),
                        Encoded)});
  CI->setMetadata(SqttMarkerHeaderMetadata, MDNode::get(Ctx, {}));
  return CI;
}

CallInst *SQTTInstrumentPass::emitBareTraceValue(IRBuilder<> &B, Value *Val,
                                                 Module *M) {
  Function *TTD =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata);
  CallInst *Trace = B.CreateCall(TTD, {Val});
  Trace->setMetadata(SqttRawPayloadMetadata, MDNode::get(B.getContext(), {}));
  return Trace;
}

bool SQTTInstrumentPass::isTraceDataCall(const CallInst *CI) {
  if (!CI)
    return false;
  const Function *Callee = CI->getCalledFunction();
  if (Callee) {
    Intrinsic::ID ID = Callee->getIntrinsicID();
    return ID == Intrinsic::amdgcn_s_ttracedata ||
           ID == Intrinsic::amdgcn_s_ttracedata_imm;
  }
  const auto *Asm = dyn_cast<InlineAsm>(CI->getCalledOperand());
  return Asm && CI->arg_size() == 2 && Asm->getAsmString() == M0TraceAsmText;
}

CallInst *SQTTInstrumentPass::emitRawTracePayload(IRBuilder<> &B, Value *Val,
                                                  Module *M, CallInst *Header) {
  // Full s_ttracedata lowers through M0, so its input must be scalar. The
  // intrinsic lowering normally inserts this readfirstlane for a divergent
  // named data value; retain that behavior before the explicit asm lowering.
  Type *I32 = Type::getInt32Ty(M->getContext());
  MDNode *Group = MDNode::getDistinct(B.getContext(), {});
  Header->setMetadata(SqttPayloadGroupMetadata, Group);
  if (Val->getType() != I32) {
    Val = B.CreateZExtOrTrunc(Val, I32);
    if (auto *I = dyn_cast<Instruction>(Val))
      I->setMetadata(SqttPayloadGroupMetadata, Group);
  }
  if (!isa<ConstantInt>(Val)) {
    Function *ReadFirstLane = Intrinsic::getOrInsertDeclaration(
        M, Intrinsic::amdgcn_readfirstlane, {I32});
    CallInst *Prep = B.CreateCall(ReadFirstLane, {Val});
    Prep->setMetadata(SqttPayloadGroupMetadata, Group);
    Val = Prep;
  }
  CallInst *Trace = emitBareTraceValue(B, Val, M);
  Trace->setMetadata(SqttPayloadGroupMetadata, Group);
  return Trace;
}

bool SQTTInstrumentPass::finalizeFullTraces(Function &F, GfxGen Gen) {
  unsigned ClockBits = Config.ShaderClockBits;
  const bool PackClock = Gen == GfxGen::GFX12 && ClockBits != 0;
  if (PackClock && ClockBits > 29)
    report_fatal_error(
        "sqtt shader clock layout must leave at least one marker ID bit");
  if (PackClock && (Config.ShaderClockShift >= 32 ||
                    Config.ShaderClockShift + ClockBits > 32))
    report_fatal_error(
        "sqtt shader clock window must fit in shader_cycles_lo bits [31:0]");

  const unsigned IdBits = PackClock ? 30 - ClockBits : 0;
  const uint32_t MaxId = PackClock ? (uint32_t(1) << IdBits) - 1u : 0;
  const uint32_t MarkerAndFlagMask =
      PackClock ? (uint32_t(1) << (IdBits + 2)) - 1u : 0;

  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  const bool UseShaderCyclesU64 = PackClock && hasShaderCyclesU64(F);
  Function *SGetReg = nullptr;
  InlineAsm *GetShaderCycles = nullptr;
  uint32_t Hwreg =
      PackClock ? getRegisterImmediate(ClockBits - 1, Config.ShaderClockShift,
                                       Gfx12ShaderCyclesLo)
                : 0;
  uint32_t ClockDestShift = PackClock ? 32 - ClockBits : 0;

  // Model M0 as a fixed output instead of a clobber. M0 is reserved on
  // AMDGPU and LLVM diagnoses `~{m0}` as undefined behavior.
  Value *M0Nop = ConstantInt::get(I32, getM0TraceNop(Gen));
  FunctionType *TraceTy = FunctionType::get(I32, {I32, I32}, false);
  InlineAsm *ImmediateTrace =
      InlineAsm::get(TraceTy, M0TraceAsmText, "={m0},i,i",
                     /*hasSideEffects=*/true);
  InlineAsm *ScalarTrace = InlineAsm::get(TraceTy, M0TraceAsmText, "={m0},s,i",
                                          /*hasSideEffects=*/true);

  static constexpr const char FilteredImmTraceAsmText[] =
      "s_cmp_lg_u32 $0, 0\n"
      "s_cbranch_scc0 .Lsqtt_skip_${:uid}\n"
      "s_ttracedata_imm $1\n"
      ".Lsqtt_skip_${:uid}:";
  InlineAsm *FilteredImmTrace =
      InlineAsm::get(FunctionType::get(Type::getVoidTy(Ctx),
                                       {I32, Type::getInt16Ty(Ctx)}, false),
                     FilteredImmTraceAsmText, "s,i", /*hasSideEffects=*/true);

  static constexpr const char FilteredTraceAsmText[] =
      "s_cmp_lg_u32 $1, 0\n"
      "s_cbranch_scc0 .Lsqtt_skip_${:uid}\n"
      "s_mov_b32 m0, $2\n"
      "s_nop $3\n"
      "s_ttracedata\n"
      ".Lsqtt_skip_${:uid}:";
  FunctionType *FilteredTraceTy =
      FunctionType::get(I32, {I32, I32, I32}, false);
  InlineAsm *FilteredImmediateTrace =
      InlineAsm::get(FilteredTraceTy, FilteredTraceAsmText, "={m0},s,i,i",
                     /*hasSideEffects=*/true);
  InlineAsm *FilteredScalarTrace =
      InlineAsm::get(FilteredTraceTy, FilteredTraceAsmText, "={m0},s,s,i",
                     /*hasSideEffects=*/true);

  bool Changed = false;
  for (BasicBlock &BB : F) {
    for (auto It = BB.begin(), End = BB.end(); It != End;) {
      auto *CI = dyn_cast<CallInst>(&*It++);
      if (!isTraceDataCall(CI))
        continue;

      bool Filtered = CI->getMetadata(SqttScopeFilterMetadata);
      bool PackThisTrace = false;
      if (PackClock && !CI->getMetadata(SqttRawPayloadMetadata)) {
        Value *Encoded = CI->getArgOperand(0);
        if (auto *Arg = dyn_cast<ConstantInt>(Encoded)) {
          // A bare exit has no ID and cannot be distinguished from
          // the numeric API in trace data, so keep all of them
          // packed.
          if (CI->getMetadata(SqttMarkerHeaderMetadata) ||
              Arg->getZExtValue() == FlagExitPrev) {
            uint32_t MarkerId = Arg->getZExtValue() >> 2;
            if (MarkerId > MaxId)
              report_fatal_error(Twine("sqtt marker ID ") + Twine(MarkerId) +
                                 " does not fit with SQTT_SHADER_CLOCK_BITS=" +
                                 Twine(ClockBits));
            PackThisTrace = true;
          }
        } else
          PackThisTrace = CI->getMetadata(SqttMarkerHeaderMetadata);
      }

      const Function *Callee = CI->getCalledFunction();
      Intrinsic::ID TraceID =
          Callee ? Callee->getIntrinsicID() : Intrinsic::not_intrinsic;
      bool AlreadyLowered = !Callee;
      if (!Filtered && !PackThisTrace &&
          (AlreadyLowered || TraceID != Intrinsic::amdgcn_s_ttracedata))
        continue;

      IRBuilder<> B(CI);
      Value *TraceValue = CI->getArgOperand(0);
      if (PackThisTrace) {
        if (TraceValue->getType() != I32)
          TraceValue = B.CreateZExtOrTrunc(TraceValue, I32);
        Value *MarkerAndFlags =
            B.CreateAnd(TraceValue, ConstantInt::get(I32, MarkerAndFlagMask));
        Value *Clock;
        if (UseShaderCyclesU64) {
          if (!GetShaderCycles)
            GetShaderCycles =
                InlineAsm::get(FunctionType::get(Type::getInt64Ty(Ctx), false),
                               "s_get_shader_cycles_u64 $0", "=s",
                               /*hasSideEffects=*/true);
          Value *CyclesLo = B.CreateTrunc(B.CreateCall(GetShaderCycles), I32);
          // The final shift below discards all but clockBits,
          // matching the right-aligned field returned by s_getreg
          // on gfx1200 and gfx1201.
          Clock = B.CreateLShr(CyclesLo,
                               ConstantInt::get(I32, Config.ShaderClockShift));
        } else {
          if (!SGetReg)
            SGetReg = Intrinsic::getOrInsertDeclaration(
                M, Intrinsic::amdgcn_s_getreg);
          Clock = B.CreateCall(SGetReg, {ConstantInt::get(I32, Hwreg)});
        }
        TraceValue = B.CreateOr(
            B.CreateShl(Clock, ConstantInt::get(I32, ClockDestShift)),
            MarkerAndFlags);
        ShaderClockBitsUsed = ClockBits;
      }

      CallInst *Replacement;
      if (!Filtered) {
        InlineAsm *TraceAsm =
            isa<ConstantInt>(TraceValue) ? ImmediateTrace : ScalarTrace;
        Replacement = B.CreateCall(TraceAsm, {TraceValue, M0Nop});
      } else {
        Value *Scope = getOrCreateScopeCheck(F, Gen);
        if (!PackThisTrace && TraceID == Intrinsic::amdgcn_s_ttracedata_imm)
          Replacement = B.CreateCall(FilteredImmTrace, {Scope, TraceValue});
        else {
          InlineAsm *TraceAsm = isa<ConstantInt>(TraceValue)
                                    ? FilteredImmediateTrace
                                    : FilteredScalarTrace;
          Replacement = B.CreateCall(TraceAsm, {Scope, TraceValue, M0Nop});
        }
      }
      Replacement->setDebugLoc(CI->getDebugLoc());
      Replacement->copyMetadata(*CI);
      CI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}
