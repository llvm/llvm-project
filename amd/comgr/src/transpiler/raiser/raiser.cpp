//===- raiser.cpp - Transpiler MC -> LLVM IR raiser ----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decodes each requested kernel's .text into a typed `DecodedInst` stream,
// dispatches each decoded instruction to its per-format handler, promotes the
// register-file allocas to SSA, and verifies the module of `amdgpu_kernel`
// functions this produces. The MC layer the decode runs on is built once and
// shared, since the kernels come from one code object.
//
// A raise reads two ISAs. The source one is what the code object was compiled
// for, and the decode is written in its terms. The target one is what the
// raised IR will be lowered for, and the wave projection reads it to translate
// a source lane into the target lane that runs it.
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/raiser.h"

#include "transpiler/decoder/amdgpu-formats.h"
#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/decoder/decode.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/decoder/opcode-map.h"
#include "transpiler/decoder/setpc-analysis.h"
#include "transpiler/raiser/handle-vop-cross-lane.h"
#include "transpiler/raiser/handlers.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/raise_failure.h"
#include "transpiler/raiser/wave-projection.h"

#include "comgr.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

using namespace llvm;

namespace COMGR::transpiler {

// Address space the kernarg segment lives in.
constexpr unsigned ConstantAddressSpace = 4;

// Identifier the raised module carries. A code object names no module of its
// own, and one raise holds every kernel of it.
constexpr StringLiteral kRaisedModuleName = "transpiler.raised";

// Minimum kernarg segment alignment the AMDGPU ABI mandates.
constexpr Align KernargSegmentAlign = Align::Constant<16>();

/// Return the LLVM denormal mode represented by an AMDHSA descriptor field.
static DenormalMode denormalMode(unsigned HardwareMode) {
  using Kind = DenormalMode::DenormalModeKind;
  switch (HardwareMode) {
  case amdhsa::FLOAT_DENORM_MODE_FLUSH_SRC_DST:
    return {Kind::PreserveSign, Kind::PreserveSign};
  case amdhsa::FLOAT_DENORM_MODE_FLUSH_DST:
    return {Kind::PreserveSign, Kind::IEEE};
  case amdhsa::FLOAT_DENORM_MODE_FLUSH_SRC:
    return {Kind::IEEE, Kind::PreserveSign};
  case amdhsa::FLOAT_DENORM_MODE_FLUSH_NONE:
    return {Kind::IEEE, Kind::IEEE};
  }
  llvm_unreachable("invalid hardware denormal mode");
}

/// Attach the floating-point attributes represented by the source descriptor.
static void setFloatingPointAttributes(Function &F, const KernelMeta &Meta,
                                       const MCSubtargetInfo &SourceSTI) {
  const unsigned DefaultDenormalMode = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64);
  const unsigned Float32DenormalMode = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32);
  const DenormalFPEnv FPEnv(denormalMode(DefaultDenormalMode),
                            denormalMode(Float32DenormalMode));
  F.addFnAttr(Attribute::get(F.getContext(), Attribute::DenormalFPEnv,
                             FPEnv.toIntValue()));

  if (!SourceSTI.hasFeature(AMDGPU::FeatureDX10ClampAndIEEEMode)) {
    return;
  }

  const bool Dx10Clamp =
      AMDHSA_BITS_GET(Meta.ComputePgmRsrc1,
                      amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_DX10_CLAMP);
  const bool IeeeMode =
      AMDHSA_BITS_GET(Meta.ComputePgmRsrc1,
                      amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_IEEE_MODE);
  F.addFnAttr("amdgpu-dx10-clamp", Dx10Clamp ? "true" : "false");
  F.addFnAttr("amdgpu-ieee", IeeeMode ? "true" : "false");
}

// Declare the lifted kernel: one opaque parameter spanning the source kernarg
// segment, so the emitted descriptor reports the source segment size and the
// ABI alignment. The raised body reads arguments as ordinary loads off the
// kernarg pointer, at the byte offsets the source metadata gives them.
static Function *declareKernel(Module &M, StringRef KernelName,
                               const KernelMeta &Meta,
                               const MCSubtargetInfo &SourceSTI) {
  LLVMContext &C = M.getContext();
  SmallVector<Type *> ParamTys;
  if (Meta.KernargSegmentSize > 0)
    ParamTys.push_back(PointerType::get(C, ConstantAddressSpace));

  FunctionType *FuncTy =
      FunctionType::get(Type::getVoidTy(C), ParamTys, /*isVarArg=*/false);
  Function *F =
      Function::Create(FuncTy, GlobalValue::ExternalLinkage, KernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);
  setFloatingPointAttributes(*F, Meta, SourceSTI);

  if (Meta.KernargSegmentSize > 0) {
    // AMDGPULowerKernelArguments honors the `align` parameter attribute only on
    // a byref kernel argument; without `byref` the segment would take the array
    // type's natural one-byte alignment.
    Type *SegmentTy =
        ArrayType::get(Type::getInt8Ty(C), Meta.KernargSegmentSize);
    F->addParamAttr(0, Attribute::getWithByRefType(C, SegmentTy));
    F->addParamAttr(0, Attribute::getWithAlignment(C, KernargSegmentAlign));
    F->getArg(0)->setName("kernarg_segment");
  }

  // The host fills the kernarg buffer from the source metadata and leaves no
  // room past the source segment, so the target ABI's hidden-argument block
  // must not be appended to it.
  F->addFnAttr("amdgpu-no-implicitarg-ptr");

  // Both attributes below take a "min,max" range, and both source sizes are
  // exact, so each is written as a range of one.
  //
  // Pin the block to the size the source kernel declared, so the backend lays
  // out workitem ids the way the source binary did.
  F->addFnAttr("amdgpu-flat-work-group-size",
               formatv("{0},{0}", Meta.MaxFlatWorkgroupSize).str());
  if (Meta.GroupSegmentFixedSize > 0) {
    // The raiser addresses LDS by absolute offset rather than through a
    // GlobalVariable, so without this the backend would emit
    // group_segment_fixed_size = 0 and treat every LDS access as out of
    // segment.
    F->addFnAttr("amdgpu-lds-size",
                 formatv("{0},{0}", Meta.GroupSegmentFixedSize).str());
  }
  return F;
}

// Lower one decoded instruction into `Ctx`'s current insertion point, routing
// it by instruction format. A format with no handler is refused rather than
// lowered as something else.
static Error raiseInst(RaiseContext &Ctx, const DecodedInst &Di) {
  using namespace AmdgpuFormat;
  OperandResolver Op{Ctx, Di};

  if (Di.VOPD)
    return handleVOPD(Ctx, Di);

  if (SIInstrFlags::isMAI(*Ctx.MC.InstrInfo, Di.Inst))
    return handleMFMA(Ctx, Di, Op);

  if (Di.TargetSpecificFlags & SOP1)
    return handleSOP1(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SOP2)
    return handleSOP2(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SOPC)
    return handleSOPC(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SOPK)
    return handleSOPK(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SOPP)
    return handleSOPP(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SMRD)
    return handleSMEM(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & FLAT)
    return handleVGLOBAL(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & MUBUF)
    return handleMUBUF(Ctx, Di);
  if (Di.TargetSpecificFlags & DS)
    return handleDS(Ctx, Di);

  constexpr uint64_t VOP1EncodingMask = VOP1 | VOP3 | DPP | SDWA | VOPD3;
  if ((Di.TargetSpecificFlags & VOP1EncodingMask) == VOP1)
    return handleVOP1(Ctx, Di, Op);
  if ((Di.TargetSpecificFlags & VOP1EncodingMask) == DPP &&
      Di.CanonOp == CanonicalOp::V_MOV_B32)
    return raiseDPPMove32(Ctx, Di, Op);

  constexpr uint64_t VOP2EncodingMask =
      VOP2 | VOP3 | VOP3P | DPP | SDWA | VOPD3;
  if ((Di.TargetSpecificFlags & VOP2EncodingMask) == VOP2) {
    return handleVOP2(Ctx, Di, Op);
  }

  constexpr uint64_t VOP3EncodingMask =
      VOP3 | VOP3P | VOPC | DPP | SDWA | VOPD3;
  if ((Di.TargetSpecificFlags & VOP3EncodingMask) == VOP3)
    return handleVOP3(Ctx, Di, Op);

  constexpr uint64_t VOP3PEncodingMask = VOP3P | DPP | VOPD3;
  if ((Di.TargetSpecificFlags & VOP3PEncodingMask) == VOP3P)
    return handleVOP3P(Ctx, Di, Op);

  constexpr uint64_t VOPCEncodingMask =
      VOPC | VOP3 | VOP3P | DPP | SDWA | VOPD3;
  if ((Di.TargetSpecificFlags & VOPCEncodingMask) == VOPC) {
    return handleVOPC(Ctx, Di, Op);
  }

  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags));
}

// The MC layer for one ISA. `Role` names which end of the raise this is, and
// only reaches diagnostics.
namespace {
struct IsaContext {
  MCState MC;
  // Bare AMDGPU processor the MC layer was built for.
  std::string Cpu;
  // Explicit code-object SRAM ECC setting; absent permits either setting.
  std::optional<bool> SramEcc;

  static Expected<IsaContext> create(StringRef Isa, StringRef Role);
};
} // namespace

Expected<IsaContext> IsaContext::create(StringRef Isa, StringRef Role) {
  // Reject a bad ISA before reaching the MC stack: createMCSubtargetInfo
  // accepts an unknown name and returns a featureless subtarget, and the
  // failure only surfaces inside createMCDisassembler, which aborts the
  // process instead of returning.
  TargetIdentifier Identifier;
  StringRef Cpu = Isa;
  std::optional<bool> SramEcc;
  if (parseTargetIdentifier(Isa, Identifier) == AMD_COMGR_STATUS_SUCCESS) {
    Cpu = Identifier.Processor;
    for (StringRef Feature : Identifier.Features) {
      if (Feature == "sramecc+")
        SramEcc = true;
      else if (Feature == "sramecc-")
        SramEcc = false;
    }
  }
  if (AMDGPU::parseArchAMDGCN(Cpu) == AMDGPU::GK_NONE)
    return RaiseFailure::general(RaiseFailureReason::BadInput,
                                 Role + " ISA '" + Isa +
                                     "' does not name an AMDGPU GPU");

  // The target side reads only the subtarget and registered target behind the
  // machine, and pays for a disassembler and a printer it never uses. That is
  // one extra MC stack per raise, against a second way of standing a subtarget
  // up that has to be kept in step with this one.
  Expected<MCState> MC = initMCState(Cpu);
  if (!MC)
    return MC.takeError();

  return IsaContext{std::move(*MC), Cpu.str(), SramEcc};
}

// What every kernel of one raise runs against: the ISA the code object was
// compiled for, the ISA it is being raised onto, and the opcode map built over
// the source MC layer. Built once per raise and outlives each kernel's context.
namespace {
struct RaiseEnvironment {
  IsaContext Source;
  IsaContext Target;
  OpcodeMap OpcMap;

  static Expected<RaiseEnvironment> create(StringRef SourceIsa,
                                           StringRef TargetIsa);
};
} // namespace

Expected<RaiseEnvironment> RaiseEnvironment::create(StringRef SourceIsa,
                                                    StringRef TargetIsa) {
  Expected<IsaContext> Source = IsaContext::create(SourceIsa, "source");
  if (!Source)
    return Source.takeError();

  Expected<IsaContext> Target = IsaContext::create(TargetIsa, "target");
  if (!Target)
    return Target.takeError();

  RaiseEnvironment Env{std::move(*Source), std::move(*Target), OpcodeMap()};
  Env.OpcMap.build(*Env.Source.MC.InstrInfo);
  return Env;
}

// Whether `Offset` falls strictly inside one of `Insts`, which must be in
// source order. An offset that leads an instruction is not inside one.
static bool isInsideDecodedInstruction(ArrayRef<DecodedInst> Insts,
                                       uint64_t Offset) {
  const DecodedInst *After =
      upper_bound(Insts, Offset, [](uint64_t Off, const DecodedInst &Di) {
        return Off < Di.Offset;
      });
  if (After == Insts.begin())
    return false;
  const DecodedInst &Di = *std::prev(After);
  return Offset > Di.Offset && Offset < Di.Offset + Di.sizeInBytes();
}

// The function symbol extent of `Extents` that covers `Offset`, or null when
// none of them does.
static const KernelSymbolExtent *
findFunctionExtent(ArrayRef<KernelSymbolExtent> Extents, uint64_t Offset) {
  const KernelSymbolExtent *Found =
      find_if(Extents, [Offset](const KernelSymbolExtent &E) {
        return Offset >= E.Offset && Offset < E.Offset + E.Size;
      });
  return Found == Extents.end() ? nullptr : Found;
}

// Fold a second decode into `Base`, keeping the instructions in source order.
// The two are decoded from disjoint extents, so neither carries an instruction
// the other already has.
static void mergeDecoded(DecodeResult &Base, DecodeResult &&Extra) {
  llvm::move(Extra.Insts, std::back_inserter(Base.Insts));
  sort(Base.Insts, [](const DecodedInst &A, const DecodedInst &B) {
    return A.Offset < B.Offset;
  });
  set_union(Base.BlockStarts, Extra.BlockStarts);
}

// The source offsets `SetPc` refused a transfer for because no decoded
// instruction starts there, ascending and distinct.
static SmallVector<uint64_t> unstartedTargets(const SetPcAnalysis &SetPc) {
  SmallVector<uint64_t> Targets;
  for (const SetPcSite &Site : make_second_range(SetPc.Sites)) {
    const SetPcUnresolvable *Refused = std::get_if<SetPcUnresolvable>(&Site);
    if (Refused && Refused->Why == SetPcRefusal::TargetNotAnInstruction)
      Targets.push_back(Refused->Subject);
  }
  // `Sites` is a DenseMap, so sorting is what keeps a raise from depending on
  // the order its buckets happen to be walked in.
  sort(Targets);
  Targets.erase(llvm::unique(Targets), Targets.end());
  return Targets;
}

namespace {
/// Track source EXEC writes for validation after register promotion.
struct ExecContainmentRequirements {
  WeakTrackingVH InitialExec;
  SmallVector<std::pair<WeakTrackingVH, const DecodedInst *>> ExecWrites;

  Error validate(Function &F, const MCState &MC,
                 const WaveProjection &Projection, Value *LaneId) const;
};
} // namespace

// For masks, track subsets of entry EXEC; for i1 predicates, track implication
// of the corresponding lane's entry bit. Integer casts cannot mix these bounds.
static bool isKnownSubsetOfEntryExec(
    const Instruction &I, const SmallPtrSetImpl<const Value *> &EntrySubsets,
    const DominatorTree &DT, const WaveProjection &Projection, Value *LaneId) {
  auto IsEntrySubset = [&](const Value *V) {
    // Only zero is a subset of every possible entry mask, including partial
    // waves. Nonzero constants need an intersection with a proven subset.
    if (const auto *C = dyn_cast<ConstantInt>(V))
      return C->isZero();
    return EntrySubsets.contains(V);
  };
  using namespace PatternMatch;
  switch (I.getOpcode()) {
  case Instruction::ICmp: {
    Value *Mask = nullptr;
    if (match(&I, m_SpecificICmp(
                      ICmpInst::ICMP_NE,
                      m_And(m_LShr(m_Value(Mask),
                                   m_And(m_Specific(LaneId),
                                         m_SpecificInt(
                                             Projection.sourceWaveSize() - 1))),
                            m_One()),
                      m_Zero())))
      return IsEntrySubset(Mask);
    return false;
  }
  case Instruction::And:
    return IsEntrySubset(I.getOperand(0)) || IsEntrySubset(I.getOperand(1));
  case Instruction::Or:
  case Instruction::Xor:
    return IsEntrySubset(I.getOperand(0)) && IsEntrySubset(I.getOperand(1));
  case Instruction::Select:
    if (I.getType()->isIntegerTy(1) && IsEntrySubset(I.getOperand(0)) &&
        match(I.getOperand(2), m_Zero()))
      return true;
    return IsEntrySubset(I.getOperand(1)) && IsEntrySubset(I.getOperand(2));
  case Instruction::PHI: {
    const PHINode &Phi = cast<PHINode>(I);
    return all_of(Phi.incoming_values(), [&](const Use &Incoming) {
      return !DT.isReachableFromEntry(Phi.getIncomingBlock(Incoming)) ||
             IsEntrySubset(Incoming);
    });
  }
  case Instruction::Trunc:
    if (const Value *Predicate = Projection.matchBallotPredicate(&I))
      return IsEntrySubset(Predicate);
    if (I.getType()->isIntegerTy(1))
      return false;
    [[fallthrough]];
  case Instruction::ZExt:
    return !I.getOperand(0)->getType()->isIntegerTy(1) &&
           IsEntrySubset(I.getOperand(0));
  default:
    return false;
  }
}

Error ExecContainmentRequirements::validate(Function &F, const MCState &MC,
                                            const WaveProjection &Projection,
                                            Value *LaneId) const {
  auto Refuse = [&](const DecodedInst &Di, const Twine &Detail) {
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnprovenExecContainment,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags), Detail);
  };

  assert(InitialExec && "entry EXEC was deleted before validation");
  const Value *EntryExec = InitialExec;
  SmallPtrSet<const Value *, 32> EntrySubsets;
  EntrySubsets.insert(EntryExec);
  DominatorTree DT(F);
  // A reachable PHI cycle has an incoming value on its first iteration.
  // Remove unsupported producers to prove the bound inductively, including
  // loop-carried EXEC saves. Unreachable cycles cannot establish a bound.
  for (const Instruction &I : instructions(F))
    if (I.getType()->isIntegerTy() && DT.isReachableFromEntry(I.getParent()))
      EntrySubsets.insert(&I);
  bool Changed;
  do {
    Changed = false;
    for (const Instruction &I : instructions(F))
      if (&I != EntryExec && EntrySubsets.contains(&I) &&
          !isKnownSubsetOfEntryExec(I, EntrySubsets, DT, Projection, LaneId))
        Changed |= EntrySubsets.erase(&I);
  } while (Changed);
  for (const auto &[Mask, Di] : ExecWrites) {
    assert(Mask && "EXEC write was deleted before validation");
    const Value *V = Mask;
    const auto *C = dyn_cast<ConstantInt>(V);
    if (!EntrySubsets.contains(V) && (!C || !C->isZero()))
      return Refuse(*Di, "projection cannot prove that EXEC only enables lanes "
                         "active at kernel entry");
  }

  return Error::success();
}

namespace {
enum class ProjectionKind {
  SameWave,
  WaveNative,
  WaveNativeDivergent,
  Replicated
};
} // namespace

// Whether `Di` names the plain workgroup barrier in `OperandName`, the field
// its format carries the barrier id in, rather than one of the barriers the
// raise keeps nothing for: the cluster and trap barriers, and the named
// barrier objects a subset of the workgroup joins.
static bool namesWorkgroupBarrier(const DecodedInst &Di,
                                  AMDGPU::OpName OperandName) {
  int16_t Index =
      COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(), OperandName);
  assert(Index >= 0 && "a split barrier encodes the barrier it names");
  std::optional<int64_t> Id = evalOperandAsConst(Di.Inst, Index);
  assert(Id && "the immediate form of a split barrier names its id inline");
  // Both halves carry the id in a 16-bit field, which reaches here
  // zero-extended from the wait and sign-extended from the arrival.
  return SignExtend64<16>(*Id) == AMDGPU::Barrier::WORKGROUP;
}

// Source offsets of the split-barrier halves that together stand for one whole
// barrier, as the offsets of both halves of every match.
//
// The source splits the workgroup barrier into an arrival that does not block
// and a wait that does. Neither half alone says what the raise can state,
// which carries only a barrier that arrives and waits at once: standing it in
// for an arrival holds a wave the source let run on, and standing it in for a
// wait makes the wave arrive a second time. A wait the arrival reaches with no
// control transfer in between is a different matter, because then the two
// always run together and as a pair they say exactly what the whole barrier
// says. Raising the pair at its wait keeps the point the source blocks at.
//
// `Insts` must be in source order and `BlockStarts` must be the final
// block-start set, since a block start between the halves means a path reaches
// one of them without the other.
static DenseSet<uint64_t>
pairSplitBarriers(ArrayRef<DecodedInst> Insts,
                  const std::set<uint64_t> &BlockStarts, const MCState &MC) {
  DenseSet<uint64_t> Paired;
  std::optional<uint64_t> PendingArrival;
  for (const DecodedInst &Di : Insts) {
    if (BlockStarts.count(Di.Offset))
      PendingArrival.reset();

    if (Di.CanonOp == CanonicalOp::S_BARRIER_SIGNAL_IMM) {
      // An arrival already pending is one no wait reached, and dropping it
      // here leaves it unmatched for the handler to refuse.
      PendingArrival.reset();
      if (namesWorkgroupBarrier(Di, AMDGPU::OpName::src0))
        PendingArrival = Di.Offset;
      continue;
    }

    if (Di.CanonOp == CanonicalOp::S_BARRIER_WAIT && PendingArrival &&
        namesWorkgroupBarrier(Di, AMDGPU::OpName::simm16)) {
      Paired.insert(*PendingArrival);
      Paired.insert(Di.Offset);
      PendingArrival.reset();
      continue;
    }

    // A block start is not raised at every control transfer: an instruction
    // trailing one without leading a block of its own is reached by nothing.
    // Matching across such a transfer would pair halves that never run
    // together.
    const MCInstrDesc &Desc = MC.InstrInfo->get(Di.Inst.getOpcode());
    if (Desc.isTerminator() || Desc.isCall())
      PendingArrival.reset();
  }
  return Paired;
}

/// Raise a decoded kernel with one projection. A failed attempt removes its
/// function before returning, including all register and analysis state.
static Error raiseDecodedKernel(const RaiseEnvironment &Env, Module &M,
                                const TextSection &Text,
                                const KernelRequest &Kernel,
                                const DecodeResult &Decoded,
                                const SetPcAnalysis &SetPc, TargetMachine &TM,
                                ProjectionKind Kind,
                                const KernelLaunchRequirements &Launch) {
  const KernelMeta &Meta = Kernel.Meta;
  LLVMContext &C = M.getContext();
  const MCSubtargetInfo &SourceSTI = *Env.Source.MC.SubtargetInfo;
  const MCSubtargetInfo &TargetSTI = *Env.Target.MC.SubtargetInfo;
  bool UseWaveNative = Kind == ProjectionKind::WaveNative ||
                       Kind == ProjectionKind::WaveNativeDivergent;
  bool UseReplicated = Kind == ProjectionKind::Replicated;
  bool CheckContainment =
      UseWaveNative || (UseReplicated && !Launch.RequiresWholeSourceWaves);
  std::unique_ptr<WaveProjection> Projection;
  if (UseReplicated)
    Projection = std::make_unique<ReplicatedDispatchProjection>(
        SourceSTI, TargetSTI, Type::getInt32Ty(C), Type::getInt64Ty(C), Launch);
  else if (UseWaveNative)
    Projection = std::make_unique<WaveNativeProjection>(
        SourceSTI, TargetSTI, Type::getInt32Ty(C), Type::getInt64Ty(C),
        Kind == ProjectionKind::WaveNativeDivergent);
  else
    Projection = std::make_unique<ReplicationProjection>(
        SourceSTI, TargetSTI, Type::getInt32Ty(C), Type::getInt64Ty(C));
  Projection->setMaxFlatWorkgroupSize(Meta.MaxFlatWorkgroupSize);

  Function *F =
      declareKernel(M, Kernel.Name, Meta, *Env.Source.MC.SubtargetInfo);
  scope_exit EraseOnFailure([&] {
    F->eraseFromParent();
    // Intrinsics used only by a failed projection attempt must not leak into
    // the module produced by a successful retry.
    for (Function &Declaration : make_early_inc_range(M))
      if (Declaration.isDeclaration() && Declaration.use_empty())
        Declaration.eraseFromParent();
  });
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  IRBuilder<> B(Entry);

  // Matched here rather than at decode, because a block start the transfers
  // contributed can fall between two halves that the decode alone left
  // adjacent.
  DenseSet<uint64_t> PairedSplitBarriers =
      pairSplitBarriers(Decoded.Insts, Decoded.BlockStarts, Env.Source.MC);

  Expected<RaiseContext> Ctx = RaiseContext::create(
      B, *Projection, Env.Source.MC, SetPc, PairedSplitBarriers, Meta,
      Text.Bytes, Text.Address, Text.ImageSections, Kernel.StartOffset,
      Kernel.EndOffset, Env.Source.SramEcc);
  if (!Ctx)
    return Ctx.takeError();

  if (UseReplicated)
    F->addFnAttr(
        "amdgpu-flat-work-group-size",
        formatv("{0},{1}", Projection->targetWaveSize(),
                alignTo(Launch.MaxWorkgroupSize, Projection->sourceWaveSize()) *
                    Projection->replicationFactor())
            .str());

  ExecContainmentRequirements Requirements;
  if (UseWaveNative) {
    // The metadata bounds the launch; it does not require that exact size.
    F->addFnAttr("amdgpu-flat-work-group-size",
                 formatv("1,{0}", Meta.MaxFlatWorkgroupSize).str());
  }
  Requirements.InitialExec = Ctx->registers().readExec();

  if (Launch.RequiredWorkgroupSize) {
    SmallVector<Metadata *, 3> Dimensions;
    std::array<uint32_t, 3> Required = *Launch.RequiredWorkgroupSize;
    if (UseReplicated)
      Required = {alignTo(Required[0] * Required[1] * Required[2],
                          Projection->sourceWaveSize()) *
                      Projection->replicationFactor(),
                  1, 1};
    unsigned Workitems = 1;
    for (unsigned I = 0; I != 3; ++I) {
      unsigned Size = Required[I];
      Workitems *= Size;
      Dimensions.push_back(ConstantAsMetadata::get(B.getInt32(Size)));
    }
    F->addFnAttr("amdgpu-flat-work-group-size",
                 formatv("{0},{0}", Workitems).str());
    F->setMetadata("reqd_work_group_size", MDNode::get(C, Dimensions));
  }

  // A block per recovered block start, all of them made before any instruction
  // is raised so a branch reaching forward finds the block it targets. The
  // kernel entry gets one too, rather than raising into the entry block the
  // allocas live in: a branch back to the first instruction would otherwise
  // give the entry block a predecessor, which LLVM does not allow.
  for (uint64_t Start : Decoded.BlockStarts)
    Ctx->defineBB(Start, BasicBlock::Create(C, formatv("bb_{0:x}", Start), F));

  // A followed callee can sit anywhere in the text section, the kernel's own
  // entry included, so where the raise starts is named rather than left to be
  // whichever block the first raised instruction leads.
  if (Projection->allowsDivergentScalarControlFlow()) {
    BasicBlock *Exit = BasicBlock::Create(C, "absent_source_wave", F);
    ReturnInst::Create(C, Exit);
    Value *Present = B.CreateICmpNE(Requirements.InitialExec, B.getInt32(0),
                                    "source_wave_present");
    B.CreateCondBr(Present, Ctx->lookupBB(Kernel.StartOffset), Exit);
  } else {
    B.CreateBr(Ctx->lookupBB(Kernel.StartOffset));
  }

  for (const DecodedInst &Di : Decoded.Insts) {
    BasicBlock *Open = B.GetInsertBlock();
    if (Decoded.BlockStarts.count(Di.Offset)) {
      BasicBlock *Next = Ctx->lookupBB(Di.Offset);
      // A source block ending in something other than a control transfer
      // reaches the block that follows it, which LLVM states as a branch.
      if (!Open->hasTerminator())
        B.CreateBr(Next);
      B.SetInsertPoint(Next);
    } else if (Open->hasTerminator()) {
      // An instruction trailing a control transfer without leading a block
      // start of its own is reached by nothing, and needs a block anyway for
      // its handler to raise into.
      B.SetInsertPoint(
          BasicBlock::Create(C, formatv("unreached_{0:x}", Di.Offset), F));
    }

    Ctx->registers().computeVGPRAdjust(Di);
    if (Error Err = raiseInst(*Ctx, Di))
      return Err;
    if (UseWaveNative || UseReplicated) {
      if (Instruction *Term = B.GetInsertBlock()->getTerminatorOrNull()) {
        if (isa<CondBrInst, SwitchInst>(Term))
          Ctx->requireScalarControlFlow(Term, Di);
      } else if (CheckContainment && instructionWritesEXEC(Di, Env.Source.MC)) {
        Requirements.ExecWrites.emplace_back(Ctx->registers().readExec(), &Di);
      }
    }
  }

  // Execution reaching the end of the extent means the code is truncated or
  // the extent is misbounded. Closing the block with a return instead would
  // hand back a kernel that reads as having run to completion. Every earlier
  // block is terminated on the way out of it, so the open one is the only
  // block that can still be missing a terminator here.
  if (!B.GetInsertBlock()->hasTerminator())
    return RaiseFailure::general(
        RaiseFailureReason::UnterminatedKernelExtent,
        "kernel extent ends without an instruction that ends the program");

  // Every source code-object address the kernel records is known once every
  // instruction is raised, so this is where the reads deferred along the way
  // are decided.
  if (Error Err = Ctx->registers().refuseDeferredSourceImageReads())
    return Err;

  DominatorTree DT(*F);
  AssumptionCache AC(*F);
  SmallVector<AllocaInst *> Allocas;
  Ctx->registers().collectAllocas(Allocas);
  PromoteMemToReg(Allocas, DT, &AC);
  if (Error Err = Ctx->validateRequiredBits())
    return Err;
  if (UseWaveNative || UseReplicated) {
    if (Error Err = Ctx->validateWaveRequirements(TM, Requirements.InitialExec))
      return Err;
  }
  if (CheckContainment) {
    Value *LaneId = Projection->emitLaneIdx(B);
    if (Error Err =
            Requirements.validate(*F, Env.Source.MC, *Projection, LaneId))
      return Err;
  }
  if (Error Err = Ctx->registers().validateEntrySgprs())
    return Err;
  EraseOnFailure.release();
  return Error::success();
}

/// Check the source effects and entry values the replicated mapping supports.
static Error validateReplicatedKernel(const MCState &MC,
                                      const DecodeResult &Decoded,
                                      const KernelMeta &Meta) {
  for (const KernelArgMeta &Arg : Meta.Args) {
    if (!StringRef(Arg.ValueKind).starts_with("hidden_"))
      continue;
    bool IsSourceGeometry =
        StringSwitch<bool>(Arg.ValueKind)
            .Cases({"hidden_global_offset_x", "hidden_global_offset_y",
                    "hidden_global_offset_z", "hidden_block_count_x",
                    "hidden_block_count_y", "hidden_block_count_z",
                    "hidden_group_size_x", "hidden_group_size_y",
                    "hidden_group_size_z", "hidden_remainder_x",
                    "hidden_remainder_y", "hidden_remainder_z",
                    "hidden_grid_dims", "hidden_dynamic_lds_size",
                    "hidden_none"},
                   true)
            .Default(false);
    if (!IsSourceGeometry)
      return RaiseFailure::general(
          RaiseFailureReason::UnsupportedWaveProjection,
          "replicated dispatch cannot reproduce hidden argument kind '" +
              Arg.ValueKind + "'");
  }

  for (const DecodedInst &Di : Decoded.Insts) {
    auto Refuse = [&](const Twine &Detail) {
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedWaveProjection,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags), Detail);
    };
    for (const MCOperand &Operand : Di.Inst)
      if (Operand.isReg() && Operand.getReg() == AMDGPU::LDS_DIRECT)
        return Refuse("replicated dispatch does not support LDS direct reads");
  }
  return Error::success();
}

// Raise one kernel into `M`. Everything this allocates -- the projection, the
// builder, the register file behind the context -- describes that one kernel
// and dies with the call; only the emitted function outlives it.
static Expected<KernelLaunchRequirements>
raiseKernel(const RaiseEnvironment &Env, Module &M, const TextSection &Text,
            const KernelRequest &Kernel,
            ArrayRef<KernelSymbolExtent> FunctionExtents, TargetMachine &TM,
            LaunchPolicy Policy) {
  const KernelMeta &Meta = Kernel.Meta;
  std::optional<std::array<uint32_t, 3>> RequiredSize =
      Kernel.WorkgroupSize ? Kernel.WorkgroupSize : Meta.RequiredWorkgroupSize;
  if (Kernel.WorkgroupSize && Meta.RequiredWorkgroupSize &&
      Kernel.WorkgroupSize != Meta.RequiredWorkgroupSize)
    return RaiseFailure::general(
        RaiseFailureReason::UnsupportedLaunch,
        "specialization does not match required source workgroup dimensions");
  if (RequiredSize) {
    uint64_t Workitems = 1;
    for (uint32_t Dimension : *RequiredSize) {
      if (!Dimension || Dimension > Meta.MaxFlatWorkgroupSize ||
          Workitems > Meta.MaxFlatWorkgroupSize / Dimension)
        return RaiseFailure::general(
            RaiseFailureReason::UnsupportedLaunch,
            "invalid specialized workgroup dimensions");
      Workitems *= Dimension;
    }
  }
  Expected<DecodeResult> Decoded = decodeKernel(
      Env.Source.MC, Env.OpcMap, Text.Bytes, Kernel.StartOffset,
      Kernel.EndOffset == 0 ? std::nullopt : std::optional(Kernel.EndOffset));
  if (!Decoded)
    return Decoded.takeError();

  // Caught here rather than at the terminator check below, which the branch
  // into the starting block satisfies without anything having been raised.
  if (Decoded->Insts.empty())
    return RaiseFailure::general(RaiseFailureReason::UnterminatedKernelExtent,
                                 "kernel extent holds no instruction");

  // A jump through a register names no offset the decode could follow, so the
  // code it reaches may be code no decode has read: an outlined helper the
  // kernel calls, or a stretch the scan stopped short of at an `s_endpgm`.
  // Reading from an offset the analysis reports as unstarted can reveal
  // further such jumps, so it asks again until nothing new turns up.
  std::optional<SetPcAnalysis> SetPc;
  for (;;) {
    Expected<SetPcAnalysis> Analyzed =
        analyzeSetPc(Decoded->Insts, Decoded->BlockStarts, Kernel.StartOffset,
                     Env.Source.MC);
    if (!Analyzed)
      return Analyzed.takeError();
    SetPc = std::move(*Analyzed);

    bool Followed = false;
    for (uint64_t Target : unstartedTargets(*SetPc)) {
      // A target left unread keeps the refusal the analysis recorded for the
      // transfer reaching it, which `raiseInst` below reports at that
      // instruction. Reading either kind would not lift the refusal: bytes
      // already decoded decode the same way a second time, and a target no
      // function symbol covers has no extent to read to.
      if (isInsideDecodedInstruction(Decoded->Insts, Target))
        continue;
      const KernelSymbolExtent *Owner =
          findFunctionExtent(FunctionExtents, Target);
      if (!Owner)
        continue;

      // Reading from the target rather than from its function's entry covers
      // both shapes at once: a call reaches the entry anyway, and a jump over
      // an `s_endpgm` reaches a point the enclosing function was already read
      // past. It also keeps this decode disjoint from what is already in hand.
      // Each round starts an instruction at a target that had none, so the
      // rounds run out.
      Expected<DecodeResult> Extra =
          decodeKernel(Env.Source.MC, Env.OpcMap, Text.Bytes, Target,
                       Owner->Offset + Owner->Size);
      if (!Extra)
        return Extra.takeError();
      mergeDecoded(*Decoded, std::move(*Extra));
      Followed = true;
    }
    if (!Followed)
      break;
  }

  // Merging the block starts here, before any block is made, is what lets the
  // handler find the block its jump targets.
  Decoded->BlockStarts.insert(SetPc->ExtraBlockStarts.begin(),
                              SetPc->ExtraBlockStarts.end());

  const MCSubtargetInfo &SourceSTI = *Env.Source.MC.SubtargetInfo;
  const MCSubtargetInfo &TargetSTI = *Env.Target.MC.SubtargetInfo;
  bool SameWaveSize = SourceSTI.hasFeature(AMDGPU::FeatureWavefrontSize32) ==
                      TargetSTI.hasFeature(AMDGPU::FeatureWavefrontSize32);
  bool UseWaveNative =
      Env.Source.Cpu == "gfx1250" && Env.Target.Cpu == "gfx942";
  if (!SameWaveSize && !UseWaveNative)
    return RaiseFailure::general(
        RaiseFailureReason::UnsupportedWaveProjection,
        "wave-size changes are supported only from gfx1250 to gfx942");

  ProjectionKind Kind =
      UseWaveNative ? ProjectionKind::WaveNative : ProjectionKind::SameWave;
  KernelLaunchRequirements Launch{KernelLaunchRequirements::Kind::Unchanged,
                                  Meta.MaxFlatWorkgroupSize, RequiredSize};
  std::array<std::optional<uint32_t>, 3> GroupSizes;
  for (const KernelArgMeta &Arg : Meta.Args) {
    std::optional<unsigned> Dimension;
    if (Arg.ValueKind == "hidden_group_size_x")
      Dimension = 0;
    else if (Arg.ValueKind == "hidden_group_size_y")
      Dimension = 1;
    else if (Arg.ValueKind == "hidden_group_size_z")
      Dimension = 2;
    if (Dimension) {
      if (Arg.Size != sizeof(uint16_t))
        return RaiseFailure::general(RaiseFailureReason::UnsupportedLaunch,
                                     "hidden group size must be two bytes");
      GroupSizes[*Dimension] = Arg.Offset;
    }
    if (Arg.ValueKind == "hidden_dynamic_lds_size") {
      if (Arg.Size != sizeof(uint32_t))
        return RaiseFailure::general(
            RaiseFailureReason::UnsupportedLaunch,
            "hidden dynamic LDS size must be four bytes");
      Launch.DynamicLDSSizeArgOffset = Arg.Offset;
    }
  }
  Error Err = raiseDecodedKernel(Env, M, Text, Kernel, *Decoded, *SetPc, TM,
                                 Kind, Launch);
  if (!Err)
    return Launch;
  if (!UseWaveNative || Policy != LaunchPolicy::AllowReplication)
    return std::move(Err);

  bool Retry = false;
  Err = handleErrors(std::move(Err),
                     [&](std::unique_ptr<RaiseFailure> Failure) -> Error {
                       switch (Failure->reason()) {
                       case RaiseFailureReason::RequiresPerSourceWaveExecution:
                       case RaiseFailureReason::NonUniformScalarState:
                       case RaiseFailureReason::UnprovenExecContainment:
                       case RaiseFailureReason::UnprovenKernelEntryExec:
                         Retry = true;
                         return Error::success();
                       default:
                         return Error(std::move(Failure));
                       }
                     });
  if (Err)
    return std::move(Err);
  assert(Retry && "handled projection failure must request a retry");

  unsigned SourceWaveSize = getWaveSize(SourceSTI);
  unsigned TargetWaveSize = getWaveSize(TargetSTI);
  assert(TargetWaveSize % SourceWaveSize == 0 &&
         "replicated dispatch requires an integer wave-size ratio");
  unsigned ReplicationFactor = TargetWaveSize / SourceWaveSize;
  bool HasMatrix = any_of(Decoded->Insts, [&](const DecodedInst &Di) {
    return SIInstrFlags::isMAI(*Env.Source.MC.InstrInfo, Di.Inst) ||
           SIInstrFlags::isWMMA(*Env.Source.MC.InstrInfo, Di.Inst);
  });
  if (RequiredSize &&
      uint64_t((*RequiredSize)[0]) * (*RequiredSize)[1] * (*RequiredSize)[2] >
          AMDGPU::getMaxFlatWorkGroupSize() / ReplicationFactor &&
      !HasMatrix) {
    Err = raiseDecodedKernel(Env, M, Text, Kernel, *Decoded, *SetPc, TM,
                             ProjectionKind::WaveNativeDivergent, Launch);
    if (!Err)
      return Launch;
    return std::move(Err);
  }

  if (Error Err = validateReplicatedKernel(Env.Source.MC, *Decoded, Meta))
    return std::move(Err);
  unsigned MaxWorkgroupSize =
      std::min(Meta.MaxFlatWorkgroupSize,
               AMDGPU::getMaxFlatWorkGroupSize() / ReplicationFactor);
  if (!MaxWorkgroupSize)
    return RaiseFailure::general(
        RaiseFailureReason::UnsupportedLaunch,
        "replicated dispatch requires room for a whole source wave");
  Launch.Mapping =
      RequiredSize && ((*RequiredSize)[1] != 1 || (*RequiredSize)[2] != 1)
          ? KernelLaunchRequirements::Kind::ReplicatedFlattened
          : KernelLaunchRequirements::Kind::Replicated1D;
  Launch.MaxWorkgroupSize = MaxWorkgroupSize;
  Launch.SourceWaveSize = SourceWaveSize;
  Launch.ReplicationFactor = ReplicationFactor;
  if (!RequiredSize && GroupSizes[0] && GroupSizes[1] && GroupSizes[2]) {
    Launch.Mapping = KernelLaunchRequirements::Kind::ReplicatedFlattened;
    Launch.WorkgroupSizeArgOffsets = {
        {*GroupSizes[0], *GroupSizes[1], *GroupSizes[2]}};
  }
  Launch.RequiresWholeSourceWaves =
      !RequiredSize && !Launch.WorkgroupSizeArgOffsets;
  auto ValidateRequiredShape = [&]() -> Error {
    if (!RequiredSize)
      return Error::success();
    const std::array<uint32_t, 3> &Size = *RequiredSize;
    uint64_t Workitems = uint64_t(Size[0]) * Size[1] * Size[2];
    if (Workitems > Launch.MaxWorkgroupSize)
      return RaiseFailure::general(
          RaiseFailureReason::UnsupportedLaunch,
          "workgroup exceeds the kernel's supported launch size");
    if (Launch.RequiresWholeSourceWaves && Workitems % SourceWaveSize != 0)
      return RaiseFailure::general(
          RaiseFailureReason::UnsupportedLaunch,
          "replicated dispatch requires whole source waves");
    return Error::success();
  };
  if (Error Err = ValidateRequiredShape())
    return std::move(Err);
  if (Launch.RequiresWholeSourceWaves && MaxWorkgroupSize < SourceWaveSize)
    return RaiseFailure::general(
        RaiseFailureReason::UnsupportedLaunch,
        "replicated dispatch requires room for a whole source wave");
  Err = raiseDecodedKernel(Env, M, Text, Kernel, *Decoded, *SetPc, TM,
                           ProjectionKind::Replicated, Launch);
  if (Err && !Launch.RequiresWholeSourceWaves) {
    Err = handleErrors(
        std::move(Err), [&](std::unique_ptr<RaiseFailure> Failure) -> Error {
          if (Failure->reason() !=
                  RaiseFailureReason::UnprovenExecContainment &&
              Failure->reason() != RaiseFailureReason::UnprovenKernelEntryExec)
            return Error(std::move(Failure));
          Launch.RequiresWholeSourceWaves = true;
          return Error::success();
        });
    if (!Err)
      Err = ValidateRequiredShape();
    if (!Err)
      Err = raiseDecodedKernel(Env, M, Text, Kernel, *Decoded, *SetPc, TM,
                               ProjectionKind::Replicated, Launch);
  }
  if (Err)
    return std::move(Err);
  return Launch;
}

Expected<RaiseResult> raiseToIR(const TextSection &Text, StringRef SourceIsa,
                                StringRef TargetIsa,
                                ArrayRef<KernelRequest> Kernels,
                                ArrayRef<KernelSymbolExtent> FunctionExtents,
                                LaunchPolicy Policy) {
  Expected<RaiseEnvironment> Env =
      RaiseEnvironment::create(SourceIsa, TargetIsa);
  if (!Env)
    return Env.takeError();

  RaiseResult Result;
  Result.Ctx = std::make_unique<LLVMContext>();
  Result.Module = std::make_unique<Module>(kRaisedModuleName, *Result.Ctx);
  Module &M = *Result.Module;
  M.setTargetTriple(Triple(kAMDGPUTriple));

  // A module with no data layout leaves every consumer to assume one, so take
  // the AMDGPU layout from a machine built for the processor the raised IR
  // will be lowered for. That machine is also what names the target here: the
  // triple carries no processor, and the raiser emits no target instructions
  // of its own for one to appear in.
  TargetOptions Opts;
  std::unique_ptr<TargetMachine> TM(Env->Target.MC.Target->createTargetMachine(
      Triple(kAMDGPUTriple), Env->Target.Cpu, /*Features=*/"", Opts,
      Reloc::PIC_));
  if (!TM)
    return RaiseFailure::general(
        RaiseFailureReason::TargetMachineCreationFailed,
        "no target machine for '" + Env->Target.Cpu + "'");
  M.setDataLayout(TM->createDataLayout());

  // A refusal is raised where the offending instruction is, which is below the
  // point that knows which kernel of the batch is being raised, so the name and
  // the ISA pair are attached here.
  for (const KernelRequest &Kernel : Kernels) {
    if (Kernel.Name.empty() || Result.LaunchRequirements.contains(Kernel.Name))
      return RaiseFailure::general(RaiseFailureReason::BadInput,
                                   "kernel names must be nonempty and unique");
    Expected<KernelLaunchRequirements> Launch =
        raiseKernel(*Env, M, Text, Kernel, FunctionExtents, *TM, Policy);
    if (!Launch)
      return RaiseFailure::withOrigin(Launch.takeError(), Kernel.Name,
                                      Env->Source.Cpu, Env->Target.Cpu);
    Result.LaunchRequirements.insert({Kernel.Name, *Launch});
  }

  // Verify once the module is whole: a kernel is only well-formed together
  // with the intrinsic declarations its neighbours may also have added.
  std::string VerifyErr;
  raw_string_ostream VerifyOs(VerifyErr);
  if (verifyModule(M, &VerifyOs))
    return RaiseFailure::general(RaiseFailureReason::IRVerificationFailed,
                                 VerifyErr);

  return Result;
}

} // namespace COMGR::transpiler
