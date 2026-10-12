//===- raise-context.h - Transpiler ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_RAISE_CONTEXT_H
#define TRANSPILER_RAISE_CONTEXT_H

#include "transpiler/common/kernel-meta.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/decoder/setpc-analysis.h"
#include "transpiler/loader/code-object-utils.h"
#include "transpiler/raiser/register-state.h"
#include "transpiler/raiser/wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace llvm {
class TargetMachine;
} // namespace llvm

namespace COMGR::transpiler {

struct DecodedInst;

// Shared state threaded through every format handler.
// Instructions used by deferred checks must remain alive until validation.
class RaiseContext {
public:
  // Build the context for the source kernel described by Meta. B must be
  // positioned in the entry block: the register file and the cross-block
  // shadow storage are allocated there, and that block is what
  // `KernelStartOffset` resolves to. Fails when the kernel descriptor and
  // the metadata disagree on the user-SGPR layout.
  static llvm::Expected<RaiseContext>
  create(llvm::IRBuilder<> &B, const WaveProjection &Projection,
         const MCState &MC, const SetPcAnalysis &SetPc,
         const llvm::DenseSet<uint64_t> &PairedSplitBarriers,
         const KernelMeta &Meta, llvm::ArrayRef<uint8_t> SourceTextBytes,
         uint64_t SourceTextBaseAddress,
         llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
         uint64_t KernelStartOffset, uint64_t KernelEndOffset,
         std::optional<bool> SourceSramEcc = std::nullopt);

  // Builder every handler emits into. Its insertion point moves as raising
  // progresses.
  llvm::IRBuilder<> &B;
  // Translation between the source and target wave sizes.
  const WaveProjection &Projection;
  // MC layer for the source ISA, shared by every kernel in the code object.
  const MCState &MC;

  // Where the source instruction at Offset transfers control, or null when it
  // makes no register-indirect transfer.
  const SetPcSite *setPcSite(uint64_t Offset) const {
    auto It = SetPc.Sites.find(Offset);
    return It == SetPc.Sites.end() ? nullptr : &It->second;
  }

  // Whether the split barrier at Offset is one half of a pair that together
  // stands for one whole barrier. A half that is not raises nothing it can
  // state on its own and is refused.
  bool isPairedSplitBarrier(uint64_t Offset) const {
    return PairedSplitBarriers.contains(Offset);
  }

  // Source architectural registers and the operand reads and writes that
  // resolve through them.
  RegisterState &registers() { return Registers; }

  /// Refuse hardware effects unsupported by the selected projection.
  llvm::Error validateHardwareEffect(const DecodedInst &Di) const;

  /// Return an error unless the source floating-point environment for Ty can
  /// be preserved for this instruction.
  llvm::Error validateFPEnvironment(const DecodedInst &Di,
                                    llvm::Type *Ty) const;
  /// Reject BF16 inputs when the source mode flushes BF16 denormals.
  llvm::Error validateBF16InputDenormMode(const DecodedInst &Di) const;

  bool sourceIeeeMode() const { return SourceIeeeMode; }

  /// Source SRAM ECC setting, or nothing when either execution mode is
  /// possible.
  std::optional<bool> sourceSramEcc() const { return SourceSramEcc; }

  /// Require masked bits to be provably zero after register promotion.
  void requireZeroBits(llvm::Value *Value, uint32_t Mask, const DecodedInst &Di,
                       const llvm::Twine &Detail);
  /// Refuse any bit requirement not established in the promoted register SSA.
  llvm::Error validateRequiredBits() const;

  /// Record an operand that must be uniform across a widened target wave.
  void requireWaveUniform(llvm::Value *Operand, const DecodedInst &Di,
                          const llvm::Twine &Detail);
  /// Require scalar branches to be uniform or to reconverge before barriers.
  void requireScalarControlFlow(llvm::Instruction *Branch,
                                const DecodedInst &Di);
  /// Require the source EXEC captured at kernel entry.
  void requireKernelEntryExec(const DecodedInst &Di);
  /// Refuse a hardware effect executed once per wave when packing source waves.
  llvm::Error requirePerWaveExecution(const DecodedInst &Di) const;
  /// Validate recorded EXEC and uniformity requirements after promotion.
  llvm::Error validateWaveRequirements(llvm::TargetMachine &TM,
                                       llvm::Value *KernelEntryExec) const;

  // Source text section, and the address the source code object loads it at.
  // PC-relative literals are materialized by reading out of these.
  llvm::ArrayRef<uint8_t> sourceTextBytes() const { return SourceTextBytes; }
  uint64_t sourceTextBaseAddress() const { return SourceTextBaseAddress; }
  // Source code-object sections a proven PC-relative address can land in.
  llvm::ArrayRef<TextSection::ImageSection> sourceImageSections() const {
    return SourceImageSections;
  }

  // Offset of the source kernel's first byte within the source text section.
  uint64_t kernelStartOffset() const { return KernelStartOffset; }
  // Offset one past the source kernel's last byte, or 0 when the kernel runs
  // to the end of the source text section.
  uint64_t kernelEndOffset() const { return KernelEndOffset; }

  // Source scratch allocation, disjoint from target spills. Null until a
  // handler needs source scratch.
  llvm::AllocaInst *scratchPrivateSegmentAlloca() const {
    return ScratchPrivateSegmentAlloca;
  }
  void setScratchPrivateSegmentAlloca(llvm::AllocaInst *Alloca) {
    ScratchPrivateSegmentAlloca = Alloca;
  }

  // Return the block raised from the source instruction at Addr. A missing
  // block is a raiser bug and aborts.
  llvm::BasicBlock *lookupBB(uint64_t Addr);

  // Record BB as the block the source instruction at Addr raises into. Two
  // blocks leading the same offset is a raiser bug and aborts.
  void defineBB(uint64_t Addr, llvm::BasicBlock *BB);

  // Target-hardware lane id (i32), emitted once per kernel and reused.
  llvm::Value *emitLaneIdx();

  // Freeze per-lane addresses when widening wave32 to wave64. New target lanes
  // may hold poison from an earlier inactive definition, which would make even
  // an EXEC-predicated memory operation undefined. Other wave-size directions
  // return the address unchanged.
  llvm::Value *freezeMemAddr(llvm::Value *Addr);

private:
  RaiseContext(llvm::IRBuilder<> &B, const WaveProjection &Projection,
               const MCState &MC, const SetPcAnalysis &SetPc,
               const llvm::DenseSet<uint64_t> &PairedSplitBarriers,
               RegisterState Registers, llvm::ArrayRef<uint8_t> SourceTextBytes,
               uint64_t SourceTextBaseAddress,
               llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
               uint64_t KernelStartOffset, uint64_t KernelEndOffset,
               unsigned SourceFloatRoundMode32,
               unsigned SourceFloatRoundMode16_64,
               bool SourceBF16InputDenormsFlush, bool SourceFp16Overflow,
               bool SourceDx10Clamp, bool SourceIeeeMode);

  // Where the kernel's register-indirect control transfers lead.
  const SetPcAnalysis &SetPc;
  // Offsets of the split-barrier halves the raise matched into whole barriers.
  const llvm::DenseSet<uint64_t> &PairedSplitBarriers;
  // Source architectural registers, allocated in the entry block.
  RegisterState Registers;

  // Hardware mode affecting partial-register memory loads.
  std::optional<bool> SourceSramEcc;

  /// Bits whose values the lowering must establish before returning IR.
  struct RequiredBits {
    llvm::WeakTrackingVH Value;
    uint32_t Mask;
    const DecodedInst *Instruction;
    std::string Detail;
  };
  llvm::SmallVector<RequiredBits> BitRequirements;
  /// A deferred check with its source diagnostic. The operand follows SSA
  /// replacements during register promotion.
  struct RequiredValue {
    llvm::WeakTrackingVH Operand;
    const DecodedInst *Instruction;
    std::string Detail;
  };
  /// Values that must agree across the entire target wave.
  llvm::SmallVector<RequiredValue> UniformityRequirements;
  /// Source EXEC values that must equal the source mask at kernel entry.
  llvm::SmallVector<RequiredValue> EntryExecRequirements;
  /// Scalar branch instructions checked for uniformity or barrier convergence.
  llvm::SmallVector<RequiredValue> ScalarControlFlowRequirements;
  // Block raised from each source instruction offset that starts one.
  llvm::DenseMap<uint64_t, llvm::BasicBlock *> OffsetToBb;

  // Source code object, read to materialize proven PC-relative literals.
  llvm::ArrayRef<uint8_t> SourceTextBytes;
  uint64_t SourceTextBaseAddress = 0;
  llvm::ArrayRef<TextSection::ImageSection> SourceImageSections;

  // Extent of the source kernel within the source text section.
  uint64_t KernelStartOffset = 0;
  uint64_t KernelEndOffset = 0;

  // Effective source floating-point modes. DX10 clamp and IEEE mode are fixed
  // on when their descriptor fields are absent.
  unsigned SourceFloatRoundMode32 = 0;
  unsigned SourceFloatRoundMode16_64 = 0;
  bool SourceBF16InputDenormsFlush = false;
  bool SourceFp16Overflow = false;
  bool SourceDx10Clamp = true;
  bool SourceIeeeMode = true;

  // Allocation backing the source private segment, made on first use.
  llvm::AllocaInst *ScratchPrivateSegmentAlloca = nullptr;
};

} // namespace COMGR::transpiler

#endif
