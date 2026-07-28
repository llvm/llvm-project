//===- bolt/Passes/Aligner.cpp - Pass for optimal code alignment ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AlignerPass class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/Aligner.h"
#include "bolt/Core/ParallelUtilities.h"

#define DEBUG_TYPE "bolt-aligner"

using namespace llvm;

namespace llvm {
namespace bolt {

// Align function to the specified byte-boundary (typically, 64) offsetting
// the function by not more than the corresponding value
static void alignMaxBytes(BinaryFunction &Function) {
  const BinaryContext &BC = Function.getBinaryContext();
  Function.setAlignment(BC.AlignFunctions);
  Function.setMaxAlignmentBytes(BC.AlignFunctionsMaxBytes);
  Function.setMaxColdAlignmentBytes(BC.AlignFunctionsMaxBytes);
}

// Align function to the specified byte-boundary (typically, 64) offsetting
// the function by not more than the minimum over
// -- the size of the function
// -- the specified number of bytes
static void alignCompact(BinaryFunction &Function,
                         const MCCodeEmitter *Emitter) {
  const BinaryContext &BC = Function.getBinaryContext();
  size_t HotSize = 0;
  size_t ColdSize = 0;

  // On AArch64, larger cold code size may lead to more veneers and higher
  // potential overhead for hot code. Minimize the cold code size.
  if (!Function.hasProfile() && BC.isAArch64()) {
    Function.setAlignment(Function.getMinAlignment());
    return;
  }

  for (const BinaryBasicBlock &BB : Function)
    if (BB.isSplit())
      ColdSize += BC.computeCodeSize(BB.begin(), BB.end(), Emitter);
    else
      HotSize += BC.computeCodeSize(BB.begin(), BB.end(), Emitter);

  Function.setAlignment(BC.AlignFunctions);
  if (HotSize > 0)
    Function.setMaxAlignmentBytes(
        std::min(size_t(BC.AlignFunctionsMaxBytes), HotSize));

  // using the same option, max-align-bytes, both for cold and hot parts of the
  // functions, as aligning cold functions typically does not affect performance
  if (ColdSize > 0)
    Function.setMaxColdAlignmentBytes(
        std::min(size_t(BC.AlignFunctionsMaxBytes), ColdSize));
}

void AlignerPass::alignBlocks(BinaryFunction &Function,
                              const MCCodeEmitter *Emitter) {
  if (!Function.hasValidProfile() || !Function.isSimple())
    return;

  const BinaryContext &BC = Function.getBinaryContext();

  const uint64_t FuncCount =
      std::max<uint64_t>(1, Function.getKnownExecutionCount());
  BinaryBasicBlock *PrevBB = nullptr;
  for (BinaryBasicBlock *BB : Function.getLayout().blocks()) {
    uint64_t Count = BB->getKnownExecutionCount();

    if (Count <= FuncCount * BC.AlignBlocksThreshold / 100) {
      PrevBB = BB;
      continue;
    }

    uint64_t FTCount = 0;
    if (PrevBB && PrevBB->getFallthrough() == BB)
      FTCount = PrevBB->getBranchInfo(*BB).Count;

    PrevBB = BB;

    if (Count < FTCount * 2)
      continue;

    const uint64_t BlockSize =
        BC.computeCodeSize(BB->begin(), BB->end(), Emitter);
    const uint64_t BytesToUse =
        std::min<uint64_t>(BC.BlockAlignment - 1, BlockSize);

    if (BC.AlignBlocksMinSize && BlockSize < BC.AlignBlocksMinSize)
      continue;

    BB->setAlignment(BC.BlockAlignment);
    BB->setAlignmentMaxBytes(BytesToUse);

    // Update stats.
    LLVM_DEBUG(
      std::unique_lock<llvm::sys::RWMutex> Lock(AlignHistogramMtx);
      AlignHistogram[BytesToUse]++;
      AlignedBlocksCount += BB->getKnownExecutionCount();
    );
  }
}

Error AlignerPass::runOnFunctions(BinaryContext &BC) {
  if (!BC.HasRelocations)
    return Error::success();

  AlignHistogram.resize(BC.BlockAlignment);

  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    // Create a separate MCCodeEmitter to allow lock free execution
    BinaryContext::IndependentCodeEmitter Emitter =
        BC.createIndependentMCCodeEmitter();

    if (BC.UseCompactAligner)
      alignCompact(BF, Emitter.MCE.get());
    else
      alignMaxBytes(BF);

    // Record the function's effective code alignment so layout passes can align
    // the tentative section base to the eventual section alignment without
    // re-scanning all functions. AssignSections (run just before this pass) has
    // assigned the output sections, so route the alignment to whichever of
    // .text / .text.cold the function actually emits into: a whole cold
    // function (and its constant island) lands entirely in .text.cold, while a
    // split function contributes its (duplicated) island and code to both.
    const uint16_t Align = std::max<uint16_t>(
        BF.getAlignment(),
        BF.hasIslandsInfo() ? BF.getConstantIslandAlignment() : uint16_t(0));
    const SmallString<32> MainSectionName = BF.getCodeSectionName();
    const bool InMainSection =
        StringRef(MainSectionName) == BC.getMainCodeSectionName();
    bool InColdSection =
        StringRef(MainSectionName) == BC.getColdCodeSectionName();
    if (!InColdSection && BF.isSplit())
      InColdSection = StringRef(BF.getCodeSectionName(FragmentNum::cold())) ==
                      BC.getColdCodeSectionName();
    BC.updateMaxCodeAlignment(Align, InMainSection, InColdSection);

    if (BC.AlignBlocks && !BC.PreserveBlocksAlignment)
      alignBlocks(BF, Emitter.MCE.get());
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_TRIVIAL, WorkFun,
      ParallelUtilities::PredicateTy(nullptr), "AlignerPass");

  LLVM_DEBUG(
    dbgs() << "BOLT-DEBUG: max bytes per basic block alignment distribution:\n";
    for (unsigned I = 1; I < AlignHistogram.size(); ++I)
      dbgs() << "  " << I << " : " << AlignHistogram[I] << '\n';

    dbgs() << "BOLT-DEBUG: total execution count of aligned blocks: "
           << AlignedBlocksCount << '\n';
  );
  return Error::success();
}

} // end namespace bolt
} // end namespace llvm
