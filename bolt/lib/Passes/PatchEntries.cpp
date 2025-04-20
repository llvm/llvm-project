//===- bolt/Passes/PatchEntries.cpp - Pass for patching function entries --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PatchEntries class that is used for patching the
// original function entry points. This ensures that only the new/optimized code
// executes and that the old code is never used. This is necessary due to
// current BOLT limitations of not being able to duplicate all function's
// associated metadata (e.g., .eh_frame, exception ranges, debug info,
// jump-tables).
//
// NOTE: A successful run of 'scanExternalRefs' can relax this requirement as
// it also ensures that old code is never executed.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/PatchEntries.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/NameResolver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"

namespace opts {
extern llvm::cl::OptionCategory BoltCategory;
extern llvm::cl::opt<unsigned> Verbosity;
} // namespace opts

namespace llvm {
namespace bolt {

Error PatchEntries::runOnFunctions(BinaryContext &BC) {
  if (!opts::ForcePatch) {
    // Mark the binary for patching if we did not create external references
    // for original code in any of functions we are not going to emit.
    bool NeedsPatching = llvm::any_of(
        llvm::make_second_range(BC.getBinaryFunctions()),
        [&](BinaryFunction &BF) {
          return !BC.shouldEmit(BF) && !BF.hasExternalRefRelocations();
        });

    if (!NeedsPatching)
      return Error::success();
  }

  if (opts::Verbosity >= 1)
    BC.outs() << "BOLT-INFO: patching entries in original code\n";

  // Calculate the size of the patch.
  static size_t PatchSize = 0;
  if (!PatchSize) {
    InstructionListType Seq;
    BC.MIB->createLongTailCall(Seq, BC.Ctx->createTempSymbol(), BC.Ctx.get());
    PatchSize = BC.computeCodeSize(Seq.begin(), Seq.end());
  }

  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    // Patch original code only for functions that will be emitted.
    if (!BC.shouldEmit(Function))
      continue;

    // Check if we can skip patching the function.
    if (!opts::ForcePatch && !Function.hasEHRanges() &&
        Function.getSize() < PatchThreshold)
      continue;

    // List of patches for function entries. We either successfully patch
    // all entries or, if we cannot patch one or more, do no patch any and
    // mark the function as ignorable.
    std::vector<Patch> PendingPatches;

    uint64_t NextValidByte = 0; // offset of the byte past the last patch
    bool Success = Function.forEachEntryPoint([&](uint64_t Offset,
                                                  const MCSymbol *Symbol) {
      if (Offset < NextValidByte) {
        if (opts::Verbosity >= 1)
          BC.outs() << "BOLT-INFO: unable to patch entry point in " << Function
                    << " at offset 0x" << Twine::utohexstr(Offset) << '\n';
        return false;
      }

      PendingPatches.emplace_back(
          Patch{Symbol, Function.getAddress() + Offset});
      NextValidByte = Offset + PatchSize;
      if (NextValidByte > Function.getMaxSize()) {
        if (opts::Verbosity >= 1)
          BC.outs() << "BOLT-INFO: function " << Function
                    << " too small to patch its entry point\n";
        return false;
      }

      return true;
    });

    if (!Success) {
      // We can't change output layout for AArch64 due to LongJmp pass
      if (BC.isAArch64()) {
        if (opts::ForcePatch) {
          BC.errs() << "BOLT-ERROR: unable to patch entries in " << Function
                    << "\n";
          return createFatalBOLTError("");
        }

        continue;
      }

      // If the original function entries cannot be patched, then we cannot
      // safely emit new function body.
      BC.errs() << "BOLT-WARNING: failed to patch entries in " << Function
                << ". The function will not be optimized.\n";
      Function.setIgnored();
      continue;
    }

    for (Patch &Patch : PendingPatches) {
      // Add instruction patch to the binary.
      InstructionListType Instructions;
      BC.MIB->createLongTailCall(Instructions, Patch.Symbol, BC.Ctx.get());
      BinaryFunction *PatchFunction = BC.createInstructionPatch(
          Patch.Address, Instructions,
          NameResolver::append(Patch.Symbol->getName(), ".org.0"));

      // Verify the size requirements.
      uint64_t HotSize, ColdSize;
      std::tie(HotSize, ColdSize) = BC.calculateEmittedSize(*PatchFunction);
      assert(!ColdSize && "unexpected cold code");
      assert(HotSize <= PatchSize && "max patch size exceeded");
    }
  }
  return Error::success();
}

} // end namespace bolt
} // end namespace llvm
