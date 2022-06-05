//===- bolt/Passes/PatchEntries.cpp - Pass for patching function entries --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PatchEntries class that is used for patching
// the original function entry points.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/PatchEntries.h"
#include "bolt/Utils/NameResolver.h"
#include "llvm/Support/CommandLine.h"

namespace opts {

extern llvm::cl::OptionCategory BoltCategory;

extern llvm::cl::opt<unsigned> Verbosity;

llvm::cl::opt<bool>
    ForcePatch("force-patch",
               llvm::cl::desc("force patching of original entry points"),
               llvm::cl::Hidden, llvm::cl::cat(BoltCategory));
}

namespace llvm {
namespace bolt {

void PatchEntries::runOnFunctions(BinaryContext &BC) {
  if (!opts::ForcePatch) {
    // Mark the binary for patching if we did not create external references
    // for original code in any of functions we are not going to emit.
    bool NeedsPatching = false;
    for (auto &BFI : BC.getBinaryFunctions()) {
      BinaryFunction &Function = BFI.second;
      if (!BC.shouldEmit(Function) && !Function.hasExternalRefRelocations()) {
        NeedsPatching = true;
        break;
      }
    }

    if (!NeedsPatching)
      return;
  }

  if (opts::Verbosity >= 1)
    outs() << "BOLT-INFO: patching entries in original code\n";

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
          outs() << "BOLT-INFO: unable to patch entry point in " << Function
                 << " at offset 0x" << Twine::utohexstr(Offset) << '\n';
        return false;
      }

      PendingPatches.emplace_back(Patch{Symbol, Function.getAddress() + Offset,
                                        Function.getFileOffset() + Offset,
                                        Function.getOriginSection()});
      NextValidByte = Offset + PatchSize;
      if (NextValidByte > Function.getMaxSize()) {
        if (opts::Verbosity >= 1)
          outs() << "BOLT-INFO: function " << Function
                 << " too small to patch its entry point\n";
        return false;
      }

      return true;
    });

    if (!Success) {
      // If the original function entries cannot be patched, then we cannot
      // safely emit new function body.
      errs() << "BOLT-WARNING: failed to patch entries in " << Function
             << ". The function will not be optimized.\n";
      Function.setIgnored();
      continue;
    }

    for (Patch &Patch : PendingPatches) {
      BinaryFunction *PatchFunction = BC.createInjectedBinaryFunction(
          NameResolver::append(Patch.Symbol->getName(), ".org.0"));
      // Force the function to be emitted at the given address.
      PatchFunction->setOutputAddress(Patch.Address);
      PatchFunction->setFileOffset(Patch.FileOffset);
      PatchFunction->setOriginSection(Patch.Section);

      InstructionListType Seq;
      BC.MIB->createLongTailCall(Seq, Patch.Symbol, BC.Ctx.get());
      PatchFunction->addBasicBlock(0)->addInstructions(Seq);

      // Verify the size requirements.
      uint64_t HotSize, ColdSize;
      std::tie(HotSize, ColdSize) = BC.calculateEmittedSize(*PatchFunction);
      assert(!ColdSize && "unexpected cold code");
      assert(HotSize <= PatchSize && "max patch size exceeded");
    }

    Function.setIsPatched(true);
  }
}

} // end namespace bolt
} // end namespace llvm
