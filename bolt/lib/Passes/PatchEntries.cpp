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
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/NameResolver.h"
#include "llvm/ADT/STLExtras.h"
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

Error PatchEntries::runOnFunctions(BinaryContext &BC) {
  BC.outs() << "BOLT-INFO: patching entries in original code\n";

  static size_t PatchSize = getPatchSize(BC);
  BC.outs() << "BOLT-INFO: patch size is " << PatchSize << "\n";

  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    // Patch original code only for functions that will be emitted.
    if (!BC.shouldEmit(Function))
      continue;

    bool MustPatch = opts::ForcePatch;

    // In relocation mode, a copy will be created and only the copy can be
    // changed. To avoid undefined behaviors, we must make the original function
    // jump to the copy.
    if (BC.HasRelocations && Function.mayChange())
      MustPatch = true;

    // Check if we can skip patching the function.
    if (!MustPatch && !Function.hasEHRanges() &&
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

      PendingPatches.emplace_back(Patch{Symbol, Function.getAddress() + Offset,
                                        Function.getFileOffset() + Offset,
                                        Function.getOriginSection()});
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
        if (MustPatch) {
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
      BinaryFunction *PatchFunction = BC.createInjectedBinaryFunction(
          NameResolver::append(Patch.Symbol->getName(), ".org.0"));
      // Force the function to be emitted at the given address.
      PatchFunction->setOutputAddress(Patch.Address);
      PatchFunction->setFileOffset(Patch.FileOffset);
      PatchFunction->setOriginSection(Patch.Section);

      InstructionListType Seq;
      BC.MIB->createLongTailCall(Seq, Patch.Symbol, BC.Ctx.get());
      PatchFunction->addBasicBlock()->addInstructions(Seq);

      // Verify the size requirements.
      uint64_t HotSize, ColdSize;
      std::tie(HotSize, ColdSize) = BC.calculateEmittedSize(*PatchFunction);
      assert(!ColdSize && "unexpected cold code");
      assert(HotSize <= PatchSize && "max patch size exceeded");
    }

    Function.setIsPatched(true);
  }
  return Error::success();
}

} // end namespace bolt
} // end namespace llvm
