//===- bolt/Rewrite/SDTRewriter.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement support for System Tap Statically-Defined Trace points stored in
// .note.stapsdt section.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/DebugData.h"
#include "bolt/Rewrite/MetadataRewriter.h"
#include "bolt/Rewrite/MetadataRewriters.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Timer.h"

using namespace llvm;
using namespace bolt;

namespace opts {
static cl::opt<bool> PrintSDTMarkers("print-sdt",
                                     cl::desc("print all SDT markers"),
                                     cl::Hidden, cl::cat(BoltCategory));
}

namespace {
class SDTRewriter final : public MetadataRewriter {
  ErrorOr<BinarySection &> SDTSection{std::errc::bad_address};

  struct SDTMarkerInfo {
    uint64_t PC;
    uint64_t Base;
    uint64_t Semaphore;
    StringRef Provider;
    StringRef Name;
    StringRef Args;

    /// The offset of PC within the note section
    unsigned PCOffset;
  };

  /// Map SDT locations to SDT markers info
  using SDTMarkersListType = std::unordered_map<uint64_t, SDTMarkerInfo>;
  SDTMarkersListType SDTMarkers;

  /// Read section to populate SDTMarkers.
  void readSection();

  void printSDTMarkers() const;

public:
  SDTRewriter(StringRef Name, BinaryContext &BC) : MetadataRewriter(Name, BC) {}

  Error preCFGInitializer() override;

  Error postEmitFinalizer() override;
};

void SDTRewriter::readSection() {
  SDTSection = BC.getUniqueSectionByName(".note.stapsdt");
  if (!SDTSection)
    return;

  StringRef Buf = SDTSection->getContents();
  DataExtractor DE = DataExtractor(Buf, BC.AsmInfo->isLittleEndian(),
                                   BC.AsmInfo->getCodePointerSize());
  uint64_t Offset = 0;

  while (DE.isValidOffset(Offset)) {
    uint32_t NameSz = DE.getU32(&Offset);
    DE.getU32(&Offset); // skip over DescSz
    uint32_t Type = DE.getU32(&Offset);
    Offset = alignTo(Offset, 4);

    if (Type != 3)
      errs() << "BOLT-WARNING: SDT note type \"" << Type
             << "\" is not expected\n";

    if (NameSz == 0)
      errs() << "BOLT-WARNING: SDT note has empty name\n";

    StringRef Name = DE.getCStr(&Offset);

    if (Name != "stapsdt")
      errs() << "BOLT-WARNING: SDT note name \"" << Name
             << "\" is not expected\n";

    // Parse description
    SDTMarkerInfo Marker;
    Marker.PCOffset = Offset;
    Marker.PC = DE.getU64(&Offset);
    Marker.Base = DE.getU64(&Offset);
    Marker.Semaphore = DE.getU64(&Offset);
    Marker.Provider = DE.getCStr(&Offset);
    Marker.Name = DE.getCStr(&Offset);
    Marker.Args = DE.getCStr(&Offset);
    Offset = alignTo(Offset, 4);
    SDTMarkers[Marker.PC] = Marker;
  }

  if (opts::PrintSDTMarkers)
    printSDTMarkers();
}

Error SDTRewriter::preCFGInitializer() {
  // Populate SDTMarkers.
  readSection();

  // Mark nop instructions referenced by SDT and the containing function.
  for (const uint64_t PC : llvm::make_first_range(SDTMarkers)) {
    BinaryFunction *BF = BC.getBinaryFunctionContainingAddress(PC);

    if (!BF || !BC.shouldEmit(*BF))
      continue;

    const uint64_t Offset = PC - BF->getAddress();
    MCInst *Inst = BF->getInstructionAtOffset(Offset);
    if (!Inst)
      return createStringError(errc::executable_format_error,
                               "no instruction matches SDT offset");

    if (!BC.MIB->isNoop(*Inst))
      return createStringError(std::make_error_code(std::errc::not_supported),
                               "nop instruction expected at SDT offset");

    BC.MIB->setOffset(*Inst, static_cast<uint32_t>(Offset));

    BF->setHasSDTMarker(true);
  }

  return Error::success();
}

Error SDTRewriter::postEmitFinalizer() {
  if (!SDTSection)
    return Error::success();

  SDTSection->registerPatcher(std::make_unique<SimpleBinaryPatcher>());

  SimpleBinaryPatcher *SDTNotePatcher =
      static_cast<SimpleBinaryPatcher *>(SDTSection->getPatcher());
  for (auto &SDTInfoKV : SDTMarkers) {
    const uint64_t OriginalAddress = SDTInfoKV.first;
    const SDTMarkerInfo &SDTInfo = SDTInfoKV.second;
    const BinaryFunction *F =
        BC.getBinaryFunctionContainingAddress(OriginalAddress);
    if (!F)
      continue;
    const uint64_t NewAddress =
        F->translateInputToOutputAddress(OriginalAddress);
    SDTNotePatcher->addLE64Patch(SDTInfo.PCOffset, NewAddress);
  }

  return Error::success();
}

void SDTRewriter::printSDTMarkers() const {
  outs() << "BOLT-INFO: Number of SDT markers is " << SDTMarkers.size() << "\n";
  for (const SDTMarkerInfo &Marker : llvm::make_second_range(SDTMarkers)) {
    outs() << "BOLT-INFO: PC: " << utohexstr(Marker.PC)
           << ", Base: " << utohexstr(Marker.Base)
           << ", Semaphore: " << utohexstr(Marker.Semaphore)
           << ", Provider: " << Marker.Provider << ", Name: " << Marker.Name
           << ", Args: " << Marker.Args << "\n";
  }
}
} // namespace

std::unique_ptr<MetadataRewriter>
llvm::bolt::createSDTRewriter(BinaryContext &BC) {
  return std::make_unique<SDTRewriter>("sdt-rewriter", BC);
}
