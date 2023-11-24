//===- bolt/Core/AddressMap.cpp - Input-output Address Map ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/AddressMap.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/BinarySection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {
namespace bolt {

const char *const AddressMap::AddressSectionName = ".bolt.addr2addr_map";
const char *const AddressMap::LabelSectionName = ".bolt.label2addr_map";

static void emitAddress(MCStreamer &Streamer, uint64_t InputAddress,
                        const MCSymbol *OutputLabel) {
  Streamer.emitIntValue(InputAddress, 8);
  Streamer.emitSymbolValue(OutputLabel, 8);
}

static void emitLabel(MCStreamer &Streamer, const MCSymbol *OutputLabel) {
  Streamer.emitIntValue(reinterpret_cast<uint64_t>(OutputLabel), 8);
  Streamer.emitSymbolValue(OutputLabel, 8);
}

void AddressMap::emit(MCStreamer &Streamer, BinaryContext &BC) {
  // Mark map sections as link-only to avoid allocation in the output file.
  const unsigned Flags = BinarySection::getFlags(/*IsReadOnly*/ true,
                                                 /*IsText*/ false,
                                                 /*IsAllocatable*/ true);
  BC.registerOrUpdateSection(AddressSectionName, ELF::SHT_PROGBITS, Flags)
      .setLinkOnly();
  BC.registerOrUpdateSection(LabelSectionName, ELF::SHT_PROGBITS, Flags)
      .setLinkOnly();

  for (const auto &[BFAddress, BF] : BC.getBinaryFunctions()) {
    if (!BF.requiresAddressMap())
      continue;

    for (const auto &BB : BF) {
      if (!BB.getLabel()->isDefined())
        continue;

      Streamer.switchSection(BC.getDataSection(LabelSectionName));
      emitLabel(Streamer, BB.getLabel());

      if (!BB.hasLocSyms())
        continue;

      Streamer.switchSection(BC.getDataSection(AddressSectionName));
      for (auto [Offset, Symbol] : BB.getLocSyms())
        emitAddress(Streamer, BFAddress + Offset, Symbol);
    }
  }
}

std::optional<AddressMap> AddressMap::parse(BinaryContext &BC) {
  auto AddressMapSection = BC.getUniqueSectionByName(AddressSectionName);
  auto LabelMapSection = BC.getUniqueSectionByName(LabelSectionName);

  if (!AddressMapSection && !LabelMapSection)
    return std::nullopt;

  AddressMap Parsed;

  const size_t EntrySize = 2 * BC.AsmInfo->getCodePointerSize();
  auto parseSection =
      [&](BinarySection &Section,
          function_ref<void(uint64_t, uint64_t)> InsertCallback) {
        StringRef Buffer = Section.getOutputContents();
        assert(Buffer.size() % EntrySize == 0 && "Unexpected address map size");

        DataExtractor DE(Buffer, BC.AsmInfo->isLittleEndian(),
                         BC.AsmInfo->getCodePointerSize());
        DataExtractor::Cursor Cursor(0);

        while (Cursor && !DE.eof(Cursor)) {
          const uint64_t Input = DE.getAddress(Cursor);
          const uint64_t Output = DE.getAddress(Cursor);
          InsertCallback(Input, Output);
        }

        assert(Cursor && "Error reading address map section");
        BC.deregisterSection(Section);
      };

  if (AddressMapSection) {
    Parsed.Address2AddressMap.reserve(AddressMapSection->getOutputSize() /
                                      EntrySize);
    parseSection(*AddressMapSection, [&](uint64_t Input, uint64_t Output) {
      if (!Parsed.Address2AddressMap.count(Input))
        Parsed.Address2AddressMap.insert({Input, Output});
    });
  }

  if (LabelMapSection) {
    Parsed.Label2AddrMap.reserve(LabelMapSection->getOutputSize() / EntrySize);
    parseSection(*LabelMapSection, [&](uint64_t Input, uint64_t Output) {
      assert(!Parsed.Label2AddrMap.count(
                 reinterpret_cast<const MCSymbol *>(Input)) &&
             "Duplicate label entry detected.");
      Parsed.Label2AddrMap.insert(
          {reinterpret_cast<const MCSymbol *>(Input), Output});
    });
  }

  return Parsed;
}

} // namespace bolt
} // namespace llvm
