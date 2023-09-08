#include "bolt/Core/AddressMap.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {
namespace bolt {

const char *const AddressMap::SectionName = ".bolt.address_map";

static void emitLabel(MCStreamer &Streamer, uint64_t InputAddress,
                      const MCSymbol *OutputLabel) {
  Streamer.emitIntValue(InputAddress, 8);
  Streamer.emitSymbolValue(OutputLabel, 8);
}

void AddressMap::emit(MCStreamer &Streamer, BinaryContext &BC) {
  Streamer.switchSection(BC.getDataSection(SectionName));

  for (const auto &[BFAddress, BF] : BC.getBinaryFunctions()) {
    if (!BF.requiresAddressMap())
      continue;

    for (const auto &BB : BF) {
      if (!BB.getLabel()->isDefined())
        continue;

      emitLabel(Streamer, BFAddress + BB.getInputAddressRange().first,
                BB.getLabel());

      if (!BB.hasLocSyms())
        continue;

      for (auto [Offset, Symbol] : BB.getLocSyms())
        emitLabel(Streamer, BFAddress + Offset, Symbol);
    }
  }
}

AddressMap AddressMap::parse(StringRef Buffer, const BinaryContext &BC) {
  const auto EntrySize = 2 * BC.AsmInfo->getCodePointerSize();
  assert(Buffer.size() % EntrySize == 0 && "Unexpected address map size");

  DataExtractor DE(Buffer, BC.AsmInfo->isLittleEndian(),
                   BC.AsmInfo->getCodePointerSize());
  DataExtractor::Cursor Cursor(0);

  AddressMap Parsed;
  Parsed.Map.reserve(Buffer.size() / EntrySize);

  while (Cursor && !DE.eof(Cursor)) {
    const auto Input = DE.getAddress(Cursor);
    const auto Output = DE.getAddress(Cursor);
    if (!Parsed.Map.count(Input))
      Parsed.Map.insert({Input, Output});
  }

  assert(Cursor && "Error reading address map section");
  return Parsed;
}

} // namespace bolt
} // namespace llvm
