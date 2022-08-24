//===- bolt/Profile/BoltAddressTranslation.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/BoltAddressTranslation.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"

#define DEBUG_TYPE "bolt-bat"

namespace llvm {
namespace bolt {

const char *BoltAddressTranslation::SECTION_NAME = ".note.bolt_bat";

void BoltAddressTranslation::writeEntriesForBB(MapTy &Map,
                                               const BinaryBasicBlock &BB,
                                               uint64_t FuncAddress) {
  const uint64_t BBOutputOffset =
      BB.getOutputAddressRange().first - FuncAddress;
  const uint32_t BBInputOffset = BB.getInputOffset();

  // Every output BB must track back to an input BB for profile collection
  // in bolted binaries. If we are missing an offset, it means this block was
  // created by a pass. We will skip writing any entries for it, and this means
  // any traffic happening in this block will map to the previous block in the
  // layout. This covers the case where an input basic block is split into two,
  // and the second one lacks any offset.
  if (BBInputOffset == BinaryBasicBlock::INVALID_OFFSET)
    return;

  LLVM_DEBUG(dbgs() << "BB " << BB.getName() << "\n");
  LLVM_DEBUG(dbgs() << "  Key: " << Twine::utohexstr(BBOutputOffset)
                    << " Val: " << Twine::utohexstr(BBInputOffset) << "\n");
  // In case of conflicts (same Key mapping to different Vals), the last
  // update takes precedence. Of course it is not ideal to have conflicts and
  // those happen when we have an empty BB that either contained only
  // NOPs or a jump to the next block (successor). Either way, the successor
  // and this deleted block will both share the same output address (the same
  // key), and we need to map back. We choose here to privilege the successor by
  // allowing it to overwrite the previously inserted key in the map.
  Map[BBOutputOffset] = BBInputOffset;

  for (const auto &IOPair : BB.getOffsetTranslationTable()) {
    const uint64_t OutputOffset = IOPair.first + BBOutputOffset;
    const uint32_t InputOffset = IOPair.second;

    // Is this the first instruction in the BB? No need to duplicate the entry.
    if (OutputOffset == BBOutputOffset)
      continue;

    LLVM_DEBUG(dbgs() << "  Key: " << Twine::utohexstr(OutputOffset) << " Val: "
                      << Twine::utohexstr(InputOffset) << " (branch)\n");
    Map.insert(
        std::pair<uint32_t, uint32_t>(OutputOffset, InputOffset | BRANCHENTRY));
  }
}

void BoltAddressTranslation::write(const BinaryContext &BC, raw_ostream &OS) {
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: Writing BOLT Address Translation Tables\n");
  for (auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction &Function = BFI.second;
    // We don't need a translation table if the body of the function hasn't
    // changed
    if (Function.isIgnored() || (!BC.HasRelocations && !Function.isSimple()))
      continue;

    LLVM_DEBUG(dbgs() << "Function name: " << Function.getPrintName() << "\n");
    LLVM_DEBUG(dbgs() << " Address reference: 0x"
                      << Twine::utohexstr(Function.getOutputAddress()) << "\n");

    MapTy Map;
    for (const BinaryBasicBlock *const BB :
         Function.getLayout().getMainFragment())
      writeEntriesForBB(Map, *BB, Function.getOutputAddress());
    Maps.emplace(Function.getOutputAddress(), std::move(Map));

    if (!Function.isSplit())
      continue;

    // Split maps
    LLVM_DEBUG(dbgs() << " Cold part\n");
    for (const FunctionFragment &FF :
         Function.getLayout().getSplitFragments()) {
      Map.clear();
      for (const BinaryBasicBlock *const BB : FF)
        writeEntriesForBB(Map, *BB, FF.getAddress());

      Maps.emplace(FF.getAddress(), std::move(Map));
      ColdPartSource.emplace(FF.getAddress(), Function.getOutputAddress());
    }
  }

  const uint32_t NumFuncs = Maps.size();
  OS.write(reinterpret_cast<const char *>(&NumFuncs), 4);
  LLVM_DEBUG(dbgs() << "Writing " << NumFuncs << " functions for BAT.\n");
  for (auto &MapEntry : Maps) {
    const uint64_t Address = MapEntry.first;
    MapTy &Map = MapEntry.second;
    const uint32_t NumEntries = Map.size();
    LLVM_DEBUG(dbgs() << "Writing " << NumEntries << " entries for 0x"
                      << Twine::utohexstr(Address) << ".\n");
    OS.write(reinterpret_cast<const char *>(&Address), 8);
    OS.write(reinterpret_cast<const char *>(&NumEntries), 4);
    for (std::pair<const uint32_t, uint32_t> &KeyVal : Map) {
      OS.write(reinterpret_cast<const char *>(&KeyVal.first), 4);
      OS.write(reinterpret_cast<const char *>(&KeyVal.second), 4);
    }
  }
  const uint32_t NumColdEntries = ColdPartSource.size();
  LLVM_DEBUG(dbgs() << "Writing " << NumColdEntries
                    << " cold part mappings.\n");
  OS.write(reinterpret_cast<const char *>(&NumColdEntries), 4);
  for (std::pair<const uint64_t, uint64_t> &ColdEntry : ColdPartSource) {
    OS.write(reinterpret_cast<const char *>(&ColdEntry.first), 8);
    OS.write(reinterpret_cast<const char *>(&ColdEntry.second), 8);
    LLVM_DEBUG(dbgs() << " " << Twine::utohexstr(ColdEntry.first) << " -> "
                      << Twine::utohexstr(ColdEntry.second) << "\n");
  }

  outs() << "BOLT-INFO: Wrote " << Maps.size() << " BAT maps\n";
  outs() << "BOLT-INFO: Wrote " << NumColdEntries
         << " BAT cold-to-hot entries\n";
}

std::error_code BoltAddressTranslation::parse(StringRef Buf) {
  DataExtractor DE = DataExtractor(Buf, true, 8);
  uint64_t Offset = 0;
  if (Buf.size() < 12)
    return make_error_code(llvm::errc::io_error);

  const uint32_t NameSz = DE.getU32(&Offset);
  const uint32_t DescSz = DE.getU32(&Offset);
  const uint32_t Type = DE.getU32(&Offset);

  if (Type != BinarySection::NT_BOLT_BAT ||
      Buf.size() + Offset < alignTo(NameSz, 4) + DescSz)
    return make_error_code(llvm::errc::io_error);

  StringRef Name = Buf.slice(Offset, Offset + NameSz);
  Offset = alignTo(Offset + NameSz, 4);
  if (Name.substr(0, 4) != "BOLT")
    return make_error_code(llvm::errc::io_error);

  if (Buf.size() - Offset < 4)
    return make_error_code(llvm::errc::io_error);

  const uint32_t NumFunctions = DE.getU32(&Offset);
  LLVM_DEBUG(dbgs() << "Parsing " << NumFunctions << " functions\n");
  for (uint32_t I = 0; I < NumFunctions; ++I) {
    if (Buf.size() - Offset < 12)
      return make_error_code(llvm::errc::io_error);

    const uint64_t Address = DE.getU64(&Offset);
    const uint32_t NumEntries = DE.getU32(&Offset);
    MapTy Map;

    LLVM_DEBUG(dbgs() << "Parsing " << NumEntries << " entries for 0x"
                      << Twine::utohexstr(Address) << "\n");
    if (Buf.size() - Offset < 8 * NumEntries)
      return make_error_code(llvm::errc::io_error);
    for (uint32_t J = 0; J < NumEntries; ++J) {
      const uint32_t OutputAddr = DE.getU32(&Offset);
      const uint32_t InputAddr = DE.getU32(&Offset);
      Map.insert(std::pair<uint32_t, uint32_t>(OutputAddr, InputAddr));
      LLVM_DEBUG(dbgs() << Twine::utohexstr(OutputAddr) << " -> "
                        << Twine::utohexstr(InputAddr) << "\n");
    }
    Maps.insert(std::pair<uint64_t, MapTy>(Address, Map));
  }

  if (Buf.size() - Offset < 4)
    return make_error_code(llvm::errc::io_error);

  const uint32_t NumColdEntries = DE.getU32(&Offset);
  LLVM_DEBUG(dbgs() << "Parsing " << NumColdEntries << " cold part mappings\n");
  for (uint32_t I = 0; I < NumColdEntries; ++I) {
    if (Buf.size() - Offset < 16)
      return make_error_code(llvm::errc::io_error);
    const uint32_t ColdAddress = DE.getU64(&Offset);
    const uint32_t HotAddress = DE.getU64(&Offset);
    ColdPartSource.insert(
        std::pair<uint64_t, uint64_t>(ColdAddress, HotAddress));
    LLVM_DEBUG(dbgs() << Twine::utohexstr(ColdAddress) << " -> "
                      << Twine::utohexstr(HotAddress) << "\n");
  }
  outs() << "BOLT-INFO: Parsed " << Maps.size() << " BAT entries\n";
  outs() << "BOLT-INFO: Parsed " << NumColdEntries
         << " BAT cold-to-hot entries\n";

  return std::error_code();
}

void BoltAddressTranslation::dump(raw_ostream &OS) {
  const size_t NumTables = Maps.size();
  OS << "BAT tables for " << NumTables << " functions:\n";
  for (const auto &MapEntry : Maps) {
    OS << "Function Address: 0x" << Twine::utohexstr(MapEntry.first) << "\n";
    OS << "BB mappings:\n";
    for (const auto &Entry : MapEntry.second) {
      const bool IsBranch = Entry.second & BRANCHENTRY;
      const uint32_t Val = Entry.second & ~BRANCHENTRY;
      OS << "0x" << Twine::utohexstr(Entry.first) << " -> "
         << "0x" << Twine::utohexstr(Val);
      if (IsBranch)
        OS << " (branch)";
      OS << "\n";
    }
    OS << "\n";
  }
  const size_t NumColdParts = ColdPartSource.size();
  if (!NumColdParts)
    return;

  OS << NumColdParts << " cold mappings:\n";
  for (const auto &Entry : ColdPartSource) {
    OS << "0x" << Twine::utohexstr(Entry.first) << " -> "
       << Twine::utohexstr(Entry.second) << "\n";
  }
  OS << "\n";
}

uint64_t BoltAddressTranslation::translate(uint64_t FuncAddress,
                                           uint64_t Offset,
                                           bool IsBranchSrc) const {
  auto Iter = Maps.find(FuncAddress);
  if (Iter == Maps.end())
    return Offset;

  const MapTy &Map = Iter->second;
  auto KeyVal = Map.upper_bound(Offset);
  if (KeyVal == Map.begin())
    return Offset;

  --KeyVal;

  const uint32_t Val = KeyVal->second & ~BRANCHENTRY;
  // Branch source addresses are translated to the first instruction of the
  // source BB to avoid accounting for modifications BOLT may have made in the
  // BB regarding deletion/addition of instructions.
  if (IsBranchSrc)
    return Val;
  return Offset - KeyVal->first + Val;
}

Optional<BoltAddressTranslation::FallthroughListTy>
BoltAddressTranslation::getFallthroughsInTrace(uint64_t FuncAddress,
                                               uint64_t From,
                                               uint64_t To) const {
  SmallVector<std::pair<uint64_t, uint64_t>, 16> Res;

  // Filter out trivial case
  if (From >= To)
    return Res;

  From -= FuncAddress;
  To -= FuncAddress;

  auto Iter = Maps.find(FuncAddress);
  if (Iter == Maps.end())
    return NoneType();

  const MapTy &Map = Iter->second;
  auto FromIter = Map.upper_bound(From);
  if (FromIter == Map.begin())
    return Res;
  // Skip instruction entries, to create fallthroughs we are only interested in
  // BB boundaries
  do {
    if (FromIter == Map.begin())
      return Res;
    --FromIter;
  } while (FromIter->second & BRANCHENTRY);

  auto ToIter = Map.upper_bound(To);
  if (ToIter == Map.begin())
    return Res;
  --ToIter;
  if (FromIter->first >= ToIter->first)
    return Res;

  for (auto Iter = FromIter; Iter != ToIter;) {
    const uint32_t Src = Iter->first;
    if (Iter->second & BRANCHENTRY) {
      ++Iter;
      continue;
    }

    ++Iter;
    while (Iter->second & BRANCHENTRY && Iter != ToIter)
      ++Iter;
    if (Iter->second & BRANCHENTRY)
      break;
    Res.emplace_back(Src, Iter->first);
  }

  return Res;
}

uint64_t BoltAddressTranslation::fetchParentAddress(uint64_t Address) const {
  auto Iter = ColdPartSource.find(Address);
  if (Iter == ColdPartSource.end())
    return 0;
  return Iter->second;
}

bool BoltAddressTranslation::enabledFor(
    llvm::object::ELFObjectFileBase *InputFile) const {
  for (const SectionRef &Section : InputFile->sections()) {
    Expected<StringRef> SectionNameOrErr = Section.getName();
    if (Error E = SectionNameOrErr.takeError())
      continue;

    if (SectionNameOrErr.get() == SECTION_NAME)
      return true;
  }
  return false;
}
} // namespace bolt
} // namespace llvm
