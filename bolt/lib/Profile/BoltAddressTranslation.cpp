//===- bolt/Profile/BoltAddressTranslation.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/BoltAddressTranslation.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"

#define DEBUG_TYPE "bolt-bat"

namespace llvm {
namespace bolt {

const char *BoltAddressTranslation::SECTION_NAME = ".note.bolt_bat";

void BoltAddressTranslation::writeEntriesForBB(
    MapTy &Map, const BinaryBasicBlock &BB, uint64_t FuncInputAddress,
    uint64_t FuncOutputAddress) const {
  const uint64_t BBOutputOffset =
      BB.getOutputAddressRange().first - FuncOutputAddress;
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
  // NB: in `writeEntriesForBB` we use the input address because hashes are
  // saved early in `saveMetadata` before output addresses are assigned.
  const BBHashMapTy &BBHashMap = getBBHashMap(FuncInputAddress);
  (void)BBHashMap;
  LLVM_DEBUG(
      dbgs() << formatv(" Hash: {0:x}\n", BBHashMap.getBBHash(BBInputOffset)));
  LLVM_DEBUG(
      dbgs() << formatv(" Index: {0}\n", BBHashMap.getBBIndex(BBInputOffset)));
  // In case of conflicts (same Key mapping to different Vals), the last
  // update takes precedence. Of course it is not ideal to have conflicts and
  // those happen when we have an empty BB that either contained only
  // NOPs or a jump to the next block (successor). Either way, the successor
  // and this deleted block will both share the same output address (the same
  // key), and we need to map back. We choose here to privilege the successor by
  // allowing it to overwrite the previously inserted key in the map.
  Map.emplace(BBOutputOffset, BBInputOffset << 1);

  const auto &IOAddressMap =
      BB.getFunction()->getBinaryContext().getIOAddressMap();

  for (const auto &[InputOffset, Sym] : BB.getLocSyms()) {
    const auto InputAddress = BB.getFunction()->getAddress() + InputOffset;
    const auto OutputAddress = IOAddressMap.lookup(InputAddress);
    assert(OutputAddress && "Unknown instruction address");
    const auto OutputOffset = *OutputAddress - FuncOutputAddress;

    // Is this the first instruction in the BB? No need to duplicate the entry.
    if (OutputOffset == BBOutputOffset)
      continue;

    LLVM_DEBUG(dbgs() << "  Key: " << Twine::utohexstr(OutputOffset) << " Val: "
                      << Twine::utohexstr(InputOffset) << " (branch)\n");
    Map.emplace(OutputOffset, (InputOffset << 1) | BRANCHENTRY);
  }
}

void BoltAddressTranslation::write(const BinaryContext &BC, raw_ostream &OS) {
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: Writing BOLT Address Translation Tables\n");
  for (auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction &Function = BFI.second;
    const uint64_t InputAddress = Function.getAddress();
    const uint64_t OutputAddress = Function.getOutputAddress();
    // We don't need a translation table if the body of the function hasn't
    // changed
    if (Function.isIgnored() || (!BC.HasRelocations && !Function.isSimple()))
      continue;

    uint32_t NumSecondaryEntryPoints = 0;
    Function.forEachEntryPoint([&](uint64_t Offset, const MCSymbol *) {
      if (!Offset)
        return true;
      ++NumSecondaryEntryPoints;
      SecondaryEntryPointsMap[OutputAddress].push_back(Offset);
      return true;
    });

    LLVM_DEBUG(dbgs() << "Function name: " << Function.getPrintName() << "\n");
    LLVM_DEBUG(dbgs() << " Address reference: 0x"
                      << Twine::utohexstr(Function.getOutputAddress()) << "\n");
    LLVM_DEBUG(dbgs() << formatv(" Hash: {0:x}\n", getBFHash(InputAddress)));
    LLVM_DEBUG(dbgs() << " Secondary Entry Points: " << NumSecondaryEntryPoints
                      << '\n');

    MapTy Map;
    for (const BinaryBasicBlock *const BB :
         Function.getLayout().getMainFragment())
      writeEntriesForBB(Map, *BB, InputAddress, OutputAddress);
    // Add entries for deleted blocks. They are still required for correct BB
    // mapping of branches modified by SCTC. By convention, they would have the
    // end of the function as output address.
    const BBHashMapTy &BBHashMap = getBBHashMap(InputAddress);
    if (BBHashMap.size() != Function.size()) {
      const uint64_t EndOffset = Function.getOutputSize();
      std::unordered_set<uint32_t> MappedInputOffsets;
      for (const BinaryBasicBlock &BB : Function)
        MappedInputOffsets.emplace(BB.getInputOffset());
      for (const auto &[InputOffset, _] : BBHashMap)
        if (!llvm::is_contained(MappedInputOffsets, InputOffset))
          Map.emplace(EndOffset, InputOffset << 1);
    }
    Maps.emplace(Function.getOutputAddress(), std::move(Map));
    ReverseMap.emplace(OutputAddress, InputAddress);

    if (!Function.isSplit())
      continue;

    // Split maps
    LLVM_DEBUG(dbgs() << " Cold part\n");
    for (const FunctionFragment &FF :
         Function.getLayout().getSplitFragments()) {
      ColdPartSource.emplace(FF.getAddress(), Function.getOutputAddress());
      Map.clear();
      for (const BinaryBasicBlock *const BB : FF)
        writeEntriesForBB(Map, *BB, InputAddress, FF.getAddress());

      Maps.emplace(FF.getAddress(), std::move(Map));
    }
  }

  // Output addresses are delta-encoded
  uint64_t PrevAddress = 0;
  writeMaps</*Cold=*/false>(Maps, PrevAddress, OS);
  writeMaps</*Cold=*/true>(Maps, PrevAddress, OS);

  BC.outs() << "BOLT-INFO: Wrote " << Maps.size() << " BAT maps\n";
  BC.outs() << "BOLT-INFO: Wrote " << FuncHashes.getNumFunctions()
            << " function and " << FuncHashes.getNumBasicBlocks()
            << " basic block hashes\n";
}

APInt BoltAddressTranslation::calculateBranchEntriesBitMask(
    MapTy &Map, size_t EqualElems) const {
  APInt BitMask(alignTo(EqualElems, 8), 0);
  size_t Index = 0;
  for (std::pair<const uint32_t, uint32_t> &KeyVal : Map) {
    if (Index == EqualElems)
      break;
    const uint32_t OutputOffset = KeyVal.second;
    if (OutputOffset & BRANCHENTRY)
      BitMask.setBit(Index);
    ++Index;
  }
  return BitMask;
}

size_t BoltAddressTranslation::getNumEqualOffsets(const MapTy &Map,
                                                  uint32_t Skew) const {
  size_t EqualOffsets = 0;
  for (const std::pair<const uint32_t, uint32_t> &KeyVal : Map) {
    const uint32_t OutputOffset = KeyVal.first;
    const uint32_t InputOffset = KeyVal.second >> 1;
    if (OutputOffset == InputOffset - Skew)
      ++EqualOffsets;
    else
      break;
  }
  return EqualOffsets;
}

template <bool Cold>
void BoltAddressTranslation::writeMaps(std::map<uint64_t, MapTy> &Maps,
                                       uint64_t &PrevAddress, raw_ostream &OS) {
  const uint32_t NumFuncs =
      llvm::count_if(llvm::make_first_range(Maps), [&](const uint64_t Address) {
        return Cold == ColdPartSource.count(Address);
      });
  encodeULEB128(NumFuncs, OS);
  LLVM_DEBUG(dbgs() << "Writing " << NumFuncs << (Cold ? " cold" : "")
                    << " functions for BAT.\n");
  size_t PrevIndex = 0;
  for (auto &MapEntry : Maps) {
    const uint64_t Address = MapEntry.first;
    // Only process cold fragments in cold mode, and vice versa.
    if (Cold != ColdPartSource.count(Address))
      continue;
    // NB: in `writeMaps` we use the input address because hashes are saved
    // early in `saveMetadata` before output addresses are assigned.
    const uint64_t HotInputAddress =
        ReverseMap[Cold ? ColdPartSource[Address] : Address];
    MapTy &Map = MapEntry.second;
    const uint32_t NumEntries = Map.size();
    LLVM_DEBUG(dbgs() << "Writing " << NumEntries << " entries for 0x"
                      << Twine::utohexstr(Address) << ".\n");
    encodeULEB128(Address - PrevAddress, OS);
    PrevAddress = Address;
    const uint32_t NumSecondaryEntryPoints =
        SecondaryEntryPointsMap.count(Address)
            ? SecondaryEntryPointsMap[Address].size()
            : 0;
    uint32_t Skew = 0;
    if (Cold) {
      auto HotEntryIt = Maps.find(ColdPartSource[Address]);
      assert(HotEntryIt != Maps.end());
      size_t HotIndex = std::distance(Maps.begin(), HotEntryIt);
      encodeULEB128(HotIndex - PrevIndex, OS);
      PrevIndex = HotIndex;
      // Skew of all input offsets for cold fragments is simply the first input
      // offset.
      Skew = Map.begin()->second >> 1;
      encodeULEB128(Skew, OS);
    } else {
      // Function hash
      size_t BFHash = getBFHash(HotInputAddress);
      LLVM_DEBUG(dbgs() << "Hash: " << formatv("{0:x}\n", BFHash));
      OS.write(reinterpret_cast<char *>(&BFHash), 8);
      // Number of basic blocks
      size_t NumBasicBlocks = NumBasicBlocksMap[HotInputAddress];
      LLVM_DEBUG(dbgs() << "Basic blocks: " << NumBasicBlocks << '\n');
      encodeULEB128(NumBasicBlocks, OS);
      // Secondary entry points
      encodeULEB128(NumSecondaryEntryPoints, OS);
      LLVM_DEBUG(dbgs() << "Secondary Entry Points: " << NumSecondaryEntryPoints
                        << '\n');
    }
    encodeULEB128(NumEntries, OS);
    // Encode the number of equal offsets (output = input - skew) in the
    // beginning of the function. Only encode one offset in these cases.
    const size_t EqualElems = getNumEqualOffsets(Map, Skew);
    encodeULEB128(EqualElems, OS);
    if (EqualElems) {
      const size_t BranchEntriesBytes = alignTo(EqualElems, 8) / 8;
      APInt BranchEntries = calculateBranchEntriesBitMask(Map, EqualElems);
      OS.write(reinterpret_cast<const char *>(BranchEntries.getRawData()),
               BranchEntriesBytes);
      LLVM_DEBUG({
        dbgs() << "BranchEntries: ";
        SmallString<8> BitMaskStr;
        BranchEntries.toString(BitMaskStr, 2, false);
        dbgs() << BitMaskStr << '\n';
      });
    }
    const BBHashMapTy &BBHashMap = getBBHashMap(HotInputAddress);
    size_t Index = 0;
    uint64_t InOffset = 0;
    size_t PrevBBIndex = 0;
    // Output and Input addresses and delta-encoded
    for (std::pair<const uint32_t, uint32_t> &KeyVal : Map) {
      const uint64_t OutputAddress = KeyVal.first + Address;
      encodeULEB128(OutputAddress - PrevAddress, OS);
      PrevAddress = OutputAddress;
      if (Index++ >= EqualElems)
        encodeSLEB128(KeyVal.second - InOffset, OS);
      InOffset = KeyVal.second; // Keeping InOffset as if BRANCHENTRY is encoded
      if ((InOffset & BRANCHENTRY) == 0) {
        const bool IsBlock = BBHashMap.isInputBlock(InOffset >> 1);
        unsigned BBIndex = IsBlock ? BBHashMap.getBBIndex(InOffset >> 1) : 0;
        size_t BBHash = IsBlock ? BBHashMap.getBBHash(InOffset >> 1) : 0;
        OS.write(reinterpret_cast<char *>(&BBHash), 8);
        // Basic block index in the input binary
        encodeULEB128(BBIndex - PrevBBIndex, OS);
        PrevBBIndex = BBIndex;
        LLVM_DEBUG(dbgs() << formatv("{0:x} -> {1:x} {2:x} {3}\n", KeyVal.first,
                                     InOffset >> 1, BBHash, BBIndex));
      }
    }
    uint32_t PrevOffset = 0;
    if (!Cold && NumSecondaryEntryPoints) {
      LLVM_DEBUG(dbgs() << "Secondary entry points: ");
      // Secondary entry point offsets, delta-encoded
      for (uint32_t Offset : SecondaryEntryPointsMap[Address]) {
        encodeULEB128(Offset - PrevOffset, OS);
        LLVM_DEBUG(dbgs() << formatv("{0:x} ", Offset));
        PrevOffset = Offset;
      }
      LLVM_DEBUG(dbgs() << '\n');
    }
  }
}

std::error_code BoltAddressTranslation::parse(raw_ostream &OS, StringRef Buf) {
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
  if (!Name.starts_with("BOLT"))
    return make_error_code(llvm::errc::io_error);

  Error Err(Error::success());
  std::vector<uint64_t> HotFuncs;
  uint64_t PrevAddress = 0;
  parseMaps</*Cold=*/false>(HotFuncs, PrevAddress, DE, Offset, Err);
  parseMaps</*Cold=*/true>(HotFuncs, PrevAddress, DE, Offset, Err);
  OS << "BOLT-INFO: Parsed " << Maps.size() << " BAT entries\n";
  return errorToErrorCode(std::move(Err));
}

template <bool Cold>
void BoltAddressTranslation::parseMaps(std::vector<uint64_t> &HotFuncs,
                                       uint64_t &PrevAddress, DataExtractor &DE,
                                       uint64_t &Offset, Error &Err) {
  const uint32_t NumFunctions = DE.getULEB128(&Offset, &Err);
  LLVM_DEBUG(dbgs() << "Parsing " << NumFunctions << (Cold ? " cold" : "")
                    << " functions\n");
  size_t HotIndex = 0;
  for (uint32_t I = 0; I < NumFunctions; ++I) {
    const uint64_t Address = PrevAddress + DE.getULEB128(&Offset, &Err);
    uint64_t HotAddress = Cold ? 0 : Address;
    PrevAddress = Address;
    uint32_t SecondaryEntryPoints = 0;
    uint64_t ColdInputSkew = 0;
    if (Cold) {
      HotIndex += DE.getULEB128(&Offset, &Err);
      HotAddress = HotFuncs[HotIndex];
      ColdPartSource.emplace(Address, HotAddress);
      ColdInputSkew = DE.getULEB128(&Offset, &Err);
    } else {
      HotFuncs.push_back(Address);
      // Function hash
      const size_t FuncHash = DE.getU64(&Offset, &Err);
      FuncHashes.addEntry(Address, FuncHash);
      LLVM_DEBUG(dbgs() << formatv("{0:x}: hash {1:x}\n", Address, FuncHash));
      // Number of basic blocks
      const size_t NumBasicBlocks = DE.getULEB128(&Offset, &Err);
      NumBasicBlocksMap.emplace(Address, NumBasicBlocks);
      LLVM_DEBUG(dbgs() << formatv("{0:x}: #bbs {1}, {2} bytes\n", Address,
                                   NumBasicBlocks,
                                   getULEB128Size(NumBasicBlocks)));
      // Secondary entry points
      SecondaryEntryPoints = DE.getULEB128(&Offset, &Err);
      LLVM_DEBUG(
          dbgs() << formatv("{0:x}: secondary entry points {1}, {2} bytes\n",
                            Address, SecondaryEntryPoints,
                            getULEB128Size(SecondaryEntryPoints)));
    }
    const uint32_t NumEntries = DE.getULEB128(&Offset, &Err);
    // Equal offsets.
    const size_t EqualElems = DE.getULEB128(&Offset, &Err);
    APInt BEBitMask;
    LLVM_DEBUG(dbgs() << formatv("Equal offsets: {0}, {1} bytes\n", EqualElems,
                                 getULEB128Size(EqualElems)));
    if (EqualElems) {
      const size_t BranchEntriesBytes = alignTo(EqualElems, 8) / 8;
      BEBitMask = APInt(alignTo(EqualElems, 8), 0);
      LoadIntFromMemory(
          BEBitMask,
          reinterpret_cast<const uint8_t *>(
              DE.getBytes(&Offset, BranchEntriesBytes, &Err).data()),
          BranchEntriesBytes);
      LLVM_DEBUG({
        dbgs() << "BEBitMask: ";
        SmallString<8> BitMaskStr;
        BEBitMask.toString(BitMaskStr, 2, false);
        dbgs() << BitMaskStr << ", " << BranchEntriesBytes << " bytes\n";
      });
    }
    MapTy Map;

    LLVM_DEBUG(dbgs() << "Parsing " << NumEntries << " entries for 0x"
                      << Twine::utohexstr(Address) << "\n");
    uint64_t InputOffset = 0;
    size_t BBIndex = 0;
    for (uint32_t J = 0; J < NumEntries; ++J) {
      const uint64_t OutputDelta = DE.getULEB128(&Offset, &Err);
      const uint64_t OutputAddress = PrevAddress + OutputDelta;
      const uint64_t OutputOffset = OutputAddress - Address;
      PrevAddress = OutputAddress;
      int64_t InputDelta = 0;
      if (J < EqualElems) {
        InputOffset = ((OutputOffset + ColdInputSkew) << 1) | BEBitMask[J];
      } else {
        InputDelta = DE.getSLEB128(&Offset, &Err);
        InputOffset += InputDelta;
      }
      Map.insert(std::pair<uint32_t, uint32_t>(OutputOffset, InputOffset));
      size_t BBHash = 0;
      size_t BBIndexDelta = 0;
      const bool IsBranchEntry = InputOffset & BRANCHENTRY;
      if (!IsBranchEntry) {
        BBHash = DE.getU64(&Offset, &Err);
        BBIndexDelta = DE.getULEB128(&Offset, &Err);
        BBIndex += BBIndexDelta;
        // Map basic block hash to hot fragment by input offset
        getBBHashMap(HotAddress).addEntry(InputOffset >> 1, BBIndex, BBHash);
      }
      LLVM_DEBUG({
        dbgs() << formatv(
            "{0:x} -> {1:x} ({2}/{3}b -> {4}/{5}b), {6:x}", OutputOffset,
            InputOffset, OutputDelta, getULEB128Size(OutputDelta), InputDelta,
            (J < EqualElems) ? 0 : getSLEB128Size(InputDelta), OutputAddress);
        if (!IsBranchEntry) {
          dbgs() << formatv(" {0:x} {1}/{2}b", BBHash, BBIndex,
                            getULEB128Size(BBIndexDelta));
        }
        dbgs() << '\n';
      });
    }
    Maps.insert(std::pair<uint64_t, MapTy>(Address, Map));
    if (!Cold && SecondaryEntryPoints) {
      uint32_t EntryPointOffset = 0;
      LLVM_DEBUG(dbgs() << "Secondary entry points: ");
      for (uint32_t EntryPointId = 0; EntryPointId != SecondaryEntryPoints;
           ++EntryPointId) {
        uint32_t OffsetDelta = DE.getULEB128(&Offset, &Err);
        EntryPointOffset += OffsetDelta;
        SecondaryEntryPointsMap[Address].push_back(EntryPointOffset);
        LLVM_DEBUG(dbgs() << formatv("{0:x}/{1}b ", EntryPointOffset,
                                     getULEB128Size(OffsetDelta)));
      }
      LLVM_DEBUG(dbgs() << '\n');
    }
  }
}

void BoltAddressTranslation::dump(raw_ostream &OS) const {
  const size_t NumTables = Maps.size();
  OS << "BAT tables for " << NumTables << " functions:\n";
  for (const auto &MapEntry : Maps) {
    const uint64_t Address = MapEntry.first;
    const uint64_t HotAddress = fetchParentAddress(Address);
    const bool IsHotFunction = HotAddress == 0;
    OS << "Function Address: 0x" << Twine::utohexstr(Address);
    if (IsHotFunction)
      OS << formatv(", hash: {0:x}", getBFHash(Address));
    OS << "\n";
    OS << "BB mappings:\n";
    const BBHashMapTy &BBHashMap =
        getBBHashMap(HotAddress ? HotAddress : Address);
    for (const auto &Entry : MapEntry.second) {
      const bool IsBranch = Entry.second & BRANCHENTRY;
      const uint32_t Val = Entry.second >> 1; // dropping BRANCHENTRY bit
      OS << "0x" << Twine::utohexstr(Entry.first) << " -> "
         << "0x" << Twine::utohexstr(Val);
      if (IsBranch)
        OS << " (branch)";
      else
        OS << formatv(" hash: {0:x}", BBHashMap.getBBHash(Val));
      OS << "\n";
    }
    if (IsHotFunction) {
      auto NumBasicBlocksIt = NumBasicBlocksMap.find(Address);
      assert(NumBasicBlocksIt != NumBasicBlocksMap.end());
      OS << "NumBlocks: " << NumBasicBlocksIt->second << '\n';
    }
    auto SecondaryEntryPointsIt = SecondaryEntryPointsMap.find(Address);
    if (SecondaryEntryPointsIt != SecondaryEntryPointsMap.end()) {
      const std::vector<uint32_t> &SecondaryEntryPoints =
          SecondaryEntryPointsIt->second;
      OS << SecondaryEntryPoints.size() << " secondary entry points:\n";
      for (uint32_t EntryPointOffset : SecondaryEntryPoints)
        OS << formatv("{0:x}\n", EntryPointOffset);
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

  const uint32_t Val = KeyVal->second >> 1; // dropping BRANCHENTRY bit
  // Branch source addresses are translated to the first instruction of the
  // source BB to avoid accounting for modifications BOLT may have made in the
  // BB regarding deletion/addition of instructions.
  if (IsBranchSrc)
    return Val;
  return Offset - KeyVal->first + Val;
}

std::optional<BoltAddressTranslation::FallthroughListTy>
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
    return std::nullopt;

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

void BoltAddressTranslation::saveMetadata(BinaryContext &BC) {
  for (BinaryFunction &BF : llvm::make_second_range(BC.getBinaryFunctions())) {
    // We don't need a translation table if the body of the function hasn't
    // changed
    if (BF.isIgnored() || (!BC.HasRelocations && !BF.isSimple()))
      continue;
    // Prepare function and block hashes
    FuncHashes.addEntry(BF.getAddress(), BF.computeHash());
    BF.computeBlockHashes();
    BBHashMapTy &BBHashMap = getBBHashMap(BF.getAddress());
    // Set BF/BB metadata
    for (const BinaryBasicBlock &BB : BF)
      BBHashMap.addEntry(BB.getInputOffset(), BB.getIndex(), BB.getHash());
    NumBasicBlocksMap.emplace(BF.getAddress(), BF.size());
  }
}

unsigned
BoltAddressTranslation::getSecondaryEntryPointId(uint64_t Address,
                                                 uint32_t Offset) const {
  auto FunctionIt = SecondaryEntryPointsMap.find(Address);
  if (FunctionIt == SecondaryEntryPointsMap.end())
    return 0;
  const std::vector<uint32_t> &Offsets = FunctionIt->second;
  auto OffsetIt = std::find(Offsets.begin(), Offsets.end(), Offset);
  if (OffsetIt == Offsets.end())
    return 0;
  // Adding one here because main entry point is not stored in BAT, and
  // enumeration for secondary entry points starts with 1.
  return OffsetIt - Offsets.begin() + 1;
}

std::pair<const BinaryFunction *, unsigned>
BoltAddressTranslation::translateSymbol(const BinaryContext &BC,
                                        const MCSymbol &Symbol,
                                        uint32_t Offset) const {
  // The symbol could be a secondary entry in a cold fragment.
  uint64_t SymbolValue = cantFail(errorOrToExpected(BC.getSymbolValue(Symbol)));

  const BinaryFunction *Callee = BC.getFunctionForSymbol(&Symbol);
  assert(Callee);

  // Containing function, not necessarily the same as symbol value.
  const uint64_t CalleeAddress = Callee->getAddress();
  const uint32_t OutputOffset = SymbolValue - CalleeAddress;

  const uint64_t ParentAddress = fetchParentAddress(CalleeAddress);
  const uint64_t HotAddress = ParentAddress ? ParentAddress : CalleeAddress;

  const BinaryFunction *ParentBF = BC.getBinaryFunctionAtAddress(HotAddress);

  const uint32_t InputOffset =
      translate(CalleeAddress, OutputOffset, /*IsBranchSrc*/ false) + Offset;

  unsigned SecondaryEntryId{0};
  if (InputOffset)
    SecondaryEntryId = getSecondaryEntryPointId(HotAddress, InputOffset);

  return std::pair(ParentBF, SecondaryEntryId);
}

} // namespace bolt
} // namespace llvm
