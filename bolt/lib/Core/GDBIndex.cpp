//===- bolt/Core/GDBIndex.cpp  - GDB Index support ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/GDBIndex.h"

using namespace llvm::bolt;
using namespace llvm::support::endian;

void GDBIndex::addGDBTypeUnitEntry(const GDBIndexTUEntry &&Entry) {
  std::lock_guard<std::mutex> Lock(GDBIndexMutex);
  if (!BC.getGdbIndexSection())
    return;
  GDBIndexTUEntryVector.emplace_back(Entry);
}

void GDBIndex::updateGdbIndexSection(
    const CUOffsetMap &CUMap, const uint32_t NumCUs,
    DebugARangesSectionWriter &ARangesSectionWriter) {
  if (!BC.getGdbIndexSection())
    return;
  // See https://sourceware.org/gdb/onlinedocs/gdb/Index-Section-Format.html
  // for .gdb_index section format.

  StringRef GdbIndexContents = BC.getGdbIndexSection()->getContents();

  const char *Data = GdbIndexContents.data();

  // Parse the header.
  const uint32_t Version = read32le(Data);
  if (Version != 7 && Version != 8) {
    errs() << "BOLT-ERROR: can only process .gdb_index versions 7 and 8\n";
    exit(1);
  }

  // Some .gdb_index generators use file offsets while others use section
  // offsets. Hence we can only rely on offsets relative to each other,
  // and ignore their absolute values.
  const uint32_t CUListOffset = read32le(Data + 4);
  const uint32_t CUTypesOffset = read32le(Data + 8);
  const uint32_t AddressTableOffset = read32le(Data + 12);
  const uint32_t SymbolTableOffset = read32le(Data + 16);
  const uint32_t ConstantPoolOffset = read32le(Data + 20);
  Data += 24;

  // Map CUs offsets to indices and verify existing index table.
  std::map<uint32_t, uint32_t> OffsetToIndexMap;
  const uint32_t CUListSize = CUTypesOffset - CUListOffset;
  const uint32_t TUListSize = AddressTableOffset - CUTypesOffset;
  const unsigned NUmCUsEncoded = CUListSize / 16;
  unsigned MaxDWARFVersion = BC.DwCtx->getMaxVersion();
  unsigned NumDWARF5TUs =
      getGDBIndexTUEntryVector().size() - BC.DwCtx->getNumTypeUnits();
  bool SkipTypeUnits = false;
  // For DWARF5 Types are in .debug_info.
  // LLD doesn't generate Types CU List, and in CU list offset
  // only includes CUs.
  // GDB 11+ includes only CUs in CU list and generates Types
  // list.
  // GDB 9 includes CUs and TUs in CU list and generates TYpes
  // list. The NumCUs is CUs + TUs, so need to modify the check.
  // For split-dwarf
  // GDB-11, DWARF5: TU units from dwo are not included.
  // GDB-11, DWARF4: TU units from dwo are included.
  if (MaxDWARFVersion >= 5)
    SkipTypeUnits = !TUListSize ? true
                                : ((NUmCUsEncoded + NumDWARF5TUs) ==
                                   BC.DwCtx->getNumCompileUnits());

  if (!((CUListSize == NumCUs * 16) ||
        (CUListSize == (NumCUs + NumDWARF5TUs) * 16))) {
    errs() << "BOLT-ERROR: .gdb_index: CU count mismatch\n";
    exit(1);
  }
  DenseSet<uint64_t> OriginalOffsets;
  for (unsigned Index = 0, PresentUnitsIndex = 0,
                Units = BC.DwCtx->getNumCompileUnits();
       Index < Units; ++Index) {
    const DWARFUnit *CU = BC.DwCtx->getUnitAtIndex(Index);
    if (SkipTypeUnits && CU->isTypeUnit())
      continue;
    const uint64_t Offset = read64le(Data);
    Data += 16;
    if (CU->getOffset() != Offset) {
      errs() << "BOLT-ERROR: .gdb_index CU offset mismatch\n";
      exit(1);
    }

    OriginalOffsets.insert(Offset);
    OffsetToIndexMap[Offset] = PresentUnitsIndex++;
  }

  // Ignore old address table.
  const uint32_t OldAddressTableSize = SymbolTableOffset - AddressTableOffset;
  // Move Data to the beginning of symbol table.
  Data += SymbolTableOffset - CUTypesOffset;

  // Calculate the size of the new address table.
  const auto IsValidAddressRange = [](const DebugAddressRange &Range) {
    return Range.HighPC > Range.LowPC;
  };

  uint32_t NewAddressTableSize = 0;
  for (const auto &CURangesPair : ARangesSectionWriter.getCUAddressRanges()) {
    const SmallVector<DebugAddressRange, 2> &Ranges = CURangesPair.second;
    NewAddressTableSize +=
        llvm::count_if(Ranges,
                       [&IsValidAddressRange](const DebugAddressRange &Range) {
                         return IsValidAddressRange(Range);
                       }) *
        20;
  }

  // Difference between old and new table (and section) sizes.
  // Could be negative.
  int32_t Delta = NewAddressTableSize - OldAddressTableSize;

  size_t NewGdbIndexSize = GdbIndexContents.size() + Delta;

  // Free'd by ExecutableFileMemoryManager.
  auto *NewGdbIndexContents = new uint8_t[NewGdbIndexSize];
  uint8_t *Buffer = NewGdbIndexContents;

  write32le(Buffer, Version);
  write32le(Buffer + 4, CUListOffset);
  write32le(Buffer + 8, CUTypesOffset);
  write32le(Buffer + 12, AddressTableOffset);
  write32le(Buffer + 16, SymbolTableOffset + Delta);
  write32le(Buffer + 20, ConstantPoolOffset + Delta);
  Buffer += 24;

  using MapEntry = std::pair<uint32_t, CUInfo>;
  std::vector<MapEntry> CUVector(CUMap.begin(), CUMap.end());
  // Remove the CUs we won't emit anyway.
  CUVector.erase(std::remove_if(CUVector.begin(), CUVector.end(),
                                [&OriginalOffsets](const MapEntry &It) {
                                  // Skipping TU for DWARF5 when they are not
                                  // included in CU list.
                                  return OriginalOffsets.count(It.first) == 0;
                                }),
                 CUVector.end());
  // Need to sort since we write out all of TUs in .debug_info before CUs.
  std::sort(CUVector.begin(), CUVector.end(),
            [](const MapEntry &E1, const MapEntry &E2) -> bool {
              return E1.second.Offset < E2.second.Offset;
            });
  // Create the original CU index -> updated CU index mapping,
  // as the sort above could've changed the order and we have to update
  // indices correspondingly in address map and constant pool.
  std::unordered_map<uint32_t, uint32_t> OriginalCUIndexToUpdatedCUIndexMap;
  OriginalCUIndexToUpdatedCUIndexMap.reserve(CUVector.size());
  for (uint32_t I = 0; I < CUVector.size(); ++I) {
    OriginalCUIndexToUpdatedCUIndexMap[OffsetToIndexMap.at(CUVector[I].first)] =
        I;
  }
  const auto RemapCUIndex = [&OriginalCUIndexToUpdatedCUIndexMap,
                             CUVectorSize = CUVector.size(),
                             TUVectorSize = getGDBIndexTUEntryVector().size()](
                                uint32_t OriginalIndex) {
    if (OriginalIndex >= CUVectorSize) {
      if (OriginalIndex >= CUVectorSize + TUVectorSize) {
        errs() << "BOLT-ERROR: .gdb_index unknown CU index\n";
        exit(1);
      }
      // The index is into TU CU List, which we don't reorder, so return as is.
      return OriginalIndex;
    }

    const auto It = OriginalCUIndexToUpdatedCUIndexMap.find(OriginalIndex);
    if (It == OriginalCUIndexToUpdatedCUIndexMap.end()) {
      errs() << "BOLT-ERROR: .gdb_index unknown CU index\n";
      exit(1);
    }

    return It->second;
  };

  // Writing out CU List <Offset, Size>
  for (auto &CUInfo : CUVector) {
    write64le(Buffer, CUInfo.second.Offset);
    // Length encoded in CU doesn't contain first 4 bytes that encode length.
    write64le(Buffer + 8, CUInfo.second.Length + 4);
    Buffer += 16;
  }
  sortGDBIndexTUEntryVector();
  // Rewrite TU CU List, since abbrevs can be different.
  // Entry example:
  // 0: offset = 0x00000000, type_offset = 0x0000001e, type_signature =
  // 0x418503b8111e9a7b Spec says " triplet, the first value is the CU offset,
  // the second value is the type offset in the CU, and the third value is the
  // type signature" Looking at what is being generated by gdb-add-index. The
  // first entry is TU offset, second entry is offset from it, and third entry
  // is the type signature.
  if (TUListSize)
    for (const GDBIndexTUEntry &Entry : getGDBIndexTUEntryVector()) {
      write64le(Buffer, Entry.UnitOffset);
      write64le(Buffer + 8, Entry.TypeDIERelativeOffset);
      write64le(Buffer + 16, Entry.TypeHash);
      Buffer += sizeof(GDBIndexTUEntry);
    }

  // Generate new address table.
  for (const std::pair<const uint64_t, DebugAddressRangesVector> &CURangesPair :
       ARangesSectionWriter.getCUAddressRanges()) {
    const uint32_t OriginalCUIndex = OffsetToIndexMap[CURangesPair.first];
    const uint32_t UpdatedCUIndex = RemapCUIndex(OriginalCUIndex);
    const DebugAddressRangesVector &Ranges = CURangesPair.second;
    for (const DebugAddressRange &Range : Ranges) {
      // Don't emit ranges that break gdb,
      // https://sourceware.org/bugzilla/show_bug.cgi?id=33247.
      // We've seen [0, 0) ranges here, for instance.
      if (IsValidAddressRange(Range)) {
        write64le(Buffer, Range.LowPC);
        write64le(Buffer + 8, Range.HighPC);
        write32le(Buffer + 16, UpdatedCUIndex);
        Buffer += 20;
      }
    }
  }

  const size_t TrailingSize =
      GdbIndexContents.data() + GdbIndexContents.size() - Data;
  assert(Buffer + TrailingSize == NewGdbIndexContents + NewGdbIndexSize &&
         "size calculation error");

  // Copy over the rest of the original data.
  memcpy(Buffer, Data, TrailingSize);

  // Fixup CU-indices in constant pool.
  const char *const OriginalConstantPoolData =
      GdbIndexContents.data() + ConstantPoolOffset;
  uint8_t *const UpdatedConstantPoolData =
      NewGdbIndexContents + ConstantPoolOffset + Delta;

  const char *OriginalSymbolTableData =
      GdbIndexContents.data() + SymbolTableOffset;
  std::set<uint32_t> CUVectorOffsets;
  // Parse the symbol map and extract constant pool CU offsets from it.
  while (OriginalSymbolTableData < OriginalConstantPoolData) {
    const uint32_t NameOffset = read32le(OriginalSymbolTableData);
    const uint32_t CUVectorOffset = read32le(OriginalSymbolTableData + 4);
    OriginalSymbolTableData += 8;

    // Iff both are zero, then the slot is considered empty in the hash-map.
    if (NameOffset || CUVectorOffset) {
      CUVectorOffsets.insert(CUVectorOffset);
    }
  }

  // Update the CU-indicies in the constant pool
  for (const auto CUVectorOffset : CUVectorOffsets) {
    const char *CurrentOriginalConstantPoolData =
        OriginalConstantPoolData + CUVectorOffset;
    uint8_t *CurrentUpdatedConstantPoolData =
        UpdatedConstantPoolData + CUVectorOffset;

    const uint32_t Num = read32le(CurrentOriginalConstantPoolData);
    CurrentOriginalConstantPoolData += 4;
    CurrentUpdatedConstantPoolData += 4;

    for (uint32_t J = 0; J < Num; ++J) {
      const uint32_t OriginalCUIndexAndAttributes =
          read32le(CurrentOriginalConstantPoolData);
      CurrentOriginalConstantPoolData += 4;

      // We only care for the index, which is the lowest 24 bits, other bits are
      // left as is.
      const uint32_t OriginalCUIndex =
          OriginalCUIndexAndAttributes & ((1 << 24) - 1);
      const uint32_t Attributes = OriginalCUIndexAndAttributes >> 24;
      const uint32_t UpdatedCUIndexAndAttributes =
          RemapCUIndex(OriginalCUIndex) | (Attributes << 24);

      write32le(CurrentUpdatedConstantPoolData, UpdatedCUIndexAndAttributes);
      CurrentUpdatedConstantPoolData += 4;
    }
  }

  // Register the new section.
  BC.registerOrUpdateNoteSection(".gdb_index", NewGdbIndexContents,
                                 NewGdbIndexSize);
}
