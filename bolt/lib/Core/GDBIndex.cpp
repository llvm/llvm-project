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
  for (unsigned Index = 0, Units = BC.DwCtx->getNumCompileUnits();
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
    OffsetToIndexMap[Offset] = Index;
  }

  // Ignore old address table.
  const uint32_t OldAddressTableSize = SymbolTableOffset - AddressTableOffset;
  // Move Data to the beginning of symbol table.
  Data += SymbolTableOffset - CUTypesOffset;

  // Calculate the size of the new address table.
  uint32_t NewAddressTableSize = 0;
  for (const auto &CURangesPair : ARangesSectionWriter.getCUAddressRanges()) {
    const SmallVector<DebugAddressRange, 2> &Ranges = CURangesPair.second;
    NewAddressTableSize += Ranges.size() * 20;
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
  // Need to sort since we write out all of TUs in .debug_info before CUs.
  std::sort(CUVector.begin(), CUVector.end(),
            [](const MapEntry &E1, const MapEntry &E2) -> bool {
              return E1.second.Offset < E2.second.Offset;
            });
  // Writing out CU List <Offset, Size>
  for (auto &CUInfo : CUVector) {
    // Skipping TU for DWARF5 when they are not included in CU list.
    if (!OriginalOffsets.count(CUInfo.first))
      continue;
    write64le(Buffer, CUInfo.second.Offset);
    // Length encoded in CU doesn't contain first 4 bytes that encode length.
    write64le(Buffer + 8, CUInfo.second.Length + 4);
    Buffer += 16;
  }

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
    const uint32_t CUIndex = OffsetToIndexMap[CURangesPair.first];
    const DebugAddressRangesVector &Ranges = CURangesPair.second;
    for (const DebugAddressRange &Range : Ranges) {
      write64le(Buffer, Range.LowPC);
      write64le(Buffer + 8, Range.HighPC);
      write32le(Buffer + 16, CUIndex);
      Buffer += 20;
    }
  }

  const size_t TrailingSize =
      GdbIndexContents.data() + GdbIndexContents.size() - Data;
  assert(Buffer + TrailingSize == NewGdbIndexContents + NewGdbIndexSize &&
         "size calculation error");

  // Copy over the rest of the original data.
  memcpy(Buffer, Data, TrailingSize);

  // Register the new section.
  BC.registerOrUpdateNoteSection(".gdb_index", NewGdbIndexContents,
                                 NewGdbIndexSize);
}
