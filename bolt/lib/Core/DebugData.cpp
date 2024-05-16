//===- bolt/Core/DebugData.cpp - Debugging information handling -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions and classes for handling debug info.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/DebugData.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/DIEBuilder.h"
#include "bolt/Utils/Utils.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAddr.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/SHA1.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#define DEBUG_TYPE "bolt-debug-info"

namespace opts {
extern llvm::cl::opt<unsigned> Verbosity;
} // namespace opts

namespace llvm {
class MCSymbol;

namespace bolt {

static void replaceLocValbyForm(DIEBuilder &DIEBldr, DIE &Die, DIEValue DIEVal,
                                dwarf::Form Format, uint64_t NewVal) {
  if (Format == dwarf::DW_FORM_loclistx)
    DIEBldr.replaceValue(&Die, DIEVal.getAttribute(), Format,
                         DIELocList(NewVal));
  else
    DIEBldr.replaceValue(&Die, DIEVal.getAttribute(), Format,
                         DIEInteger(NewVal));
}

std::optional<AttrInfo>
findAttributeInfo(const DWARFDie DIE,
                  const DWARFAbbreviationDeclaration *AbbrevDecl,
                  uint32_t Index) {
  const DWARFUnit &U = *DIE.getDwarfUnit();
  uint64_t Offset =
      AbbrevDecl->getAttributeOffsetFromIndex(Index, DIE.getOffset(), U);
  std::optional<DWARFFormValue> Value =
      AbbrevDecl->getAttributeValueFromOffset(Index, Offset, U);
  if (!Value)
    return std::nullopt;
  // AttributeSpec
  const DWARFAbbreviationDeclaration::AttributeSpec *AttrVal =
      AbbrevDecl->attributes().begin() + Index;
  uint32_t ValSize = 0;
  std::optional<int64_t> ValSizeOpt = AttrVal->getByteSize(U);
  if (ValSizeOpt) {
    ValSize = static_cast<uint32_t>(*ValSizeOpt);
  } else {
    DWARFDataExtractor DebugInfoData = U.getDebugInfoExtractor();
    uint64_t NewOffset = Offset;
    DWARFFormValue::skipValue(Value->getForm(), DebugInfoData, &NewOffset,
                              U.getFormParams());
    // This includes entire size of the entry, which might not be just the
    // encoding part. For example for DW_AT_loc it will include expression
    // location.
    ValSize = NewOffset - Offset;
  }
  return AttrInfo{*Value, DIE.getAbbreviationDeclarationPtr(), Offset, ValSize};
}

std::optional<AttrInfo> findAttributeInfo(const DWARFDie DIE,
                                          dwarf::Attribute Attr) {
  if (!DIE.isValid())
    return std::nullopt;
  const DWARFAbbreviationDeclaration *AbbrevDecl =
      DIE.getAbbreviationDeclarationPtr();
  if (!AbbrevDecl)
    return std::nullopt;
  std::optional<uint32_t> Index = AbbrevDecl->findAttributeIndex(Attr);
  if (!Index)
    return std::nullopt;
  return findAttributeInfo(DIE, AbbrevDecl, *Index);
}

const DebugLineTableRowRef DebugLineTableRowRef::NULL_ROW{0, 0};

LLVM_ATTRIBUTE_UNUSED
static void printLE64(const std::string &S) {
  for (uint32_t I = 0, Size = S.size(); I < Size; ++I) {
    errs() << Twine::utohexstr(S[I]);
    errs() << Twine::utohexstr((int8_t)S[I]);
  }
  errs() << "\n";
}

// Writes address ranges to Writer as pairs of 64-bit (address, size).
// If RelativeRange is true, assumes the address range to be written must be of
// the form (begin address, range size), otherwise (begin address, end address).
// Terminates the list by writing a pair of two zeroes.
// Returns the number of written bytes.
static uint64_t
writeAddressRanges(raw_svector_ostream &Stream,
                   const DebugAddressRangesVector &AddressRanges,
                   const bool WriteRelativeRanges = false) {
  for (const DebugAddressRange &Range : AddressRanges) {
    support::endian::write(Stream, Range.LowPC, llvm::endianness::little);
    support::endian::write(
        Stream, WriteRelativeRanges ? Range.HighPC - Range.LowPC : Range.HighPC,
        llvm::endianness::little);
  }
  // Finish with 0 entries.
  support::endian::write(Stream, 0ULL, llvm::endianness::little);
  support::endian::write(Stream, 0ULL, llvm::endianness::little);
  return AddressRanges.size() * 16 + 16;
}

DebugRangesSectionWriter::DebugRangesSectionWriter() {
  RangesBuffer = std::make_unique<DebugBufferVector>();
  RangesStream = std::make_unique<raw_svector_ostream>(*RangesBuffer);

  // Add an empty range as the first entry;
  SectionOffset +=
      writeAddressRanges(*RangesStream.get(), DebugAddressRangesVector{});
  Kind = RangesWriterKind::DebugRangesWriter;
}

uint64_t DebugRangesSectionWriter::addRanges(
    DebugAddressRangesVector &&Ranges,
    std::map<DebugAddressRangesVector, uint64_t> &CachedRanges) {
  if (Ranges.empty())
    return getEmptyRangesOffset();

  const auto RI = CachedRanges.find(Ranges);
  if (RI != CachedRanges.end())
    return RI->second;

  const uint64_t EntryOffset = addRanges(Ranges);
  CachedRanges.emplace(std::move(Ranges), EntryOffset);

  return EntryOffset;
}

uint64_t DebugRangesSectionWriter::addRanges(DebugAddressRangesVector &Ranges) {
  if (Ranges.empty())
    return getEmptyRangesOffset();

  // Reading the SectionOffset and updating it should be atomic to guarantee
  // unique and correct offsets in patches.
  std::lock_guard<std::mutex> Lock(WriterMutex);
  const uint32_t EntryOffset = SectionOffset;
  SectionOffset += writeAddressRanges(*RangesStream.get(), Ranges);

  return EntryOffset;
}

uint64_t DebugRangesSectionWriter::getSectionOffset() {
  std::lock_guard<std::mutex> Lock(WriterMutex);
  return SectionOffset;
}

DebugAddrWriter *DebugRangeListsSectionWriter::AddrWriter = nullptr;

uint64_t DebugRangeListsSectionWriter::addRanges(
    DebugAddressRangesVector &&Ranges,
    std::map<DebugAddressRangesVector, uint64_t> &CachedRanges) {
  return addRanges(Ranges);
}

struct LocListsRangelistsHeader {
  UnitLengthType UnitLength; // Size of loclist entries section, not including
                             // size of header.
  VersionType Version;
  AddressSizeType AddressSize;
  SegmentSelectorType SegmentSelector;
  OffsetEntryCountType OffsetEntryCount;
};

static std::unique_ptr<DebugBufferVector>
getDWARF5Header(const LocListsRangelistsHeader &Header) {
  std::unique_ptr<DebugBufferVector> HeaderBuffer =
      std::make_unique<DebugBufferVector>();
  std::unique_ptr<raw_svector_ostream> HeaderStream =
      std::make_unique<raw_svector_ostream>(*HeaderBuffer);

  // 7.29 length of the set of entries for this compilation unit, not including
  // the length field itself
  const uint32_t HeaderSize =
      getDWARF5RngListLocListHeaderSize() - sizeof(UnitLengthType);

  support::endian::write(*HeaderStream, Header.UnitLength + HeaderSize,
                         llvm::endianness::little);
  support::endian::write(*HeaderStream, Header.Version,
                         llvm::endianness::little);
  support::endian::write(*HeaderStream, Header.AddressSize,
                         llvm::endianness::little);
  support::endian::write(*HeaderStream, Header.SegmentSelector,
                         llvm::endianness::little);
  support::endian::write(*HeaderStream, Header.OffsetEntryCount,
                         llvm::endianness::little);
  return HeaderBuffer;
}

struct OffsetEntry {
  uint32_t Index;
  uint32_t StartOffset;
  uint32_t EndOffset;
};
template <typename DebugVector, typename ListEntry, typename DebugAddressEntry>
static bool emitWithBase(raw_ostream &OS, const DebugVector &Entries,
                         DebugAddrWriter &AddrWriter, DWARFUnit &CU,
                         uint32_t &Index, const ListEntry BaseAddressx,
                         const ListEntry OffsetPair,
                         const std::function<void(uint32_t)> &Func) {
  if (Entries.size() < 2)
    return false;
  uint64_t Base = Entries[Index].LowPC;
  std::vector<OffsetEntry> Offsets;
  uint8_t TempBuffer[64];
  while (Index < Entries.size()) {
    const DebugAddressEntry &Entry = Entries[Index];
    if (Entry.LowPC == 0)
      break;
    // In case rnglists or loclists are not sorted.
    if (Base > Entry.LowPC)
      break;
    uint32_t StartOffset = Entry.LowPC - Base;
    uint32_t EndOffset = Entry.HighPC - Base;
    if (encodeULEB128(EndOffset, TempBuffer) > 2)
      break;
    Offsets.push_back({Index, StartOffset, EndOffset});
    ++Index;
  }

  if (Offsets.size() < 2) {
    Index -= Offsets.size();
    return false;
  }

  support::endian::write(OS, static_cast<uint8_t>(BaseAddressx),
                         llvm::endianness::little);
  uint32_t BaseIndex = AddrWriter.getIndexFromAddress(Base, CU);
  encodeULEB128(BaseIndex, OS);
  for (auto &OffsetEntry : Offsets) {
    support::endian::write(OS, static_cast<uint8_t>(OffsetPair),
                           llvm::endianness::little);
    encodeULEB128(OffsetEntry.StartOffset, OS);
    encodeULEB128(OffsetEntry.EndOffset, OS);
    Func(OffsetEntry.Index);
  }
  return true;
}

uint64_t
DebugRangeListsSectionWriter::addRanges(DebugAddressRangesVector &Ranges) {
  std::lock_guard<std::mutex> Lock(WriterMutex);

  RangeEntries.push_back(CurrentOffset);
  std::sort(
      Ranges.begin(), Ranges.end(),
      [](const DebugAddressRange &R1, const DebugAddressRange &R2) -> bool {
        return R1.LowPC < R2.LowPC;
      });
  for (unsigned I = 0; I < Ranges.size();) {
    if (emitWithBase<DebugAddressRangesVector, dwarf::RnglistEntries,
                     DebugAddressRange>(*CUBodyStream, Ranges, *AddrWriter, *CU,
                                        I, dwarf::DW_RLE_base_addressx,
                                        dwarf::DW_RLE_offset_pair,
                                        [](uint32_t Index) -> void {}))
      continue;

    const DebugAddressRange &Range = Ranges[I];
    support::endian::write(*CUBodyStream,
                           static_cast<uint8_t>(dwarf::DW_RLE_startx_length),
                           llvm::endianness::little);
    uint32_t Index = AddrWriter->getIndexFromAddress(Range.LowPC, *CU);
    encodeULEB128(Index, *CUBodyStream);
    encodeULEB128(Range.HighPC - Range.LowPC, *CUBodyStream);
    ++I;
  }

  support::endian::write(*CUBodyStream,
                         static_cast<uint8_t>(dwarf::DW_RLE_end_of_list),
                         llvm::endianness::little);
  CurrentOffset = CUBodyBuffer->size();
  return RangeEntries.size() - 1;
}

void DebugRangeListsSectionWriter::finalizeSection() {
  std::unique_ptr<DebugBufferVector> CUArrayBuffer =
      std::make_unique<DebugBufferVector>();
  std::unique_ptr<raw_svector_ostream> CUArrayStream =
      std::make_unique<raw_svector_ostream>(*CUArrayBuffer);
  constexpr uint32_t SizeOfArrayEntry = 4;
  const uint32_t SizeOfArraySection = RangeEntries.size() * SizeOfArrayEntry;
  for (uint32_t Offset : RangeEntries)
    support::endian::write(*CUArrayStream, Offset + SizeOfArraySection,
                           llvm::endianness::little);

  std::unique_ptr<DebugBufferVector> Header = getDWARF5Header(
      {static_cast<uint32_t>(SizeOfArraySection + CUBodyBuffer.get()->size()),
       5, 8, 0, static_cast<uint32_t>(RangeEntries.size())});
  *RangesStream << *Header;
  *RangesStream << *CUArrayBuffer;
  *RangesStream << *CUBodyBuffer;
  SectionOffset = RangesBuffer->size();
}

void DebugRangeListsSectionWriter::initSection(DWARFUnit &Unit) {
  CUBodyBuffer = std::make_unique<DebugBufferVector>();
  CUBodyStream = std::make_unique<raw_svector_ostream>(*CUBodyBuffer);
  RangeEntries.clear();
  CurrentOffset = 0;
  CU = &Unit;
}

void DebugARangesSectionWriter::addCURanges(uint64_t CUOffset,
                                            DebugAddressRangesVector &&Ranges) {
  std::lock_guard<std::mutex> Lock(CUAddressRangesMutex);
  CUAddressRanges.emplace(CUOffset, std::move(Ranges));
}

void DebugARangesSectionWriter::writeARangesSection(
    raw_svector_ostream &RangesStream, const CUOffsetMap &CUMap) const {
  // For reference on the format of the .debug_aranges section, see the DWARF4
  // specification, section 6.1.4 Lookup by Address
  // http://www.dwarfstd.org/doc/DWARF4.pdf
  for (const auto &CUOffsetAddressRangesPair : CUAddressRanges) {
    const uint64_t Offset = CUOffsetAddressRangesPair.first;
    const DebugAddressRangesVector &AddressRanges =
        CUOffsetAddressRangesPair.second;

    // Emit header.

    // Size of this set: 8 (size of the header) + 4 (padding after header)
    // + 2*sizeof(uint64_t) bytes for each of the ranges, plus an extra
    // pair of uint64_t's for the terminating, zero-length range.
    // Does not include size field itself.
    uint32_t Size = 8 + 4 + 2 * sizeof(uint64_t) * (AddressRanges.size() + 1);

    // Header field #1: set size.
    support::endian::write(RangesStream, Size, llvm::endianness::little);

    // Header field #2: version number, 2 as per the specification.
    support::endian::write(RangesStream, static_cast<uint16_t>(2),
                           llvm::endianness::little);

    assert(CUMap.count(Offset) && "Original CU offset is not found in CU Map");
    // Header field #3: debug info offset of the correspondent compile unit.
    support::endian::write(
        RangesStream, static_cast<uint32_t>(CUMap.find(Offset)->second.Offset),
        llvm::endianness::little);

    // Header field #4: address size.
    // 8 since we only write ELF64 binaries for now.
    RangesStream << char(8);

    // Header field #5: segment size of target architecture.
    RangesStream << char(0);

    // Padding before address table - 4 bytes in the 64-bit-pointer case.
    support::endian::write(RangesStream, static_cast<uint32_t>(0),
                           llvm::endianness::little);

    writeAddressRanges(RangesStream, AddressRanges, true);
  }
}

DebugAddrWriter::DebugAddrWriter(BinaryContext *BC) : BC(BC) {
  Buffer = std::make_unique<AddressSectionBuffer>();
  AddressStream = std::make_unique<raw_svector_ostream>(*Buffer);
}

void DebugAddrWriter::AddressForDWOCU::dump() {
  std::vector<IndexAddressPair> SortedMap(indexToAddressBegin(),
                                          indexToAdddessEnd());
  // Sorting address in increasing order of indices.
  llvm::sort(SortedMap, llvm::less_first());
  for (auto &Pair : SortedMap)
    dbgs() << Twine::utohexstr(Pair.second) << "\t" << Pair.first << "\n";
}
uint32_t DebugAddrWriter::getIndexFromAddress(uint64_t Address, DWARFUnit &CU) {
  std::lock_guard<std::mutex> Lock(WriterMutex);
  const uint64_t CUID = getCUID(CU);
  if (!AddressMaps.count(CUID))
    AddressMaps[CUID] = AddressForDWOCU();

  AddressForDWOCU &Map = AddressMaps[CUID];
  auto Entry = Map.find(Address);
  if (Entry == Map.end()) {
    auto Index = Map.getNextIndex();
    Entry = Map.insert(Address, Index).first;
  }
  return Entry->second;
}

static void updateAddressBase(DIEBuilder &DIEBlder, DebugAddrWriter &AddrWriter,
                              DWARFUnit &CU, const uint64_t Offset) {
  DIE *Die = DIEBlder.getUnitDIEbyUnit(CU);
  DIEValue GnuAddrBaseAttrInfo = Die->findAttribute(dwarf::DW_AT_GNU_addr_base);
  DIEValue AddrBaseAttrInfo = Die->findAttribute(dwarf::DW_AT_addr_base);
  dwarf::Form BaseAttrForm;
  dwarf::Attribute BaseAttr;
  // For cases where Skeleton CU does not have DW_AT_GNU_addr_base
  if (!GnuAddrBaseAttrInfo && CU.getVersion() < 5)
    return;

  if (GnuAddrBaseAttrInfo) {
    BaseAttrForm = GnuAddrBaseAttrInfo.getForm();
    BaseAttr = GnuAddrBaseAttrInfo.getAttribute();
  }

  if (AddrBaseAttrInfo) {
    BaseAttrForm = AddrBaseAttrInfo.getForm();
    BaseAttr = AddrBaseAttrInfo.getAttribute();
  }

  if (GnuAddrBaseAttrInfo || AddrBaseAttrInfo) {
    DIEBlder.replaceValue(Die, BaseAttr, BaseAttrForm, DIEInteger(Offset));
  } else if (CU.getVersion() >= 5) {
    // A case where we were not using .debug_addr section, but after update
    // now using it.
    DIEBlder.addValue(Die, dwarf::DW_AT_addr_base, dwarf::DW_FORM_sec_offset,
                      DIEInteger(Offset));
  }
}

void DebugAddrWriter::update(DIEBuilder &DIEBlder, DWARFUnit &CU) {
  // Handling the case where debug information is a mix of Debug fission and
  // monolithic.
  if (!CU.getDWOId())
    return;
  const uint64_t CUID = getCUID(CU);
  auto AM = AddressMaps.find(CUID);
  // Adding to map even if it did not contribute to .debug_addr.
  // The Skeleton CU might still have DW_AT_GNU_addr_base.
  uint64_t Offset = Buffer->size();
  // If does not exist this CUs DWO section didn't contribute to .debug_addr.
  if (AM == AddressMaps.end())
    return;
  std::vector<IndexAddressPair> SortedMap(AM->second.indexToAddressBegin(),
                                          AM->second.indexToAdddessEnd());
  // Sorting address in increasing order of indices.
  llvm::sort(SortedMap, llvm::less_first());

  uint8_t AddrSize = CU.getAddressByteSize();
  uint32_t Counter = 0;
  auto WriteAddress = [&](uint64_t Address) -> void {
    ++Counter;
    switch (AddrSize) {
    default:
      assert(false && "Address Size is invalid.");
      break;
    case 4:
      support::endian::write(*AddressStream, static_cast<uint32_t>(Address),
                             llvm::endianness::little);
      break;
    case 8:
      support::endian::write(*AddressStream, Address, llvm::endianness::little);
      break;
    }
  };

  for (const IndexAddressPair &Val : SortedMap) {
    while (Val.first > Counter)
      WriteAddress(0);
    WriteAddress(Val.second);
  }
  updateAddressBase(DIEBlder, *this, CU, Offset);
}

void DebugAddrWriterDwarf5::update(DIEBuilder &DIEBlder, DWARFUnit &CU) {
  // Need to layout all sections within .debug_addr
  // Within each section sort Address by index.
  const endianness Endian = BC->DwCtx->isLittleEndian()
                                ? llvm::endianness::little
                                : llvm::endianness::big;
  const DWARFSection &AddrSec = BC->DwCtx->getDWARFObj().getAddrSection();
  DWARFDataExtractor AddrData(BC->DwCtx->getDWARFObj(), AddrSec,
                              Endian == llvm::endianness::little, 0);
  DWARFDebugAddrTable AddrTable;
  DIDumpOptions DumpOpts;
  constexpr uint32_t HeaderSize = 8;
  const uint64_t CUID = getCUID(CU);
  const uint8_t AddrSize = CU.getAddressByteSize();
  auto AMIter = AddressMaps.find(CUID);
  // A case where CU has entry in .debug_addr, but we don't modify addresses
  // for it.
  if (AMIter == AddressMaps.end()) {
    AMIter = AddressMaps.insert({CUID, AddressForDWOCU()}).first;
    std::optional<uint64_t> BaseOffset = CU.getAddrOffsetSectionBase();
    if (!BaseOffset)
      return;
    // Address base offset is to the first entry.
    // The size of header is 8 bytes.
    uint64_t Offset = *BaseOffset - HeaderSize;
    auto Iter = UnmodifiedAddressOffsets.find(Offset);
    if (Iter != UnmodifiedAddressOffsets.end()) {
      updateAddressBase(DIEBlder, *this, CU, Iter->getSecond());
      return;
    }
    UnmodifiedAddressOffsets[Offset] = Buffer->size() + HeaderSize;
    if (Error Err = AddrTable.extract(AddrData, &Offset, 5, AddrSize,
                                      DumpOpts.WarningHandler)) {
      DumpOpts.RecoverableErrorHandler(std::move(Err));
      return;
    }

    uint32_t Index = 0;
    for (uint64_t Addr : AddrTable.getAddressEntries())
      AMIter->second.insert(Addr, Index++);
  }

  updateAddressBase(DIEBlder, *this, CU, Buffer->size() + HeaderSize);

  std::vector<IndexAddressPair> SortedMap(AMIter->second.indexToAddressBegin(),
                                          AMIter->second.indexToAdddessEnd());
  // Sorting address in increasing order of indices.
  llvm::sort(SortedMap, llvm::less_first());
  // Writing out Header
  const uint32_t Length = SortedMap.size() * AddrSize + 4;
  support::endian::write(*AddressStream, Length, Endian);
  support::endian::write(*AddressStream, static_cast<uint16_t>(5), Endian);
  support::endian::write(*AddressStream, static_cast<uint8_t>(AddrSize),
                         Endian);
  support::endian::write(*AddressStream, static_cast<uint8_t>(0), Endian);

  uint32_t Counter = 0;
  auto writeAddress = [&](uint64_t Address) -> void {
    ++Counter;
    switch (AddrSize) {
    default:
      llvm_unreachable("Address Size is invalid.");
      break;
    case 4:
      support::endian::write(*AddressStream, static_cast<uint32_t>(Address),
                             Endian);
      break;
    case 8:
      support::endian::write(*AddressStream, Address, Endian);
      break;
    }
  };

  for (const IndexAddressPair &Val : SortedMap) {
    while (Val.first > Counter)
      writeAddress(0);
    writeAddress(Val.second);
  }
}

void DebugLocWriter::init() {
  LocBuffer = std::make_unique<DebugBufferVector>();
  LocStream = std::make_unique<raw_svector_ostream>(*LocBuffer);
  // Writing out empty location list to which all references to empty location
  // lists will point.
  if (!LocSectionOffset && DwarfVersion < 5) {
    const char Zeroes[16] = {0};
    *LocStream << StringRef(Zeroes, 16);
    LocSectionOffset += 16;
  }
}

uint32_t DebugLocWriter::LocSectionOffset = 0;
void DebugLocWriter::addList(DIEBuilder &DIEBldr, DIE &Die, DIEValue &AttrInfo,
                             DebugLocationsVector &LocList) {
  if (LocList.empty()) {
    replaceLocValbyForm(DIEBldr, Die, AttrInfo, AttrInfo.getForm(),
                        DebugLocWriter::EmptyListOffset);
    return;
  }
  // Since there is a separate DebugLocWriter for each thread,
  // we don't need a lock to read the SectionOffset and update it.
  const uint32_t EntryOffset = LocSectionOffset;

  for (const DebugLocationEntry &Entry : LocList) {
    support::endian::write(*LocStream, static_cast<uint64_t>(Entry.LowPC),
                           llvm::endianness::little);
    support::endian::write(*LocStream, static_cast<uint64_t>(Entry.HighPC),
                           llvm::endianness::little);
    support::endian::write(*LocStream, static_cast<uint16_t>(Entry.Expr.size()),
                           llvm::endianness::little);
    *LocStream << StringRef(reinterpret_cast<const char *>(Entry.Expr.data()),
                            Entry.Expr.size());
    LocSectionOffset += 2 * 8 + 2 + Entry.Expr.size();
  }
  LocStream->write_zeros(16);
  LocSectionOffset += 16;
  LocListDebugInfoPatches.push_back({0xdeadbeee, EntryOffset}); // never seen
                                                                // use
  replaceLocValbyForm(DIEBldr, Die, AttrInfo, AttrInfo.getForm(), EntryOffset);
}

std::unique_ptr<DebugBufferVector> DebugLocWriter::getBuffer() {
  return std::move(LocBuffer);
}

// DWARF 4: 2.6.2
void DebugLocWriter::finalize(DIEBuilder &DIEBldr, DIE &Die) {}

static void writeEmptyListDwarf5(raw_svector_ostream &Stream) {
  support::endian::write(Stream, static_cast<uint32_t>(4),
                         llvm::endianness::little);
  support::endian::write(Stream, static_cast<uint8_t>(dwarf::DW_LLE_start_end),
                         llvm::endianness::little);

  const char Zeroes[16] = {0};
  Stream << StringRef(Zeroes, 16);
  encodeULEB128(0, Stream);
  support::endian::write(Stream,
                         static_cast<uint8_t>(dwarf::DW_LLE_end_of_list),
                         llvm::endianness::little);
}

static void writeLegacyLocList(DIEValue &AttrInfo,
                               DebugLocationsVector &LocList,
                               DIEBuilder &DIEBldr, DIE &Die,
                               DebugAddrWriter &AddrWriter,
                               DebugBufferVector &LocBuffer, DWARFUnit &CU,
                               raw_svector_ostream &LocStream) {
  if (LocList.empty()) {
    replaceLocValbyForm(DIEBldr, Die, AttrInfo, AttrInfo.getForm(),
                        DebugLocWriter::EmptyListOffset);
    return;
  }

  const uint32_t EntryOffset = LocBuffer.size();
  for (const DebugLocationEntry &Entry : LocList) {
    support::endian::write(LocStream,
                           static_cast<uint8_t>(dwarf::DW_LLE_startx_length),
                           llvm::endianness::little);
    const uint32_t Index = AddrWriter.getIndexFromAddress(Entry.LowPC, CU);
    encodeULEB128(Index, LocStream);

    support::endian::write(LocStream,
                           static_cast<uint32_t>(Entry.HighPC - Entry.LowPC),
                           llvm::endianness::little);
    support::endian::write(LocStream, static_cast<uint16_t>(Entry.Expr.size()),
                           llvm::endianness::little);
    LocStream << StringRef(reinterpret_cast<const char *>(Entry.Expr.data()),
                           Entry.Expr.size());
  }
  support::endian::write(LocStream,
                         static_cast<uint8_t>(dwarf::DW_LLE_end_of_list),
                         llvm::endianness::little);
  replaceLocValbyForm(DIEBldr, Die, AttrInfo, AttrInfo.getForm(), EntryOffset);
}

static void writeDWARF5LocList(uint32_t &NumberOfEntries, DIEValue &AttrInfo,
                               DebugLocationsVector &LocList, DIE &Die,
                               DIEBuilder &DIEBldr, DebugAddrWriter &AddrWriter,
                               DebugBufferVector &LocBodyBuffer,
                               std::vector<uint32_t> &RelativeLocListOffsets,
                               DWARFUnit &CU,
                               raw_svector_ostream &LocBodyStream) {

  replaceLocValbyForm(DIEBldr, Die, AttrInfo, dwarf::DW_FORM_loclistx,
                      NumberOfEntries);

  RelativeLocListOffsets.push_back(LocBodyBuffer.size());
  ++NumberOfEntries;
  if (LocList.empty()) {
    writeEmptyListDwarf5(LocBodyStream);
    return;
  }

  std::vector<uint64_t> OffsetsArray;
  auto writeExpression = [&](uint32_t Index) -> void {
    const DebugLocationEntry &Entry = LocList[Index];
    encodeULEB128(Entry.Expr.size(), LocBodyStream);
    LocBodyStream << StringRef(
        reinterpret_cast<const char *>(Entry.Expr.data()), Entry.Expr.size());
  };
  for (unsigned I = 0; I < LocList.size();) {
    if (emitWithBase<DebugLocationsVector, dwarf::LoclistEntries,
                     DebugLocationEntry>(LocBodyStream, LocList, AddrWriter, CU,
                                         I, dwarf::DW_LLE_base_addressx,
                                         dwarf::DW_LLE_offset_pair,
                                         writeExpression))
      continue;

    const DebugLocationEntry &Entry = LocList[I];
    support::endian::write(LocBodyStream,
                           static_cast<uint8_t>(dwarf::DW_LLE_startx_length),
                           llvm::endianness::little);
    const uint32_t Index = AddrWriter.getIndexFromAddress(Entry.LowPC, CU);
    encodeULEB128(Index, LocBodyStream);
    encodeULEB128(Entry.HighPC - Entry.LowPC, LocBodyStream);
    writeExpression(I);
    ++I;
  }

  support::endian::write(LocBodyStream,
                         static_cast<uint8_t>(dwarf::DW_LLE_end_of_list),
                         llvm::endianness::little);
}

void DebugLoclistWriter::addList(DIEBuilder &DIEBldr, DIE &Die,
                                 DIEValue &AttrInfo,
                                 DebugLocationsVector &LocList) {
  if (DwarfVersion < 5)
    writeLegacyLocList(AttrInfo, LocList, DIEBldr, Die, *AddrWriter, *LocBuffer,
                       CU, *LocStream);
  else
    writeDWARF5LocList(NumberOfEntries, AttrInfo, LocList, Die, DIEBldr,
                       *AddrWriter, *LocBodyBuffer, RelativeLocListOffsets, CU,
                       *LocBodyStream);
}

uint32_t DebugLoclistWriter::LoclistBaseOffset = 0;
void DebugLoclistWriter::finalizeDWARF5(DIEBuilder &DIEBldr, DIE &Die) {
  if (LocBodyBuffer->empty()) {
    DIEValue LocListBaseAttrInfo =
        Die.findAttribute(dwarf::DW_AT_loclists_base);
    // Pointing to first one, because it doesn't matter. There are no uses of it
    // in this CU.
    if (!isSplitDwarf() && LocListBaseAttrInfo.getType())
      DIEBldr.replaceValue(&Die, dwarf::DW_AT_loclists_base,
                           LocListBaseAttrInfo.getForm(),
                           DIEInteger(getDWARF5RngListLocListHeaderSize()));
    return;
  }

  std::unique_ptr<DebugBufferVector> LocArrayBuffer =
      std::make_unique<DebugBufferVector>();
  std::unique_ptr<raw_svector_ostream> LocArrayStream =
      std::make_unique<raw_svector_ostream>(*LocArrayBuffer);

  const uint32_t SizeOfArraySection = NumberOfEntries * sizeof(uint32_t);
  // Write out IndexArray
  for (uint32_t RelativeOffset : RelativeLocListOffsets)
    support::endian::write(
        *LocArrayStream,
        static_cast<uint32_t>(SizeOfArraySection + RelativeOffset),
        llvm::endianness::little);

  std::unique_ptr<DebugBufferVector> Header = getDWARF5Header(
      {static_cast<uint32_t>(SizeOfArraySection + LocBodyBuffer.get()->size()),
       5, 8, 0, NumberOfEntries});
  *LocStream << *Header;
  *LocStream << *LocArrayBuffer;
  *LocStream << *LocBodyBuffer;

  if (!isSplitDwarf()) {
    DIEValue LocListBaseAttrInfo =
        Die.findAttribute(dwarf::DW_AT_loclists_base);
    if (LocListBaseAttrInfo.getType()) {
      DIEBldr.replaceValue(
          &Die, dwarf::DW_AT_loclists_base, LocListBaseAttrInfo.getForm(),
          DIEInteger(LoclistBaseOffset + getDWARF5RngListLocListHeaderSize()));
    } else {
      DIEBldr.addValue(&Die, dwarf::DW_AT_loclists_base,
                       dwarf::DW_FORM_sec_offset,
                       DIEInteger(LoclistBaseOffset + Header->size()));
    }
    LoclistBaseOffset += LocBuffer->size();
  }
  clearList(RelativeLocListOffsets);
  clearList(*LocArrayBuffer);
  clearList(*LocBodyBuffer);
}

void DebugLoclistWriter::finalize(DIEBuilder &DIEBldr, DIE &Die) {
  if (DwarfVersion >= 5)
    finalizeDWARF5(DIEBldr, Die);
}

DebugAddrWriter *DebugLoclistWriter::AddrWriter = nullptr;

static std::string encodeLE(size_t ByteSize, uint64_t NewValue) {
  std::string LE64(ByteSize, 0);
  for (size_t I = 0; I < ByteSize; ++I) {
    LE64[I] = NewValue & 0xff;
    NewValue >>= 8;
  }
  return LE64;
}

void SimpleBinaryPatcher::addBinaryPatch(uint64_t Offset,
                                         std::string &&NewValue,
                                         uint32_t OldValueSize) {
  Patches.emplace_back(Offset, std::move(NewValue));
}

void SimpleBinaryPatcher::addBytePatch(uint64_t Offset, uint8_t Value) {
  auto Str = std::string(1, Value);
  Patches.emplace_back(Offset, std::move(Str));
}

void SimpleBinaryPatcher::addLEPatch(uint64_t Offset, uint64_t NewValue,
                                     size_t ByteSize) {
  Patches.emplace_back(Offset, encodeLE(ByteSize, NewValue));
}

void SimpleBinaryPatcher::addUDataPatch(uint64_t Offset, uint64_t Value,
                                        uint32_t OldValueSize) {
  std::string Buff;
  raw_string_ostream OS(Buff);
  encodeULEB128(Value, OS, OldValueSize);

  Patches.emplace_back(Offset, std::move(Buff));
}

void SimpleBinaryPatcher::addLE64Patch(uint64_t Offset, uint64_t NewValue) {
  addLEPatch(Offset, NewValue, 8);
}

void SimpleBinaryPatcher::addLE32Patch(uint64_t Offset, uint32_t NewValue,
                                       uint32_t OldValueSize) {
  addLEPatch(Offset, NewValue, 4);
}

std::string SimpleBinaryPatcher::patchBinary(StringRef BinaryContents) {
  std::string BinaryContentsStr = std::string(BinaryContents);
  for (const auto &Patch : Patches) {
    uint32_t Offset = Patch.first;
    const std::string &ByteSequence = Patch.second;
    assert(Offset + ByteSequence.size() <= BinaryContents.size() &&
           "Applied patch runs over binary size.");
    for (uint64_t I = 0, Size = ByteSequence.size(); I < Size; ++I) {
      BinaryContentsStr[Offset + I] = ByteSequence[I];
    }
  }
  return BinaryContentsStr;
}

void DebugStrOffsetsWriter::initialize(DWARFUnit &Unit) {
  if (Unit.getVersion() < 5)
    return;
  const DWARFSection &StrOffsetsSection = Unit.getStringOffsetSection();
  const std::optional<StrOffsetsContributionDescriptor> &Contr =
      Unit.getStringOffsetsTableContribution();
  if (!Contr)
    return;
  const uint8_t DwarfOffsetByteSize = Contr->getDwarfOffsetByteSize();
  assert(DwarfOffsetByteSize == 4 &&
         "Dwarf String Offsets Byte Size is not supported.");
  StrOffsets.reserve(Contr->Size);
  for (uint64_t Offset = 0; Offset < Contr->Size; Offset += DwarfOffsetByteSize)
    StrOffsets.push_back(support::endian::read32le(
        StrOffsetsSection.Data.data() + Contr->Base + Offset));
}

void DebugStrOffsetsWriter::updateAddressMap(uint32_t Index, uint32_t Address) {
  IndexToAddressMap[Index] = Address;
  StrOffsetSectionWasModified = true;
}

void DebugStrOffsetsWriter::finalizeSection(DWARFUnit &Unit,
                                            DIEBuilder &DIEBldr) {
  std::optional<AttrInfo> AttrVal =
      findAttributeInfo(Unit.getUnitDIE(), dwarf::DW_AT_str_offsets_base);
  if (!AttrVal && !Unit.isDWOUnit())
    return;
  std::optional<uint64_t> Val = std::nullopt;
  if (AttrVal) {
    Val = AttrVal->V.getAsSectionOffset();
  } else {
    if (!Unit.isDWOUnit())
      BC.errs() << "BOLT-WARNING: [internal-dwarf-error]: "
                   "DW_AT_str_offsets_base Value not present\n";
    Val = 0;
  }
  DIE &Die = *DIEBldr.getUnitDIEbyUnit(Unit);
  DIEValue StrListBaseAttrInfo =
      Die.findAttribute(dwarf::DW_AT_str_offsets_base);
  auto RetVal = ProcessedBaseOffsets.find(*Val);
  // Handling re-use of str-offsets section.
  if (RetVal == ProcessedBaseOffsets.end() || StrOffsetSectionWasModified) {
    initialize(Unit);
    // Update String Offsets that were modified.
    for (const auto &Entry : IndexToAddressMap)
      StrOffsets[Entry.first] = Entry.second;
    // Writing out the header for each section.
    support::endian::write(*StrOffsetsStream,
                           static_cast<uint32_t>(StrOffsets.size() * 4 + 4),
                           llvm::endianness::little);
    support::endian::write(*StrOffsetsStream, static_cast<uint16_t>(5),
                           llvm::endianness::little);
    support::endian::write(*StrOffsetsStream, static_cast<uint16_t>(0),
                           llvm::endianness::little);

    uint64_t BaseOffset = StrOffsetsBuffer->size();
    ProcessedBaseOffsets[*Val] = BaseOffset;
    if (StrListBaseAttrInfo.getType())
      DIEBldr.replaceValue(&Die, dwarf::DW_AT_str_offsets_base,
                           StrListBaseAttrInfo.getForm(),
                           DIEInteger(BaseOffset));
    for (const uint32_t Offset : StrOffsets)
      support::endian::write(*StrOffsetsStream, Offset,
                             llvm::endianness::little);
  } else {
    DIEBldr.replaceValue(&Die, dwarf::DW_AT_str_offsets_base,
                         StrListBaseAttrInfo.getForm(),
                         DIEInteger(RetVal->second));
  }

  StrOffsetSectionWasModified = false;
  clear();
}

void DebugStrWriter::create() {
  StrBuffer = std::make_unique<DebugStrBufferVector>();
  StrStream = std::make_unique<raw_svector_ostream>(*StrBuffer);
}

void DebugStrWriter::initialize() {
  StringRef StrSection;
  if (IsDWO)
    StrSection = DwCtx.getDWARFObj().getStrDWOSection();
  else
    StrSection = DwCtx.getDWARFObj().getStrSection();
  (*StrStream) << StrSection;
}

uint32_t DebugStrWriter::addString(StringRef Str) {
  std::lock_guard<std::mutex> Lock(WriterMutex);
  if (StrBuffer->empty())
    initialize();
  auto Offset = StrBuffer->size();
  (*StrStream) << Str;
  StrStream->write_zeros(1);
  return Offset;
}

static void emitDwarfSetLineAddrAbs(MCStreamer &OS,
                                    MCDwarfLineTableParams Params,
                                    int64_t LineDelta, uint64_t Address,
                                    int PointerSize) {
  // emit the sequence to set the address
  OS.emitIntValue(dwarf::DW_LNS_extended_op, 1);
  OS.emitULEB128IntValue(PointerSize + 1);
  OS.emitIntValue(dwarf::DW_LNE_set_address, 1);
  OS.emitIntValue(Address, PointerSize);

  // emit the sequence for the LineDelta (from 1) and a zero address delta.
  MCDwarfLineAddr::Emit(&OS, Params, LineDelta, 0);
}

static inline void emitBinaryDwarfLineTable(
    MCStreamer *MCOS, MCDwarfLineTableParams Params,
    const DWARFDebugLine::LineTable *Table,
    const std::vector<DwarfLineTable::RowSequence> &InputSequences) {
  if (InputSequences.empty())
    return;

  constexpr uint64_t InvalidAddress = UINT64_MAX;
  unsigned FileNum = 1;
  unsigned LastLine = 1;
  unsigned Column = 0;
  unsigned Flags = DWARF2_LINE_DEFAULT_IS_STMT ? DWARF2_FLAG_IS_STMT : 0;
  unsigned Isa = 0;
  unsigned Discriminator = 0;
  uint64_t LastAddress = InvalidAddress;
  uint64_t PrevEndOfSequence = InvalidAddress;
  const MCAsmInfo *AsmInfo = MCOS->getContext().getAsmInfo();

  auto emitEndOfSequence = [&](uint64_t Address) {
    MCDwarfLineAddr::Emit(MCOS, Params, INT64_MAX, Address - LastAddress);
    FileNum = 1;
    LastLine = 1;
    Column = 0;
    Flags = DWARF2_LINE_DEFAULT_IS_STMT ? DWARF2_FLAG_IS_STMT : 0;
    Isa = 0;
    Discriminator = 0;
    LastAddress = InvalidAddress;
  };

  for (const DwarfLineTable::RowSequence &Sequence : InputSequences) {
    const uint64_t SequenceStart =
        Table->Rows[Sequence.FirstIndex].Address.Address;

    // Check if we need to mark the end of the sequence.
    if (PrevEndOfSequence != InvalidAddress && LastAddress != InvalidAddress &&
        PrevEndOfSequence != SequenceStart) {
      emitEndOfSequence(PrevEndOfSequence);
    }

    for (uint32_t RowIndex = Sequence.FirstIndex;
         RowIndex <= Sequence.LastIndex; ++RowIndex) {
      const DWARFDebugLine::Row &Row = Table->Rows[RowIndex];
      int64_t LineDelta = static_cast<int64_t>(Row.Line) - LastLine;
      const uint64_t Address = Row.Address.Address;

      if (FileNum != Row.File) {
        FileNum = Row.File;
        MCOS->emitInt8(dwarf::DW_LNS_set_file);
        MCOS->emitULEB128IntValue(FileNum);
      }
      if (Column != Row.Column) {
        Column = Row.Column;
        MCOS->emitInt8(dwarf::DW_LNS_set_column);
        MCOS->emitULEB128IntValue(Column);
      }
      if (Discriminator != Row.Discriminator &&
          MCOS->getContext().getDwarfVersion() >= 4) {
        Discriminator = Row.Discriminator;
        unsigned Size = getULEB128Size(Discriminator);
        MCOS->emitInt8(dwarf::DW_LNS_extended_op);
        MCOS->emitULEB128IntValue(Size + 1);
        MCOS->emitInt8(dwarf::DW_LNE_set_discriminator);
        MCOS->emitULEB128IntValue(Discriminator);
      }
      if (Isa != Row.Isa) {
        Isa = Row.Isa;
        MCOS->emitInt8(dwarf::DW_LNS_set_isa);
        MCOS->emitULEB128IntValue(Isa);
      }
      if (Row.IsStmt != Flags) {
        Flags = Row.IsStmt;
        MCOS->emitInt8(dwarf::DW_LNS_negate_stmt);
      }
      if (Row.BasicBlock)
        MCOS->emitInt8(dwarf::DW_LNS_set_basic_block);
      if (Row.PrologueEnd)
        MCOS->emitInt8(dwarf::DW_LNS_set_prologue_end);
      if (Row.EpilogueBegin)
        MCOS->emitInt8(dwarf::DW_LNS_set_epilogue_begin);

      // The end of the sequence is not normal in the middle of the input
      // sequence, but could happen, e.g. for assembly code.
      if (Row.EndSequence) {
        emitEndOfSequence(Address);
      } else {
        if (LastAddress == InvalidAddress)
          emitDwarfSetLineAddrAbs(*MCOS, Params, LineDelta, Address,
                                  AsmInfo->getCodePointerSize());
        else
          MCDwarfLineAddr::Emit(MCOS, Params, LineDelta, Address - LastAddress);

        LastAddress = Address;
        LastLine = Row.Line;
      }

      Discriminator = 0;
    }
    PrevEndOfSequence = Sequence.EndAddress;
  }

  // Finish with the end of the sequence.
  if (LastAddress != InvalidAddress)
    emitEndOfSequence(PrevEndOfSequence);
}

// This function is similar to the one from MCDwarfLineTable, except it handles
// end-of-sequence entries differently by utilizing line entries with
// DWARF2_FLAG_END_SEQUENCE flag.
static inline void emitDwarfLineTable(
    MCStreamer *MCOS, MCSection *Section,
    const MCLineSection::MCDwarfLineEntryCollection &LineEntries) {
  unsigned FileNum = 1;
  unsigned LastLine = 1;
  unsigned Column = 0;
  unsigned Flags = DWARF2_LINE_DEFAULT_IS_STMT ? DWARF2_FLAG_IS_STMT : 0;
  unsigned Isa = 0;
  unsigned Discriminator = 0;
  MCSymbol *LastLabel = nullptr;
  const MCAsmInfo *AsmInfo = MCOS->getContext().getAsmInfo();

  // Loop through each MCDwarfLineEntry and encode the dwarf line number table.
  for (const MCDwarfLineEntry &LineEntry : LineEntries) {
    if (LineEntry.getFlags() & DWARF2_FLAG_END_SEQUENCE) {
      MCOS->emitDwarfAdvanceLineAddr(INT64_MAX, LastLabel, LineEntry.getLabel(),
                                     AsmInfo->getCodePointerSize());
      FileNum = 1;
      LastLine = 1;
      Column = 0;
      Flags = DWARF2_LINE_DEFAULT_IS_STMT ? DWARF2_FLAG_IS_STMT : 0;
      Isa = 0;
      Discriminator = 0;
      LastLabel = nullptr;
      continue;
    }

    int64_t LineDelta = static_cast<int64_t>(LineEntry.getLine()) - LastLine;

    if (FileNum != LineEntry.getFileNum()) {
      FileNum = LineEntry.getFileNum();
      MCOS->emitInt8(dwarf::DW_LNS_set_file);
      MCOS->emitULEB128IntValue(FileNum);
    }
    if (Column != LineEntry.getColumn()) {
      Column = LineEntry.getColumn();
      MCOS->emitInt8(dwarf::DW_LNS_set_column);
      MCOS->emitULEB128IntValue(Column);
    }
    if (Discriminator != LineEntry.getDiscriminator() &&
        MCOS->getContext().getDwarfVersion() >= 2) {
      Discriminator = LineEntry.getDiscriminator();
      unsigned Size = getULEB128Size(Discriminator);
      MCOS->emitInt8(dwarf::DW_LNS_extended_op);
      MCOS->emitULEB128IntValue(Size + 1);
      MCOS->emitInt8(dwarf::DW_LNE_set_discriminator);
      MCOS->emitULEB128IntValue(Discriminator);
    }
    if (Isa != LineEntry.getIsa()) {
      Isa = LineEntry.getIsa();
      MCOS->emitInt8(dwarf::DW_LNS_set_isa);
      MCOS->emitULEB128IntValue(Isa);
    }
    if ((LineEntry.getFlags() ^ Flags) & DWARF2_FLAG_IS_STMT) {
      Flags = LineEntry.getFlags();
      MCOS->emitInt8(dwarf::DW_LNS_negate_stmt);
    }
    if (LineEntry.getFlags() & DWARF2_FLAG_BASIC_BLOCK)
      MCOS->emitInt8(dwarf::DW_LNS_set_basic_block);
    if (LineEntry.getFlags() & DWARF2_FLAG_PROLOGUE_END)
      MCOS->emitInt8(dwarf::DW_LNS_set_prologue_end);
    if (LineEntry.getFlags() & DWARF2_FLAG_EPILOGUE_BEGIN)
      MCOS->emitInt8(dwarf::DW_LNS_set_epilogue_begin);

    MCSymbol *Label = LineEntry.getLabel();

    // At this point we want to emit/create the sequence to encode the delta
    // in line numbers and the increment of the address from the previous
    // Label and the current Label.
    MCOS->emitDwarfAdvanceLineAddr(LineDelta, LastLabel, Label,
                                   AsmInfo->getCodePointerSize());
    Discriminator = 0;
    LastLine = LineEntry.getLine();
    LastLabel = Label;
  }

  assert(LastLabel == nullptr && "end of sequence expected");
}

void DwarfLineTable::emitCU(MCStreamer *MCOS, MCDwarfLineTableParams Params,
                            std::optional<MCDwarfLineStr> &LineStr,
                            BinaryContext &BC) const {
  if (!RawData.empty()) {
    assert(MCLineSections.getMCLineEntries().empty() &&
           InputSequences.empty() &&
           "cannot combine raw data with new line entries");
    MCOS->emitLabel(getLabel());
    MCOS->emitBytes(RawData);
    return;
  }

  MCSymbol *LineEndSym = Header.Emit(MCOS, Params, LineStr).second;

  // Put out the line tables.
  for (const auto &LineSec : MCLineSections.getMCLineEntries())
    emitDwarfLineTable(MCOS, LineSec.first, LineSec.second);

  // Emit line tables for the original code.
  emitBinaryDwarfLineTable(MCOS, Params, InputTable, InputSequences);

  // This is the end of the section, so set the value of the symbol at the end
  // of this section (that was used in a previous expression).
  MCOS->emitLabel(LineEndSym);
}

// Helper function to parse .debug_line_str, and populate one we are using.
// For functions that we do not modify we output them as raw data.
// Re-constructing .debug_line_str so that offsets are correct for those
// debug line tables.
// Bonus is that when we output a final binary we can re-use .debug_line_str
// section. So we don't have to do the SHF_ALLOC trick we did with
// .debug_line.
static void parseAndPopulateDebugLineStr(BinarySection &LineStrSection,
                                         MCDwarfLineStr &LineStr,
                                         BinaryContext &BC) {
  DataExtractor StrData(LineStrSection.getContents(),
                        BC.DwCtx->isLittleEndian(), 0);
  uint64_t Offset = 0;
  while (StrData.isValidOffset(Offset)) {
    const uint64_t StrOffset = Offset;
    Error Err = Error::success();
    const char *CStr = StrData.getCStr(&Offset, &Err);
    if (Err) {
      BC.errs() << "BOLT-ERROR: could not extract string from .debug_line_str";
      continue;
    }
    const size_t NewOffset = LineStr.addString(CStr);
    assert(StrOffset == NewOffset &&
           "New offset in .debug_line_str doesn't match original offset");
    (void)StrOffset;
    (void)NewOffset;
  }
}

void DwarfLineTable::emit(BinaryContext &BC, MCStreamer &Streamer) {
  MCAssembler &Assembler =
      static_cast<MCObjectStreamer *>(&Streamer)->getAssembler();

  MCDwarfLineTableParams Params = Assembler.getDWARFLinetableParams();

  auto &LineTables = BC.getDwarfLineTables();

  // Bail out early so we don't switch to the debug_line section needlessly and
  // in doing so create an unnecessary (if empty) section.
  if (LineTables.empty())
    return;
  // In a v5 non-split line table, put the strings in a separate section.
  std::optional<MCDwarfLineStr> LineStr;
  ErrorOr<BinarySection &> LineStrSection =
      BC.getUniqueSectionByName(".debug_line_str");

  // Some versions of GCC output DWARF5 .debug_info, but DWARF4 or lower
  // .debug_line, so need to check if section exists.
  if (LineStrSection) {
    LineStr.emplace(*BC.Ctx);
    parseAndPopulateDebugLineStr(*LineStrSection, *LineStr, BC);
  }

  // Switch to the section where the table will be emitted into.
  Streamer.switchSection(BC.MOFI->getDwarfLineSection());

  const uint16_t DwarfVersion = BC.Ctx->getDwarfVersion();
  // Handle the rest of the Compile Units.
  for (auto &CUIDTablePair : LineTables) {
    Streamer.getContext().setDwarfVersion(
        CUIDTablePair.second.getDwarfVersion());
    CUIDTablePair.second.emitCU(&Streamer, Params, LineStr, BC);
  }

  // Resetting DWARF version for rest of the flow.
  BC.Ctx->setDwarfVersion(DwarfVersion);

  // Still need to write the section out for the ExecutionEngine, and temp in
  // memory object we are constructing.
  if (LineStr)
    LineStr->emitSection(&Streamer);
}

} // namespace bolt
} // namespace llvm
