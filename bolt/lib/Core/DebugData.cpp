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
#include "bolt/Utils/Utils.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
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
#include <limits>
#include <unordered_map>

#define DEBUG_TYPE "bolt-debug-info"

namespace opts {
extern llvm::cl::opt<unsigned> Verbosity;
} // namespace opts

namespace llvm {
class MCSymbol;

namespace bolt {

/// Finds attributes FormValue and Offset.
///
/// \param DIE die to look up in.
/// \param Index the attribute index to extract.
/// \return an optional AttrInfo with DWARFFormValue and Offset.
Optional<AttrInfo>
findAttributeInfo(const DWARFDie DIE,
                  const DWARFAbbreviationDeclaration *AbbrevDecl,
                  uint32_t Index) {
  const DWARFUnit &U = *DIE.getDwarfUnit();
  uint64_t Offset =
      AbbrevDecl->getAttributeOffsetFromIndex(Index, DIE.getOffset(), U);
  Optional<DWARFFormValue> Value =
      AbbrevDecl->getAttributeValueFromOffset(Index, Offset, U);
  if (!Value)
    return None;
  // AttributeSpec
  const DWARFAbbreviationDeclaration::AttributeSpec *AttrVal =
      AbbrevDecl->attributes().begin() + Index;
  uint32_t ValSize = 0;
  Optional<int64_t> ValSizeOpt = AttrVal->getByteSize(U);
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

  return AttrInfo{*Value, Offset, ValSize};
}

const DebugLineTableRowRef DebugLineTableRowRef::NULL_ROW{0, 0};

namespace {

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
uint64_t writeAddressRanges(raw_svector_ostream &Stream,
                            const DebugAddressRangesVector &AddressRanges,
                            const bool WriteRelativeRanges = false) {
  for (const DebugAddressRange &Range : AddressRanges) {
    support::endian::write(Stream, Range.LowPC, support::little);
    support::endian::write(
        Stream, WriteRelativeRanges ? Range.HighPC - Range.LowPC : Range.HighPC,
        support::little);
  }
  // Finish with 0 entries.
  support::endian::write(Stream, 0ULL, support::little);
  support::endian::write(Stream, 0ULL, support::little);
  return AddressRanges.size() * 16 + 16;
}

} // namespace

DebugRangesSectionWriter::DebugRangesSectionWriter() {
  RangesBuffer = std::make_unique<DebugBufferVector>();
  RangesStream = std::make_unique<raw_svector_ostream>(*RangesBuffer);

  // Add an empty range as the first entry;
  SectionOffset +=
      writeAddressRanges(*RangesStream.get(), DebugAddressRangesVector{});
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

uint64_t
DebugRangesSectionWriter::addRanges(const DebugAddressRangesVector &Ranges) {
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
    support::endian::write(RangesStream, Size, support::little);

    // Header field #2: version number, 2 as per the specification.
    support::endian::write(RangesStream, static_cast<uint16_t>(2),
                           support::little);

    assert(CUMap.count(Offset) && "Original CU offset is not found in CU Map");
    // Header field #3: debug info offset of the correspondent compile unit.
    support::endian::write(
        RangesStream, static_cast<uint32_t>(CUMap.find(Offset)->second.Offset),
        support::little);

    // Header field #4: address size.
    // 8 since we only write ELF64 binaries for now.
    RangesStream << char(8);

    // Header field #5: segment size of target architecture.
    RangesStream << char(0);

    // Padding before address table - 4 bytes in the 64-bit-pointer case.
    support::endian::write(RangesStream, static_cast<uint32_t>(0),
                           support::little);

    writeAddressRanges(RangesStream, AddressRanges, true);
  }
}

DebugAddrWriter::DebugAddrWriter(BinaryContext *Bc) { BC = Bc; }

void DebugAddrWriter::AddressForDWOCU::dump() {
  std::vector<IndexAddressPair> SortedMap(indexToAddressBegin(),
                                          indexToAdddessEnd());
  // Sorting address in increasing order of indices.
  std::sort(SortedMap.begin(), SortedMap.end(),
            [](const IndexAddressPair &A, const IndexAddressPair &B) {
              return A.first < B.first;
            });
  for (auto &Pair : SortedMap)
    dbgs() << Twine::utohexstr(Pair.second) << "\t" << Pair.first << "\n";
}
uint32_t DebugAddrWriter::getIndexFromAddress(uint64_t Address,
                                              uint64_t DWOId) {
  std::lock_guard<std::mutex> Lock(WriterMutex);
  if (!AddressMaps.count(DWOId))
    AddressMaps[DWOId] = AddressForDWOCU();

  AddressForDWOCU &Map = AddressMaps[DWOId];
  auto Entry = Map.find(Address);
  if (Entry == Map.end()) {
    auto Index = Map.getNextIndex();
    Entry = Map.insert(Address, Index).first;
  }
  return Entry->second;
}

// Case1) Address is not in map insert in to AddresToIndex and IndexToAddres
// Case2) Address is in the map but Index is higher or equal. Need to update
// IndexToAddrss. Case3) Address is in the map but Index is lower. Need to
// update AddressToIndex and IndexToAddress
void DebugAddrWriter::addIndexAddress(uint64_t Address, uint32_t Index,
                                      uint64_t DWOId) {
  std::lock_guard<std::mutex> Lock(WriterMutex);
  AddressForDWOCU &Map = AddressMaps[DWOId];
  auto Entry = Map.find(Address);
  if (Entry != Map.end()) {
    if (Entry->second > Index)
      Map.updateAddressToIndex(Address, Index);
    Map.updateIndexToAddrss(Address, Index);
  } else {
    Map.insert(Address, Index);
  }
}

AddressSectionBuffer DebugAddrWriter::finalize() {
  // Need to layout all sections within .debug_addr
  // Within each section sort Address by index.
  AddressSectionBuffer Buffer;
  raw_svector_ostream AddressStream(Buffer);
  for (std::unique_ptr<DWARFUnit> &CU : BC->DwCtx->compile_units()) {
    Optional<uint64_t> DWOId = CU->getDWOId();
    // Handling the case wehre debug information is a mix of Debug fission and
    // monolitic.
    if (!DWOId)
      continue;
    auto AM = AddressMaps.find(*DWOId);
    // Adding to map even if it did not contribute to .debug_addr.
    // The Skeleton CU will still have DW_AT_GNU_addr_base.
    DWOIdToOffsetMap[*DWOId] = Buffer.size();
    // If does not exist this CUs DWO section didn't contribute to .debug_addr.
    if (AM == AddressMaps.end())
      continue;
    std::vector<IndexAddressPair> SortedMap(AM->second.indexToAddressBegin(),
                                            AM->second.indexToAdddessEnd());
    // Sorting address in increasing order of indices.
    std::sort(SortedMap.begin(), SortedMap.end(),
              [](const IndexAddressPair &A, const IndexAddressPair &B) {
                return A.first < B.first;
              });

    uint8_t AddrSize = CU->getAddressByteSize();
    uint32_t Counter = 0;
    auto WriteAddress = [&](uint64_t Address) -> void {
      ++Counter;
      switch (AddrSize) {
      default:
        assert(false && "Address Size is invalid.");
        break;
      case 4:
        support::endian::write(AddressStream, static_cast<uint32_t>(Address),
                               support::little);
        break;
      case 8:
        support::endian::write(AddressStream, Address, support::little);
        break;
      }
    };

    for (const IndexAddressPair &Val : SortedMap) {
      while (Val.first > Counter)
        WriteAddress(0);
      WriteAddress(Val.second);
    }
  }

  return Buffer;
}

uint64_t DebugAddrWriter::getOffset(uint64_t DWOId) {
  auto Iter = DWOIdToOffsetMap.find(DWOId);
  assert(Iter != DWOIdToOffsetMap.end() &&
         "Offset in to.debug_addr was not found for DWO ID.");
  return Iter->second;
}

DebugLocWriter::DebugLocWriter(BinaryContext *BC) {
  LocBuffer = std::make_unique<DebugBufferVector>();
  LocStream = std::make_unique<raw_svector_ostream>(*LocBuffer);
}

void DebugLocWriter::addList(uint64_t AttrOffset,
                             DebugLocationsVector &&LocList) {
  if (LocList.empty()) {
    EmptyAttrLists.push_back(AttrOffset);
    return;
  }
  // Since there is a separate DebugLocWriter for each thread,
  // we don't need a lock to read the SectionOffset and update it.
  const uint32_t EntryOffset = SectionOffset;

  for (const DebugLocationEntry &Entry : LocList) {
    support::endian::write(*LocStream, static_cast<uint64_t>(Entry.LowPC),
                           support::little);
    support::endian::write(*LocStream, static_cast<uint64_t>(Entry.HighPC),
                           support::little);
    support::endian::write(*LocStream, static_cast<uint16_t>(Entry.Expr.size()),
                           support::little);
    *LocStream << StringRef(reinterpret_cast<const char *>(Entry.Expr.data()),
                            Entry.Expr.size());
    SectionOffset += 2 * 8 + 2 + Entry.Expr.size();
  }
  LocStream->write_zeros(16);
  SectionOffset += 16;
  LocListDebugInfoPatches.push_back({AttrOffset, EntryOffset});
}

void DebugLoclistWriter::addList(uint64_t AttrOffset,
                                 DebugLocationsVector &&LocList) {
  Patches.push_back({AttrOffset, std::move(LocList)});
}

std::unique_ptr<DebugBufferVector> DebugLocWriter::getBuffer() {
  return std::move(LocBuffer);
}

// DWARF 4: 2.6.2
void DebugLocWriter::finalize(uint64_t SectionOffset,
                              SimpleBinaryPatcher &DebugInfoPatcher) {
  for (const auto LocListDebugInfoPatchType : LocListDebugInfoPatches) {
    uint64_t Offset = SectionOffset + LocListDebugInfoPatchType.LocListOffset;
    DebugInfoPatcher.addLE32Patch(LocListDebugInfoPatchType.DebugInfoAttrOffset,
                                  Offset);
  }

  for (uint64_t DebugInfoAttrOffset : EmptyAttrLists)
    DebugInfoPatcher.addLE32Patch(DebugInfoAttrOffset,
                                  DebugLocWriter::EmptyListOffset);
}

void DebugLoclistWriter::finalize(uint64_t SectionOffset,
                                  SimpleBinaryPatcher &DebugInfoPatcher) {
  for (LocPatch &Patch : Patches) {
    if (Patch.LocList.empty()) {
      DebugInfoPatcher.addLE32Patch(Patch.AttrOffset,
                                    DebugLocWriter::EmptyListOffset);
      continue;
    }
    const uint32_t EntryOffset = LocBuffer->size();
    for (const DebugLocationEntry &Entry : Patch.LocList) {
      support::endian::write(*LocStream,
                             static_cast<uint8_t>(dwarf::DW_LLE_startx_length),
                             support::little);
      uint32_t Index = AddrWriter->getIndexFromAddress(Entry.LowPC, DWOId);
      encodeULEB128(Index, *LocStream);

      // TODO: Support DWARF5
      support::endian::write(*LocStream,
                             static_cast<uint32_t>(Entry.HighPC - Entry.LowPC),
                             support::little);
      support::endian::write(*LocStream,
                             static_cast<uint16_t>(Entry.Expr.size()),
                             support::little);
      *LocStream << StringRef(reinterpret_cast<const char *>(Entry.Expr.data()),
                              Entry.Expr.size());
    }
    support::endian::write(*LocStream,
                           static_cast<uint8_t>(dwarf::DW_LLE_end_of_list),
                           support::little);
    DebugInfoPatcher.addLE32Patch(Patch.AttrOffset, EntryOffset);
    clearList(Patch.LocList);
  }
  clearList(Patches);
}

DebugAddrWriter *DebugLoclistWriter::AddrWriter = nullptr;

void DebugInfoBinaryPatcher::addUnitBaseOffsetLabel(uint64_t Offset) {
  Offset -= DWPUnitOffset;
  std::lock_guard<std::mutex> Lock(WriterMutex);
  DebugPatches.emplace_back(new DWARFUnitOffsetBaseLabel(Offset));
}

void DebugInfoBinaryPatcher::addDestinationReferenceLabel(uint64_t Offset) {
  Offset -= DWPUnitOffset;
  std::lock_guard<std::mutex> Lock(WriterMutex);
  auto RetVal = DestinationLabels.insert(Offset);
  if (!RetVal.second)
    return;

  DebugPatches.emplace_back(new DestinationReferenceLabel(Offset));
}

static std::string encodeLE(size_t ByteSize, uint64_t NewValue) {
  std::string LE64(ByteSize, 0);
  for (size_t I = 0; I < ByteSize; ++I) {
    LE64[I] = NewValue & 0xff;
    NewValue >>= 8;
  }
  return LE64;
}

void DebugInfoBinaryPatcher::insertNewEntry(const DWARFDie &DIE,
                                            uint32_t Value) {
  std::string StrValue = encodeLE(4, Value);
  insertNewEntry(DIE, std::move(StrValue));
}

void DebugInfoBinaryPatcher::insertNewEntry(const DWARFDie &DIE,
                                            std::string &&Value) {
  const DWARFAbbreviationDeclaration *AbbrevDecl =
      DIE.getAbbreviationDeclarationPtr();

  // In case this DIE has no attributes.
  uint32_t Offset = DIE.getOffset() + 1;
  size_t NumOfAttributes = AbbrevDecl->getNumAttributes();
  if (NumOfAttributes) {
    Optional<AttrInfo> Val =
        findAttributeInfo(DIE, AbbrevDecl, NumOfAttributes - 1);
    assert(Val && "Invalid Value.");

    Offset = Val->Offset + Val->Size - DWPUnitOffset;
  }
  std::lock_guard<std::mutex> Lock(WriterMutex);
  DebugPatches.emplace_back(new NewDebugEntry(Offset, std::move(Value)));
}

void DebugInfoBinaryPatcher::addReferenceToPatch(uint64_t Offset,
                                                 uint32_t DestinationOffset,
                                                 uint32_t OldValueSize,
                                                 dwarf::Form Form) {
  Offset -= DWPUnitOffset;
  DestinationOffset -= DWPUnitOffset;
  std::lock_guard<std::mutex> Lock(WriterMutex);
  DebugPatches.emplace_back(
      new DebugPatchReference(Offset, OldValueSize, DestinationOffset, Form));
}

void DebugInfoBinaryPatcher::addUDataPatch(uint64_t Offset, uint64_t NewValue,
                                           uint32_t OldValueSize) {
  Offset -= DWPUnitOffset;
  std::lock_guard<std::mutex> Lock(WriterMutex);
  DebugPatches.emplace_back(
      new DebugPatchVariableSize(Offset, OldValueSize, NewValue));
}

void DebugInfoBinaryPatcher::addLE64Patch(uint64_t Offset, uint64_t NewValue) {
  Offset -= DWPUnitOffset;
  std::lock_guard<std::mutex> Lock(WriterMutex);
  DebugPatches.emplace_back(new DebugPatch64(Offset, NewValue));
}

void DebugInfoBinaryPatcher::addLE32Patch(uint64_t Offset, uint32_t NewValue,
                                          uint32_t OldValueSize) {
  Offset -= DWPUnitOffset;
  std::lock_guard<std::mutex> Lock(WriterMutex);
  if (OldValueSize == 4)
    DebugPatches.emplace_back(new DebugPatch32(Offset, NewValue));
  else if (OldValueSize == 8)
    DebugPatches.emplace_back(new DebugPatch64to32(Offset, NewValue));
  else
    DebugPatches.emplace_back(
        new DebugPatch32GenericSize(Offset, NewValue, OldValueSize));
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

CUOffsetMap DebugInfoBinaryPatcher::computeNewOffsets(DWARFContext &DWCtx,
                                                      bool IsDWOContext) {
  CUOffsetMap CUMap;
  std::sort(DebugPatches.begin(), DebugPatches.end(),
            [](const UniquePatchPtrType &V1, const UniquePatchPtrType &V2) {
              if (V1.get()->Offset == V2.get()->Offset) {
                if (V1->Kind == DebugPatchKind::NewDebugEntry &&
                    V2->Kind == DebugPatchKind::NewDebugEntry)
                  return reinterpret_cast<const NewDebugEntry *>(V1.get())
                             ->CurrentOrder <
                         reinterpret_cast<const NewDebugEntry *>(V2.get())
                             ->CurrentOrder;

                // This is a case where we are modifying first entry of next
                // DIE, and adding a new one.
                return V1->Kind == DebugPatchKind::NewDebugEntry;
              }
              return V1.get()->Offset < V2.get()->Offset;
            });

  DWARFUnitVector::compile_unit_range CompileUnits =
      IsDWOContext ? DWCtx.dwo_compile_units() : DWCtx.compile_units();

  for (const std::unique_ptr<DWARFUnit> &CU : CompileUnits)
    CUMap[CU->getOffset()] = {static_cast<uint32_t>(CU->getOffset()),
                              static_cast<uint32_t>(CU->getLength())};

  // Calculating changes in .debug_info size from Patches to build a map of old
  // to updated reference destination offsets.
  uint32_t PreviousOffset = 0;
  int32_t PreviousChangeInSize = 0;
  for (UniquePatchPtrType &PatchBase : DebugPatches) {
    Patch *P = PatchBase.get();
    switch (P->Kind) {
    default:
      continue;
    case DebugPatchKind::PatchValue64to32: {
      PreviousChangeInSize -= 4;
      break;
    }
    case DebugPatchKind::PatchValue32GenericSize: {
      DebugPatch32GenericSize *DPVS =
          reinterpret_cast<DebugPatch32GenericSize *>(P);
      PreviousChangeInSize += 4 - DPVS->OldValueSize;
      break;
    }
    case DebugPatchKind::PatchValueVariable: {
      DebugPatchVariableSize *DPV =
          reinterpret_cast<DebugPatchVariableSize *>(P);
      std::string Temp;
      raw_string_ostream OS(Temp);
      encodeULEB128(DPV->Value, OS);
      PreviousChangeInSize += Temp.size() - DPV->OldValueSize;
      break;
    }
    case DebugPatchKind::DestinationReferenceLabel: {
      DestinationReferenceLabel *DRL =
          reinterpret_cast<DestinationReferenceLabel *>(P);
      OldToNewOffset[DRL->Offset] =
          DRL->Offset + ChangeInSize + PreviousChangeInSize;
      break;
    }
    case DebugPatchKind::ReferencePatchValue: {
      // This doesn't look to be a common case, so will always encode as 4 bytes
      // to reduce algorithmic complexity.
      DebugPatchReference *RDP = reinterpret_cast<DebugPatchReference *>(P);
      if (RDP->PatchInfo.IndirectRelative) {
        PreviousChangeInSize += 4 - RDP->PatchInfo.OldValueSize;
        assert(RDP->PatchInfo.OldValueSize <= 4 &&
               "Variable encoding reference greater than 4 bytes.");
      }
      break;
    }
    case DebugPatchKind::DWARFUnitOffsetBaseLabel: {
      DWARFUnitOffsetBaseLabel *BaseLabel =
          reinterpret_cast<DWARFUnitOffsetBaseLabel *>(P);
      uint32_t CUOffset = BaseLabel->Offset;
      ChangeInSize += PreviousChangeInSize;
      uint32_t CUOffsetUpdate = CUOffset + ChangeInSize;
      CUMap[CUOffset].Offset = CUOffsetUpdate;
      CUMap[PreviousOffset].Length += PreviousChangeInSize;
      PreviousChangeInSize = 0;
      PreviousOffset = CUOffset;
      break;
    }
    case DebugPatchKind::NewDebugEntry: {
      NewDebugEntry *NDE = reinterpret_cast<NewDebugEntry *>(P);
      PreviousChangeInSize += NDE->Value.size();
      break;
    }
    }
  }
  CUMap[PreviousOffset].Length += PreviousChangeInSize;
  return CUMap;
}
uint32_t DebugInfoBinaryPatcher::NewDebugEntry::OrderCounter = 0;

std::string DebugInfoBinaryPatcher::patchBinary(StringRef BinaryContents) {
  std::string NewBinaryContents;
  NewBinaryContents.reserve(BinaryContents.size() + ChangeInSize);
  uint32_t StartOffset = 0;
  uint32_t DwarfUnitBaseOffset = 0;
  uint32_t OldValueSize = 0;
  uint32_t Offset = 0;
  std::string ByteSequence;
  std::vector<std::pair<uint32_t, uint32_t>> LengthPatches;
  // Wasting one entry to avoid checks for first.
  LengthPatches.push_back({0, 0});

  // Applying all the patches replacing current entry.
  // This might change the size of .debug_info section.
  for (const UniquePatchPtrType &PatchBase : DebugPatches) {
    Patch *P = PatchBase.get();
    switch (P->Kind) {
    default:
      continue;
    case DebugPatchKind::ReferencePatchValue: {
      DebugPatchReference *RDP = reinterpret_cast<DebugPatchReference *>(P);
      uint32_t DestinationOffset = RDP->DestinationOffset;
      assert(OldToNewOffset.count(DestinationOffset) &&
             "Destination Offset for reference not updated.");
      uint32_t UpdatedOffset = OldToNewOffset[DestinationOffset];
      Offset = RDP->Offset;
      OldValueSize = RDP->PatchInfo.OldValueSize;
      if (RDP->PatchInfo.DirectRelative) {
        UpdatedOffset -= DwarfUnitBaseOffset;
        ByteSequence = encodeLE(OldValueSize, UpdatedOffset);
        // In theory reference for DW_FORM_ref{1,2,4,8} can be right on the edge
        // and overflow if later debug information grows.
        if (ByteSequence.size() > OldValueSize)
          errs() << "BOLT-ERROR: Relative reference of size "
                 << Twine::utohexstr(OldValueSize)
                 << " overflows with the new encoding.\n";
      } else if (RDP->PatchInfo.DirectAbsolute) {
        ByteSequence = encodeLE(OldValueSize, UpdatedOffset);
      } else if (RDP->PatchInfo.IndirectRelative) {
        UpdatedOffset -= DwarfUnitBaseOffset;
        ByteSequence.clear();
        raw_string_ostream OS(ByteSequence);
        encodeULEB128(UpdatedOffset, OS, 4);
      } else {
        llvm_unreachable("Invalid Reference form.");
      }
      break;
    }
    case DebugPatchKind::PatchValue32: {
      DebugPatch32 *P32 = reinterpret_cast<DebugPatch32 *>(P);
      Offset = P32->Offset;
      OldValueSize = 4;
      ByteSequence = encodeLE(4, P32->Value);
      break;
    }
    case DebugPatchKind::PatchValue64to32: {
      DebugPatch64to32 *P64to32 = reinterpret_cast<DebugPatch64to32 *>(P);
      Offset = P64to32->Offset;
      OldValueSize = 8;
      ByteSequence = encodeLE(4, P64to32->Value);
      break;
    }
    case DebugPatchKind::PatchValue32GenericSize: {
      DebugPatch32GenericSize *DPVS =
          reinterpret_cast<DebugPatch32GenericSize *>(P);
      Offset = DPVS->Offset;
      OldValueSize = DPVS->OldValueSize;
      ByteSequence = encodeLE(4, DPVS->Value);
      break;
    }
    case DebugPatchKind::PatchValueVariable: {
      DebugPatchVariableSize *PV =
          reinterpret_cast<DebugPatchVariableSize *>(P);
      Offset = PV->Offset;
      OldValueSize = PV->OldValueSize;
      ByteSequence.clear();
      raw_string_ostream OS(ByteSequence);
      encodeULEB128(PV->Value, OS);
      break;
    }
    case DebugPatchKind::PatchValue64: {
      DebugPatch64 *P64 = reinterpret_cast<DebugPatch64 *>(P);
      Offset = P64->Offset;
      OldValueSize = 8;
      ByteSequence = encodeLE(8, P64->Value);
      break;
    }
    case DebugPatchKind::DWARFUnitOffsetBaseLabel: {
      DWARFUnitOffsetBaseLabel *BaseLabel =
          reinterpret_cast<DWARFUnitOffsetBaseLabel *>(P);
      Offset = BaseLabel->Offset;
      OldValueSize = 0;
      ByteSequence.clear();
      auto &Patch = LengthPatches.back();
      // Length to copy between last patch entry and next compile unit.
      uint32_t RemainingLength = Offset - StartOffset;
      uint32_t NewCUOffset = NewBinaryContents.size() + RemainingLength;
      DwarfUnitBaseOffset = NewCUOffset;
      // Length of previous CU = This CU Offset - sizeof(length) - last CU
      // Offset.
      Patch.second = NewCUOffset - 4 - Patch.first;
      LengthPatches.push_back({NewCUOffset, 0});
      break;
    }
    case DebugPatchKind::NewDebugEntry: {
      NewDebugEntry *NDE = reinterpret_cast<NewDebugEntry *>(P);
      Offset = NDE->Offset;
      OldValueSize = 0;
      ByteSequence = NDE->Value;
      break;
    }
    }

    assert((P->Kind == DebugPatchKind::NewDebugEntry ||
            Offset + ByteSequence.size() <= BinaryContents.size()) &&
           "Applied patch runs over binary size.");
    uint32_t Length = Offset - StartOffset;
    NewBinaryContents.append(BinaryContents.substr(StartOffset, Length).data(),
                             Length);
    NewBinaryContents.append(ByteSequence.data(), ByteSequence.size());
    StartOffset = Offset + OldValueSize;
  }
  uint32_t Length = BinaryContents.size() - StartOffset;
  NewBinaryContents.append(BinaryContents.substr(StartOffset, Length).data(),
                           Length);
  DebugPatches.clear();

  // Patching lengths of CUs
  auto &Patch = LengthPatches.back();
  Patch.second = NewBinaryContents.size() - 4 - Patch.first;
  for (uint32_t J = 1, Size = LengthPatches.size(); J < Size; ++J) {
    const auto &Patch = LengthPatches[J];
    ByteSequence = encodeLE(4, Patch.second);
    Offset = Patch.first;
    for (uint64_t I = 0, Size = ByteSequence.size(); I < Size; ++I)
      NewBinaryContents[Offset + I] = ByteSequence[I];
  }

  return NewBinaryContents;
}

void DebugStrWriter::create() {
  StrBuffer = std::make_unique<DebugStrBufferVector>();
  StrStream = std::make_unique<raw_svector_ostream>(*StrBuffer);
}

void DebugStrWriter::initialize() {
  auto StrSection = BC->DwCtx->getDWARFObj().getStrSection();
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

void DebugAbbrevWriter::addUnitAbbreviations(DWARFUnit &Unit) {
  const DWARFAbbreviationDeclarationSet *Abbrevs = Unit.getAbbreviations();
  if (!Abbrevs)
    return;

  const PatchesTy &UnitPatches = Patches[&Unit];
  const AbbrevEntryTy &AbbrevEntries = NewAbbrevEntries[&Unit];

  // We are duplicating abbrev sections, to handle the case where for one CU we
  // modify it, but for another we don't.
  auto UnitDataPtr = std::make_unique<AbbrevData>();
  AbbrevData &UnitData = *UnitDataPtr.get();
  UnitData.Buffer = std::make_unique<DebugBufferVector>();
  UnitData.Stream = std::make_unique<raw_svector_ostream>(*UnitData.Buffer);

  raw_svector_ostream &OS = *UnitData.Stream.get();

  // Returns true if AbbrevData is re-used, false otherwise.
  auto hashAndAddAbbrev = [&](StringRef AbbrevData) -> bool {
    llvm::SHA1 Hasher;
    Hasher.update(AbbrevData);
    StringRef Key = Hasher.final();
    auto Iter = AbbrevDataCache.find(Key);
    if (Iter != AbbrevDataCache.end()) {
      UnitsAbbrevData[&Unit] = Iter->second.get();
      return true;
    }
    AbbrevDataCache[Key] = std::move(UnitDataPtr);
    UnitsAbbrevData[&Unit] = &UnitData;
    return false;
  };
  // Take a fast path if there are no patches to apply. Simply copy the original
  // contents.
  if (UnitPatches.empty() && AbbrevEntries.empty()) {
    StringRef AbbrevSectionContents =
        Unit.isDWOUnit() ? Unit.getContext().getDWARFObj().getAbbrevDWOSection()
                         : Unit.getContext().getDWARFObj().getAbbrevSection();
    StringRef AbbrevContents;

    const DWARFUnitIndex &CUIndex = Unit.getContext().getCUIndex();
    if (!CUIndex.getRows().empty()) {
      // Handle DWP section contribution.
      const DWARFUnitIndex::Entry *DWOEntry =
          CUIndex.getFromHash(*Unit.getDWOId());
      if (!DWOEntry)
        return;

      const DWARFUnitIndex::Entry::SectionContribution *DWOContrubution =
          DWOEntry->getContribution(DWARFSectionKind::DW_SECT_ABBREV);
      AbbrevContents = AbbrevSectionContents.substr(DWOContrubution->Offset,
                                                    DWOContrubution->Length);
    } else if (!Unit.isDWOUnit()) {
      const uint64_t StartOffset = Unit.getAbbreviationsOffset();

      // We know where the unit's abbreviation set starts, but not where it ends
      // as such data is not readily available. Hence, we have to build a sorted
      // list of start addresses and find the next starting address to determine
      // the set boundaries.
      //
      // FIXME: if we had a full access to DWARFDebugAbbrev::AbbrDeclSets
      // we wouldn't have to build our own sorted list for the quick lookup.
      if (AbbrevSetOffsets.empty()) {
        for_each(
            *Unit.getContext().getDebugAbbrev(),
            [&](const std::pair<uint64_t, DWARFAbbreviationDeclarationSet> &P) {
              AbbrevSetOffsets.push_back(P.first);
            });
        sort(AbbrevSetOffsets);
      }
      auto It = upper_bound(AbbrevSetOffsets, StartOffset);
      const uint64_t EndOffset =
          It == AbbrevSetOffsets.end() ? AbbrevSectionContents.size() : *It;
      AbbrevContents = AbbrevSectionContents.slice(StartOffset, EndOffset);
    } else {
      // For DWO unit outside of DWP, we expect the entire section to hold
      // abbreviations for this unit only.
      AbbrevContents = AbbrevSectionContents;
    }

    if (!hashAndAddAbbrev(AbbrevContents)) {
      OS.reserveExtraSpace(AbbrevContents.size());
      OS << AbbrevContents;
    }
    return;
  }

  for (auto I = Abbrevs->begin(), E = Abbrevs->end(); I != E; ++I) {
    const DWARFAbbreviationDeclaration &Abbrev = *I;
    auto Patch = UnitPatches.find(&Abbrev);

    encodeULEB128(Abbrev.getCode(), OS);
    encodeULEB128(Abbrev.getTag(), OS);
    encodeULEB128(Abbrev.hasChildren(), OS);
    for (const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec :
         Abbrev.attributes()) {
      if (Patch != UnitPatches.end()) {
        bool Patched = false;
        // Patches added later take a precedence over earlier ones.
        for (auto I = Patch->second.rbegin(), E = Patch->second.rend(); I != E;
             ++I) {
          if (I->OldAttr != AttrSpec.Attr)
            continue;

          encodeULEB128(I->NewAttr, OS);
          encodeULEB128(I->NewAttrForm, OS);
          Patched = true;
          break;
        }
        if (Patched)
          continue;
      }

      encodeULEB128(AttrSpec.Attr, OS);
      encodeULEB128(AttrSpec.Form, OS);
      if (AttrSpec.isImplicitConst())
        encodeSLEB128(AttrSpec.getImplicitConstValue(), OS);
    }
    const auto Entries = AbbrevEntries.find(&Abbrev);
    // Adding new Abbrevs for inserted entries.
    if (Entries != AbbrevEntries.end()) {
      for (const AbbrevEntry &Entry : Entries->second) {
        encodeULEB128(Entry.Attr, OS);
        encodeULEB128(Entry.Form, OS);
      }
    }
    encodeULEB128(0, OS);
    encodeULEB128(0, OS);
  }
  encodeULEB128(0, OS);

  hashAndAddAbbrev(OS.str());
}

std::unique_ptr<DebugBufferVector> DebugAbbrevWriter::finalize() {
  // Used to create determinism for writing out abbrevs.
  std::vector<AbbrevData *> Abbrevs;
  if (DWOId) {
    // We expect abbrev_offset to always be zero for DWO units as there
    // should be one CU per DWO, and TUs should share the same abbreviation
    // set with the CU.
    // For DWP AbbreviationsOffset is an Abbrev contribution in the DWP file, so
    // can be none zero. Thus we are skipping the check for DWP.
    bool IsDWP = !Context.getCUIndex().getRows().empty();
    if (!IsDWP) {
      for (const std::unique_ptr<DWARFUnit> &Unit : Context.dwo_units()) {
        if (Unit->getAbbreviationsOffset() != 0) {
          errs() << "BOLT-ERROR: detected DWO unit with non-zero abbr_offset. "
                    "Unable to update debug info.\n";
          exit(1);
        }
      }
    }

    DWARFUnit *Unit = Context.getDWOCompileUnitForHash(*DWOId);
    // Issue abbreviations for the DWO CU only.
    addUnitAbbreviations(*Unit);
    AbbrevData *Abbrev = UnitsAbbrevData[Unit];
    Abbrevs.push_back(Abbrev);
  } else {
    Abbrevs.reserve(Context.getNumCompileUnits() + Context.getNumTypeUnits());
    std::unordered_set<AbbrevData *> ProcessedAbbrevs;
    // Add abbreviations from compile and type non-DWO units.
    for (const std::unique_ptr<DWARFUnit> &Unit : Context.normal_units()) {
      addUnitAbbreviations(*Unit);
      AbbrevData *Abbrev = UnitsAbbrevData[Unit.get()];
      if (!ProcessedAbbrevs.insert(Abbrev).second)
        continue;
      Abbrevs.push_back(Abbrev);
    }
  }

  DebugBufferVector ReturnBuffer;
  // Pre-calculate the total size of abbrev section.
  uint64_t Size = 0;
  for (const AbbrevData *UnitData : Abbrevs)
    Size += UnitData->Buffer->size();

  ReturnBuffer.reserve(Size);

  uint64_t Pos = 0;
  for (AbbrevData *UnitData : Abbrevs) {
    ReturnBuffer.append(*UnitData->Buffer);
    UnitData->Offset = Pos;
    Pos += UnitData->Buffer->size();

    UnitData->Buffer.reset();
    UnitData->Stream.reset();
  }

  return std::make_unique<DebugBufferVector>(ReturnBuffer);
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
        MCOS->getContext().getDwarfVersion() >= 4) {
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
                            Optional<MCDwarfLineStr> &LineStr,
                            BinaryContext &BC) const {
  if (!RawData.empty()) {
    assert(MCLineSections.getMCLineEntries().empty() &&
           InputSequences.empty() &&
           "cannot combine raw data with new line entries");
    MCOS->emitLabel(getLabel());
    MCOS->emitBytes(RawData);

    // Emit fake relocation for RuntimeDyld to always allocate the section.
    //
    // FIXME: remove this once RuntimeDyld stops skipping allocatable sections
    //        without relocations.
    MCOS->emitRelocDirective(
        *MCConstantExpr::create(0, *BC.Ctx), "BFD_RELOC_NONE",
        MCSymbolRefExpr::create(getLabel(), *BC.Ctx), SMLoc(), *BC.STI);

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
  Optional<MCDwarfLineStr> LineStr(None);
  if (BC.Ctx->getDwarfVersion() >= 5)
    LineStr = MCDwarfLineStr(*BC.Ctx);

  // Switch to the section where the table will be emitted into.
  Streamer.SwitchSection(BC.MOFI->getDwarfLineSection());

  // Handle the rest of the Compile Units.
  for (auto &CUIDTablePair : LineTables) {
    CUIDTablePair.second.emitCU(&Streamer, Params, LineStr, BC);
  }
}

} // namespace bolt
} // namespace llvm
