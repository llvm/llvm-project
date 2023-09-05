//===- bolt/Core/DebugData.h - Debugging information handling ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declaration of classes that represent and serialize
// DWARF-related entities.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_DEBUG_DATA_H
#define BOLT_CORE_DEBUG_DATA_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define DWARF2_FLAG_END_SEQUENCE (1 << 4)

namespace llvm {

namespace bolt {

class DIEBuilder;
struct AttrInfo {
  DWARFFormValue V;
  const DWARFAbbreviationDeclaration *AbbrevDecl;
  uint64_t Offset;
  uint32_t Size; // Size of the attribute.
};

/// Finds attributes FormValue and Offset.
///
/// \param DIE die to look up in.
/// \param AbbrevDecl abbrev declaration for the die.
/// \param Index an index in Abbrev declaration entry.
std::optional<AttrInfo>
findAttributeInfo(const DWARFDie DIE,
                  const DWARFAbbreviationDeclaration *AbbrevDecl,
                  uint32_t Index);

/// Finds attributes FormValue and Offset.
///
/// \param DIE die to look up in.
/// \param Attr the attribute to extract.
/// \return an optional AttrInfo with DWARFFormValue and Offset.
std::optional<AttrInfo> findAttributeInfo(const DWARFDie DIE,
                                          dwarf::Attribute Attr);

// DWARF5 Header in order of encoding.
// Types represent encodnig sizes.
using UnitLengthType = uint32_t;
using VersionType = uint16_t;
using AddressSizeType = uint8_t;
using SegmentSelectorType = uint8_t;
using OffsetEntryCountType = uint32_t;
/// Get DWARF5 Header size.
/// Rangelists and Loclists have the same header.
constexpr uint32_t getDWARF5RngListLocListHeaderSize() {
  return sizeof(UnitLengthType) + sizeof(VersionType) +
         sizeof(AddressSizeType) + sizeof(SegmentSelectorType) +
         sizeof(OffsetEntryCountType);
}

class BinaryContext;

/// Address range representation. Takes less space than DWARFAddressRange.
struct DebugAddressRange {
  uint64_t LowPC{0};
  uint64_t HighPC{0};

  DebugAddressRange() = default;

  DebugAddressRange(uint64_t LowPC, uint64_t HighPC)
      : LowPC(LowPC), HighPC(HighPC) {}
};

static inline bool operator<(const DebugAddressRange &LHS,
                             const DebugAddressRange &RHS) {
  return std::tie(LHS.LowPC, LHS.HighPC) < std::tie(RHS.LowPC, RHS.HighPC);
}

/// DebugAddressRangesVector - represents a set of absolute address ranges.
using DebugAddressRangesVector = SmallVector<DebugAddressRange, 2>;

/// Address range with location used by .debug_loc section.
/// More compact than DWARFLocationEntry and uses absolute addresses.
struct DebugLocationEntry {
  uint64_t LowPC;
  uint64_t HighPC;
  SmallVector<uint8_t, 4> Expr;
};

using DebugLocationsVector = SmallVector<DebugLocationEntry, 4>;

/// References a row in a DWARFDebugLine::LineTable by the DWARF
/// Context index of the DWARF Compile Unit that owns the Line Table and the row
/// index. This is tied to our IR during disassembly so that we can later update
/// .debug_line information. RowIndex has a base of 1, which means a RowIndex
/// of 1 maps to the first row of the line table and a RowIndex of 0 is invalid.
struct DebugLineTableRowRef {
  uint32_t DwCompileUnitIndex;
  uint32_t RowIndex;

  const static DebugLineTableRowRef NULL_ROW;

  bool operator==(const DebugLineTableRowRef &Rhs) const {
    return DwCompileUnitIndex == Rhs.DwCompileUnitIndex &&
           RowIndex == Rhs.RowIndex;
  }

  bool operator!=(const DebugLineTableRowRef &Rhs) const {
    return !(*this == Rhs);
  }

  static DebugLineTableRowRef fromSMLoc(const SMLoc &Loc) {
    union {
      decltype(Loc.getPointer()) Ptr;
      DebugLineTableRowRef Ref;
    } U;
    U.Ptr = Loc.getPointer();
    return U.Ref;
  }

  SMLoc toSMLoc() const {
    union {
      decltype(SMLoc().getPointer()) Ptr;
      DebugLineTableRowRef Ref;
    } U;
    U.Ref = *this;
    return SMLoc::getFromPointer(U.Ptr);
  }
};

/// Common buffer vector used for debug info handling.
using DebugBufferVector = SmallVector<char, 16>;

/// Map of old CU offset to new offset and length.
struct CUInfo {
  uint32_t Offset;
  uint32_t Length;
};
using CUOffsetMap = std::map<uint32_t, CUInfo>;

enum class RangesWriterKind { DebugRangesWriter, DebugRangeListsWriter };
/// Serializes the .debug_ranges DWARF section.
class DebugRangesSectionWriter {
public:
  DebugRangesSectionWriter();

  DebugRangesSectionWriter(RangesWriterKind K) : Kind(K){};

  virtual ~DebugRangesSectionWriter(){};

  /// Add ranges with caching.
  virtual uint64_t
  addRanges(DebugAddressRangesVector &&Ranges,
            std::map<DebugAddressRangesVector, uint64_t> &CachedRanges);

  /// Add ranges and return offset into section.
  virtual uint64_t addRanges(DebugAddressRangesVector &Ranges);

  /// Returns an offset of an empty address ranges list that is always written
  /// to .debug_ranges
  uint64_t getEmptyRangesOffset() const { return EmptyRangesOffset; }

  /// Returns the SectionOffset.
  uint64_t getSectionOffset();

  /// Returns a buffer containing Ranges.
  virtual std::unique_ptr<DebugBufferVector> releaseBuffer() {
    return std::move(RangesBuffer);
  }

  RangesWriterKind getKind() const { return Kind; }

  static bool classof(const DebugRangesSectionWriter *Writer) {
    return Writer->getKind() == RangesWriterKind::DebugRangesWriter;
  }

  /// Writes out range lists for a current CU being processed.
  void virtual finalizeSection(){};

  /// Needs to be invoked before each \p CU is processed.
  void virtual initSection(DWARFUnit &CU){};

protected:
  std::unique_ptr<DebugBufferVector> RangesBuffer;

  std::unique_ptr<raw_svector_ostream> RangesStream;

  std::mutex WriterMutex;

  /// Current offset in the section (updated as new entries are written).
  /// Starts with 16 since the first 16 bytes are reserved for an empty range.
  uint32_t SectionOffset{0};

  /// Offset of an empty address ranges list.
  static constexpr uint64_t EmptyRangesOffset{0};

private:
  RangesWriterKind Kind;
};

class DebugAddrWriter;
class DebugRangeListsSectionWriter : public DebugRangesSectionWriter {
public:
  DebugRangeListsSectionWriter()
      : DebugRangesSectionWriter(RangesWriterKind::DebugRangeListsWriter) {
    RangesBuffer = std::make_unique<DebugBufferVector>();
    RangesStream = std::make_unique<raw_svector_ostream>(*RangesBuffer);
  };
  virtual ~DebugRangeListsSectionWriter(){};

  static void setAddressWriter(DebugAddrWriter *AddrW) { AddrWriter = AddrW; }

  /// Add ranges with caching.
  uint64_t addRanges(
      DebugAddressRangesVector &&Ranges,
      std::map<DebugAddressRangesVector, uint64_t> &CachedRanges) override;

  /// Add ranges and return offset into section.
  uint64_t addRanges(DebugAddressRangesVector &Ranges) override;

  std::unique_ptr<DebugBufferVector> releaseBuffer() override {
    return std::move(RangesBuffer);
  }

  /// Needs to be invoked before each \p CU is processed.
  void initSection(DWARFUnit &CU) override;

  /// Writes out range lists for a current CU being processed.
  void finalizeSection() override;

  // Returns true if section is empty.
  bool empty() { return RangesBuffer->empty(); }

  static bool classof(const DebugRangesSectionWriter *Writer) {
    return Writer->getKind() == RangesWriterKind::DebugRangeListsWriter;
  }

private:
  static DebugAddrWriter *AddrWriter;
  /// Used to find unique CU ID.
  DWARFUnit *CU;
  /// Current relative offset of range list entry within this CUs rangelist
  /// body.
  uint32_t CurrentOffset{0};
  /// Contains relative offset of each range list entry.
  SmallVector<uint32_t, 1> RangeEntries;

  std::unique_ptr<DebugBufferVector> CUBodyBuffer;
  std::unique_ptr<raw_svector_ostream> CUBodyStream;
};

/// Serializes the .debug_aranges DWARF section.
class DebugARangesSectionWriter {
public:
  /// Add ranges for CU matching \p CUOffset.
  void addCURanges(uint64_t CUOffset, DebugAddressRangesVector &&Ranges);

  /// Writes .debug_aranges with the added ranges to the MCObjectWriter.
  /// Takes in \p RangesStream to write into, and \p CUMap which maps CU
  /// original offsets to new ones.
  void writeARangesSection(raw_svector_ostream &RangesStream,
                           const CUOffsetMap &CUMap) const;

  /// Resets the writer to a clear state.
  void reset() { CUAddressRanges.clear(); }

  /// Map DWARFCompileUnit index to ranges.
  using CUAddressRangesType = std::map<uint64_t, DebugAddressRangesVector>;

  /// Return ranges for a given CU.
  const CUAddressRangesType &getCUAddressRanges() const {
    return CUAddressRanges;
  }

private:
  /// Map from compile unit offset to the list of address intervals that belong
  /// to that compile unit. Each interval is a pair
  /// (first address, interval size).
  CUAddressRangesType CUAddressRanges;

  std::mutex CUAddressRangesMutex;
};

using IndexAddressPair = std::pair<uint32_t, uint64_t>;
using AddressToIndexMap = std::unordered_map<uint64_t, uint32_t>;
using IndexToAddressMap = std::unordered_map<uint32_t, uint64_t>;
using AddressSectionBuffer = SmallVector<char, 4>;
class DebugAddrWriter {
public:
  DebugAddrWriter() = delete;
  DebugAddrWriter(BinaryContext *BC_);
  virtual ~DebugAddrWriter(){};
  /// Given an address returns an index in .debug_addr.
  /// Adds Address to map.
  uint32_t getIndexFromAddress(uint64_t Address, DWARFUnit &CU);

  /// Write out entries in to .debug_addr section for CUs.
  virtual void update(DIEBuilder &DIEBlder, DWARFUnit &CUs);

  /// Return buffer with all the entries in .debug_addr already writen out using
  /// update(...).
  virtual AddressSectionBuffer &finalize() { return *Buffer; }

  /// Returns False if .debug_addr section was created..
  bool isInitialized() const { return !AddressMaps.empty(); }

protected:
  class AddressForDWOCU {
  public:
    AddressToIndexMap::iterator find(uint64_t Address) {
      return AddressToIndex.find(Address);
    }
    AddressToIndexMap::iterator end() { return AddressToIndex.end(); }
    AddressToIndexMap::iterator begin() { return AddressToIndex.begin(); }

    IndexToAddressMap::iterator indexToAdddessEnd() {
      return IndexToAddress.end();
    }
    IndexToAddressMap::iterator indexToAddressBegin() {
      return IndexToAddress.begin();
    }
    uint32_t getNextIndex() {
      while (IndexToAddress.count(CurrentIndex))
        ++CurrentIndex;
      return CurrentIndex;
    }

    /// Inserts elements in to IndexToAddress and AddressToIndex.
    /// Follows the same semantics as unordered_map insert.
    std::pair<AddressToIndexMap::iterator, bool> insert(uint64_t Address,
                                                        uint32_t Index) {
      IndexToAddress.insert({Index, Address});
      return AddressToIndex.insert({Address, Index});
    }

    /// Updates AddressToIndex Map.
    /// Follows the same symantics as unordered map [].
    void updateAddressToIndex(uint64_t Address, uint32_t Index) {
      AddressToIndex[Address] = Index;
    }

    /// Updates IndexToAddress Map.
    /// Follows the same symantics as unordered map [].
    void updateIndexToAddrss(uint64_t Address, uint32_t Index) {
      IndexToAddress[Index] = Address;
    }

    void dump();

  private:
    AddressToIndexMap AddressToIndex;
    IndexToAddressMap IndexToAddress;
    uint32_t CurrentIndex{0};
  };

  virtual uint64_t getCUID(DWARFUnit &Unit) {
    assert(Unit.getDWOId() && "Unit is not Skeleton CU.");
    return *Unit.getDWOId();
  }

  BinaryContext *BC;
  /// Maps DWOID to AddressForDWOCU.
  std::unordered_map<uint64_t, AddressForDWOCU> AddressMaps;
  /// Mutex used for parallel processing of debug info.
  std::mutex WriterMutex;
  std::unique_ptr<AddressSectionBuffer> Buffer;
  std::unique_ptr<raw_svector_ostream> AddressStream;
  /// Used to track sections that were not modified so that they can be re-used.
  DenseMap<uint64_t, uint64_t> UnmodifiedAddressOffsets;
};

class DebugAddrWriterDwarf5 : public DebugAddrWriter {
public:
  DebugAddrWriterDwarf5() = delete;
  DebugAddrWriterDwarf5(BinaryContext *BC) : DebugAddrWriter(BC) {}

  /// Write out entries in to .debug_addr section for CUs.
  virtual void update(DIEBuilder &DIEBlder, DWARFUnit &CUs) override;

protected:
  /// Given DWARFUnit \p Unit returns either DWO ID or it's offset within
  /// .debug_info.
  uint64_t getCUID(DWARFUnit &Unit) override {
    if (Unit.isDWOUnit()) {
      DWARFUnit *SkeletonCU = Unit.getLinkedUnit();
      return SkeletonCU->getOffset();
    }
    return Unit.getOffset();
  }
};

/// This class is NOT thread safe.
using DebugStrOffsetsBufferVector = SmallVector<char, 16>;
class DebugStrOffsetsWriter {
public:
  DebugStrOffsetsWriter() {
    StrOffsetsBuffer = std::make_unique<DebugStrOffsetsBufferVector>();
    StrOffsetsStream = std::make_unique<raw_svector_ostream>(*StrOffsetsBuffer);
  }

  /// Initializes Buffer and Stream.
  void initialize(const DWARFSection &StrOffsetsSection,
                  const std::optional<StrOffsetsContributionDescriptor> Contr);

  /// Update Str offset in .debug_str in .debug_str_offsets.
  void updateAddressMap(uint32_t Index, uint32_t Address);

  /// Writes out current sections entry into .debug_str_offsets.
  void finalizeSection(DWARFUnit &Unit, DIEBuilder &DIEBldr);

  /// Returns False if no strings were added to .debug_str.
  bool isFinalized() const { return !StrOffsetsBuffer->empty(); }

  /// Returns buffer containing .debug_str_offsets.
  std::unique_ptr<DebugStrOffsetsBufferVector> releaseBuffer() {
    return std::move(StrOffsetsBuffer);
  }

private:
  std::unique_ptr<DebugStrOffsetsBufferVector> StrOffsetsBuffer;
  std::unique_ptr<raw_svector_ostream> StrOffsetsStream;
  std::map<uint32_t, uint32_t> IndexToAddressMap;
  std::unordered_map<uint64_t, uint64_t> ProcessedBaseOffsets;
  // Section size not including header.
  uint32_t CurrentSectionSize{0};
  bool StrOffsetSectionWasModified = false;
};

using DebugStrBufferVector = SmallVector<char, 16>;
class DebugStrWriter {
public:
  DebugStrWriter() = delete;
  DebugStrWriter(BinaryContext &BC) : BC(BC) { create(); }
  std::unique_ptr<DebugStrBufferVector> releaseBuffer() {
    return std::move(StrBuffer);
  }

  /// Adds string to .debug_str.
  /// On first invokation it initializes internal data stractures.
  uint32_t addString(StringRef Str);

  /// Returns False if no strings were added to .debug_str.
  bool isInitialized() const { return !StrBuffer->empty(); }

private:
  /// Mutex used for parallel processing of debug info.
  std::mutex WriterMutex;
  /// Initializes Buffer and Stream.
  void initialize();
  /// Creats internal data stractures.
  void create();
  std::unique_ptr<DebugStrBufferVector> StrBuffer;
  std::unique_ptr<raw_svector_ostream> StrStream;
  BinaryContext &BC;
};

enum class LocWriterKind { DebugLocWriter, DebugLoclistWriter };

/// Serializes part of a .debug_loc DWARF section with LocationLists.
class SimpleBinaryPatcher;
class DebugLocWriter {
protected:
  DebugLocWriter(uint8_t DwarfVersion, LocWriterKind Kind)
      : DwarfVersion(DwarfVersion), Kind(Kind) {
    init();
  }

public:
  DebugLocWriter() { init(); };
  virtual ~DebugLocWriter(){};

  /// Writes out location lists and stores internal patches.
  virtual void addList(DIEBuilder &DIEBldr, DIE &Die, DIEValue &AttrInfo,
                       DebugLocationsVector &LocList);

  /// Writes out locations in to a local buffer, and adds Debug Info patches.
  virtual void finalize(DIEBuilder &DIEBldr, DIE &Die);

  /// Return internal buffer.
  virtual std::unique_ptr<DebugBufferVector> getBuffer();

  /// Returns DWARF version.
  uint8_t getDwarfVersion() const { return DwarfVersion; }

  /// Offset of an empty location list.
  static constexpr uint32_t EmptyListOffset = 0;

  LocWriterKind getKind() const { return Kind; }

  static bool classof(const DebugLocWriter *Writer) {
    return Writer->getKind() == LocWriterKind::DebugLocWriter;
  }

protected:
  std::unique_ptr<DebugBufferVector> LocBuffer;
  std::unique_ptr<raw_svector_ostream> LocStream;
  /// Current offset in the section (updated as new entries are written).
  /// Starts with 0 here since this only writes part of a full location lists
  /// section. In the final section, for DWARF4, the first 16 bytes are reserved
  /// for an empty list.
  static uint32_t LocSectionOffset;
  uint8_t DwarfVersion{4};
  LocWriterKind Kind{LocWriterKind::DebugLocWriter};

private:
  /// Inits all the related data structures.
  void init();
  struct LocListDebugInfoPatchType {
    uint64_t DebugInfoAttrOffset;
    uint64_t LocListOffset;
  };
  using VectorLocListDebugInfoPatchType =
      std::vector<LocListDebugInfoPatchType>;
  /// The list of debug info patches to be made once individual
  /// location list writers have been filled
  VectorLocListDebugInfoPatchType LocListDebugInfoPatches;
};

class DebugLoclistWriter : public DebugLocWriter {
public:
  ~DebugLoclistWriter() {}
  DebugLoclistWriter() = delete;
  DebugLoclistWriter(DWARFUnit &Unit, uint8_t DV, bool SD)
      : DebugLocWriter(DV, LocWriterKind::DebugLoclistWriter), CU(Unit),
        IsSplitDwarf(SD) {
    assert(DebugLoclistWriter::AddrWriter &&
           "Please use SetAddressWriter to initialize "
           "DebugAddrWriter before instantiation.");
    if (DwarfVersion >= 5) {
      LocBodyBuffer = std::make_unique<DebugBufferVector>();
      LocBodyStream = std::make_unique<raw_svector_ostream>(*LocBodyBuffer);
    } else {
      // Writing out empty location list to which all references to empty
      // location lists will point.
      const char Zeroes[16] = {0};
      *LocStream << StringRef(Zeroes, 16);
    }
  }

  static void setAddressWriter(DebugAddrWriter *AddrW) { AddrWriter = AddrW; }

  /// Stores location lists internally to be written out during finalize phase.
  virtual void addList(DIEBuilder &DIEBldr, DIE &Die, DIEValue &AttrInfo,
                       DebugLocationsVector &LocList) override;

  /// Writes out locations in to a local buffer and applies debug info patches.
  void finalize(DIEBuilder &DIEBldr, DIE &Die) override;

  /// Returns CU ID.
  /// For Skelton CU it is a CU Offset.
  /// For DWO CU it is a DWO ID.
  uint64_t getCUID() const {
    return CU.isDWOUnit() ? *CU.getDWOId() : CU.getOffset();
  }

  LocWriterKind getKind() const { return DebugLocWriter::getKind(); }

  static bool classof(const DebugLocWriter *Writer) {
    return Writer->getKind() == LocWriterKind::DebugLoclistWriter;
  }

  bool isSplitDwarf() const { return IsSplitDwarf; }

  constexpr static uint32_t InvalidIndex = UINT32_MAX;

private:
  /// Writes out locations in to a local buffer and applies debug info patches.
  void finalizeDWARF5(DIEBuilder &DIEBldr, DIE &Die);

  static DebugAddrWriter *AddrWriter;
  DWARFUnit &CU;
  bool IsSplitDwarf{false};
  // Used for DWARF5 to store location lists before being finalized.
  std::unique_ptr<DebugBufferVector> LocBodyBuffer;
  std::unique_ptr<raw_svector_ostream> LocBodyStream;
  std::vector<uint32_t> RelativeLocListOffsets;
  uint32_t NumberOfEntries{0};
  static uint32_t LoclistBaseOffset;
};

/// Abstract interface for classes that apply modifications to a binary string.
class BinaryPatcher {
public:
  virtual ~BinaryPatcher() {}
  /// Applies modifications to the copy of binary string \p BinaryContents .
  /// Implementations do not need to guarantee that size of a new \p
  /// BinaryContents remains unchanged.
  virtual std::string patchBinary(StringRef BinaryContents) = 0;
};

/// Applies simple modifications to a binary string, such as directly replacing
/// the contents of a certain portion with a string or an integer.
class SimpleBinaryPatcher : public BinaryPatcher {
private:
  std::vector<std::pair<uint32_t, std::string>> Patches;

  /// Adds a patch to replace the contents of \p ByteSize bytes with the integer
  /// \p NewValue encoded in little-endian, with the least-significant byte
  /// being written at the offset \p Offset.
  void addLEPatch(uint64_t Offset, uint64_t NewValue, size_t ByteSize);

  /// RangeBase for DWO DebugInfo Patcher.
  uint64_t RangeBase{0};

  /// Gets reset to false when setRangeBase is invoked.
  /// Gets set to true when getRangeBase is called
  uint64_t WasRangeBaseUsed{false};

public:
  virtual ~SimpleBinaryPatcher() {}

  /// Adds a patch to replace the contents of the binary string starting at the
  /// specified \p Offset with the string \p NewValue.
  /// The \p OldValueSize is the size of the old value that will be replaced.
  void addBinaryPatch(uint64_t Offset, std::string &&NewValue,
                      uint32_t OldValueSize);

  /// Adds a patch to replace the contents of a single byte of the string, at
  /// the offset \p Offset, with the value \Value.
  void addBytePatch(uint64_t Offset, uint8_t Value);

  /// Adds a patch to put the integer \p NewValue encoded as a 64-bit
  /// little-endian value at offset \p Offset.
  virtual void addLE64Patch(uint64_t Offset, uint64_t NewValue);

  /// Adds a patch to put the integer \p NewValue encoded as a 32-bit
  /// little-endian value at offset \p Offset.
  /// The \p OldValueSize is the size of the old value that will be replaced.
  virtual void addLE32Patch(uint64_t Offset, uint32_t NewValue,
                            uint32_t OldValueSize = 4);

  /// Add a patch at \p Offset with \p Value using unsigned LEB128 encoding with
  /// size \p OldValueSize.
  /// The \p OldValueSize is the size of the old value that will be replaced.
  virtual void addUDataPatch(uint64_t Offset, uint64_t Value,
                             uint32_t OldValueSize);

  /// Setting DW_AT_GNU_ranges_base
  void setRangeBase(uint64_t Rb) {
    WasRangeBaseUsed = false;
    RangeBase = Rb;
  }

  /// Gets DW_AT_GNU_ranges_base
  uint64_t getRangeBase() {
    WasRangeBaseUsed = true;
    return RangeBase;
  }

  /// Proxy for if we broke up low_pc/high_pc to ranges.
  bool getWasRangBasedUsed() const { return WasRangeBaseUsed; }

  /// This function takes in \p BinaryContents, applies patches to it and
  /// returns an updated string.
  std::string patchBinary(StringRef BinaryContents) override;
};

/// Similar to MCDwarfLineEntry, but identifies the location by its address
/// instead of MCLabel.
class BinaryDwarfLineEntry : public MCDwarfLoc {
  uint64_t Address;

public:
  // Constructor to create an BinaryDwarfLineEntry given a symbol and the dwarf
  // loc.
  BinaryDwarfLineEntry(uint64_t Address, const MCDwarfLoc loc)
      : MCDwarfLoc(loc), Address(Address) {}

  uint64_t getAddress() const { return Address; }
};

/// Line number information for the output binary. One instance per CU.
///
/// For any given CU, we may:
///   1. Generate new line table using:
///     a) emitted code: getMCLineSections().addEntry()
///     b) information from the input line table: addLineTableSequence()
/// or
///   2. Copy line table from the input file: addRawContents().
class DwarfLineTable {
public:
  /// Line number information on contiguous code region from the input binary.
  /// It is represented by [FirstIndex, LastIndex] rows range in the input
  /// line table, and the end address of the sequence used for issuing the end
  /// of the sequence directive.
  struct RowSequence {
    uint32_t FirstIndex;
    uint32_t LastIndex;
    uint64_t EndAddress;
  };

private:
  MCDwarfLineTableHeader Header;

  /// MC line tables for the code generated via MC layer.
  MCLineSection MCLineSections;

  /// Line info for the original code. To be merged with tables for new code.
  const DWARFDebugLine::LineTable *InputTable{nullptr};
  std::vector<RowSequence> InputSequences;

  /// Raw data representing complete debug line section for the unit.
  StringRef RawData;

  /// DWARF Version
  uint16_t DwarfVersion;

public:
  /// Emit line info for all units in the binary context.
  static void emit(BinaryContext &BC, MCStreamer &Streamer);

  /// Emit the Dwarf file and the line tables for a given CU.
  void emitCU(MCStreamer *MCOS, MCDwarfLineTableParams Params,
              std::optional<MCDwarfLineStr> &LineStr, BinaryContext &BC) const;

  Expected<unsigned> tryGetFile(StringRef &Directory, StringRef &FileName,
                                std::optional<MD5::MD5Result> Checksum,
                                std::optional<StringRef> Source,
                                uint16_t DwarfVersion,
                                unsigned FileNumber = 0) {
    assert(RawData.empty() && "cannot use with raw data");
    return Header.tryGetFile(Directory, FileName, Checksum, Source,
                             DwarfVersion, FileNumber);
  }

  /// Return label at the start of the emitted debug line for the unit.
  MCSymbol *getLabel() const { return Header.Label; }

  void setLabel(MCSymbol *Label) { Header.Label = Label; }

  /// Sets the root file \p Directory, \p FileName, optional \p CheckSum, and
  /// optional \p Source.
  void setRootFile(StringRef Directory, StringRef FileName,
                   std::optional<MD5::MD5Result> Checksum,
                   std::optional<StringRef> Source) {
    Header.setRootFile(Directory, FileName, Checksum, Source);
  }

  /// Access to MC line info.
  MCLineSection &getMCLineSections() { return MCLineSections; }

  /// Add line information using the sequence from the input line \p Table.
  void addLineTableSequence(const DWARFDebugLine::LineTable *Table,
                            uint32_t FirstRow, uint32_t LastRow,
                            uint64_t EndOfSequenceAddress) {
    assert((!InputTable || InputTable == Table) &&
           "expected same table for CU");
    InputTable = Table;
    InputSequences.emplace_back(
        RowSequence{FirstRow, LastRow, EndOfSequenceAddress});
  }

  /// Indicate that for the unit we should emit specified contents instead of
  /// generating a new line info table.
  void addRawContents(StringRef DebugLineContents) {
    RawData = DebugLineContents;
  }

  /// Sets DWARF version for this line table.
  void setDwarfVersion(uint16_t V) { DwarfVersion = V; }

  // Returns DWARF Version for this line table.
  uint16_t getDwarfVersion() const { return DwarfVersion; }
};
} // namespace bolt
} // namespace llvm

#endif
