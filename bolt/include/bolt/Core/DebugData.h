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
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define DWARF2_FLAG_END_SEQUENCE (1 << 4)

namespace llvm {

namespace bolt {

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
  virtual uint64_t addRanges(const DebugAddressRangesVector &Ranges);

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
  virtual uint64_t addRanges(
      DebugAddressRangesVector &&Ranges,
      std::map<DebugAddressRangesVector, uint64_t> &CachedRanges) override;

  /// Add ranges and return offset into section.
  virtual uint64_t addRanges(const DebugAddressRangesVector &Ranges) override;

  virtual std::unique_ptr<DebugBufferVector> releaseBuffer() override {
    return std::move(RangesBuffer);
  }

  /// Needs to be invoked before each \p CU is processed.
  void virtual initSection(DWARFUnit &CU) override;

  /// Writes out range lists for a current CU being processed.
  void virtual finalizeSection() override;

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

  /// Adds {\p Address, \p Index} to \p CU.
  void addIndexAddress(uint64_t Address, uint32_t Index, DWARFUnit &CU);

  /// Creates consolidated .debug_addr section, and builds DWOID to offset map.
  virtual AddressSectionBuffer finalize();

  /// Given DWARFUnit \p Unit returns offset of this CU in to .debug_addr
  /// section.
  virtual uint64_t getOffset(DWARFUnit &Unit);

  /// Returns False if .debug_addr section was created..
  bool isInitialized() const { return !AddressMaps.empty(); }

protected:
  class AddressForDWOCU {
  public:
    AddressToIndexMap::iterator find(uint64_t Adddress) {
      return AddressToIndex.find(Adddress);
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
  /// Maps DWOID to offset within .debug_addr section.
  std::unordered_map<uint64_t, uint64_t> DWOIdToOffsetMap;
  /// Mutex used for parallel processing of debug info.
  std::mutex WriterMutex;
};

class DebugAddrWriterDwarf5 : public DebugAddrWriter {
public:
  DebugAddrWriterDwarf5() = delete;
  DebugAddrWriterDwarf5(BinaryContext *BC) : DebugAddrWriter(BC) {}

  /// Creates consolidated .debug_addr section, and builds DWOID to offset map.
  virtual AddressSectionBuffer finalize() override;
  /// Given DWARFUnit \p Unit returns offset of this CU in to .debug_addr
  /// section.
  virtual uint64_t getOffset(DWARFUnit &Unit) override;

protected:
  /// Given DWARFUnit \p Unit returns either DWO ID or it's offset within
  /// .debug_info.
  virtual uint64_t getCUID(DWARFUnit &Unit) override {
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
                  const Optional<StrOffsetsContributionDescriptor> Contr);

  /// Update Str offset in .debug_str in .debug_str_offsets.
  void updateAddressMap(uint32_t Index, uint32_t Address);

  /// Writes out current sections entry into .debug_str_offsets.
  void finalizeSection();

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
  // Section size not including header.
  uint32_t CurrentSectionSize{0};
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
public:
  DebugLocWriter() = delete;
  DebugLocWriter(BinaryContext *BC);
  virtual ~DebugLocWriter(){};

  /// Writes out location lists and stores internal patches.
  virtual void addList(uint64_t AttrOffset, uint32_t LocListIndex,
                       DebugLocationsVector &&LocList);

  /// Writes out locations in to a local buffer, and adds Debug Info patches.
  virtual void finalize(uint64_t SectionOffset,
                        SimpleBinaryPatcher &DebugInfoPatcher);

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
  /// section. In the final section, the first 16 bytes are reserved for an
  /// empty list.
  uint32_t SectionOffset{0};
  uint8_t DwarfVersion{4};
  LocWriterKind Kind{LocWriterKind::DebugLocWriter};

private:
  struct LocListDebugInfoPatchType {
    uint64_t DebugInfoAttrOffset;
    uint64_t LocListOffset;
  };
  using VectorLocListDebugInfoPatchType =
      std::vector<LocListDebugInfoPatchType>;
  /// The list of debug info patches to be made once individual
  /// location list writers have been filled
  VectorLocListDebugInfoPatchType LocListDebugInfoPatches;

  using VectorEmptyLocListAttributes = std::vector<uint64_t>;
  /// Contains all the attributes pointing to empty location list.
  VectorEmptyLocListAttributes EmptyAttrLists;
};

class DebugLoclistWriter : public DebugLocWriter {
public:
  ~DebugLoclistWriter() {}
  DebugLoclistWriter() = delete;
  DebugLoclistWriter(BinaryContext *BC, DWARFUnit &Unit,
                     uint32_t LocListsBaseAttrOffset, uint8_t DV, bool SD)
      : DebugLocWriter(BC), CU(Unit),
        LocListsBaseAttrOffset(LocListsBaseAttrOffset), IsSplitDwarf(SD) {
    Kind = LocWriterKind::DebugLoclistWriter;
    DwarfVersion = DV;
    assert(DebugLoclistWriter::AddrWriter &&
           "Please use SetAddressWriter to initialize "
           "DebugAddrWriter before instantiation.");
  }

  static void setAddressWriter(DebugAddrWriter *AddrW) { AddrWriter = AddrW; }

  /// Stores location lists internally to be written out during finalize phase.
  virtual void addList(uint64_t AttrOffset, uint32_t LocListIndex,
                       DebugLocationsVector &&LocList) override;

  /// Writes out locations in to a local buffer and applies debug info patches.
  void finalize(uint64_t SectionOffset,
                SimpleBinaryPatcher &DebugInfoPatcher) override;

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
  constexpr static uint32_t InvalidLocListsBaseAttrOffset = UINT32_MAX;

private:
  /// Writes out locations in to a local buffer and applies debug info patches.
  void finalizeDWARFLegacy(uint64_t SectionOffset,
                           SimpleBinaryPatcher &DebugInfoPatcher);

  /// Writes out locations in to a local buffer and applies debug info patches.
  void finalizeDWARF5(uint64_t SectionOffset,
                      SimpleBinaryPatcher &DebugInfoPatcher);

  struct LocPatch {
    uint64_t AttrOffset{0};
    uint32_t Index;
    DebugLocationsVector LocList;
  };
  using LocPatchVec = SmallVector<LocPatch, 4>;
  LocPatchVec Patches;

  class Patch {
  public:
    Patch() = delete;
    Patch(uint64_t O, uint64_t A) : Offset(O), Address(A) {}
    uint64_t Offset{0};
    uint64_t Address{0};
  };
  static DebugAddrWriter *AddrWriter;
  DWARFUnit &CU;
  uint32_t LocListsBaseAttrOffset{InvalidLocListsBaseAttrOffset};
  bool IsSplitDwarf{false};
};

enum class PatcherKind { SimpleBinaryPatcher, DebugInfoBinaryPatcher };
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

  virtual PatcherKind getKind() const {
    return PatcherKind::SimpleBinaryPatcher;
  }

  static bool classof(const SimpleBinaryPatcher *Patcher) {
    return Patcher->getKind() == PatcherKind::SimpleBinaryPatcher;
  }

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
  virtual std::string patchBinary(StringRef BinaryContents) override;
};

class DebugInfoBinaryPatcher : public SimpleBinaryPatcher {
public:
  enum class DebugPatchKind {
    PatchBaseClass,
    PatchValue32,
    PatchValue64to32,
    PatchValue32GenericSize,
    PatchValue64,
    PatchValueVariable,
    ReferencePatchValue,
    DWARFUnitOffsetBaseLabel,
    DestinationReferenceLabel,
    NewDebugEntry
  };

  struct Patch {
    Patch(uint32_t O, DebugPatchKind K) : Offset(O), Kind(K) {}
    DebugPatchKind getKind() const { return Kind; }

    static bool classof(const Patch *Writer) {
      return Writer->getKind() == DebugPatchKind::PatchBaseClass;
    }

    uint32_t Offset;
    DebugPatchKind Kind;
  };

  struct DebugPatch64to32 : public Patch {
    DebugPatch64to32(uint32_t O, uint32_t V)
        : Patch(O, DebugPatchKind::PatchValue64to32) {
      Value = V;
    }
    DebugPatchKind getKind() const { return Kind; }

    static bool classof(const Patch *Writer) {
      return Writer->getKind() == DebugPatchKind::PatchValue64to32;
    }
    uint32_t Value;
  };

  struct DebugPatch32 : public Patch {
    DebugPatch32(uint32_t O, uint32_t V)
        : Patch(O, DebugPatchKind::PatchValue32) {
      Value = V;
    }
    DebugPatchKind getKind() const { return Kind; }

    static bool classof(const Patch *Writer) {
      return Writer->getKind() == DebugPatchKind::PatchValue32;
    }
    uint32_t Value;
  };

  /// Patch for 4 byte entry, where original entry size is not 4 bytes or 8
  /// bytes.
  struct DebugPatch32GenericSize : public Patch {
    DebugPatch32GenericSize(uint32_t O, uint32_t V, uint32_t OVS)
        : Patch(O, DebugPatchKind::PatchValue32GenericSize) {
      Value = V;
      OldValueSize = OVS;
    }

    static bool classof(const Patch *Writer) {
      return Writer->getKind() == DebugPatchKind::PatchValue32GenericSize;
    }
    uint32_t Value;
    uint32_t OldValueSize;
  };

  struct DebugPatch64 : public Patch {
    DebugPatch64(uint32_t O, uint64_t V)
        : Patch(O, DebugPatchKind::PatchValue64) {
      Value = V;
    }

    static bool classof(const Patch *Writer) {
      return Writer->getKind() == DebugPatchKind::PatchValue64;
    }
    uint64_t Value;
  };

  struct DebugPatchVariableSize : public Patch {
    DebugPatchVariableSize(uint32_t O, uint32_t OVS, uint32_t V)
        : Patch(O, DebugPatchKind::PatchValueVariable) {
      OldValueSize = OVS;
      Value = V;
    }

    static bool classof(const Patch *Writer) {
      return Writer->getKind() == DebugPatchKind::PatchValueVariable;
    }
    uint32_t OldValueSize;
    uint32_t Value;
  };

  struct DebugPatchReference : public Patch {
    struct Data {
      uint32_t OldValueSize : 4;
      uint32_t DirectRelative : 1;
      uint32_t IndirectRelative : 1;
      uint32_t DirectAbsolute : 1;
    };
    DebugPatchReference(uint32_t O, uint32_t OVS, uint32_t DO, dwarf::Form F)
        : Patch(O, DebugPatchKind::ReferencePatchValue) {
      PatchInfo.OldValueSize = OVS;
      DestinationOffset = DO;
      PatchInfo.DirectRelative =
          F == dwarf::DW_FORM_ref1 || F == dwarf::DW_FORM_ref2 ||
          F == dwarf::DW_FORM_ref4 || F == dwarf::DW_FORM_ref8;
      PatchInfo.IndirectRelative = F == dwarf::DW_FORM_ref_udata;
      PatchInfo.DirectAbsolute = F == dwarf::DW_FORM_ref_addr;
    }

    static bool classof(const Patch *Writer) {
      return Writer->getKind() == DebugPatchKind::ReferencePatchValue;
    }
    Data PatchInfo;
    uint32_t DestinationOffset;
  };

  struct DWARFUnitOffsetBaseLabel : public Patch {
    DWARFUnitOffsetBaseLabel(uint32_t O)
        : Patch(O, DebugPatchKind::DWARFUnitOffsetBaseLabel) {}

    static bool classof(const Patch *Writer) {
      return Writer->getKind() == DebugPatchKind::DWARFUnitOffsetBaseLabel;
    }
  };

  struct DestinationReferenceLabel : public Patch {
    DestinationReferenceLabel() = delete;
    DestinationReferenceLabel(uint32_t O)
        : Patch(O, DebugPatchKind::DestinationReferenceLabel) {}

    static bool classof(const Patch *Writer) {
      return Writer->getKind() == DebugPatchKind::DestinationReferenceLabel;
    }
  };

  struct NewDebugEntry : public Patch {
    NewDebugEntry() = delete;
    NewDebugEntry(uint32_t O, std::string &&V)
        : Patch(O, DebugPatchKind::NewDebugEntry) {
      CurrentOrder = NewDebugEntry::OrderCounter++;
      Value = std::move(V);
    }

    static bool classof(const Patch *Writer) {
      return Writer->getKind() == DebugPatchKind::NewDebugEntry;
    }
    static uint32_t OrderCounter;
    uint32_t CurrentOrder;
    std::string Value;
  };

  virtual PatcherKind getKind() const override {
    return PatcherKind::DebugInfoBinaryPatcher;
  }

  static bool classof(const SimpleBinaryPatcher *Patcher) {
    return Patcher->getKind() == PatcherKind::DebugInfoBinaryPatcher;
  }

  /// This function takes in \p BinaryContents, and re-writes it with new
  /// patches inserted into it. It returns an updated string.
  virtual std::string patchBinary(StringRef BinaryContents) override;

  /// Adds a patch to put the integer \p NewValue encoded as a 64-bit
  /// little-endian value at offset \p Offset.
  virtual void addLE64Patch(uint64_t Offset, uint64_t NewValue) override;

  /// Adds a patch to put the integer \p NewValue encoded as a 32-bit
  /// little-endian value at offset \p Offset.
  /// The \p OldValueSize is the size of the old value that will be replaced.
  virtual void addLE32Patch(uint64_t Offset, uint32_t NewValue,
                            uint32_t OldValueSize = 4) override;

  /// Add a patch at \p Offset with \p Value using unsigned LEB128 encoding with
  /// size \p OldValueSize.
  /// The \p OldValueSize is the size of the old value that will be replaced.
  virtual void addUDataPatch(uint64_t Offset, uint64_t Value,
                             uint32_t OldValueSize) override;

  /// Adds a label \p Offset for DWARF UNit.
  /// Used to recompute relative references.
  void addUnitBaseOffsetLabel(uint64_t Offset);

  /// Adds a Label for destination. Either relative or explicit reference.
  void addDestinationReferenceLabel(uint64_t Offset);

  /// Adds a reference at \p Offset to patch with new fully resolved \p
  /// DestinationOffset . The \p OldValueSize is the original size of entry in
  /// the DIE. The \p Form is the form of the entry.
  void addReferenceToPatch(uint64_t Offset, uint32_t DestinationOffset,
                           uint32_t OldValueSize, dwarf::Form Form);

  /// Inserts a new uint32_t \p Value at the end of \p DIE .
  void insertNewEntry(const DWARFDie &DIE, uint32_t);

  /// Inserts a new encoded \p Value at the end of \p DIE .
  void insertNewEntry(const DWARFDie &DIE, std::string &&Value);

  /// Clears unordered set for DestinationLabels.
  void clearDestinationLabels() { DestinationLabels.clear(); }

  /// Sets DWARF Units offset, \p  DWPOffset , within DWP file.
  void setDWPOffset(uint64_t DWPOffset) { DWPUnitOffset = DWPOffset; }

  /// When this function is invoked all of the DebugInfo Patches must be done.
  /// Returns a map of old CU offsets to new offsets and new sizes.
  CUOffsetMap computeNewOffsets(DWARFContext &DWCtx, bool IsDWOContext);

private:
  struct PatchDeleter {
    void operator()(Patch *P) const {
      Patch *Pp = reinterpret_cast<Patch *>(P);
      switch (Pp->Kind) {
      default:
        llvm_unreachable(
            "Not all delete instances handled for Patch derived classes");
      case DebugPatchKind::PatchValue32:
        delete reinterpret_cast<DebugPatch32 *>(P);
        break;
      case DebugPatchKind::PatchValue64to32:
        delete reinterpret_cast<DebugPatch64to32 *>(P);
        break;
      case DebugPatchKind::PatchValue32GenericSize:
        delete reinterpret_cast<DebugPatch32GenericSize *>(P);
        break;
      case DebugPatchKind::PatchValue64:
        delete reinterpret_cast<DebugPatch64 *>(P);
        break;
      case DebugPatchKind::PatchValueVariable:
        delete reinterpret_cast<DebugPatchVariableSize *>(P);
        break;
      case DebugPatchKind::ReferencePatchValue:
        delete reinterpret_cast<DebugPatchReference *>(P);
        break;
      case DebugPatchKind::DWARFUnitOffsetBaseLabel:
        delete reinterpret_cast<DWARFUnitOffsetBaseLabel *>(P);
        break;
      case DebugPatchKind::DestinationReferenceLabel:
        delete reinterpret_cast<DestinationReferenceLabel *>(P);
        break;
      case DebugPatchKind::NewDebugEntry:
        delete reinterpret_cast<NewDebugEntry *>(P);
        break;
      }
    }
  };
  using UniquePatchPtrType = std::unique_ptr<Patch, PatchDeleter>;

  uint64_t DWPUnitOffset{0};
  int32_t ChangeInSize{0};
  std::vector<UniquePatchPtrType> DebugPatches;
  /// Mutex used for parallel processing of debug info.
  std::mutex WriterMutex;
  /// Stores fully resolved addresses of DIEs that are being referenced.
  std::unordered_set<uint32_t> DestinationLabels;
  /// Map of original debug info references to new ones.
  std::unordered_map<uint32_t, uint32_t> OldToNewOffset;
};

/// Class to facilitate modifying and writing abbreviation sections.
class DebugAbbrevWriter {
  /// Mutex used for parallel processing of debug info.
  std::mutex WriterMutex;

  /// Offsets of abbreviation sets in normal .debug_abbrev section.
  std::vector<uint64_t> AbbrevSetOffsets;

  /// Abbrev data set for a single unit.
  struct AbbrevData {
    uint64_t Offset{0}; ///< Offset of the data in the final section.
    std::unique_ptr<DebugBufferVector> Buffer;
    std::unique_ptr<raw_svector_ostream> Stream;
  };
  /// Map original unit to abbreviations data.
  std::unordered_map<const DWARFUnit *, AbbrevData *> UnitsAbbrevData;

  /// Map from Hash Signature to AbbrevData.
  llvm::StringMap<std::unique_ptr<AbbrevData>> AbbrevDataCache;

  /// Attributes substitution (patch) information.
  struct PatchInfo {
    dwarf::Attribute OldAttr;
    dwarf::Attribute NewAttr;
    uint8_t NewAttrForm;
  };

  struct AbbrevEntry {
    dwarf::Attribute Attr;
    dwarf::Form Form;
  };

  using PatchesTy = std::unordered_map<const DWARFAbbreviationDeclaration *,
                                       SmallVector<PatchInfo, 2>>;
  std::unordered_map<const DWARFUnit *, PatchesTy> Patches;

  using AbbrevEntryTy = std::unordered_map<const DWARFAbbreviationDeclaration *,
                                           SmallVector<AbbrevEntry, 2>>;
  std::unordered_map<const DWARFUnit *, AbbrevEntryTy> NewAbbrevEntries;

  /// DWARF context containing abbreviations.
  DWARFContext &Context;

  /// DWO ID used to identify unit contribution in DWP.
  Optional<uint64_t> DWOId;

  /// Add abbreviations from compile/type \p Unit to the writer.
  void addUnitAbbreviations(DWARFUnit &Unit);

public:
  /// Create an abbrev section writer for abbreviations in \p Context.
  /// If no \p DWOId is given, all normal (non-DWO) abbreviations in the
  /// \p Context are handled. Otherwise, only abbreviations associated with
  /// the compile unit matching \p DWOId in DWP or DWO will be covered by
  /// an instance of this class.
  ///
  /// NOTE: Type unit abbreviations are not handled separately for DWOs.
  ///       Most of the time, using type units with DWO is not a good idea.
  ///       If type units are used, the caller is responsible for verifying
  ///       that abbreviations are shared by CU and TUs.
  DebugAbbrevWriter(DWARFContext &Context, Optional<uint64_t> DWOId = None)
      : Context(Context), DWOId(DWOId) {}

  DebugAbbrevWriter(const DebugAbbrevWriter &) = delete;
  DebugAbbrevWriter &operator=(const DebugAbbrevWriter &) = delete;

  DebugAbbrevWriter(DebugAbbrevWriter &&) = delete;
  DebugAbbrevWriter &operator=(DebugAbbrevWriter &&) = delete;

  virtual ~DebugAbbrevWriter() = default;

  /// Substitute attribute \p AttrTag in abbreviation declaration \p Abbrev
  /// belonging to CU \p Unit with new attribute \p NewAttrTag having
  /// \p NewAttrForm form.
  void addAttributePatch(const DWARFUnit &Unit,
                         const DWARFAbbreviationDeclaration *Abbrev,
                         dwarf::Attribute AttrTag, dwarf::Attribute NewAttrTag,
                         uint8_t NewAttrForm) {
    assert(&Unit.getContext() == &Context &&
           "cannot update attribute from a different DWARF context");
    std::lock_guard<std::mutex> Lock(WriterMutex);
    Patches[&Unit][Abbrev].emplace_back(
        PatchInfo{AttrTag, NewAttrTag, NewAttrForm});
  }

  /// Adds attribute \p AttrTag and \p NewAttrForm in abbreviation declaration
  /// \p Abbrev belonging to CU \p Unit .
  void addAttribute(const DWARFUnit &Unit,
                    const DWARFAbbreviationDeclaration *Abbrev,
                    dwarf::Attribute AttrTag, dwarf::Form AttrForm) {
    assert(&Unit.getContext() == &Context &&
           "cannot update attribute from a different DWARF context");
    std::lock_guard<std::mutex> Lock(WriterMutex);
    bool AlreadyAdded = false;
    for (AbbrevEntry &E : NewAbbrevEntries[&Unit][Abbrev])
      if (E.Attr == AttrTag) {
        AlreadyAdded = true;
        break;
      }

    if (AlreadyAdded)
      return;
    NewAbbrevEntries[&Unit][Abbrev].emplace_back(
        AbbrevEntry{AttrTag, AttrForm});
  }

  /// Return a buffer with concatenated abbrev sections for all CUs and TUs
  /// in the associated DWARF context. Section offsets could be queried using
  /// getAbbreviationsOffsetForUnit() interface. For DWP, we are using DWOId
  /// to return only that unit's contribution to abbrev section.
  std::unique_ptr<DebugBufferVector> finalize();

  /// Return an offset in the finalized abbrev section corresponding to CU/TU.
  uint64_t getAbbreviationsOffsetForUnit(const DWARFUnit &Unit) {
    assert(!DWOId && "offsets are tracked for non-DWO units only");
    assert(UnitsAbbrevData.count(&Unit) && "no abbrev data found for unit");
    return UnitsAbbrevData[&Unit]->Offset;
  }
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
              Optional<MCDwarfLineStr> &LineStr, BinaryContext &BC) const;

  Expected<unsigned> tryGetFile(StringRef &Directory, StringRef &FileName,
                                Optional<MD5::MD5Result> Checksum,
                                Optional<StringRef> Source,
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
                   Optional<MD5::MD5Result> Checksum,
                   Optional<StringRef> Source) {
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

struct AttrInfo {
  DWARFFormValue V;
  uint64_t Offset;
  uint32_t Size; // Size of the attribute.
};

Optional<AttrInfo>
findAttributeInfo(const DWARFDie DIE,
                  const DWARFAbbreviationDeclaration *AbbrevDecl,
                  uint32_t Index);

} // namespace bolt
} // namespace llvm

#endif
