//===- bolt/Core/DebugNames.h - Debug names support ---*- C++
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declaration of classes required for generation of
// .debug_names section.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_DEBUG_NAMES_H
#define BOLT_CORE_DEBUG_NAMES_H

#include "bolt/Core/DebugData.h"
#include "llvm/CodeGen/AccelTable.h"

namespace llvm {
namespace bolt {
class BOLTDWARF5AccelTableData : public DWARF5AccelTableData {
public:
  BOLTDWARF5AccelTableData(const uint64_t DieOffset,
                           const std::optional<uint64_t> DefiningParentOffset,
                           const unsigned DieTag, const unsigned UnitID,
                           const bool IsParentRoot, const bool IsTU,
                           const std::optional<unsigned> SecondUnitID)
      : DWARF5AccelTableData(DieOffset, DefiningParentOffset, DieTag, UnitID,
                             IsTU),
        SecondUnitID(SecondUnitID), IsParentRoot(IsParentRoot) {}

  uint64_t getDieOffset() const { return DWARF5AccelTableData::getDieOffset(); }
  unsigned getDieTag() const { return DWARF5AccelTableData::getDieTag(); }
  unsigned getUnitID() const { return DWARF5AccelTableData::getUnitID(); }
  bool isTU() const { return DWARF5AccelTableData::isTU(); }
  bool isParentRoot() const { return IsParentRoot; }
  std::optional<unsigned> getSecondUnitID() const { return SecondUnitID; }

  void setPatchOffset(uint64_t PatchOffset) { OffsetVal = PatchOffset; }
  uint64_t getPatchOffset() const { return std::get<uint64_t>(OffsetVal); }

private:
  std::optional<unsigned> SecondUnitID;
  bool IsParentRoot;
};

class DWARF5AcceleratorTable {
public:
  DWARF5AcceleratorTable(const bool CreateDebugNames, BinaryContext &BC,
                         DebugStrWriter &MainBinaryStrWriter);
  ~DWARF5AcceleratorTable() {
    for (DebugNamesAbbrev *Abbrev : AbbreviationsVector)
      Abbrev->~DebugNamesAbbrev();
  }
  /// Add DWARF5 Accelerator table entry.
  /// Input is DWARFUnit being processed, DIE that belongs to it, potential
  /// DWOID if the Unit comes from a DWO section, and potential parent entry.
  std::optional<BOLTDWARF5AccelTableData *>
  addAccelTableEntry(DWARFUnit &Unit, const DIE &Die,
                     const std::optional<uint64_t> &DWOID,
                     const uint32_t NumberParentsInChain,
                     std::optional<BOLTDWARF5AccelTableData *> &Parent);
  /// Set current unit being processed.
  void setCurrentUnit(DWARFUnit &Unit, const uint64_t UnitStartOffset);
  /// Emit Accelerator table.
  void emitAccelTable();
  /// Returns true if the table was crated.
  bool isCreated() const { return NeedToCreate; }
  /// Returns buffer containing the accelerator table.
  std::unique_ptr<DebugBufferVector> releaseBuffer() {
    return std::move(FullTableBuffer);
  }
  /// Adds a DIE that is referenced across CUs.
  void addCrossCUDie(DWARFUnit *Unit, const DIE *Die) {
    CrossCUDies.insert({Die->getOffset(), {Unit, Die}});
  }
  /// Returns true if the DIE can generate an entry for a cross cu reference.
  /// This only checks TAGs of a DIE because when this is invoked DIE might not
  /// be fully constructed.
  bool canGenerateEntryWithCrossCUReference(
      const DWARFUnit &Unit, const DIE &Die,
      const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec);

private:
  BinaryContext &BC;
  bool NeedToCreate = false;
  BumpPtrAllocator Allocator;
  DebugStrWriter &MainBinaryStrWriter;
  StringRef StrSection;
  uint64_t CurrentUnitOffset = 0;
  const DWARFUnit *CurrentUnit = nullptr;
  std::unordered_map<uint32_t, uint32_t> AbbrevTagToIndexMap;
  /// Contains a map of TU hashes to a Foreign TU indecies.
  /// This is used to reduce the size of Foreign TU list since there could be
  /// multiple TUs with the same hash.
  DenseMap<uint64_t, uint32_t> TUHashToIndexMap;

  /// Represents a group of entries with identical name (and hence, hash value).
  struct HashData {
    uint64_t StrOffset;
    uint32_t HashValue;
    uint32_t EntryOffset;
    std::vector<BOLTDWARF5AccelTableData *> Values;
  };
  using HashList = std::vector<HashData *>;
  using BucketList = std::vector<HashList>;
  /// Contains all the offsets of CUs.
  SmallVector<uint32_t, 1> CUList;
  /// Contains all the offsets of local TUs.
  SmallVector<uint32_t, 1> LocalTUList;
  /// Contains all the type hashes for split dwarf TUs.
  SmallVector<uint64_t, 1> ForeignTUList;
  using StringEntries =
      MapVector<std::string, HashData, llvm::StringMap<unsigned>>;
  StringEntries Entries;
  /// FoldingSet that uniques the abbreviations.
  FoldingSet<DebugNamesAbbrev> AbbreviationsSet;
  /// Vector containing DebugNames abbreviations for iteration in order.
  SmallVector<DebugNamesAbbrev *, 5> AbbreviationsVector;
  /// The bump allocator to use when creating DIEAbbrev objects in the uniqued
  /// storage container.
  BumpPtrAllocator Alloc;
  uint32_t BucketCount = 0;
  uint32_t UniqueHashCount = 0;
  uint32_t AbbrevTableSize = 0;
  uint32_t CUIndexEncodingSize = 4;
  uint32_t TUIndexEncodingSize = 4;
  uint32_t AugmentationStringSize = 0;
  dwarf::Form CUIndexForm = dwarf::DW_FORM_data4;
  dwarf::Form TUIndexForm = dwarf::DW_FORM_data4;

  BucketList Buckets;

  std::unique_ptr<DebugBufferVector> FullTableBuffer;
  std::unique_ptr<raw_svector_ostream> FullTableStream;
  std::unique_ptr<DebugBufferVector> StrBuffer;
  std::unique_ptr<raw_svector_ostream> StrStream;
  std::unique_ptr<DebugBufferVector> EntriesBuffer;
  std::unique_ptr<raw_svector_ostream> Entriestream;
  std::unique_ptr<DebugBufferVector> AugStringBuffer;
  std::unique_ptr<raw_svector_ostream> AugStringtream;
  llvm::DenseMap<llvm::hash_code, uint64_t> StrCacheToOffsetMap;
  // Contains DWO ID to CUList Index.
  llvm::DenseMap<uint64_t, uint32_t> CUOffsetsToPatch;
  // Contains a map of Entry ID to Entry relative offset.
  llvm::DenseMap<uint64_t, uint32_t> EntryRelativeOffsets;
  llvm::DenseMap<uint64_t, std::pair<DWARFUnit *, const DIE *>> CrossCUDies;
  /// Adds Unit to either CUList, LocalTUList or ForeignTUList.
  /// Input Unit being processed, and DWO ID if Unit is being processed comes
  /// from a DWO section.
  void addUnit(DWARFUnit &Unit, const std::optional<uint64_t> &DWOID);
  /// Returns number of buckets in .debug_name table.
  ArrayRef<HashList> getBuckets() const { return Buckets; }
  /// Get encoding for a given attribute.
  std::optional<DWARF5AccelTable::UnitIndexAndEncoding>
  getIndexForEntry(const BOLTDWARF5AccelTableData &Value) const;
  /// Get encoding for a given attribute for second index.
  /// Returns nullopt if there is no second index.
  std::optional<DWARF5AccelTable::UnitIndexAndEncoding>
  getSecondIndexForEntry(const BOLTDWARF5AccelTableData &Value) const;
  /// Uniquify Entries.
  void finalize();
  /// Computes bucket count.
  void computeBucketCount();
  /// Populate Abbreviations Map.
  void populateAbbrevsMap();
  /// Write Entries.
  void writeEntries();
  /// Write an Entry.
  void writeEntry(BOLTDWARF5AccelTableData &Entry);
  /// Write augmentation_string for BOLT.
  void writeAugmentationString();
  /// Emit out Header for DWARF5 Accelerator table.
  void emitHeader() const;
  /// Emit out CU list.
  void emitCUList() const;
  /// Emit out TU List. Combination of LocalTUList and ForeignTUList.
  void emitTUList() const;
  /// Emit buckets.
  void emitBuckets() const;
  /// Emit hashes for hash table.
  void emitHashes() const;
  /// Emit string offsets for hash table.
  void emitStringOffsets() const;
  /// Emit Entry Offsets for hash table.
  void emitOffsets() const;
  /// Emit abbreviation table.
  void emitAbbrevs();
  /// Emit entries.
  void emitData();
  /// Emit augmentation string.
  void emitAugmentationString() const;
  /// Creates a new entry for a given DIE.
  std::optional<BOLTDWARF5AccelTableData *>
  addEntry(DWARFUnit &DU, const DIE &CurrDie,
           const std::optional<uint64_t> &DWOID,
           const std::optional<BOLTDWARF5AccelTableData *> &Parent,
           const std::optional<std::string> &Name,
           const uint32_t NumberParentsInChain);
  /// Returns UnitID for a given DWARFUnit.
  uint32_t getUnitID(const DWARFUnit &Unit,
                     const std::optional<uint64_t> &DWOID, bool &IsTU);
  std::optional<std::string> getName(DWARFUnit &DU,
                                     const std::optional<uint64_t> &DWOID,
                                     const std::string &NameToUse,
                                     DIEValue ValName);
  /// Processes a DIE with references to other DIEs for DW_AT_name and
  /// DW_AT_linkage_name resolution.
  /// If DW_AT_name exists method creates a new entry for this DIE and returns
  /// it.
  std::optional<BOLTDWARF5AccelTableData *> processReferencedDie(
      DWARFUnit &Unit, const DIE &Die, const std::optional<uint64_t> &DWOID,
      const std::optional<BOLTDWARF5AccelTableData *> &Parent,
      const std::string &NameToUse, const uint32_t NumberParentsInChain,
      const dwarf::Attribute &Attr);
};
} // namespace bolt
} // namespace llvm
#endif
