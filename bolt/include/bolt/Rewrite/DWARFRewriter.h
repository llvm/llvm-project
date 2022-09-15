//===- bolt/Rewrite/DWARFRewriter.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_DWARF_REWRITER_H
#define BOLT_REWRITE_DWARF_REWRITER_H

#include "bolt/Core/DIEBuilder.h"
#include "bolt/Core/DebugData.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/MC/MCAsmLayout.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {

namespace bolt {

class BinaryContext;

class DWARFRewriter {
public:
  DWARFRewriter() = delete;
  /// Contains information about TU so we can write out correct entries in GDB
  /// index.
  struct GDBIndexTUEntry {
    uint64_t UnitOffset;
    uint64_t TypeHash;
    uint64_t TypeDIERelativeOffset;
  };
  /// Contains information for CU or TU so we can output correct {cu, tu}-index.
  struct UnitMeta {
    uint64_t Offset;
    uint64_t Length;
    uint64_t TUHash;
  };

private:
  BinaryContext &BC;

  std::mutex DWARFRewriterMutex;

  /// Stores and serializes information that will be put into the
  /// .debug_ranges DWARF section.
  std::unique_ptr<DebugRangesSectionWriter> LegacyRangesSectionWriter;

  /// Stores and serializes information that will be put into the
  /// .debug_rnglists DWARF section.
  std::unique_ptr<DebugRangeListsSectionWriter> RangeListsSectionWriter;

  /// Stores and serializes information that will be put into the
  /// .debug_aranges DWARF section.
  std::unique_ptr<DebugARangesSectionWriter> ARangesSectionWriter;

  /// Stores and serializes information that will be put into the
  /// .debug_addr DWARF section.
  std::unique_ptr<DebugAddrWriter> AddrWriter;

  /// Stores and serializes information that will be put in to the
  /// .debug_addr DWARF section.
  /// Does not do de-duplication.
  std::unique_ptr<DebugStrWriter> StrWriter;

  /// Stores and serializes information that will be put in to the
  /// .debug_str_offsets DWARF section.
  std::unique_ptr<DebugStrOffsetsWriter> StrOffstsWriter;

  using LocWriters = std::map<uint64_t, std::unique_ptr<DebugLocWriter>>;
  /// Use a separate location list writer for each compilation unit
  LocWriters LocListWritersByCU;

  using RangeListsDWOWriers =
      std::unordered_map<uint64_t,
                         std::unique_ptr<DebugRangeListsSectionWriter>>;
  /// Store Rangelists writer for each DWO CU.
  RangeListsDWOWriers RangeListsWritersByCU;

  std::mutex LocListDebugInfoPatchesMutex;

  /// Dwo id specific its .debug_info.dwo section content.
  std::unordered_map<uint64_t, std::string> DwoDebugInfoMap;

  /// Dwo id specific its .debug_abbrev.dwo section content.
  std::unordered_map<uint64_t, std::string> DwoDebugAbbrevMap;

  /// Dwo id specific its .debug_types.dwo section content.
  std::unordered_map<uint64_t, std::string> DwoDebugTypeMap;

  /// Dwo id specific its RangesBase.
  std::unordered_map<uint64_t, uint64_t> DwoRangesBase;

  std::unordered_map<DWARFUnit *, uint64_t> LineTablePatchMap;
  std::unordered_map<DWARFUnit *, uint64_t> TypeUnitRelocMap;

  /// Entries for GDB Index Types CU List
  using GDBIndexTUEntryType = std::vector<GDBIndexTUEntry>;
  GDBIndexTUEntryType GDBIndexTUEntryVector;

  using UnitMetaVectorType = std::vector<UnitMeta>;
  using TUnitMetaDwoMapType = std::unordered_map<uint64_t, UnitMetaVectorType>;
  using CUnitMetaDwoMapType = std::unordered_map<uint64_t, UnitMeta>;
  CUnitMetaDwoMapType CUnitMetaDwoMap;
  TUnitMetaDwoMapType TUnitMetaDwoMap;

  /// DWARFLegacy is all DWARF versions before DWARF 5.
  enum class DWARFVersion { DWARFLegacy, DWARF5 };

  /// Update debug info for all DIEs in \p Unit.
  void updateUnitDebugInfo(DWARFUnit &Unit, DIEBuilder &DIEBldr,
                           DebugLocWriter &DebugLocWriter,
                           DebugRangesSectionWriter &RangesSectionWriter,
                           std::optional<uint64_t> RangesBase = std::nullopt);

  /// Patches the binary for an object's address ranges to be updated.
  /// The object can be anything that has associated address ranges via either
  /// DW_AT_low/high_pc or DW_AT_ranges (i.e. functions, lexical blocks, etc).
  /// \p DebugRangesOffset is the offset in .debug_ranges of the object's
  /// new address ranges in the output binary.
  /// \p Unit Compile unit the object belongs to.
  /// \p DIE is the object's DIE in the input binary.
  /// \p RangesBase if present, update \p DIE to use  DW_AT_GNU_ranges_base
  ///    attribute.
  void updateDWARFObjectAddressRanges(
      DWARFUnit &Unit, DIEBuilder &DIEBldr, DIE &Die,
      uint64_t DebugRangesOffset, uint64_t LowPCToUse,
      std::optional<uint64_t> RangesBase = std::nullopt);

  std::unique_ptr<DebugBufferVector>
  makeFinalLocListsSection(DWARFVersion Version);

  /// Finalize debug sections in the main binary.
  CUOffsetMap finalizeDebugSections(DIEBuilder &DIEBlder);

  /// Patches the binary for DWARF address ranges (e.g. in functions and lexical
  /// blocks) to be updated.
  void updateDebugAddressRanges();

  /// Rewrite .gdb_index section if present.
  void updateGdbIndexSection(CUOffsetMap &CUMap, uint32_t NumCUs);

  /// Output .dwo files.
  void writeDWOFiles(std::unordered_map<uint64_t, std::string> &DWOIdToName);

  /// Output .dwp files.
  void writeDWP(std::unordered_map<uint64_t, std::string> &DWOIdToName);

  /// DWARFDie contains a pointer to a DIE and hence gets invalidated once the
  /// embedded DIE is destroyed. This wrapper class stores a DIE internally and
  /// could be cast to a DWARFDie that is valid even after the initial DIE is
  /// destroyed.
  struct DWARFDieWrapper {
    DWARFUnit *Unit;
    DWARFDebugInfoEntry DIE;

    DWARFDieWrapper(DWARFUnit *Unit, DWARFDebugInfoEntry DIE)
        : Unit(Unit), DIE(DIE) {}

    DWARFDieWrapper(DWARFDie &Die)
        : Unit(Die.getDwarfUnit()), DIE(*Die.getDebugInfoEntry()) {}

    operator DWARFDie() { return DWARFDie(Unit, &DIE); }
  };

  /// Update \p DIE that was using DW_AT_(low|high)_pc with DW_AT_ranges offset.
  /// Updates to the DIE should be synced with abbreviation updates using the
  /// function above.
  void convertToRangesPatchDebugInfo(
      DWARFUnit &Unit, DIEBuilder &DIEBldr, DIE &Die,
      uint64_t RangesSectionOffset, DIEValue &LowPCAttrInfo,
      DIEValue &HighPCAttrInfo, uint64_t LowPCToUse,
      std::optional<uint64_t> RangesBase = std::nullopt);

  /// Adds a \p Str to .debug_str section.
  /// Uses \p AttrInfoVal to either update entry in a DIE for legacy DWARF using
  /// \p DebugInfoPatcher, or for DWARF5 update an index in .debug_str_offsets
  /// for this contribution of \p Unit.
  void addStringHelper(DIEBuilder &DIEBldr, DIE &Die, const DWARFUnit &Unit,
                       DIEValue &DIEAttrInfo, StringRef Str);

public:
  DWARFRewriter(BinaryContext &BC) : BC(BC) {}

  /// Main function for updating the DWARF debug info.
  void updateDebugInfo();

  /// Update stmt_list for CUs based on the new .debug_line \p Layout.
  void updateLineTableOffsets(const MCAsmLayout &Layout);

  /// Given a \p DWOId, return its DebugLocWriter if it exists.
  DebugLocWriter *getDebugLocWriter(uint64_t DWOId) {
    auto Iter = LocListWritersByCU.find(DWOId);
    return Iter == LocListWritersByCU.end() ? nullptr
                                            : LocListWritersByCU[DWOId].get();
  }

  StringRef getDwoDebugInfoStr(uint64_t DWOId) {
    return DwoDebugInfoMap[DWOId];
  }

  StringRef getDwoDebugAbbrevStr(uint64_t DWOId) {
    return DwoDebugAbbrevMap[DWOId];
  }

  StringRef getDwoDebugTypeStr(uint64_t DWOId) {
    return DwoDebugTypeMap[DWOId];
  }

  uint64_t getDwoRangesBase(uint64_t DWOId) { return DwoRangesBase[DWOId]; }

  void setDwoDebugInfoStr(uint64_t DWOId, StringRef Str) {
    DwoDebugInfoMap[DWOId] = Str.str();
  }

  void setDwoDebugAbbrevStr(uint64_t DWOId, StringRef Str) {
    DwoDebugAbbrevMap[DWOId] = Str.str();
  }

  void setDwoDebugTypeStr(uint64_t DWOId, StringRef Str) {
    DwoDebugTypeMap[DWOId] = Str.str();
  }

  void setDwoRangesBase(uint64_t DWOId, uint64_t RangesBase) {
    DwoRangesBase[DWOId] = RangesBase;
  }

  /// Adds an GDBIndexTUEntry if .gdb_index seciton exists.
  void addGDBTypeUnitEntry(const GDBIndexTUEntry &&Entry);

  /// Returns all entries needed for Types CU list
  const GDBIndexTUEntryType &getGDBIndexTUEntryVector() const {
    return GDBIndexTUEntryVector;
  }

  /// Stores meta data for each CU per DWO ID. It's used to create cu-index for
  /// DWARF5.
  void addCUnitMetaEntry(const uint64_t DWOId, const UnitMeta &Entry) {
    auto RetVal = CUnitMetaDwoMap.insert({DWOId, Entry});
    if (!RetVal.second)
      errs() << "BOLT-WARNING: [internal-dwarf-error]: Trying to set CU meta "
                "data twice for DWOID: "
             << Twine::utohexstr(DWOId) << ".\n";
  }

  /// Stores meta data for each TU per DWO ID. It's used to create cu-index for
  /// DWARF5.
  void addTUnitMetaEntry(const uint64_t DWOId, const UnitMeta &Entry) {
    TUnitMetaDwoMap[DWOId].emplace_back(Entry);
  }

  /// Returns Meta data for TUs in offset increasing order.
  UnitMetaVectorType &getTUnitMetaEntries(const uint64_t DWOId) {
    return TUnitMetaDwoMap[DWOId];
  }
  /// Returns Meta data for TUs in offset increasing order.
  const UnitMeta &getCUnitMetaEntry(const uint64_t DWOId) {
    return CUnitMetaDwoMap[DWOId];
  }
};

} // namespace bolt
} // namespace llvm

#endif
