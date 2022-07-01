//===- bolt/Rewrite/DWARFRewriter.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_DWARF_REWRITER_H
#define BOLT_REWRITE_DWARF_REWRITER_H

#include "bolt/Core/DebugData.h"
#include "llvm/MC/MCAsmLayout.h"
#include <cstdint>
#include <memory>
#include <mutex>
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
  using DebugTypesSignaturesPerCUMap =
      std::unordered_map<uint64_t, std::unordered_set<uint64_t>>;

private:
  BinaryContext &BC;

  std::mutex DebugInfoPatcherMutex;

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

  /// .debug_abbrev section writer for the main binary.
  std::unique_ptr<DebugAbbrevWriter> AbbrevWriter;

  using LocWriters = std::map<uint64_t, std::unique_ptr<DebugLocWriter>>;
  /// Use a separate location list writer for each compilation unit
  LocWriters LocListWritersByCU;

  using RangeListsDWOWriers =
      std::unordered_map<uint64_t,
                         std::unique_ptr<DebugRangeListsSectionWriter>>;
  /// Store Rangelists writer for each DWO CU.
  RangeListsDWOWriers RangeListsWritersByCU;

  using DebugAbbrevDWOWriters =
      std::unordered_map<uint64_t, std::unique_ptr<DebugAbbrevWriter>>;
  /// Abbrev section writers for DWOs.
  DebugAbbrevDWOWriters BinaryDWOAbbrevWriters;

  using DebugInfoDWOPatchers =
      std::unordered_map<uint64_t, std::unique_ptr<SimpleBinaryPatcher>>;
  /// Binary patchers for DWO debug_info sections.
  DebugInfoDWOPatchers BinaryDWODebugInfoPatchers;

  /// Stores all the Type Signatures for DWO CU.
  DebugTypesSignaturesPerCUMap TypeSignaturesPerCU;

  std::mutex LocListDebugInfoPatchesMutex;

  /// DWARFLegacy is all DWARF versions before DWARF 5.
  enum class DWARFVersion { DWARFLegacy, DWARF5 };

  /// Update debug info for all DIEs in \p Unit.
  void updateUnitDebugInfo(DWARFUnit &Unit,
                           DebugInfoBinaryPatcher &DebugInfoPatcher,
                           DebugAbbrevWriter &AbbrevWriter,
                           DebugLocWriter &DebugLocWriter,
                           DebugRangesSectionWriter &RangesWriter,
                           Optional<uint64_t> RangesBase = None);

  /// Patches the binary for an object's address ranges to be updated.
  /// The object can be anything that has associated address ranges via either
  /// DW_AT_low/high_pc or DW_AT_ranges (i.e. functions, lexical blocks, etc).
  /// \p DebugRangesOffset is the offset in .debug_ranges of the object's
  /// new address ranges in the output binary.
  /// \p Unit Compile unit the object belongs to.
  /// \p DIE is the object's DIE in the input binary.
  /// \p RangesBase if present, update \p DIE to use  DW_AT_GNU_ranges_base
  ///    attribute.
  void updateDWARFObjectAddressRanges(const DWARFDie DIE,
                                      uint64_t DebugRangesOffset,
                                      SimpleBinaryPatcher &DebugInfoPatcher,
                                      DebugAbbrevWriter &AbbrevWriter,
                                      Optional<uint64_t> RangesBase = None);

  std::unique_ptr<DebugBufferVector>
  makeFinalLocListsSection(DebugInfoBinaryPatcher &DebugInfoPatcher,
                           DWARFVersion Version);

  /// Finalize debug sections in the main binary.
  CUOffsetMap finalizeDebugSections(DebugInfoBinaryPatcher &DebugInfoPatcher);

  /// Patches the binary for DWARF address ranges (e.g. in functions and lexical
  /// blocks) to be updated.
  void updateDebugAddressRanges();

  /// Rewrite .gdb_index section if present.
  void updateGdbIndexSection(CUOffsetMap &CUMap);

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

  /// DIEs with abbrevs that were not converted to DW_AT_ranges.
  /// We only update those when all DIEs have been processed to guarantee that
  /// the abbrev (which is shared) is intact.
  using PendingRangesType = std::unordered_map<
      const DWARFAbbreviationDeclaration *,
      std::vector<std::pair<DWARFDieWrapper, DebugAddressRange>>>;

  /// Convert \p Abbrev from using a simple DW_AT_(low|high)_pc range to
  /// DW_AT_ranges with optional \p RangesBase.
  void convertToRangesPatchAbbrev(const DWARFUnit &Unit,
                                  const DWARFAbbreviationDeclaration *Abbrev,
                                  DebugAbbrevWriter &AbbrevWriter,
                                  Optional<uint64_t> RangesBase = None);

  /// Update \p DIE that was using DW_AT_(low|high)_pc with DW_AT_ranges offset.
  /// Updates to the DIE should be synced with abbreviation updates using the
  /// function above.
  void convertToRangesPatchDebugInfo(DWARFDie DIE, uint64_t RangesSectionOffset,
                                     SimpleBinaryPatcher &DebugInfoPatcher,
                                     Optional<uint64_t> RangesBase = None);

  /// Helper function for creating and returning per-DWO patchers/writers.
  template <class T, class Patcher>
  Patcher *getBinaryDWOPatcherHelper(T &BinaryPatchers, uint64_t DwoId) {
    std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
    auto Iter = BinaryPatchers.find(DwoId);
    if (Iter == BinaryPatchers.end()) {
      // Using make_pair instead of {} to work around bug in older version of
      // the library. https://timsong-cpp.github.io/lwg-issues/2354
      Iter = BinaryPatchers
                 .insert(std::make_pair(DwoId, std::make_unique<Patcher>()))
                 .first;
    }

    return static_cast<Patcher *>(Iter->second.get());
  }

  /// Adds a \p Str to .debug_str section.
  /// Uses \p AttrInfoVal to either update entry in a DIE for legacy DWARF using
  /// \p DebugInfoPatcher, or for DWARF5 update an index in .debug_str_offsets
  /// for this contribution of \p Unit.
  void addStringHelper(DebugInfoBinaryPatcher &DebugInfoPatcher,
                       const DWARFUnit &Unit, const AttrInfo &AttrInfoVal,
                       StringRef Str);

public:
  DWARFRewriter(BinaryContext &BC) : BC(BC) {}

  /// Main function for updating the DWARF debug info.
  void updateDebugInfo();

  /// Update stmt_list for CUs based on the new .debug_line \p Layout.
  void updateLineTableOffsets(const MCAsmLayout &Layout);

  /// Returns a DWO Debug Info Patcher for DWO ID.
  /// Creates a new instance if it does not already exist.
  SimpleBinaryPatcher *getBinaryDWODebugInfoPatcher(uint64_t DwoId) {
    return getBinaryDWOPatcherHelper<DebugInfoDWOPatchers,
                                     DebugInfoBinaryPatcher>(
        BinaryDWODebugInfoPatchers, DwoId);
  }

  /// Creates abbrev writer for DWO unit with \p DWOId.
  DebugAbbrevWriter *createBinaryDWOAbbrevWriter(DWARFContext &Context,
                                                 uint64_t DWOId) {
    std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
    auto &Entry = BinaryDWOAbbrevWriters[DWOId];
    Entry = std::make_unique<DebugAbbrevWriter>(Context, DWOId);
    return Entry.get();
  }

  /// Returns DWO abbrev writer for \p DWOId. The writer must exist.
  DebugAbbrevWriter *getBinaryDWOAbbrevWriter(uint64_t DWOId) {
    auto Iter = BinaryDWOAbbrevWriters.find(DWOId);
    assert(Iter != BinaryDWOAbbrevWriters.end() && "writer does not exist");
    return Iter->second.get();
  }

  /// Given a \p DWOId, return its DebugLocWriter if it exists.
  DebugLocWriter *getDebugLocWriter(uint64_t DWOId) {
    auto Iter = LocListWritersByCU.find(DWOId);
    return Iter == LocListWritersByCU.end() ? nullptr
                                            : LocListWritersByCU[DWOId].get();
  }
};

} // namespace bolt
} // namespace llvm

#endif
