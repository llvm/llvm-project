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
#include "bolt/Core/DebugNames.h"
#include "bolt/Core/GDBIndex.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/ToolOutputFile.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
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

  /// Stores ranges writer for each DWO CU.
  std::unordered_map<uint64_t, std::unique_ptr<DebugRangesSectionWriter>>
      LegacyRangesWritersByCU;

  /// Stores address writer for each CU.
  std::unordered_map<uint64_t, std::unique_ptr<DebugAddrWriter>>
      AddressWritersByCU;

  std::mutex LocListDebugInfoPatchesMutex;

  std::unordered_map<DWARFUnit *, uint64_t> LineTablePatchMap;
  std::unordered_map<const DWARFUnit *, uint64_t> TypeUnitRelocMap;

  /// Entries for GDB Index Types CU List
  using GDBIndexTUEntryType = std::vector<GDBIndexTUEntry>;
  GDBIndexTUEntryType GDBIndexTUEntryVector;

  /// DWARFLegacy is all DWARF versions before DWARF 5.
  enum class DWARFVersion { DWARFLegacy, DWARF5 };

  /// Used to track last CU offset for GDB Index.
  uint32_t CUOffset{0};

  /// Update debug info for all DIEs in \p Unit.
  void updateUnitDebugInfo(DWARFUnit &Unit, DIEBuilder &DIEBldr,
                           DebugLocWriter &DebugLocWriter,
                           DebugRangesSectionWriter &RangesSectionWriter,
                           DebugAddrWriter &AddressWriter,
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
      uint64_t DebugRangesOffset,
      std::optional<uint64_t> RangesBase = std::nullopt);

  std::unique_ptr<DebugBufferVector>
  makeFinalLocListsSection(DWARFVersion Version);

  /// Finalize type sections in the main binary.
  CUOffsetMap finalizeTypeSections(DIEBuilder &DIEBlder, DIEStreamer &Streamer,
                                   GDBIndex &GDBIndexSection);

  /// Process and write out CUs that are passsed in.
  void finalizeCompileUnits(DIEBuilder &DIEBlder, DIEStreamer &Streamer,
                            CUOffsetMap &CUMap,
                            const std::list<DWARFUnit *> &CUs,
                            DebugAddrWriter &FinalAddrWriter);

  /// Finalize debug sections in the main binary.
  void finalizeDebugSections(DIEBuilder &DIEBlder,
                             DWARF5AcceleratorTable &DebugNamesTable,
                             DIEStreamer &Streamer, raw_svector_ostream &ObjOS,
                             CUOffsetMap &CUMap,
                             DebugAddrWriter &FinalAddrWriter);

  /// Patches the binary for DWARF address ranges (e.g. in functions and lexical
  /// blocks) to be updated.
  void updateDebugAddressRanges();

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
      DIEValue &HighPCAttrInfo,
      std::optional<uint64_t> RangesBase = std::nullopt);

public:
  DWARFRewriter(BinaryContext &BC) : BC(BC) {}

  /// Main function for updating the DWARF debug info.
  void updateDebugInfo();

  /// Update stmt_list for CUs based on the new .debug_line \p Layout.
  void updateLineTableOffsets(const MCAssembler &Asm);

  using OverriddenSectionsMap = std::unordered_map<DWARFSectionKind, StringRef>;
  /// Output .dwo files.
  void writeDWOFiles(DWARFUnit &, const OverriddenSectionsMap &,
                     const std::string &, DebugLocWriter &,
                     DebugStrOffsetsWriter &, DebugStrWriter &,
                     DebugRangesSectionWriter &);
  using KnownSectionsEntry = std::pair<MCSection *, DWARFSectionKind>;
};

} // namespace bolt
} // namespace llvm

#endif
