//===- bolt/Core/DIEBuilder.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the DIEBuilder class, which is the
/// base class for Debug Information IR construction.
///
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_DIE_BUILDER_H
#define BOLT_CORE_DIE_BUILDER_H

#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/DebugNames.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/Allocator.h"

#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {

namespace bolt {

class DIEStreamer;
class DebugStrOffsetsWriter;

class DIEBuilder {
  friend DIEStreamer;

public:
  /// Wrapper around DIE so we can access DIEs easily.
  struct DIEInfo {
    DIE *Die;
    uint32_t DieId;
    uint32_t UnitId;
  };

  /// Contains information for the CU level of DWARF.
  struct DWARFUnitInfo {
    // Contains all the DIEs for the current unit.
    // Accessed by DIE ID.
    std::vector<std::unique_ptr<DIEInfo>> DieInfoVector;
    DIE *UnitDie = nullptr;
    uint32_t UnitId = 0;
    uint32_t UnitOffset = 0;
    uint32_t UnitLength = 0;
    bool IsConstructed = false;
    // A map of DIE offsets in original DWARF section to DIE ID.
    // Whih is used to access DieInfoVector.
    std::unordered_map<uint64_t, uint32_t> DIEIDMap;

    // Some STL implementations don't have a noexcept move constructor for
    // unordered_map (e.g. https://github.com/microsoft/STL/issues/165 explains
    // why the Microsoft STL doesn't). In that case, the default move
    // constructor generated for DWARFUnitInfo isn't noexcept either, and thus
    // resizing a vector of DWARFUnitInfo will copy elements instead of moving
    // them (https://en.cppreference.com/w/cpp/utility/move_if_noexcept).
    // DWARFUnitInfo isn't copyable though, since the DieInfoVector member is a
    // vector of unique_ptrs and unique_ptr isn't copyable, so using a vector of
    // DWARFUnitInfo causes build errors. Explicitly marking DWARFUnitInfo as
    // non-copyable forces vector resizes to move instead and fixes the issue.
    DWARFUnitInfo() = default;
    DWARFUnitInfo(const DWARFUnitInfo &) = delete;
    DWARFUnitInfo(DWARFUnitInfo &&) = default;
  };

  enum class ProcessingType { DWARF4TUs, DWARF5TUs, CUs };

private:
  /// Contains information so that we we can update references in locexpr after
  /// we calculated all the final DIE offsets.
  struct LocWithReference {
    LocWithReference(std::vector<uint8_t> &&BlockData, DWARFUnit &U, DIE &Die,
                     dwarf::Form Form, dwarf::Attribute Attr)
        : BlockData(BlockData), U(U), Die(Die), Form(Form), Attr(Attr) {}
    std::vector<uint8_t> BlockData;
    DWARFUnit &U;
    DIE &Die;
    dwarf::Form Form;
    dwarf::Attribute Attr;
  };
  /// Contains information so that we can update cross CU references, after we
  /// calculated all the final DIE offsets.
  struct AddrReferenceInfo {
    AddrReferenceInfo(DIEInfo *Die,
                      DWARFAbbreviationDeclaration::AttributeSpec Spec)
        : Dst(Die), AttrSpec(Spec) {}
    DIEInfo *Dst;
    DWARFAbbreviationDeclaration::AttributeSpec AttrSpec;
  };

  struct State {
    /// A map of Units to Unit Index.
    std::unordered_map<uint64_t, uint32_t> UnitIDMap;
    /// A map of Type Units to Type DIEs.
    std::unordered_map<DWARFUnit *, DIE *> TypeDIEMap;
    std::list<DWARFUnit *> DUList;
    std::vector<DWARFUnitInfo> CloneUnitCtxMap;
    std::vector<std::pair<DIEInfo *, AddrReferenceInfo>> AddrReferences;
    std::vector<DWARFUnit *> DWARF4TUVector;
    std::vector<DWARFUnit *> DWARF5TUVector;
    std::vector<DWARFUnit *> DWARFCUVector;
    std::vector<LocWithReference> LocWithReferencesToProcess;
    BumpPtrAllocator DIEAlloc;
    ProcessingType Type;
    std::unordered_set<uint64_t> DWARFDieAddressesParsed;
  };

  std::unique_ptr<State> BuilderState;
  FoldingSet<DIEAbbrev> AbbreviationsSet;
  std::vector<std::unique_ptr<DIEAbbrev>> Abbreviations;
  BinaryContext &BC;
  DWARFContext *DwarfContext{nullptr};
  DWARFUnit *SkeletonCU{nullptr};
  uint64_t UnitSize{0};
  /// Adds separate UnitSize counter for updating DebugNames
  /// so there is no dependency between the functions.
  uint64_t DebugNamesUnitSize{0};
  llvm::DenseSet<uint64_t> AllProcessed;
  DWARF5AcceleratorTable &DebugNamesTable;
  // Unordered map to handle name collision if output DWO directory is
  // specified.
  std::unordered_map<std::string, uint32_t> NameToIndexMap;

  /// Returns current state of the DIEBuilder
  State &getState() { return *BuilderState.get(); }

  /// Resolve the reference in DIE, if target is not loaded into IR,
  /// pre-allocate it. \p RefCU will be updated to the Unit specific by \p
  /// RefValue.
  DWARFDie resolveDIEReference(
      const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
      const uint64_t ReffOffset, DWARFUnit *&RefCU,
      DWARFDebugInfoEntry &DwarfDebugInfoEntry);

  /// Clone one attribute according to the format. \return the size of this
  /// attribute.
  void
  cloneAttribute(DIE &Die, const DWARFDie &InputDIE, DWARFUnit &U,
                 const DWARFFormValue &Val,
                 const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec);

  /// Clone an attribute in string format.
  void cloneStringAttribute(
      DIE &Die, const DWARFUnit &U,
      const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
      const DWARFFormValue &Val);

  /// Clone an attribute in reference format.
  void cloneDieOffsetReferenceAttribute(
      DIE &Die, const DWARFUnit &U, const DWARFDie &InputDIE,
      const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec, uint64_t Ref);

  /// Clone an attribute in block format.
  void cloneBlockAttribute(
      DIE &Die, DWARFUnit &U,
      const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
      const DWARFFormValue &Val);

  enum class CloneExpressionStage { INIT, PATCH };
  /// Clone an attribute in expression format. \p OutputBuffer will hold the
  /// output content.
  /// Returns true if Expression contains a reference.
  bool cloneExpression(const DataExtractor &Data,
                       const DWARFExpression &Expression, DWARFUnit &U,
                       SmallVectorImpl<uint8_t> &OutputBuffer,
                       const CloneExpressionStage &Stage);

  /// Clone an attribute in address format.
  void cloneAddressAttribute(
      DIE &Die, const DWARFUnit &U,
      const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
      const DWARFFormValue &Val);

  /// Clone an attribute in refsig format.
  void cloneRefsigAttribute(
      DIE &Die, const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
      const DWARFFormValue &Val);

  /// Clone an attribute in scalar format.
  void cloneScalarAttribute(
      DIE &Die, const DWARFDie &InputDIE,
      const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
      const DWARFFormValue &Val);

  /// Clone an attribute in loclist format.
  void cloneLoclistAttrubute(
      DIE &Die, const DWARFDie &InputDIE,
      const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
      const DWARFFormValue &Val);

  /// Update references once the layout is finalized.
  void updateReferences();

  /// Update the Offset and Size of DIE.
  /// Along with current CU, and DIE being processed and the new DIE offset to
  /// be updated, it takes in Parents vector that can be empty if this DIE has
  /// no parents.
  uint32_t finalizeDIEs(DWARFUnit &CU, DIE &Die, uint32_t &CurOffset);

  /// Populates DebugNames table.
  void populateDebugNamesTable(DWARFUnit &CU, const DIE &Die,
                               std::optional<BOLTDWARF5AccelTableData *> Parent,
                               uint32_t NumberParentsInChain);

  void registerUnit(DWARFUnit &DU, bool NeedSort);

  /// \return the unique ID of \p U if it exists.
  std::optional<uint32_t> getUnitId(const DWARFUnit &DU);

  DWARFUnitInfo &getUnitInfo(uint32_t UnitId) {
    return getState().CloneUnitCtxMap[UnitId];
  }

  DIEInfo &getDIEInfo(uint32_t UnitId, uint32_t DIEId) {
    if (getState().CloneUnitCtxMap[UnitId].DieInfoVector.size() > DIEId)
      return *getState().CloneUnitCtxMap[UnitId].DieInfoVector[DIEId].get();

    BC.errs()
        << "BOLT-WARNING: [internal-dwarf-error]: The DIE is not allocated "
           "before looking up, some"
        << "unexpected corner cases happened.\n";
    return *getState().CloneUnitCtxMap[UnitId].DieInfoVector.front().get();
  }

  std::optional<uint32_t> getAllocDIEId(const DWARFUnit &DU,
                                        const uint64_t Offset) {
    const DWARFUnitInfo &DWARFUnitInfo = getUnitInfoByDwarfUnit(DU);
    auto Iter = DWARFUnitInfo.DIEIDMap.find(Offset);
    return (Iter == DWARFUnitInfo.DIEIDMap.end())
               ? std::nullopt
               : std::optional<uint32_t>(Iter->second);
  }
  std::optional<uint32_t> getAllocDIEId(const DWARFUnit &DU,
                                        const DWARFDie &DDie) {
    return getAllocDIEId(DU, DDie.getOffset());
  }

  // To avoid overhead, do not use this unless we do get the DWARFUnitInfo
  // first. We can use getDIEInfo with UnitId and DieId
  DIEInfo &getDIEInfoByDwarfDie(DWARFDie &DwarfDie) {
    DWARFUnit &DwarfUnit = *DwarfDie.getDwarfUnit();
    std::optional<uint32_t> UnitId = getUnitId(DwarfUnit);
    std::optional<uint32_t> HasDieId = getAllocDIEId(DwarfUnit, DwarfDie);
    assert(HasDieId);

    return getDIEInfo(*UnitId, *HasDieId);
  }

  uint32_t allocDIE(const DWARFUnit &DU, const DWARFDie &DDie,
                    BumpPtrAllocator &Alloc, const uint32_t UId);

  /// Construct IR for \p DU. \p DUOffsetList specific the Unit in current
  /// Section.
  void constructFromUnit(DWARFUnit &DU);

  /// Construct a DIE for \p DDie in \p U. \p DUOffsetList specific the Unit in
  /// current Section.
  DIE *constructDIEFast(DWARFDie &DDie, DWARFUnit &U, uint32_t UnitId);

  /// Returns true if this DIEBUilder is for DWO Unit.
  bool isDWO() const { return SkeletonCU != nullptr; }

public:
  DIEBuilder(BinaryContext &BC, DWARFContext *DwarfContext,
             DWARF5AcceleratorTable &DebugNamesTable,
             DWARFUnit *SkeletonCU = nullptr);

  /// Returns enum to what we are currently processing.
  ProcessingType getCurrentProcessingState() { return getState().Type; }

  /// Constructs IR for Type Units.
  void buildTypeUnits(DebugStrOffsetsWriter *StrOffsetWriter = nullptr,
                      const bool Init = true);
  /// Constructs IR for all the CUs.
  void buildCompileUnits(const bool Init = true);
  /// Constructs IR for CUs in a vector.
  void buildCompileUnits(const std::vector<DWARFUnit *> &CUs);
  /// Preventing implicit conversions.
  template <class T> void buildCompileUnits(T) = delete;
  /// Builds DWO Unit. For DWARF5 this includes the type units.
  void buildDWOUnit(DWARFUnit &U);

  /// Returns DWARFUnitInfo for DWARFUnit
  DWARFUnitInfo &getUnitInfoByDwarfUnit(const DWARFUnit &DwarfUnit) {
    std::optional<uint32_t> UnitId = getUnitId(DwarfUnit);
    return getUnitInfo(*UnitId);
  }

  const std::vector<std::unique_ptr<DIEInfo>> &getDIEsByUnit(DWARFUnit &DU) {
    DWARFUnitInfo &U = getUnitInfoByDwarfUnit(DU);
    return U.DieInfoVector;
  }
  std::vector<std::unique_ptr<DIEAbbrev>> &getAbbrevs() {
    return Abbreviations;
  }
  DIE *getTypeDIE(DWARFUnit &DU) {
    if (getState().TypeDIEMap.count(&DU))
      return getState().TypeDIEMap[&DU];

    BC.errs()
        << "BOLT-ERROR: unable to find TypeUnit for Type Unit at offset 0x"
        << DU.getOffset() << "\n";
    return nullptr;
  }

  std::vector<DWARFUnit *> &getDWARF4TUVector() {
    return getState().DWARF4TUVector;
  }
  std::vector<DWARFUnit *> &getDWARF5TUVector() {
    return getState().DWARF5TUVector;
  }
  std::vector<DWARFUnit *> &getDWARFCUVector() {
    return getState().DWARFCUVector;
  }
  /// Returns list of CUs for which IR was build.
  std::list<DWARFUnit *> &getProcessedCUs() { return getState().DUList; }
  bool isEmpty() { return getState().CloneUnitCtxMap.empty(); }

  DIE *getUnitDIEbyUnit(const DWARFUnit &DU) {
    const DWARFUnitInfo &U = getUnitInfoByDwarfUnit(DU);
    return U.UnitDie;
  }

  /// Generate and populate all Abbrevs.
  void generateAbbrevs();
  void generateUnitAbbrevs(DIE *Die);
  void assignAbbrev(DIEAbbrev &Abbrev);

  /// Finish current DIE construction.
  void finish();

  /// Update debug names table.
  void updateDebugNamesTable();

  // Interface to edit DIE
  template <class T> T *allocateDIEValue() {
    return new (getState().DIEAlloc) T;
  }

  DIEValueList::value_iterator addValue(DIEValueList *Die, const DIEValue &V) {
    return Die->addValue(getState().DIEAlloc, V);
  }

  template <class T>
  DIEValueList::value_iterator addValue(DIEValueList *Die,
                                        dwarf::Attribute Attribute,
                                        dwarf::Form Form, T &&Value) {
    return Die->addValue(getState().DIEAlloc, Attribute, Form,
                         std::forward<T>(Value));
  }

  template <class T>
  bool replaceValue(DIEValueList *Die, dwarf::Attribute Attribute,
                    dwarf::Form Form, T &&NewValue) {
    return Die->replaceValue(getState().DIEAlloc, Attribute, Form,
                             std::forward<T>(NewValue));
  }

  template <class T>
  bool replaceValue(DIEValueList *Die, dwarf::Attribute Attribute,
                    dwarf::Attribute NewAttribute, dwarf::Form Form,
                    T &&NewValue) {
    return Die->replaceValue(getState().DIEAlloc, Attribute, NewAttribute, Form,
                             std::forward<T>(NewValue));
  }

  bool replaceValue(DIEValueList *Die, dwarf::Attribute Attribute,
                    dwarf::Form Form, DIEValue &NewValue) {
    return Die->replaceValue(getState().DIEAlloc, Attribute, Form, NewValue);
  }

  bool deleteValue(DIEValueList *Die, dwarf::Attribute Attribute) {
    return Die->deleteValue(Attribute);
  }
  /// Updates DWO Name and Compilation directory for Skeleton CU \p Unit.
  std::string updateDWONameCompDir(DebugStrOffsetsWriter &StrOffstsWriter,
                                   DebugStrWriter &StrWriter,
                                   DWARFUnit &SkeletonCU,
                                   std::optional<StringRef> DwarfOutputPath,
                                   std::optional<StringRef> DWONameToUse);
  /// Updates DWO Name and Compilation directory for Type Units.
  void updateDWONameCompDirForTypes(DebugStrOffsetsWriter &StrOffstsWriter,
                                    DebugStrWriter &StrWriter, DWARFUnit &Unit,
                                    std::optional<StringRef> DwarfOutputPath,
                                    const StringRef DWOName);
};
} // namespace bolt
} // namespace llvm

#endif
