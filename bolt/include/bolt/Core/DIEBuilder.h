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

#include "llvm/CodeGen/DIE.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace llvm {

namespace bolt {
class DIEStreamer;

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
  };

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

  /// A map of Units to Unit Index.
  std::unordered_map<uint64_t, uint32_t> UnitIDMap;
  /// A map of Type Units to Type DIEs.
  std::unordered_map<DWARFUnit *, DIE *> TypeDIEMap;
  std::vector<DWARFUnit *> DUList;
  std::vector<DWARFUnitInfo> CloneUnitCtxMap;
  std::vector<std::pair<DIEInfo *, AddrReferenceInfo>> AddrReferences;
  FoldingSet<DIEAbbrev> AbbreviationsSet;
  std::vector<std::unique_ptr<DIEAbbrev>> Abbreviations;
  std::vector<DWARFUnit *> DWARF4TUVector;
  std::vector<LocWithReference> LocWithReferencesToProcess;
  BumpPtrAllocator DIEAlloc;

  /// Resolve the reference in DIE, if target is not loaded into IR,
  /// pre-allocate it. \p RefCU will be updated to the Unit specific by \p
  /// RefValue.
  DWARFDie resolveDIEReference(const DWARFFormValue &RefValue,
                               DWARFUnit *&RefCU,
                               DWARFDebugInfoEntry &DwarfDebugInfoEntry,
                               const std::vector<DWARFUnit *> &DUOffsetList);

  /// Resolve the reference in DIE, if target is not loaded into IR,
  /// pre-allocate it. \p RefCU will be updated to the Unit specific by \p
  /// RefValue.
  DWARFDie resolveDIEReference(const uint64_t ReffOffset, DWARFUnit *&RefCU,
                               DWARFDebugInfoEntry &DwarfDebugInfoEntry,
                               const std::vector<DWARFUnit *> &DUOffsetList);

  /// Clone one attribute according to the format. \return the size of this
  /// attribute.
  void
  cloneAttribute(DIE &Die, const DWARFDie &InputDIE, DWARFUnit &U,
                 const DWARFFormValue &Val,
                 const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
                 const std::vector<DWARFUnit *> &DUOffsetList);

  /// Clone an attribute in string format.
  void cloneStringAttribute(
      DIE &Die, const DWARFUnit &U,
      const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
      const DWARFFormValue &Val);

  /// Clone an attribute in reference format.
  void cloneDieReferenceAttribute(
      DIE &Die, const DWARFUnit &U, const DWARFDie &InputDIE,
      const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
      const DWARFFormValue &Val, const std::vector<DWARFUnit *> &DUOffsetList);

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
  uint32_t computeDIEOffset(const DWARFUnit &CU, DIE &Die, uint32_t &CurOffset);

  void registerUnit(DWARFUnit &DU);

  /// \return the unique ID of \p U if it exists.
  std::optional<uint32_t> getUnitId(const DWARFUnit &DU);

  DWARFUnitInfo &getUnitInfo(uint32_t UnitId) {
    return CloneUnitCtxMap[UnitId];
  }

  DIEInfo &getDIEInfo(uint32_t UnitId, uint32_t DIEId) {
    if (CloneUnitCtxMap[UnitId].DieInfoVector.size() > DIEId)
      return *CloneUnitCtxMap[UnitId].DieInfoVector[DIEId].get();

    errs() << "BOLT-WARNING: [internal-dwarf-error]: The DIE is not allocated "
              "before looking up, some"
           << "unexpected corner cases happened.\n";
    return *CloneUnitCtxMap[UnitId].DieInfoVector.front().get();
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
  void constructFromUnit(DWARFUnit &DU, std::vector<DWARFUnit *> &DUOffsetList);

  /// Construct a DIE for \p DDie in \p U. \p DUOffsetList specific the Unit in
  /// current Section.
  DIE *constructDIEFast(DWARFDie &DDie, DWARFUnit &U, uint32_t UnitId,
                        std::vector<DWARFUnit *> &DUOffsetList);

public:
  DIEBuilder(DWARFContext *DwarfContext, bool IsDWO = false);

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
    if (TypeDIEMap.count(&DU))
      return TypeDIEMap[&DU];

    errs() << "BOLT-ERROR: unable to find TypeUnit for Type Unit at offset 0x"
           << DU.getOffset() << "\n";
    return nullptr;
  }

  std::vector<DWARFUnit *> getDWARF4TUVector() { return DWARF4TUVector; }
  bool isEmpty() { return CloneUnitCtxMap.empty(); }

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

  // Interface to edit DIE
  template <class T> T *allocateDIEValue() { return new (DIEAlloc) T; }

  DIEValueList::value_iterator addValue(DIEValueList *Die, const DIEValue &V) {
    return Die->addValue(DIEAlloc, V);
  }

  template <class T>
  DIEValueList::value_iterator addValue(DIEValueList *Die,
                                        dwarf::Attribute Attribute,
                                        dwarf::Form Form, T &&Value) {
    return Die->addValue(DIEAlloc, Attribute, Form, std::forward<T>(Value));
  }

  template <class T>
  bool replaceValue(DIEValueList *Die, dwarf::Attribute Attribute,
                    dwarf::Form Form, T &&NewValue) {
    return Die->replaceValue(DIEAlloc, Attribute, Form,
                             std::forward<T>(NewValue));
  }

  template <class T>
  bool replaceValue(DIEValueList *Die, dwarf::Attribute Attribute,
                    dwarf::Attribute NewAttribute, dwarf::Form Form,
                    T &&NewValue) {
    return Die->replaceValue(DIEAlloc, Attribute, NewAttribute, Form,
                             std::forward<T>(NewValue));
  }

  bool replaceValue(DIEValueList *Die, dwarf::Attribute Attribute,
                    dwarf::Form Form, DIEValue &NewValue) {
    return Die->replaceValue(DIEAlloc, Attribute, Form, NewValue);
  }

  template <class T>
  bool deleteValue(DIEValueList *Die, dwarf::Attribute Attribute) {
    return Die->deleteValue(Attribute);
  }
};
} // namespace bolt
} // namespace llvm

#endif
