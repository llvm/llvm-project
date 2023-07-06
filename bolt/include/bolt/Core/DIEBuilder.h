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
#include "llvm/DWARFLinker/DWARFLinkerCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <forward_list>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llvm {

namespace bolt {

class DIEBuilder {
  struct DIEInfo {
    DIE *Die;
    uint32_t DieId;
    uint32_t UnitId;
    uint32_t CanonicalDIEOffset;
  };

  struct UnitInfo {
    std::list<DIEInfo> DieInfoList;
    uint32_t UnitId;
    uint32_t UnitOffset;
    bool Isconstructed = false;
    uint32_t NewDieId = 0;
    std::unordered_map<uint64_t, uint32_t> DIEIDMap;
    std::unordered_map<uint32_t, DIEInfo *> DIEId2InfoMap;
  };

  struct ForwardReferenceInfo {
    ForwardReferenceInfo(DIEInfo *Die,
                         DWARFAbbreviationDeclaration::AttributeSpec spec)
        : Dst(Die), AttrSpec(spec) {}
    DIEInfo *Dst;
    DWARFAbbreviationDeclaration::AttributeSpec AttrSpec;
  };

  bool IsBuilt = false;
  std::unordered_map<DWARFUnit *, std::vector<DIE *>> UnitDIEs;
  std::unordered_map<DWARFUnit *, uint32_t> UnitIDMap;
  std::unordered_map<DWARFUnit *, DIE *> TypeDIEMap;
  std::vector<DWARFUnit *> DUList;
  std::vector<UnitInfo> CloneUnitCtxMap;
  std::vector<std::pair<DIEInfo *, ForwardReferenceInfo>> ForwardReferences;
  FoldingSet<DIEAbbrev> AbbreviationsSet;
  std::vector<std::unique_ptr<DIEAbbrev>> Abbreviations;
  std::vector<DWARFUnit *> DWARF4TUVector;
  std::vector<DWARFUnit *> DWARF4CUVector;
  BumpPtrAllocator DIEAlloc;

  /// Resolve the reference in DIE, if target is not loaded into IR,
  /// pre-allocate it. \p RefCU will be updated to the Unit specific by \p
  /// RefValue.
  DWARFDie
  resolveDIEReference(const DWARFFormValue &RefValue,
                      DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
                      DWARFUnit *&RefCU,
                      DWARFDebugInfoEntry &DwarfDebugInfoEntry,
                      const std::vector<DWARFUnit *> &DUOffsetList);

  /// Clone one attribute according the format. \return the size of this
  /// attribute.
  uint32_t
  cloneAttribute(DIE &Die, const DWARFDie &InputDIE, DWARFUnit &U,
                 const DWARFFormValue &Val,
                 const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
                 unsigned AttrSize, bool IsLittleEndian,
                 std::vector<DWARFUnit *> &DUOffsetList);

  /// Clone an attribute in string format.
  uint32_t cloneStringAttribute(
      DIE &Die, DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
      unsigned AttrSize, const DWARFFormValue &Val, const DWARFUnit &U);

  /// Clone an attribute in reference format.
  uint32_t cloneDieReferenceAttribute(
      DIE &Die, const DWARFDie &InputDIE,
      DWARFAbbreviationDeclaration::AttributeSpec AttrSpec, unsigned AttrSize,
      const DWARFFormValue &Val, DWARFUnit &U,
      std::vector<DWARFUnit *> &DUOffsetList);

  /// Clone an attribute in block format.
  uint32_t
  cloneBlockAttribute(DIE &Die, DWARFUnit &U,
                      DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
                      const DWARFFormValue &Val, unsigned AttrSize,
                      bool IsLittleEndian);

  /// Clone an attribute in expression format. \p OutputBuffer will hold the
  /// output content.
  void cloneExpression(DataExtractor &Data, DWARFExpression &Expression,
                       DWARFUnit &U, SmallVectorImpl<uint8_t> &OutputBuffer);

  /// Clone an attribute in address format.
  uint32_t
  cloneAddressAttribute(DIE &Die,
                        DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
                        const DWARFFormValue &Val, const DWARFUnit &U);

  /// Clone an attribute in Refsig format.
  uint32_t
  cloneRefsigAttribute(DIE &Die,
                       DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
                       unsigned AttrSize, const DWARFFormValue &Val);

  /// Clone an attribute in scalar format.
  uint32_t
  cloneScalarAttribute(DIE &Die, const DWARFDie &InputDIE,
                       DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
                       const DWARFFormValue &Val, unsigned AttrSize);

  /// Clone an attribute in loclist format.
  uint32_t
  cloneLoclistAttrubute(DIE &Die, const DWARFDie &InputDIE,
                        DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
                        const DWARFFormValue &Val, unsigned AttrSize);

  /// Update the Cross-CU reference offset.
  void computeReference();

  /// Update the Offset and Size of DIE.
  uint32_t computeDIEOffset(DWARFUnit &CU, DIE &Die, uint32_t &CurOffset);

  void registerUnit(DWARFUnit &Unit) {
    UnitIDMap[&Unit] = DUList.size();
    DUList.push_back(&Unit);
  }

  /// \return the unique ID of \p U if it exists.
  std::optional<uint32_t> getUnitId(DWARFUnit &DU) {
    if (UnitIDMap.count(&DU))
      return UnitIDMap[&DU];
    return std::nullopt;
  }

  UnitInfo &getUnitInfo(uint32_t UnitId) { return CloneUnitCtxMap[UnitId]; }

  DIEInfo &getDIEInfo(uint32_t UnitId, uint32_t DIEId) {
    if (CloneUnitCtxMap[UnitId].DIEId2InfoMap.count(DIEId))
      return *CloneUnitCtxMap[UnitId].DIEId2InfoMap[DIEId];

    errs() << "BOLT-ERROR: The DIE is not allocated before looking up, some"
           << "unexpected corner cases happened.\n";
    return CloneUnitCtxMap[UnitId].DieInfoList.front();
  }

  UnitInfo &getUnitInfoByDwarfUnit(DWARFUnit &DwarfUnit) {
    std::optional<uint32_t> UnitId = getUnitId(DwarfUnit);
    return getUnitInfo(*UnitId);
  }

  std::optional<uint32_t> getAllocDIEId(DWARFUnit &DU, DWARFDie &DDie) {
    UnitInfo &UnitInfo = getUnitInfoByDwarfUnit(DU);
    uint64_t Offset = DDie.getOffset();

    if (!UnitInfo.DIEIDMap.count(Offset))
      return std::nullopt;
    return UnitInfo.DIEIDMap[Offset];
  }

  std::optional<uint32_t> getAllocDIEId(DWARFUnit &DU, uint64_t Offset) {
    UnitInfo &UnitInfo = getUnitInfoByDwarfUnit(DU);

    if (!UnitInfo.DIEIDMap.count(Offset))
      return std::nullopt;
    return UnitInfo.DIEIDMap[Offset];
  }

  // To avoid overhead, do not use this unless we do get the UnitInfo first.
  // We can use getDIEInfo with UnitId and DieId
  DIEInfo &getDIEInfoByDwarfDie(DWARFDie *DwarfDie) {
    DWARFUnit *DwarfUnit = DwarfDie->getDwarfUnit();
    std::optional<uint32_t> UnitId = getUnitId(*DwarfUnit);
    std::optional<uint32_t> hasDIEId = getAllocDIEId(*DwarfUnit, *DwarfDie);
    assert(hasDIEId.has_value());

    return getDIEInfo(*UnitId, *hasDIEId);
  }

  std::optional<uint32_t> allocDIE(DWARFUnit &DU, DWARFDie &DDie,
                                   BumpPtrAllocator &Alloc, uint32_t UId,
                                   uint32_t offset = 0);

  uint32_t AllocDIEId(DWARFUnit &DU) {
    UnitInfo &UnitInfo = getUnitInfoByDwarfUnit(DU);
    return UnitInfo.NewDieId++;
  }

  /// Construct IR for \p DU. \p DUOffsetList specific the Unit in current
  /// Section.
  void constructFromUnit(DWARFUnit &DU, std::vector<DWARFUnit *> &DUOffsetList);

  /// Construct a DIE for \p DDie in \p U. \p DUOffsetList specific the Unit in
  /// current Section.
  DIE *constructDIEFast(DWARFDie &DDie, DWARFUnit &U, bool IsLittleEndian,
                        uint32_t UnitId,
                        std::vector<DWARFUnit *> &DUOffsetList);

public:
  DIEBuilder(DWARFContext *DwarfContext, bool IsDWO = false);

  std::vector<DIE *> getDIEsByUnit(DWARFUnit &DU) { return UnitDIEs[&DU]; }
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
  bool isEmpty() { return !IsBuilt; }

  DIE *getUnitDIEbyUnit(DWARFUnit &DU) {
    assert(UnitDIEs.count(&DU) && UnitDIEs[&DU].size() &&
           "DU is not constructed in IR");
    return UnitDIEs[&DU].front();
  }

  /// Generate and populate all Abbrevs.
  void generateAbbrevs();
  void generateUnitAbbrevs(DIE *die);
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
