//===- bolt/Core/DIEBuilder.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/DIEBuilder.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/ParallelUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFTypeUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"
#include "llvm/ObjectYAML/DWARFYAML.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/YAMLTraits.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt"
namespace opts {
extern cl::opt<unsigned> Verbosity;
}
namespace llvm {
namespace bolt {

void DIEBuilder::updateReferences() {
  for (auto &[SrcDIEInfo, ReferenceInfo] : getState().AddrReferences) {
    DIEInfo *DstDIEInfo = ReferenceInfo.Dst;
    DWARFUnitInfo &DstUnitInfo = getUnitInfo(DstDIEInfo->UnitId);
    dwarf::Attribute Attr = ReferenceInfo.AttrSpec.Attr;
    dwarf::Form Form = ReferenceInfo.AttrSpec.Form;

    const uint64_t NewAddr =
        DstDIEInfo->Die->getOffset() + DstUnitInfo.UnitOffset;
    SrcDIEInfo->Die->replaceValue(getState().DIEAlloc, Attr, Form,
                                  DIEInteger(NewAddr));
  }

  // Handling referenes in location expressions.
  for (LocWithReference &LocExpr : getState().LocWithReferencesToProcess) {
    SmallVector<uint8_t, 32> Buffer;
    DataExtractor Data(StringRef((const char *)LocExpr.BlockData.data(),
                                 LocExpr.BlockData.size()),
                       LocExpr.U.isLittleEndian(),
                       LocExpr.U.getAddressByteSize());
    DWARFExpression Expr(Data, LocExpr.U.getAddressByteSize(),
                         LocExpr.U.getFormParams().Format);
    cloneExpression(Data, Expr, LocExpr.U, Buffer, CloneExpressionStage::PATCH);

    DIEValueList *AttrVal;
    if (LocExpr.Form == dwarf::DW_FORM_exprloc) {
      DIELoc *DL = new (getState().DIEAlloc) DIELoc;
      DL->setSize(Buffer.size());
      AttrVal = static_cast<DIEValueList *>(DL);
    } else {
      DIEBlock *DBL = new (getState().DIEAlloc) DIEBlock;
      DBL->setSize(Buffer.size());
      AttrVal = static_cast<DIEValueList *>(DBL);
    }
    for (auto Byte : Buffer)
      AttrVal->addValue(getState().DIEAlloc, static_cast<dwarf::Attribute>(0),
                        dwarf::DW_FORM_data1, DIEInteger(Byte));

    DIEValue Value;
    if (LocExpr.Form == dwarf::DW_FORM_exprloc)
      Value =
          DIEValue(dwarf::Attribute(LocExpr.Attr), dwarf::Form(LocExpr.Form),
                   static_cast<DIELoc *>(AttrVal));
    else
      Value =
          DIEValue(dwarf::Attribute(LocExpr.Attr), dwarf::Form(LocExpr.Form),
                   static_cast<DIEBlock *>(AttrVal));

    LocExpr.Die.replaceValue(getState().DIEAlloc, LocExpr.Attr, LocExpr.Form,
                             Value);
  }

  return;
}

uint32_t DIEBuilder::allocDIE(const DWARFUnit &DU, const DWARFDie &DDie,
                              BumpPtrAllocator &Alloc, const uint32_t UId) {
  DWARFUnitInfo &DWARFUnitInfo = getUnitInfo(UId);
  const uint64_t DDieOffset = DDie.getOffset();
  if (DWARFUnitInfo.DIEIDMap.count(DDieOffset))
    return DWARFUnitInfo.DIEIDMap[DDieOffset];

  DIE *Die = DIE::get(Alloc, dwarf::Tag(DDie.getTag()));
  // This handles the case where there is a DIE ref which points to
  // invalid DIE. This prevents assert when IR is written out.
  // Also it makes debugging easier.
  // DIE dump is not very useful.
  // It's nice to know original offset from which this DIE was constructed.
  Die->setOffset(DDie.getOffset());
  if (opts::Verbosity >= 1)
    getState().DWARFDieAddressesParsed.insert(DDie.getOffset());
  const uint32_t DId = DWARFUnitInfo.DieInfoVector.size();
  DWARFUnitInfo.DIEIDMap[DDieOffset] = DId;
  DWARFUnitInfo.DieInfoVector.emplace_back(
      std::make_unique<DIEInfo>(DIEInfo{Die, DId, UId}));
  return DId;
}

void DIEBuilder::constructFromUnit(DWARFUnit &DU) {
  std::optional<uint32_t> UnitId = getUnitId(DU);
  if (!UnitId) {
    errs() << "BOLT-WARNING: [internal-dwarf-error]: "
           << "Skip Unit at " << Twine::utohexstr(DU.getOffset()) << "\n";
    return;
  }

  const uint32_t UnitHeaderSize = DU.getHeaderSize();
  uint64_t DIEOffset = DU.getOffset() + UnitHeaderSize;
  uint64_t NextCUOffset = DU.getNextUnitOffset();
  DWARFDataExtractor DebugInfoData = DU.getDebugInfoExtractor();
  DWARFDebugInfoEntry DIEEntry;
  std::vector<DIE *> CurParentDIEStack;
  std::vector<uint32_t> Parents;
  uint32_t TUTypeOffset = 0;

  if (DWARFTypeUnit *TU = dyn_cast_or_null<DWARFTypeUnit>(&DU))
    TUTypeOffset = TU->getTypeOffset();

  assert(DebugInfoData.isValidOffset(NextCUOffset - 1));
  Parents.push_back(UINT32_MAX);
  do {
    const bool IsTypeDIE = (TUTypeOffset == DIEOffset - DU.getOffset());
    if (!DIEEntry.extractFast(DU, &DIEOffset, DebugInfoData, NextCUOffset,
                              Parents.back()))
      break;

    if (const DWARFAbbreviationDeclaration *AbbrDecl =
            DIEEntry.getAbbreviationDeclarationPtr()) {
      DWARFDie DDie(&DU, &DIEEntry);

      DIE *CurDIE = constructDIEFast(DDie, DU, *UnitId);
      DWARFUnitInfo &UI = getUnitInfo(*UnitId);
      // Can't rely on first element in DieVector due to cross CU forward
      // references.
      if (!UI.UnitDie)
        UI.UnitDie = CurDIE;
      if (IsTypeDIE)
        getState().TypeDIEMap[&DU] = CurDIE;

      if (!CurParentDIEStack.empty())
        CurParentDIEStack.back()->addChild(CurDIE);

      if (AbbrDecl->hasChildren())
        CurParentDIEStack.push_back(CurDIE);
    } else {
      // NULL DIE: finishes current children scope.
      CurParentDIEStack.pop_back();
    }
  } while (CurParentDIEStack.size() > 0);

  getState().CloneUnitCtxMap[*UnitId].IsConstructed = true;
}

DIEBuilder::DIEBuilder(DWARFContext *DwarfContext, bool IsDWO)
    : DwarfContext(DwarfContext), IsDWO(IsDWO) {}

static unsigned int getCUNum(DWARFContext *DwarfContext, bool IsDWO) {
  unsigned int CUNum = IsDWO ? DwarfContext->getNumDWOCompileUnits()
                             : DwarfContext->getNumCompileUnits();
  CUNum += IsDWO ? DwarfContext->getNumDWOTypeUnits()
                 : DwarfContext->getNumTypeUnits();
  return CUNum;
}

void DIEBuilder::buildTypeUnits(DebugStrOffsetsWriter *StrOffsetWriter,
                                const bool Init) {
  if (Init)
    BuilderState.reset(new State());

  const DWARFUnitIndex &TUIndex = DwarfContext->getTUIndex();
  if (!TUIndex.getRows().empty()) {
    for (auto &Row : TUIndex.getRows()) {
      uint64_t Signature = Row.getSignature();
      // manually populate TypeUnit to UnitVector
      DwarfContext->getTypeUnitForHash(DwarfContext->getMaxVersion(), Signature,
                                       true);
    }
  }
  const unsigned int CUNum = getCUNum(DwarfContext, IsDWO);
  getState().CloneUnitCtxMap.resize(CUNum);
  DWARFContext::unit_iterator_range CU4TURanges =
      IsDWO ? DwarfContext->dwo_types_section_units()
            : DwarfContext->types_section_units();

  getState().Type = ProcessingType::DWARF4TUs;
  for (std::unique_ptr<DWARFUnit> &DU : CU4TURanges)
    registerUnit(*DU.get(), false);

  for (std::unique_ptr<DWARFUnit> &DU : CU4TURanges)
    constructFromUnit(*DU.get());

  DWARFContext::unit_iterator_range CURanges =
      IsDWO ? DwarfContext->dwo_info_section_units()
            : DwarfContext->info_section_units();

  // This handles DWARF4 CUs and DWARF5 CU/TUs.
  // Creating a vector so that for reference handling only DWARF5 CU/TUs are
  // used, and not DWARF4 TUs.
  getState().Type = ProcessingType::DWARF5TUs;
  for (std::unique_ptr<DWARFUnit> &DU : CURanges) {
    if (!DU->isTypeUnit())
      continue;
    registerUnit(*DU.get(), false);
  }

  for (DWARFUnit *DU : getState().DWARF5TUVector) {
    constructFromUnit(*DU);
    if (StrOffsetWriter)
      StrOffsetWriter->finalizeSection(*DU, *this);
  }
}

void DIEBuilder::buildCompileUnits(const bool Init) {
  if (Init)
    BuilderState.reset(new State());

  unsigned int CUNum = getCUNum(DwarfContext, IsDWO);
  getState().CloneUnitCtxMap.resize(CUNum);
  DWARFContext::unit_iterator_range CURanges =
      IsDWO ? DwarfContext->dwo_info_section_units()
            : DwarfContext->info_section_units();

  // This handles DWARF4 CUs and DWARF5 CU/TUs.
  // Creating a vector so that for reference handling only DWARF5 CU/TUs are
  // used, and not DWARF4 TUs.getState().DUList
  getState().Type = ProcessingType::CUs;
  for (std::unique_ptr<DWARFUnit> &DU : CURanges) {
    if (DU->isTypeUnit())
      continue;
    registerUnit(*DU.get(), false);
  }

  // Using DULIst since it can be modified by cross CU refrence resolution.
  for (DWARFUnit *DU : getState().DUList) {
    if (DU->isTypeUnit())
      continue;
    constructFromUnit(*DU);
  }
}
void DIEBuilder::buildCompileUnits(const std::vector<DWARFUnit *> &CUs) {
  BuilderState.reset(new State());
  // Allocating enough for current batch being processed.
  // In real use cases we either processing a batch of CUs with no cross
  // references, or if they do have them it is due to LTO. With clang they will
  // share the same abbrev table. In either case this vector will not grow.
  getState().CloneUnitCtxMap.resize(CUs.size());
  getState().Type = ProcessingType::CUs;
  for (DWARFUnit *CU : CUs)
    registerUnit(*CU, false);

  for (DWARFUnit *DU : getState().DUList)
    constructFromUnit(*DU);
}

void DIEBuilder::buildDWOUnit(DWARFUnit &U) {
  BuilderState.release();
  BuilderState = std::make_unique<State>();
  buildTypeUnits(nullptr, false);
  getState().Type = ProcessingType::CUs;
  registerUnit(U, false);
  constructFromUnit(U);
}

DIE *DIEBuilder::constructDIEFast(DWARFDie &DDie, DWARFUnit &U,
                                  uint32_t UnitId) {

  std::optional<uint32_t> Idx = getAllocDIEId(U, DDie);
  if (Idx) {
    DWARFUnitInfo &DWARFUnitInfo = getUnitInfo(UnitId);
    DIEInfo &DieInfo = getDIEInfo(UnitId, *Idx);
    if (DWARFUnitInfo.IsConstructed && DieInfo.Die)
      return DieInfo.Die;
  } else {
    Idx = allocDIE(U, DDie, getState().DIEAlloc, UnitId);
  }

  DIEInfo &DieInfo = getDIEInfo(UnitId, *Idx);

  uint64_t Offset = DDie.getOffset();
  uint64_t NextOffset = Offset;
  DWARFDataExtractor Data = U.getDebugInfoExtractor();
  DWARFDebugInfoEntry DDIEntry;

  if (DDIEntry.extractFast(U, &NextOffset, Data, U.getNextUnitOffset(), 0))
    assert(NextOffset - U.getOffset() <= Data.getData().size() &&
           "NextOffset OOB");

  SmallString<40> DIECopy(Data.getData().substr(Offset, NextOffset - Offset));
  Data =
      DWARFDataExtractor(DIECopy, Data.isLittleEndian(), Data.getAddressSize());

  const DWARFAbbreviationDeclaration *Abbrev =
      DDie.getAbbreviationDeclarationPtr();
  uint64_t AttrOffset = getULEB128Size(Abbrev->getCode());

  using AttrSpec = DWARFAbbreviationDeclaration::AttributeSpec;
  for (const AttrSpec &AttrSpec : Abbrev->attributes()) {
    DWARFFormValue Val(AttrSpec.Form);
    Val.extractValue(Data, &AttrOffset, U.getFormParams(), &U);
    cloneAttribute(*DieInfo.Die, DDie, U, Val, AttrSpec);
  }
  return DieInfo.Die;
}

static DWARFUnit *
getUnitForOffset(DIEBuilder &Builder, DWARFContext &DWCtx,
                 const uint64_t Offset,
                 const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec) {
  auto findUnit = [&](std::vector<DWARFUnit *> &Units) -> DWARFUnit * {
    auto CUIter = llvm::upper_bound(Units, Offset,
                                    [](uint64_t LHS, const DWARFUnit *RHS) {
                                      return LHS < RHS->getNextUnitOffset();
                                    });
    static std::vector<DWARFUnit *> CUOffsets;
    static std::once_flag InitVectorFlag;
    auto initCUVector = [&]() {
      CUOffsets.reserve(DWCtx.getNumCompileUnits());
      for (const std::unique_ptr<DWARFUnit> &CU : DWCtx.compile_units())
        CUOffsets.emplace_back(CU.get());
    };
    DWARFUnit *CU = CUIter != Units.end() ? *CUIter : nullptr;
    // Above algorithm breaks when there is only one CU, and reference is
    // outside of it. Fall through slower path, that searches all the CUs.
    // For example when src and destination of cross CU references have
    // different abbrev section.
    if (!CU ||
        (CU && AttrSpec.Form == dwarf::DW_FORM_ref_addr &&
         !(CU->getOffset() < Offset && CU->getNextUnitOffset() > Offset))) {
      // This is a work around for XCode clang. There is a build error when we
      // pass DWCtx.compile_units() to llvm::upper_bound
      std::call_once(InitVectorFlag, initCUVector);
      auto CUIter = std::upper_bound(CUOffsets.begin(), CUOffsets.end(), Offset,
                                     [](uint64_t LHS, const DWARFUnit *RHS) {
                                       return LHS < RHS->getNextUnitOffset();
                                     });
      CU = CUIter != CUOffsets.end() ? (*CUIter) : nullptr;
    }
    return CU;
  };

  switch (Builder.getCurrentProcessingState()) {
  case DIEBuilder::ProcessingType::DWARF4TUs:
    return findUnit(Builder.getDWARF4TUVector());
  case DIEBuilder::ProcessingType::DWARF5TUs:
    return findUnit(Builder.getDWARF5TUVector());
  case DIEBuilder::ProcessingType::CUs:
    return findUnit(Builder.getDWARFCUVector());
  };

  return nullptr;
}

uint32_t DIEBuilder::computeDIEOffset(const DWARFUnit &CU, DIE &Die,
                                      uint32_t &CurOffset) {
  getState().DWARFDieAddressesParsed.erase(Die.getOffset());
  uint32_t CurSize = 0;
  Die.setOffset(CurOffset);
  for (DIEValue &Val : Die.values())
    CurSize += Val.sizeOf(CU.getFormParams());
  CurSize += getULEB128Size(Die.getAbbrevNumber());
  CurOffset += CurSize;

  for (DIE &Child : Die.children()) {
    uint32_t ChildSize = computeDIEOffset(CU, Child, CurOffset);
    CurSize += ChildSize;
  }
  // for children end mark.
  if (Die.hasChildren()) {
    CurSize += sizeof(uint8_t);
    CurOffset += sizeof(uint8_t);
  }

  Die.setSize(CurSize);

  return CurSize;
}

void DIEBuilder::finish() {
  auto computeOffset = [&](const DWARFUnit &CU,
                           uint64_t &UnitStartOffset) -> void {
    DIE *UnitDIE = getUnitDIEbyUnit(CU);
    uint32_t HeaderSize = CU.getHeaderSize();
    uint32_t CurOffset = HeaderSize;
    computeDIEOffset(CU, *UnitDIE, CurOffset);

    DWARFUnitInfo &CurUnitInfo = getUnitInfoByDwarfUnit(CU);
    CurUnitInfo.UnitOffset = UnitStartOffset;
    CurUnitInfo.UnitLength = HeaderSize + UnitDIE->getSize();
    UnitStartOffset += CurUnitInfo.UnitLength;
  };
  // Computing offsets for .debug_types section.
  // It's processed first when CU is registered so will be at the begginnig of
  // the vector.
  uint64_t TypeUnitStartOffset = 0;
  for (const DWARFUnit *CU : getState().DUList) {
    // We process DWARF$ types first.
    if (!(CU->getVersion() < 5 && CU->isTypeUnit()))
      break;
    computeOffset(*CU, TypeUnitStartOffset);
  }

  for (const DWARFUnit *CU : getState().DUList) {
    // Skipping DWARF4 types.
    if (CU->getVersion() < 5 && CU->isTypeUnit())
      continue;
    computeOffset(*CU, UnitSize);
  }
  if (opts::Verbosity >= 1) {
    if (!getState().DWARFDieAddressesParsed.empty())
      dbgs() << "Referenced DIE offsets not in .debug_info\n";
    for (const uint64_t Address : getState().DWARFDieAddressesParsed) {
      dbgs() << Twine::utohexstr(Address) << "\n";
    }
  }
  updateReferences();
}

DWARFDie DIEBuilder::resolveDIEReference(
    const DWARFFormValue &RefValue,
    const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    DWARFUnit *&RefCU, DWARFDebugInfoEntry &DwarfDebugInfoEntry) {
  assert(RefValue.isFormClass(DWARFFormValue::FC_Reference));
  uint64_t RefOffset = *RefValue.getAsReference();
  return resolveDIEReference(AttrSpec, RefOffset, RefCU, DwarfDebugInfoEntry);
}

DWARFDie DIEBuilder::resolveDIEReference(
    const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const uint64_t RefOffset, DWARFUnit *&RefCU,
    DWARFDebugInfoEntry &DwarfDebugInfoEntry) {
  uint64_t TmpRefOffset = RefOffset;
  if ((RefCU =
           getUnitForOffset(*this, *DwarfContext, TmpRefOffset, AttrSpec))) {
    /// Trying to add to current working set in case it's cross CU reference.
    registerUnit(*RefCU, true);
    DWARFDataExtractor DebugInfoData = RefCU->getDebugInfoExtractor();
    if (DwarfDebugInfoEntry.extractFast(*RefCU, &TmpRefOffset, DebugInfoData,
                                        RefCU->getNextUnitOffset(), 0)) {
      // In a file with broken references, an attribute might point to a NULL
      // DIE.
      DWARFDie RefDie = DWARFDie(RefCU, &DwarfDebugInfoEntry);
      if (!RefDie.isNULL()) {
        std::optional<uint32_t> UnitId = getUnitId(*RefCU);

        // forward reference
        if (UnitId && !getState().CloneUnitCtxMap[*UnitId].IsConstructed &&
            !getAllocDIEId(*RefCU, RefDie))
          allocDIE(*RefCU, RefDie, getState().DIEAlloc, *UnitId);
        return RefDie;
      }
      errs() << "BOLT-WARNING: [internal-dwarf-error]: invalid referenced DIE "
                "at offset: "
             << Twine::utohexstr(RefOffset) << ".\n";

    } else {
      errs() << "BOLT-WARNING: [internal-dwarf-error]: could not parse "
                "referenced DIE at offset: "
             << Twine::utohexstr(RefOffset) << ".\n";
    }
  } else {
    errs() << "BOLT-WARNING: [internal-dwarf-error]: could not find referenced "
              "CU. Referenced DIE offset: "
           << Twine::utohexstr(RefOffset) << ".\n";
  }
  return DWARFDie();
}

void DIEBuilder::cloneDieReferenceAttribute(
    DIE &Die, const DWARFUnit &U, const DWARFDie &InputDIE,
    const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const DWARFFormValue &Val) {
  const uint64_t Ref = *Val.getAsReference();

  DIE *NewRefDie = nullptr;
  DWARFUnit *RefUnit = nullptr;

  DWARFDebugInfoEntry DDIEntry;
  const DWARFDie RefDie = resolveDIEReference(Val, AttrSpec, RefUnit, DDIEntry);

  if (!RefDie)
    return;

  const std::optional<uint32_t> UnitId = getUnitId(*RefUnit);
  const std::optional<uint32_t> IsAllocId = getAllocDIEId(*RefUnit, RefDie);
  assert(IsAllocId.has_value() && "Encountered unexpected unallocated DIE.");
  const uint32_t DIEId = *IsAllocId;
  DIEInfo &DieInfo = getDIEInfo(*UnitId, DIEId);

  if (!DieInfo.Die) {
    assert(Ref > InputDIE.getOffset());
    (void)Ref;
    errs() << "BOLT-WARNING: [internal-dwarf-error]: encounter unexpected "
              "unallocated DIE. Should be alloc!\n";
    // We haven't cloned this DIE yet. Just create an empty one and
    // store it. It'll get really cloned when we process it.
    DieInfo.Die = DIE::get(getState().DIEAlloc, dwarf::Tag(RefDie.getTag()));
  }
  NewRefDie = DieInfo.Die;

  if (AttrSpec.Form == dwarf::DW_FORM_ref_addr) {
    // no matter forward reference or backward reference, we are supposed
    // to calculate them in `finish` due to the possible modification of
    // the DIE.
    DWARFDie CurDie = const_cast<DWARFDie &>(InputDIE);
    DIEInfo *CurDieInfo = &getDIEInfoByDwarfDie(CurDie);
    getState().AddrReferences.push_back(
        std::make_pair(CurDieInfo, AddrReferenceInfo(&DieInfo, AttrSpec)));

    Die.addValue(getState().DIEAlloc, AttrSpec.Attr, dwarf::DW_FORM_ref_addr,
                 DIEInteger(0xDEADBEEF));
    return;
  }

  Die.addValue(getState().DIEAlloc, AttrSpec.Attr, AttrSpec.Form,
               DIEEntry(*NewRefDie));
}

void DIEBuilder::cloneStringAttribute(
    DIE &Die, const DWARFUnit &U,
    const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const DWARFFormValue &Val) {
  if (AttrSpec.Form == dwarf::DW_FORM_string) {
    Expected<const char *> StrAddr = Val.getAsCString();
    if (!StrAddr) {
      consumeError(StrAddr.takeError());
      return;
    }
    Die.addValue(getState().DIEAlloc, AttrSpec.Attr, dwarf::DW_FORM_string,
                 new (getState().DIEAlloc)
                     DIEInlineString(StrAddr.get(), getState().DIEAlloc));
  } else {
    std::optional<uint64_t> OffsetIndex = Val.getRawUValue();
    Die.addValue(getState().DIEAlloc, AttrSpec.Attr, AttrSpec.Form,
                 DIEInteger(*OffsetIndex));
  }
}

bool DIEBuilder::cloneExpression(const DataExtractor &Data,
                                 const DWARFExpression &Expression,
                                 DWARFUnit &U,
                                 SmallVectorImpl<uint8_t> &OutputBuffer,
                                 const CloneExpressionStage &Stage) {
  using Encoding = DWARFExpression::Operation::Encoding;
  using Descr = DWARFExpression::Operation::Description;
  uint64_t OpOffset = 0;
  bool DoesContainReference = false;
  for (const DWARFExpression::Operation &Op : Expression) {
    const Descr &Description = Op.getDescription();
    // DW_OP_const_type is variable-length and has 3
    // operands. Thus far we only support 2.
    if ((Description.Op.size() == 2 &&
         Description.Op[0] == Encoding::BaseTypeRef) ||
        (Description.Op.size() == 2 &&
         Description.Op[1] == Encoding::BaseTypeRef &&
         Description.Op[0] != Encoding::Size1))
      outs() << "BOLT-WARNING: [internal-dwarf-error]: unsupported DW_OP "
                "encoding.\n";

    if ((Description.Op.size() == 1 &&
         Description.Op[0] == Encoding::BaseTypeRef) ||
        (Description.Op.size() == 2 &&
         Description.Op[1] == Encoding::BaseTypeRef &&
         Description.Op[0] == Encoding::Size1)) {
      // This code assumes that the other non-typeref operand fits into 1
      // byte.
      assert(OpOffset < Op.getEndOffset());
      const uint32_t ULEBsize = Op.getEndOffset() - OpOffset - 1;
      (void)ULEBsize;
      assert(ULEBsize <= 16);

      // Copy over the operation.
      OutputBuffer.push_back(Op.getCode());
      uint64_t RefOffset;
      if (Description.Op.size() == 1) {
        RefOffset = Op.getRawOperand(0);
      } else {
        OutputBuffer.push_back(Op.getRawOperand(0));
        RefOffset = Op.getRawOperand(1);
      }
      uint32_t Offset = 0;
      if (RefOffset > 0 || Op.getCode() != dwarf::DW_OP_convert) {
        DoesContainReference = true;
        std::optional<uint32_t> RefDieID =
            getAllocDIEId(U, U.getOffset() + RefOffset);
        std::optional<uint32_t> RefUnitID = getUnitId(U);
        if (RefDieID.has_value() && RefUnitID.has_value()) {
          DIEInfo &RefDieInfo = getDIEInfo(*RefUnitID, *RefDieID);
          if (DIE *Clone = RefDieInfo.Die)
            Offset = Stage == CloneExpressionStage::INIT ? RefOffset
                                                         : Clone->getOffset();
          else
            errs() << "BOLT-WARNING: [internal-dwarf-error]: base type ref "
                      "doesn't point to "
                      "DW_TAG_base_type.\n";
        }
      }
      uint8_t ULEB[16];
      // Hard coding to max size so size doesn't change when we update the
      // offset.
      encodeULEB128(Offset, ULEB, 4);
      ArrayRef<uint8_t> ULEBbytes(ULEB, 4);
      OutputBuffer.append(ULEBbytes.begin(), ULEBbytes.end());
    } else {
      // Copy over everything else unmodified.
      const StringRef Bytes = Data.getData().slice(OpOffset, Op.getEndOffset());
      OutputBuffer.append(Bytes.begin(), Bytes.end());
    }
    OpOffset = Op.getEndOffset();
  }
  return DoesContainReference;
}

void DIEBuilder::cloneBlockAttribute(
    DIE &Die, DWARFUnit &U,
    const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const DWARFFormValue &Val) {
  DIEValueList *Attr;
  DIEValue Value;
  DIELoc *Loc = nullptr;
  DIEBlock *Block = nullptr;

  if (AttrSpec.Form == dwarf::DW_FORM_exprloc) {
    Loc = new (getState().DIEAlloc) DIELoc;
  } else if (doesFormBelongToClass(AttrSpec.Form, DWARFFormValue::FC_Block,
                                   U.getVersion())) {
    Block = new (getState().DIEAlloc) DIEBlock;
  } else {
    errs() << "BOLT-WARNING: [internal-dwarf-error]: Unexpected Form value in "
              "cloneBlockAttribute\n";
    return;
  }
  Attr = Loc ? static_cast<DIEValueList *>(Loc)
             : static_cast<DIEValueList *>(Block);

  SmallVector<uint8_t, 32> Buffer;
  ArrayRef<uint8_t> Bytes = *Val.getAsBlock();
  if (DWARFAttribute::mayHaveLocationExpr(AttrSpec.Attr) &&
      (Val.isFormClass(DWARFFormValue::FC_Block) ||
       Val.isFormClass(DWARFFormValue::FC_Exprloc))) {
    DataExtractor Data(StringRef((const char *)Bytes.data(), Bytes.size()),
                       U.isLittleEndian(), U.getAddressByteSize());
    DWARFExpression Expr(Data, U.getAddressByteSize(),
                         U.getFormParams().Format);
    if (cloneExpression(Data, Expr, U, Buffer, CloneExpressionStage::INIT))
      getState().LocWithReferencesToProcess.emplace_back(
          Bytes.vec(), U, Die, AttrSpec.Form, AttrSpec.Attr);
    Bytes = Buffer;
  }
  for (auto Byte : Bytes)
    Attr->addValue(getState().DIEAlloc, static_cast<dwarf::Attribute>(0),
                   dwarf::DW_FORM_data1, DIEInteger(Byte));

  if (Loc)
    Loc->setSize(Bytes.size());
  else
    Block->setSize(Bytes.size());

  if (Loc)
    Value = DIEValue(dwarf::Attribute(AttrSpec.Attr),
                     dwarf::Form(AttrSpec.Form), Loc);
  else
    Value = DIEValue(dwarf::Attribute(AttrSpec.Attr),
                     dwarf::Form(AttrSpec.Form), Block);
  Die.addValue(getState().DIEAlloc, Value);
}

void DIEBuilder::cloneAddressAttribute(
    DIE &Die, const DWARFUnit &U,
    const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const DWARFFormValue &Val) {
  Die.addValue(getState().DIEAlloc, AttrSpec.Attr, AttrSpec.Form,
               DIEInteger(Val.getRawUValue()));
}

void DIEBuilder::cloneRefsigAttribute(
    DIE &Die, DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const DWARFFormValue &Val) {
  const std::optional<uint64_t> SigVal = Val.getRawUValue();
  Die.addValue(getState().DIEAlloc, AttrSpec.Attr, dwarf::DW_FORM_ref_sig8,
               DIEInteger(*SigVal));
}

void DIEBuilder::cloneScalarAttribute(
    DIE &Die, const DWARFDie &InputDIE,
    const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const DWARFFormValue &Val) {
  uint64_t Value;

  if (auto OptionalValue = Val.getAsUnsignedConstant())
    Value = *OptionalValue;
  else if (auto OptionalValue = Val.getAsSignedConstant())
    Value = *OptionalValue;
  else if (auto OptionalValue = Val.getAsSectionOffset())
    Value = *OptionalValue;
  else {
    errs() << "BOLT-WARNING: [internal-dwarf-error]: Unsupported scalar "
              "attribute form. Dropping "
              "attribute.\n";
    return;
  }

  Die.addValue(getState().DIEAlloc, AttrSpec.Attr, AttrSpec.Form,
               DIEInteger(Value));
}

void DIEBuilder::cloneLoclistAttrubute(
    DIE &Die, const DWARFDie &InputDIE,
    const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const DWARFFormValue &Val) {
  std::optional<uint64_t> Value = std::nullopt;

  if (auto OptionalValue = Val.getAsUnsignedConstant())
    Value = OptionalValue;
  else if (auto OptionalValue = Val.getAsSignedConstant())
    Value = OptionalValue;
  else if (auto OptionalValue = Val.getAsSectionOffset())
    Value = OptionalValue;
  else
    errs() << "BOLT-WARNING: [internal-dwarf-error]: Unsupported scalar "
              "attribute form. Dropping "
              "attribute.\n";

  if (!Value.has_value())
    return;

  Die.addValue(getState().DIEAlloc, AttrSpec.Attr, AttrSpec.Form,
               DIELocList(*Value));
}

void DIEBuilder::cloneAttribute(
    DIE &Die, const DWARFDie &InputDIE, DWARFUnit &U, const DWARFFormValue &Val,
    const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec) {
  switch (AttrSpec.Form) {
  case dwarf::DW_FORM_strp:
  case dwarf::DW_FORM_string:
  case dwarf::DW_FORM_strx:
  case dwarf::DW_FORM_strx1:
  case dwarf::DW_FORM_strx2:
  case dwarf::DW_FORM_strx3:
  case dwarf::DW_FORM_strx4:
  case dwarf::DW_FORM_GNU_str_index:
  case dwarf::DW_FORM_line_strp:
    cloneStringAttribute(Die, U, AttrSpec, Val);
    break;
  case dwarf::DW_FORM_ref_addr:
  case dwarf::DW_FORM_ref1:
  case dwarf::DW_FORM_ref2:
  case dwarf::DW_FORM_ref4:
  case dwarf::DW_FORM_ref8:
    cloneDieReferenceAttribute(Die, U, InputDIE, AttrSpec, Val);
    break;
  case dwarf::DW_FORM_block:
  case dwarf::DW_FORM_block1:
  case dwarf::DW_FORM_block2:
  case dwarf::DW_FORM_block4:
  case dwarf::DW_FORM_exprloc:
    cloneBlockAttribute(Die, U, AttrSpec, Val);
    break;
  case dwarf::DW_FORM_addr:
  case dwarf::DW_FORM_addrx:
  case dwarf::DW_FORM_GNU_addr_index:
    cloneAddressAttribute(Die, U, AttrSpec, Val);
    break;
  case dwarf::DW_FORM_data1:
  case dwarf::DW_FORM_data2:
  case dwarf::DW_FORM_data4:
  case dwarf::DW_FORM_data8:
  case dwarf::DW_FORM_udata:
  case dwarf::DW_FORM_sdata:
  case dwarf::DW_FORM_sec_offset:
  case dwarf::DW_FORM_rnglistx:
  case dwarf::DW_FORM_flag:
  case dwarf::DW_FORM_flag_present:
  case dwarf::DW_FORM_implicit_const:
    cloneScalarAttribute(Die, InputDIE, AttrSpec, Val);
    break;
  case dwarf::DW_FORM_loclistx:
    cloneLoclistAttrubute(Die, InputDIE, AttrSpec, Val);
    break;
  case dwarf::DW_FORM_ref_sig8:
    cloneRefsigAttribute(Die, AttrSpec, Val);
    break;
  default:
    errs() << "BOLT-WARNING: [internal-dwarf-error]: Unsupported attribute "
              "form " +
                  dwarf::FormEncodingString(AttrSpec.Form).str() +
                  " in cloneAttribute. Dropping.";
  }
}
void DIEBuilder::assignAbbrev(DIEAbbrev &Abbrev) {
  // Check the set for priors.
  FoldingSetNodeID ID;
  Abbrev.Profile(ID);
  void *InsertToken;
  DIEAbbrev *InSet = AbbreviationsSet.FindNodeOrInsertPos(ID, InsertToken);

  // If it's newly added.
  if (InSet) {
    // Assign existing abbreviation number.
    Abbrev.setNumber(InSet->getNumber());
  } else {
    // Add to abbreviation list.
    Abbreviations.push_back(
        std::make_unique<DIEAbbrev>(Abbrev.getTag(), Abbrev.hasChildren()));
    for (const auto &Attr : Abbrev.getData())
      Abbreviations.back()->AddAttribute(Attr.getAttribute(), Attr.getForm());
    AbbreviationsSet.InsertNode(Abbreviations.back().get(), InsertToken);
    // Assign the unique abbreviation number.
    Abbrev.setNumber(Abbreviations.size());
    Abbreviations.back()->setNumber(Abbreviations.size());
  }
}

void DIEBuilder::generateAbbrevs() {
  if (isEmpty())
    return;

  for (DWARFUnit *DU : getState().DUList) {
    DIE *UnitDIE = getUnitDIEbyUnit(*DU);
    generateUnitAbbrevs(UnitDIE);
  }
}

void DIEBuilder::generateUnitAbbrevs(DIE *Die) {
  DIEAbbrev NewAbbrev = Die->generateAbbrev();

  if (Die->hasChildren())
    NewAbbrev.setChildrenFlag(dwarf::DW_CHILDREN_yes);
  assignAbbrev(NewAbbrev);
  Die->setAbbrevNumber(NewAbbrev.getNumber());

  for (auto &Child : Die->children()) {
    generateUnitAbbrevs(&Child);
  }
}

static uint64_t getHash(const DWARFUnit &DU) {
  // Before DWARF5 TU units are in their own section, so at least one offset,
  // first one, will be the same as CUs in .debug_info.dwo section
  if (DU.getVersion() < 5 && DU.isTypeUnit()) {
    const uint64_t TypeUnitHash =
        cast_or_null<DWARFTypeUnit>(&DU)->getTypeHash();
    const uint64_t Offset = DU.getOffset();
    return llvm::hash_combine(llvm::hash_value(TypeUnitHash),
                              llvm::hash_value(Offset));
  }
  return DU.getOffset();
}

void DIEBuilder::registerUnit(DWARFUnit &DU, bool NeedSort) {
  auto IterGlobal = AllProcessed.insert(getHash(DU));
  // If DU is already in a current working set or was already processed we can
  // skip it.
  if (!IterGlobal.second)
    return;
  if (getState().Type == ProcessingType::DWARF4TUs) {
    getState().DWARF4TUVector.push_back(&DU);
  } else if (getState().Type == ProcessingType::DWARF5TUs) {
    getState().DWARF5TUVector.push_back(&DU);
  } else {
    getState().DWARFCUVector.push_back(&DU);
    /// Sorting for cross CU reference resolution.
    if (NeedSort)
      std::sort(getState().DWARFCUVector.begin(),
                getState().DWARFCUVector.end(),
                [](const DWARFUnit *A, const DWARFUnit *B) {
                  return A->getOffset() < B->getOffset();
                });
  }
  getState().UnitIDMap[getHash(DU)] = getState().DUList.size();
  // This handles the case where we do have cross cu references, but CUs do not
  // share the same abbrev table.
  if (getState().DUList.size() == getState().CloneUnitCtxMap.size())
    getState().CloneUnitCtxMap.emplace_back();
  getState().DUList.push_back(&DU);
}

std::optional<uint32_t> DIEBuilder::getUnitId(const DWARFUnit &DU) {
  auto Iter = getState().UnitIDMap.find(getHash(DU));
  if (Iter != getState().UnitIDMap.end())
    return Iter->second;
  return std::nullopt;
}

} // namespace bolt
} // namespace llvm
