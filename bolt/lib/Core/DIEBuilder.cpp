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
#include <string>
#include <unordered_map>
#include <utility>

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt"
namespace llvm {
namespace bolt {

void DIEBuilder::computeReference() {
  for (auto &[SrcDIEInfo, ReferenceInfo] : ForwardReferences) {
    DIEInfo *DstDIEInfo = ReferenceInfo.Dst;
    UnitInfo &DstUnitInfo = getUnitInfo(DstDIEInfo->UnitId);
    dwarf::Attribute Attr = ReferenceInfo.AttrSpec.Attr;
    dwarf::Form Form = ReferenceInfo.AttrSpec.Form;
    uint64_t CUBase = 0;

    // If DWARF 4, type unit is store in .debug_types. So we need to calibrate
    // the begin of .debug_info to the first Compile Unit offset.
    if (!DWARF4CUVector.empty()) {
      UnitInfo &FirstCUInfo = getUnitInfoByDwarfUnit(*DWARF4CUVector.front());
      CUBase = FirstCUInfo.UnitOffset;
    }

    uint64_t NewAddr =
        DstDIEInfo->Die->getOffset() + DstUnitInfo.UnitOffset - CUBase;
    SrcDIEInfo->Die->replaceValue(DIEAlloc, Attr, Form, DIEInteger(NewAddr));
  }

  return;
}

std::optional<uint32_t> DIEBuilder::allocDIE(DWARFUnit &DU, DWARFDie &DDie,
                                             BumpPtrAllocator &Alloc,
                                             uint32_t UId, uint32_t offset) {
  auto &UnitInfo = getUnitInfo(UId);
  auto DDieOffset = DDie.getOffset();
  if (UnitInfo.DIEIDMap.count(DDieOffset))
    return UnitInfo.DIEIDMap[DDieOffset];
  uint32_t DId = AllocDIEId(DU);

  DIE *Die = DIE::get(Alloc, dwarf::Tag(DDie.getTag()));
  UnitInfo.DIEIDMap[DDieOffset] = DId;
  UnitInfo.DieInfoList.push_back(DIEInfo{Die, DId, UId, offset});
  UnitInfo.DIEId2InfoMap[DId] = &UnitInfo.DieInfoList.back();

  return DId;
}

void DIEBuilder::constructFromUnit(DWARFUnit &DU,
                                   std::vector<DWARFUnit *> &DUOffsetList) {
  std::optional<uint32_t> UnitId = getUnitId(DU);
  if (!UnitId.has_value()) {
    errs() << "BOLT-WARNING: " << format("Skip Unit at 0x%x\n", DU.getOffset());
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
  bool IsTypeDIE = false;
  bool IsCUDIE = true;

  if (DWARFTypeUnit *TU = dyn_cast_or_null<DWARFTypeUnit>(&DU)) {
    TUTypeOffset = TU->getTypeOffset();
  }

  assert(DebugInfoData.isValidOffset(NextCUOffset - 1));
  Parents.push_back(UINT32_MAX);
  do {
    if (TUTypeOffset == DIEOffset - DU.getOffset()) {
      IsTypeDIE = true;
    }

    if (!DIEEntry.extractFast(DU, &DIEOffset, DebugInfoData, NextCUOffset,
                              Parents.back()))
      break;

    if (const DWARFAbbreviationDeclaration *AbbrDecl =
            DIEEntry.getAbbreviationDeclarationPtr()) {
      DWARFDie DDie(&DU, &DIEEntry);

      DIE *CurDIE = constructDIEFast(DDie, DU, DU.getContext().isLittleEndian(),
                                     *UnitId, DUOffsetList);
      if (IsTypeDIE) {
        TypeDIEMap[&DU] = CurDIE;
        IsTypeDIE = false;
      }

      if (!CurParentDIEStack.empty())
        CurParentDIEStack.back()->addChild(CurDIE);

      if (AbbrDecl->hasChildren()) {
        CurParentDIEStack.push_back(CurDIE);
      } else if (IsCUDIE) {
        // Stop if we have single compile unit die w/o children.
        break;
      }
    } else {
      // NULL DIE: finishes current children scope.
      CurParentDIEStack.pop_back();
    }

    if (IsCUDIE)
      IsCUDIE = false;
  } while (CurParentDIEStack.size() > 0);

  CloneUnitCtxMap[*UnitId].Isconstructed = true;
}

DIEBuilder::DIEBuilder(DWARFContext *DwarfContext, bool IsDWO) {
  outs() << "BOLT-INFO: Constructing DIE...\n";
  IsBuilt = true;

  const DWARFUnitIndex &TUIndex = DwarfContext->getTUIndex();
  if (!TUIndex.getRows().empty()) {
    for (auto &Row : TUIndex.getRows()) {
      uint64_t Signature = Row.getSignature();

      // manually populate TypeUnit to UnitVector
      DwarfContext->getTypeUnitForHash(DwarfContext->getMaxVersion(), Signature,
                                       true);
    }
  }

  uint32_t MaxVersion =
      IsDWO ? DwarfContext->getMaxDWOVersion() : DwarfContext->getMaxVersion();
  unsigned int CUNum = IsDWO ? DwarfContext->getNumDWOCompileUnits()
                             : DwarfContext->getNumCompileUnits();
  DWARFContext::compile_unit_range CU4Ranges =
      IsDWO ? DwarfContext->dwo_compile_units() : DwarfContext->compile_units();
  DWARFContext::unit_iterator_range CU5Ranges =
      IsDWO ? DwarfContext->dwo_info_section_units()
            : DwarfContext->info_section_units();

  if (MaxVersion >= 5) {
    CUNum = IsDWO ? DwarfContext->getNumDWOCompileUnits() +
                        DwarfContext->getNumDWOTypeUnits()
                  : DwarfContext->getNumCompileUnits() +
                        DwarfContext->getNumTypeUnits();
  }

  CloneUnitCtxMap = std::vector<UnitInfo>(CUNum);

  if (MaxVersion >= 5) {
    for (std::unique_ptr<DWARFUnit> &DU : CU5Ranges) {
      if (!DU.get())
        continue;
      registerUnit(*DU.get());
    }
    for (std::unique_ptr<DWARFUnit> &DU : CU5Ranges) {
      if (!DU.get())
        continue;
      constructFromUnit(*DU.get(), DUList);
    }
  } else {
    DWARFContext::unit_iterator_range CU4TURanges =
        IsDWO ? DwarfContext->dwo_types_section_units()
              : DwarfContext->types_section_units();
    for (std::unique_ptr<DWARFUnit> &DU : CU4TURanges) {
      CloneUnitCtxMap.resize(CloneUnitCtxMap.size() + 1);
      registerUnit(*DU.get());
      DWARF4TUVector.push_back(DU.get());
    }
    for (std::unique_ptr<DWARFUnit> &DU : CU4TURanges) {
      constructFromUnit(*DU.get(), DWARF4TUVector);
    }

    for (std::unique_ptr<DWARFUnit> &DU : CU4Ranges) {
      registerUnit(*DU.get());
      DWARF4CUVector.push_back(DU.get());
    }
    for (std::unique_ptr<DWARFUnit> &DU : CU4Ranges) {
      constructFromUnit(*DU.get(), DWARF4CUVector);
    }
  }
  outs() << "BOLT-INFO: Finish constructing DIE\n";
}

DIE *DIEBuilder::constructDIEFast(DWARFDie &DDie, DWARFUnit &U,
                                  bool IsLittleEndian, uint32_t UnitId,
                                  std::vector<DWARFUnit *> &DUOffsetList) {

  std::optional<uint32_t> Idx = getAllocDIEId(U, DDie);
  if (Idx.has_value()) {
    UnitInfo &UnitInfo = getUnitInfo(UnitId);
    DIEInfo &DieInfo = getDIEInfo(UnitId, *Idx);
    if (UnitInfo.Isconstructed && DieInfo.Die)
      return DieInfo.Die;
  } else {
    Idx = allocDIE(U, DDie, DIEAlloc, UnitId);
  }

  DIEInfo &DieInfo = getDIEInfo(UnitId, *Idx);
  DIE *Die = DieInfo.Die;
  UnitDIEs[&U].push_back(Die);

  uint64_t Offset = DDie.getOffset();
  uint64_t NextOffset = Offset;
  DWARFDataExtractor Data = U.getDebugInfoExtractor();
  DWARFDebugInfoEntry DDIEntry;

  if (DDIEntry.extractFast(U, &NextOffset, Data, U.getNextUnitOffset(), 0)) {
    assert(NextOffset - U.getOffset() <= Data.getData().size() &&
           "NextOffset OOB");
  }

  SmallString<40> DIECopy(Data.getData().substr(Offset, NextOffset - Offset));
  Data =
      DWARFDataExtractor(DIECopy, Data.isLittleEndian(), Data.getAddressSize());
  Offset = 0;

  const DWARFAbbreviationDeclaration *Abbrev =
      DDie.getAbbreviationDeclarationPtr();
  Offset += getULEB128Size(Abbrev->getCode());

  for (const auto &AttrSpec : Abbrev->attributes()) {
    DWARFFormValue Val(AttrSpec.Form);
    uint64_t AttrSize = Offset;
    Val.extractValue(Data, &Offset, U.getFormParams(), &U);
    AttrSize = Offset - AttrSize;
    cloneAttribute(*Die, DDie, U, Val, AttrSpec, AttrSize, IsLittleEndian,
                   DUOffsetList);
  }
  return Die;
}

static DWARFUnit *getUnitForOffset(const std::vector<DWARFUnit *> &Units,
                                   uint64_t Offset) {
  auto CU =
      llvm::upper_bound(Units, Offset, [](uint64_t LHS, const DWARFUnit *RHS) {
        return LHS < RHS->getNextUnitOffset();
      });
  return CU != Units.end() ? *CU : nullptr;
}

uint32_t DIEBuilder::computeDIEOffset(DWARFUnit &CU, DIE &Die,
                                      uint32_t &CurOffset) {
  uint32_t CurSize = 0;

  Die.setOffset(CurOffset);
  for (DIEValue &Val : Die.values()) {
    CurSize += Val.sizeOf(CU.getFormParams());
  }
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
  uint64_t UnitStartOffset = 0;

  for (DWARFUnit *CU : DUList) {
    DIE *UnitDIE = getUnitDIEbyUnit(*CU);
    uint32_t HeaderSize = CU->getHeaderSize();
    uint32_t CurOffset = HeaderSize;
    computeDIEOffset(*CU, *UnitDIE, CurOffset);

    UnitInfo &CurUnitInfo = getUnitInfoByDwarfUnit(*CU);
    CurUnitInfo.UnitOffset = UnitStartOffset;
    UnitStartOffset += HeaderSize + UnitDIE->getSize();
  }

  computeReference();
}

DWARFDie DIEBuilder::resolveDIEReference(
    const DWARFFormValue &RefValue,
    DWARFAbbreviationDeclaration::AttributeSpec AttrSpec, DWARFUnit *&RefCU,
    DWARFDebugInfoEntry &DwarfDebugInfoEntry,
    const std::vector<DWARFUnit *> &DUOffsetList) {
  assert(RefValue.isFormClass(DWARFFormValue::FC_Reference));
  uint64_t RefOffset = *RefValue.getAsReference();

  if ((RefCU = getUnitForOffset(DUOffsetList, RefOffset))) {
    DWARFDataExtractor DebugInfoData = RefCU->getDebugInfoExtractor();
    if (DwarfDebugInfoEntry.extractFast(*RefCU, &RefOffset, DebugInfoData,
                                        RefCU->getNextUnitOffset(), 0)) {
      // In a file with broken references, an attribute might point to a NULL
      // DIE.
      DWARFDie RefDie = DWARFDie(RefCU, &DwarfDebugInfoEntry);
      if (!RefDie.isNULL()) {
        std::optional<uint32_t> UnitId = getUnitId(*RefCU);

        // forward reference
        if (UnitId.has_value() && !CloneUnitCtxMap[*UnitId].Isconstructed) {
          std::optional<uint32_t> IsAllocId = getAllocDIEId(*RefCU, RefDie);
          if (!IsAllocId.has_value()) {
            // forward reference but need allocate a empty one
            IsAllocId = allocDIE(*RefCU, RefDie, DIEAlloc, *UnitId);
          }

          uint32_t DIEId = *IsAllocId;
          DIEInfo &DieInfo = getDIEInfo(*UnitId, DIEId);
          DieInfo.CanonicalDIEOffset = 0xDEADBEEF;
        }
        return RefDie;
      }
    }
  }

  llvm_unreachable("could not find referenced CU\n");
  return DWARFDie();
}

uint32_t DIEBuilder::cloneDieReferenceAttribute(
    DIE &Die, const DWARFDie &InputDIE,
    DWARFAbbreviationDeclaration::AttributeSpec AttrSpec, unsigned AttrSize,
    const DWARFFormValue &Val, DWARFUnit &U,
    std::vector<DWARFUnit *> &DUOffsetList) {
  uint64_t Ref = *Val.getAsReference();

  DIE *NewRefDie = nullptr;
  DWARFUnit *RefUnit = nullptr;

  DWARFDebugInfoEntry DDIEntry;
  DWARFDie RefDie =
      resolveDIEReference(Val, AttrSpec, RefUnit, DDIEntry, DUOffsetList);

  if (!RefDie || AttrSpec.Attr == dwarf::DW_AT_sibling)
    return 0;

  std::optional<uint32_t> UnitId = getUnitId(*RefUnit);
  std::optional<uint32_t> IsAllocId = getAllocDIEId(*RefUnit, RefDie);
  if (!IsAllocId.has_value())
    llvm_unreachable(
        "[error] encounter unexpected unallocated DIE. Should be alloc!");
  uint32_t DIEId = *IsAllocId;
  DIEInfo &DieInfo = getDIEInfo(*UnitId, DIEId);

  if (!DieInfo.Die) {
    assert(Ref > InputDIE.getOffset());
    llvm_unreachable(
        "[error] encounter unexpected unallocated DIE. Should be alloc!");
    // We haven't cloned this DIE yet. Just create an empty one and
    // store it. It'll get really cloned when we process it.
    DieInfo.Die = DIE::get(DIEAlloc, dwarf::Tag(RefDie.getTag()));
  }
  NewRefDie = DieInfo.Die;

  if (AttrSpec.Form == dwarf::DW_FORM_ref_addr) {
    // no matter forward reference or backward reference, we are supposed
    // to calculate them in `finish` due to the possible modification of
    // the DIE.
    DWARFDie CurDie = const_cast<DWARFDie &>(InputDIE);
    DIEInfo *CurDieInfo = &getDIEInfoByDwarfDie(&CurDie);
    ForwardReferences.push_back(
        std::make_pair(CurDieInfo, ForwardReferenceInfo(&DieInfo, AttrSpec)));

    Die.addValue(DIEAlloc, AttrSpec.Attr, dwarf::DW_FORM_ref_addr,
                 DIEInteger(0xDEADBEEF));
    return U.getRefAddrByteSize();
  }

  Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
               dwarf::Form(AttrSpec.Form), DIEEntry(*NewRefDie));

  return AttrSize;
}

uint32_t DIEBuilder::cloneStringAttribute(
    DIE &Die, DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    unsigned AttrSize, const DWARFFormValue &Val, const DWARFUnit &U) {
  if (AttrSpec.Form == dwarf::DW_FORM_string) {
    auto StrAddr = Val.getAsCString();
    if (!StrAddr) {
      consumeError(StrAddr.takeError());
      return AttrSize;
    }
    Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
                 dwarf::DW_FORM_string,
                 new (DIEAlloc) DIEInlineString(StrAddr.get(), DIEAlloc));
  } else {
    std::optional<uint64_t> OffsetIndex = Val.getRawUValue();
    Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr), AttrSpec.Form,
                 DIEInteger(*OffsetIndex));
  }
  return AttrSize;
}

void DIEBuilder::cloneExpression(DataExtractor &Data,
                                 DWARFExpression &Expression, DWARFUnit &U,
                                 SmallVectorImpl<uint8_t> &OutputBuffer) {
  using Encoding = DWARFExpression::Operation::Encoding;

  uint64_t OpOffset = 0;
  for (auto &Op : Expression) {
    auto Description = Op.getDescription();
    // DW_OP_const_type is variable-length and has 3
    // operands. Thus far we only support 2.
    if ((Description.Op.size() == 2 &&
         Description.Op[0] == Encoding::BaseTypeRef) ||
        (Description.Op.size() == 2 &&
         Description.Op[1] == Encoding::BaseTypeRef &&
         Description.Op[0] != Encoding::Size1))
      outs() << "BOLT-INFO: Unsupported DW_OP encoding.\n";

    if ((Description.Op.size() == 1 &&
         Description.Op[0] == Encoding::BaseTypeRef) ||
        (Description.Op.size() == 2 &&
         Description.Op[1] == Encoding::BaseTypeRef &&
         Description.Op[0] == Encoding::Size1)) {
      // This code assumes that the other non-typeref operand fits into 1
      // byte.
      assert(OpOffset < Op.getEndOffset());
      uint32_t ULEBsize = Op.getEndOffset() - OpOffset - 1;
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
        DWARFDie RefDie = U.getDIEForOffset(RefOffset);
        std::optional<uint32_t> RefDieID = getAllocDIEId(U, RefOffset);
        std::optional<uint32_t> RefUnitID = getUnitId(U);
        if (RefDieID.has_value() && RefUnitID.has_value()) {
          DIEInfo &RefDieInfo = getDIEInfo(*RefUnitID, *RefDieID);
          if (DIE *Clone = RefDieInfo.Die)
            Offset = RefDie.getOffset();
          else
            errs() << "BOLT-WARNING: base type ref doesn't point to "
                      "DW_TAG_base_type.\n";
        }
      }
      uint8_t ULEB[16];
      unsigned RealSize = encodeULEB128(Offset, ULEB, ULEBsize);
      if (RealSize > ULEBsize) {
        // Emit the generic type as a fallback.
        RealSize = encodeULEB128(0, ULEB, ULEBsize);
        errs() << "BOLT-WARNING: base type ref doesn't fit.\n";
      }
      assert(RealSize == ULEBsize && "padding failed");
      ArrayRef<uint8_t> ULEBbytes(ULEB, ULEBsize);
      OutputBuffer.append(ULEBbytes.begin(), ULEBbytes.end());
    } else {
      // Copy over everything else unmodified.
      StringRef Bytes = Data.getData().slice(OpOffset, Op.getEndOffset());
      OutputBuffer.append(Bytes.begin(), Bytes.end());
    }
    OpOffset = Op.getEndOffset();
  }
}

uint32_t DIEBuilder::cloneBlockAttribute(
    DIE &Die, DWARFUnit &U,
    DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const DWARFFormValue &Val, unsigned AttrSize, bool IsLittleEndian) {
  DIEValueList *Attr;
  DIEValue Value;
  DIELoc *Loc = nullptr;
  DIEBlock *Block = nullptr;

  if (AttrSpec.Form == dwarf::DW_FORM_exprloc) {
    Loc = new (DIEAlloc) DIELoc;
  } else if (doesFormBelongToClass(AttrSpec.Form, DWARFFormValue::FC_Block,
                                   U.getVersion())) {
    Block = new (DIEAlloc) DIEBlock;
  } else {
    errs() << "BOLT-WARNING: Unexpected Form value in "
              "cloneBlockAttribute\n";
    return 0;
  }
  Attr = Loc ? static_cast<DIEValueList *>(Loc)
             : static_cast<DIEValueList *>(Block);

  if (Loc)
    Value = DIEValue(dwarf::Attribute(AttrSpec.Attr),
                     dwarf::Form(AttrSpec.Form), Loc);
  else
    Value = DIEValue(dwarf::Attribute(AttrSpec.Attr),
                     dwarf::Form(AttrSpec.Form), Block);

  SmallVector<uint8_t, 32> Buffer;
  ArrayRef<uint8_t> Bytes = *Val.getAsBlock();
  if (DWARFAttribute::mayHaveLocationExpr(AttrSpec.Attr) &&
      (Val.isFormClass(DWARFFormValue::FC_Block) ||
       Val.isFormClass(DWARFFormValue::FC_Exprloc))) {
    DataExtractor Data(StringRef((const char *)Bytes.data(), Bytes.size()),
                       IsLittleEndian, U.getAddressByteSize());
    DWARFExpression Expr(Data, U.getAddressByteSize(),
                         U.getFormParams().Format);
    cloneExpression(Data, Expr, U, Buffer);
    Bytes = Buffer;
  }
  for (auto Byte : Bytes)
    Attr->addValue(DIEAlloc, static_cast<dwarf::Attribute>(0),
                   dwarf::DW_FORM_data1, DIEInteger(Byte));

  if (Loc)
    Loc->setSize(Bytes.size());
  else
    Block->setSize(Bytes.size());

  Die.addValue(DIEAlloc, Value);
  return AttrSize;
}

uint32_t DIEBuilder::cloneAddressAttribute(
    DIE &Die, DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const DWARFFormValue &Val, const DWARFUnit &U) {
  Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
               dwarf::Form(AttrSpec.Form), DIEInteger(Val.getRawUValue()));
  return U.getAddressByteSize();
}

uint32_t DIEBuilder::cloneRefsigAttribute(
    DIE &Die, DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    unsigned AttrSize, const DWARFFormValue &Val) {
  std::optional<uint64_t> SigVal = Val.getRawUValue();
  Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
               dwarf::DW_FORM_ref_sig8, DIEInteger(*SigVal));
  return AttrSize;
}

uint32_t DIEBuilder::cloneScalarAttribute(
    DIE &Die, const DWARFDie &InputDIE,
    DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const DWARFFormValue &Val, unsigned AttrSize) {
  uint64_t Value;

  if (auto OptionalValue = Val.getAsUnsignedConstant())
    Value = *OptionalValue;
  else if (auto OptionalValue = Val.getAsSignedConstant())
    Value = *OptionalValue;
  else if (auto OptionalValue = Val.getAsSectionOffset())
    Value = *OptionalValue;
  else {
    errs() << "BOLT-WARNING: Unsupported scalar attribute form. Dropping "
              "attribute.\n";
    return 0;
  }

  Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
               dwarf::Form(AttrSpec.Form), DIEInteger(Value));
  return AttrSize;
}

uint32_t DIEBuilder::cloneLoclistAttrubute(
    DIE &Die, const DWARFDie &InputDIE,
    DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    const DWARFFormValue &Val, unsigned AttrSize) {
  uint64_t Value;

  if (auto OptionalValue = Val.getAsUnsignedConstant())
    Value = *OptionalValue;
  else if (auto OptionalValue = Val.getAsSignedConstant())
    Value = *OptionalValue;
  else if (auto OptionalValue = Val.getAsSectionOffset())
    Value = *OptionalValue;
  else {
    errs() << "BOLT-WARNING: Unsupported scalar attribute form. Dropping "
              "attribute.\n";
    return 0;
  }

  Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
               dwarf::Form(AttrSpec.Form), DIELocList(Value));
  return AttrSize;
}

uint32_t DIEBuilder::cloneAttribute(
    DIE &Die, const DWARFDie &InputDIE, DWARFUnit &U, const DWARFFormValue &Val,
    const DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
    unsigned AttrSize, bool IsLittleEndian,
    std::vector<DWARFUnit *> &DUOffsetList) {
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
    return cloneStringAttribute(Die, AttrSpec, AttrSize, Val, U);
  case dwarf::DW_FORM_ref_addr:
  case dwarf::DW_FORM_ref1:
  case dwarf::DW_FORM_ref2:
  case dwarf::DW_FORM_ref4:
  case dwarf::DW_FORM_ref8:
    return cloneDieReferenceAttribute(Die, InputDIE, AttrSpec, AttrSize, Val, U,
                                      DUOffsetList);
  case dwarf::DW_FORM_block:
  case dwarf::DW_FORM_block1:
  case dwarf::DW_FORM_block2:
  case dwarf::DW_FORM_block4:
  case dwarf::DW_FORM_exprloc:
    return cloneBlockAttribute(Die, U, AttrSpec, Val, AttrSize, IsLittleEndian);
  case dwarf::DW_FORM_addr:
  case dwarf::DW_FORM_addrx:
  case dwarf::DW_FORM_GNU_addr_index:
    return cloneAddressAttribute(Die, AttrSpec, Val, U);
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
    return cloneScalarAttribute(Die, InputDIE, AttrSpec, Val, AttrSize);
  case dwarf::DW_FORM_loclistx:
    return cloneLoclistAttrubute(Die, InputDIE, AttrSpec, Val, AttrSize);
  case dwarf::DW_FORM_ref_sig8:
    return cloneRefsigAttribute(Die, AttrSpec, AttrSize, Val);
  default:
    std::string Msg = "Unsupported attribute form " +
                      dwarf::FormEncodingString(AttrSpec.Form).str() +
                      " in cloneAttribute. Dropping.";
    llvm_unreachable(Msg.c_str());
  }

  return 0;
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
  if (!IsBuilt)
    return;

  for (DWARFUnit *DU : DUList) {
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

} // namespace bolt
} // namespace llvm
