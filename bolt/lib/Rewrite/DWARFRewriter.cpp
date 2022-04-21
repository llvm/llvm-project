//===- bolt/Rewrite/DWARFRewriter.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/DWARFRewriter.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/DebugData.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DWP/DWP.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

LLVM_ATTRIBUTE_UNUSED
static void printDie(const DWARFDie &DIE) {
  DIDumpOptions DumpOpts;
  DumpOpts.ShowForm = true;
  DumpOpts.Verbose = true;
  DumpOpts.ChildRecurseDepth = 0;
  DumpOpts.ShowChildren = 0;
  DIE.dump(dbgs(), 0, DumpOpts);
}

namespace llvm {
namespace bolt {
/// Finds attributes FormValue and Offset.
///
/// \param DIE die to look up in.
/// \param Attr the attribute to extract.
/// \return an optional AttrInfo with DWARFFormValue and Offset.
static Optional<AttrInfo> findAttributeInfo(const DWARFDie DIE,
                                            dwarf::Attribute Attr) {
  if (!DIE.isValid())
    return None;
  const DWARFAbbreviationDeclaration *AbbrevDecl =
      DIE.getAbbreviationDeclarationPtr();
  if (!AbbrevDecl)
    return None;
  Optional<uint32_t> Index = AbbrevDecl->findAttributeIndex(Attr);
  if (!Index)
    return None;
  return findAttributeInfo(DIE, AbbrevDecl, *Index);
}

/// Finds attributes FormValue and Offset.
///
/// \param DIE die to look up in.
/// \param Attrs finds the first attribute that matches and extracts it.
/// \return an optional AttrInfo with DWARFFormValue and Offset.
Optional<AttrInfo> findAttributeInfo(const DWARFDie DIE,
                                     std::vector<dwarf::Attribute> Attrs) {
  for (dwarf::Attribute &Attr : Attrs)
    if (Optional<AttrInfo> Info = findAttributeInfo(DIE, Attr))
      return Info;
  return None;
}
} // namespace bolt
} // namespace llvm

using namespace llvm;
using namespace llvm::support::endian;
using namespace object;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltCategory;
extern cl::opt<unsigned> Verbosity;
extern cl::opt<std::string> OutputFilename;

static cl::opt<bool>
KeepARanges("keep-aranges",
  cl::desc("keep or generate .debug_aranges section if .gdb_index is written"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<bool>
DeterministicDebugInfo("deterministic-debuginfo",
  cl::desc("disables parallel execution of tasks that may produce"
           "nondeterministic debug info"),
  cl::init(true),
  cl::cat(BoltCategory));

static cl::opt<std::string> DwarfOutputPath(
    "dwarf-output-path",
    cl::desc("Path to where .dwo files or dwp file will be written out to."),
    cl::init(""), cl::cat(BoltCategory));

static cl::opt<bool>
    WriteDWP("write-dwp",
             cl::desc("output a single dwarf package file (dwp) instead of "
                      "multiple non-relocatable dwarf object files (dwo)."),
             cl::init(false), cl::cat(BoltCategory));

static cl::opt<bool>
    DebugSkeletonCu("debug-skeleton-cu",
                    cl::desc("prints out offsetrs for abbrev and debu_info of "
                             "Skeleton CUs that get patched."),
                    cl::ZeroOrMore, cl::Hidden, cl::init(false),
                    cl::cat(BoltCategory));
} // namespace opts

/// Returns DWO Name to be used. Handles case where user specifies output DWO
/// directory, and there are duplicate names. Assumes DWO ID is unique.
static std::string
getDWOName(llvm::DWARFUnit &CU,
           std::unordered_map<std::string, uint32_t> *NameToIndexMap,
           std::unordered_map<uint64_t, std::string> &DWOIdToName) {
  llvm::Optional<uint64_t> DWOId = CU.getDWOId();
  assert(DWOId && "DWO ID not found.");
  (void)DWOId;
  auto NameIter = DWOIdToName.find(*DWOId);
  if (NameIter != DWOIdToName.end())
    return NameIter->second;

  std::string DWOName = dwarf::toString(
      CU.getUnitDIE().find({dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}),
      "");
  assert(!DWOName.empty() &&
         "DW_AT_dwo_name/DW_AT_GNU_dwo_name does not exists.");
  if (NameToIndexMap && !opts::DwarfOutputPath.empty()) {
    auto Iter = NameToIndexMap->find(DWOName);
    if (Iter == NameToIndexMap->end())
      Iter = NameToIndexMap->insert({DWOName, 0}).first;
    DWOName.append(std::to_string(Iter->second));
    ++Iter->second;
  }
  DWOName.append(".dwo");
  DWOIdToName[*DWOId] = DWOName;
  return DWOName;
}

void DWARFRewriter::updateDebugInfo() {
  ErrorOr<BinarySection &> DebugInfo = BC.getUniqueSectionByName(".debug_info");
  if (!DebugInfo)
    return;

  auto *DebugInfoPatcher =
      static_cast<DebugInfoBinaryPatcher *>(DebugInfo->getPatcher());

  ARangesSectionWriter = std::make_unique<DebugARangesSectionWriter>();
  StrWriter = std::make_unique<DebugStrWriter>(&BC);
  AbbrevWriter = std::make_unique<DebugAbbrevWriter>(*BC.DwCtx);

  if (BC.isDWARF5Used()) {
    // Disabling none deterministic mode for dwarf5, to keep implementation
    // simpler.
    opts::DeterministicDebugInfo = true;
    AddrWriter = std::make_unique<DebugAddrWriterDwarf5>(&BC);
    RangesSectionWriter = std::make_unique<DebugRangeListsSectionWriter>();
    DebugRangeListsSectionWriter::setAddressWriter(AddrWriter.get());
  } else {
    AddrWriter = std::make_unique<DebugAddrWriter>(&BC);
    RangesSectionWriter = std::make_unique<DebugRangesSectionWriter>();
  }

  DebugLoclistWriter::setAddressWriter(AddrWriter.get());

  size_t CUIndex = 0;
  for (std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
    if (CU->getVersion() >= 5) {
      uint32_t AttrInfoOffset =
          DebugLoclistWriter::InvalidLocListsBaseAttrOffset;
      if (Optional<AttrInfo> AttrInfoVal =
              findAttributeInfo(CU->getUnitDIE(), dwarf::DW_AT_loclists_base))
        AttrInfoOffset = AttrInfoVal->Offset;

      LocListWritersByCU[CUIndex] = std::make_unique<DebugLoclistWriter>(
          &BC, CU->isDWOUnit() ? *CU->getDWOId() : CU->getOffset(),
          AttrInfoOffset, 5, false);
    } else {
      LocListWritersByCU[CUIndex] = std::make_unique<DebugLocWriter>(&BC);
    }
    ++CUIndex;
  }

  // Unordered maps to handle name collision if output DWO directory is
  // specified.
  std::unordered_map<std::string, uint32_t> NameToIndexMap;
  std::unordered_map<uint64_t, std::string> DWOIdToName;
  std::mutex AccessMutex;

  auto updateDWONameCompDir = [&](DWARFUnit &Unit) -> void {
    const DWARFDie &DIE = Unit.getUnitDIE();
    Optional<AttrInfo> AttrInfoVal = findAttributeInfo(
        DIE, {dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name});
    (void)AttrInfoVal;
    assert(AttrInfoVal && "Skeleton CU doesn't have dwo_name.");

    std::string ObjectName = "";

    {
      std::lock_guard<std::mutex> Lock(AccessMutex);
      ObjectName = getDWOName(Unit, &NameToIndexMap, DWOIdToName);
    }

    uint32_t NewOffset = StrWriter->addString(ObjectName.c_str());
    DebugInfoPatcher->addLE32Patch(AttrInfoVal->Offset, NewOffset,
                                   AttrInfoVal->Size);

    AttrInfoVal = findAttributeInfo(DIE, dwarf::DW_AT_comp_dir);
    (void)AttrInfoVal;
    assert(AttrInfoVal && "DW_AT_comp_dir is not in Skeleton CU.");

    if (!opts::DwarfOutputPath.empty()) {
      uint32_t NewOffset = StrWriter->addString(opts::DwarfOutputPath.c_str());
      DebugInfoPatcher->addLE32Patch(AttrInfoVal->Offset, NewOffset,
                                     AttrInfoVal->Size);
    }
  };

  auto processUnitDIE = [&](size_t CUIndex, DWARFUnit *Unit) {
    // Check if the unit is a skeleton and we need special updates for it and
    // its matching split/DWO CU.
    Optional<DWARFUnit *> SplitCU;
    Optional<uint64_t> RangesBase;
    llvm::Optional<uint64_t> DWOId = Unit->getDWOId();
    if (DWOId)
      SplitCU = BC.getDWOCU(*DWOId);

    DebugLocWriter *DebugLocWriter = nullptr;
    // Skipping CUs that failed to load.
    if (SplitCU) {
      updateDWONameCompDir(*Unit);

      // Assuming there is unique DWOID per binary. i.e. two or more CUs don't
      // have same DWO ID.
      assert(LocListWritersByCU.count(*DWOId) == 0 &&
             "LocList writer for DWO unit already exists.");
      {
        std::lock_guard<std::mutex> Lock(AccessMutex);
        DebugLocWriter =
            LocListWritersByCU
                .insert({*DWOId, std::make_unique<DebugLoclistWriter>(
                                     &BC, *DWOId, 0, Unit->getVersion(), true)})
                .first->second.get();
      }
      DebugInfoBinaryPatcher *DwoDebugInfoPatcher =
          llvm::cast<DebugInfoBinaryPatcher>(
              getBinaryDWODebugInfoPatcher(*DWOId));
      RangesBase = RangesSectionWriter->getSectionOffset();
      DWARFContext *DWOCtx = BC.getDWOContext();
      // Setting this CU offset with DWP to normalize DIE offsets to uint32_t
      if (DWOCtx && !DWOCtx->getCUIndex().getRows().empty())
        DwoDebugInfoPatcher->setDWPOffset((*SplitCU)->getOffset());
      DwoDebugInfoPatcher->setRangeBase(*RangesBase);
      DwoDebugInfoPatcher->addUnitBaseOffsetLabel((*SplitCU)->getOffset());
      DebugAbbrevWriter *DWOAbbrevWriter =
          createBinaryDWOAbbrevWriter((*SplitCU)->getContext(), *DWOId);
      updateUnitDebugInfo(*(*SplitCU), *DwoDebugInfoPatcher, *DWOAbbrevWriter,
                          *DebugLocWriter, *RangesSectionWriter);
      DwoDebugInfoPatcher->clearDestinationLabels();
      if (!DwoDebugInfoPatcher->getWasRangBasedUsed())
        RangesBase = None;
    }

    {
      std::lock_guard<std::mutex> Lock(AccessMutex);
      DebugLocWriter = LocListWritersByCU[CUIndex].get();
    }
    if (Unit->getVersion() >= 5) {
      RangesBase = RangesSectionWriter->getSectionOffset() +
                   getDWARF5RngListLocListHeaderSize();
      reinterpret_cast<DebugRangeListsSectionWriter *>(
          RangesSectionWriter.get())
          ->initSection(Unit->getOffset());
    }

    DebugInfoPatcher->addUnitBaseOffsetLabel(Unit->getOffset());
    updateUnitDebugInfo(*Unit, *DebugInfoPatcher, *AbbrevWriter,
                        *DebugLocWriter, *RangesSectionWriter, RangesBase);
    if (Unit->getVersion() >= 5)
      reinterpret_cast<DebugRangeListsSectionWriter *>(
          RangesSectionWriter.get())
          ->finalizeSection();
  };

  CUIndex = 0;
  if (opts::NoThreads || opts::DeterministicDebugInfo) {
    for (std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
      processUnitDIE(CUIndex, CU.get());
      if (CU->getVersion() >= 5)
        ++CUIndex;
    }
  } else {
    // Update unit debug info in parallel
    ThreadPool &ThreadPool = ParallelUtilities::getThreadPool();
    for (std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
      ThreadPool.async(processUnitDIE, CUIndex, CU.get());
      CUIndex++;
    }
    ThreadPool.wait();
  }

  DebugInfoPatcher->clearDestinationLabels();
  CUOffsetMap OffsetMap = finalizeDebugSections(*DebugInfoPatcher);

  if (opts::WriteDWP)
    writeDWP(DWOIdToName);
  else
    writeDWOFiles(DWOIdToName);

  updateGdbIndexSection(OffsetMap);
}

static uint64_t getCUId(DWARFUnit &Unit) {
  if (Unit.getVersion() >= 5)
    return Unit.getOffset();

  assert(Unit.isDWOUnit() && "Unit is not Skeleton CU.");
  return *Unit.getDWOId();
}

void DWARFRewriter::updateUnitDebugInfo(
    DWARFUnit &Unit, DebugInfoBinaryPatcher &DebugInfoPatcher,
    DebugAbbrevWriter &AbbrevWriter, DebugLocWriter &DebugLocWriter,
    DebugRangesSectionWriter &RangesSectionWriter,
    Optional<uint64_t> RangesBase) {
  // Cache debug ranges so that the offset for identical ranges could be reused.
  std::map<DebugAddressRangesVector, uint64_t> CachedRanges;

  uint64_t DIEOffset = Unit.getOffset() + Unit.getHeaderSize();
  uint64_t NextCUOffset = Unit.getNextUnitOffset();
  DWARFDebugInfoEntry Die;
  DWARFDataExtractor DebugInfoData = Unit.getDebugInfoExtractor();
  uint32_t Depth = 0;

  while (
      DIEOffset < NextCUOffset &&
      Die.extractFast(Unit, &DIEOffset, DebugInfoData, NextCUOffset, Depth)) {
    if (const DWARFAbbreviationDeclaration *AbbrDecl =
            Die.getAbbreviationDeclarationPtr()) {
      if (AbbrDecl->hasChildren())
        ++Depth;
    } else {
      // NULL entry.
      if (Depth > 0)
        --Depth;
      if (Depth == 0)
        break;
    }

    DWARFDie DIE(&Unit, &Die);

    switch (DIE.getTag()) {
    case dwarf::DW_TAG_compile_unit: {
      auto ModuleRangesOrError = DIE.getAddressRanges();
      if (!ModuleRangesOrError) {
        consumeError(ModuleRangesOrError.takeError());
        break;
      }
      DWARFAddressRangesVector &ModuleRanges = *ModuleRangesOrError;
      DebugAddressRangesVector OutputRanges =
          BC.translateModuleAddressRanges(ModuleRanges);
      const uint64_t RangesSectionOffset =
          RangesSectionWriter.addRanges(OutputRanges);
      if (!Unit.isDWOUnit())
        ARangesSectionWriter->addCURanges(Unit.getOffset(),
                                          std::move(OutputRanges));
      updateDWARFObjectAddressRanges(DIE, RangesSectionOffset, DebugInfoPatcher,
                                     AbbrevWriter, RangesBase);
      break;
    }
    case dwarf::DW_TAG_subprogram: {
      // Get function address either from ranges or [LowPC, HighPC) pair.
      uint64_t Address;
      uint64_t SectionIndex, HighPC;
      if (!DIE.getLowAndHighPC(Address, HighPC, SectionIndex)) {
        Expected<DWARFAddressRangesVector> RangesOrError =
            DIE.getAddressRanges();
        if (!RangesOrError) {
          consumeError(RangesOrError.takeError());
          break;
        }
        DWARFAddressRangesVector Ranges = *RangesOrError;
        // Not a function definition.
        if (Ranges.empty())
          break;

        Address = Ranges.front().LowPC;
      }

      // Clear cached ranges as the new function will have its own set.
      CachedRanges.clear();

      DebugAddressRangesVector FunctionRanges;
      if (const BinaryFunction *Function =
              BC.getBinaryFunctionAtAddress(Address))
        FunctionRanges = Function->getOutputAddressRanges();

      if (FunctionRanges.empty())
        FunctionRanges.push_back({0, 0});

      updateDWARFObjectAddressRanges(
          DIE, RangesSectionWriter.addRanges(FunctionRanges), DebugInfoPatcher,
          AbbrevWriter);

      break;
    }
    case dwarf::DW_TAG_lexical_block:
    case dwarf::DW_TAG_inlined_subroutine:
    case dwarf::DW_TAG_try_block:
    case dwarf::DW_TAG_catch_block: {
      uint64_t RangesSectionOffset = RangesSectionWriter.getEmptyRangesOffset();
      Expected<DWARFAddressRangesVector> RangesOrError = DIE.getAddressRanges();
      const BinaryFunction *Function =
          RangesOrError && !RangesOrError->empty()
              ? BC.getBinaryFunctionContainingAddress(
                    RangesOrError->front().LowPC)
              : nullptr;
      if (Function) {
        DebugAddressRangesVector OutputRanges =
            Function->translateInputToOutputRanges(*RangesOrError);
        LLVM_DEBUG(if (OutputRanges.empty() != RangesOrError->empty()) {
          dbgs() << "BOLT-DEBUG: problem with DIE at 0x"
                 << Twine::utohexstr(DIE.getOffset()) << " in CU at 0x"
                 << Twine::utohexstr(Unit.getOffset()) << '\n';
        });
        RangesSectionOffset = RangesSectionWriter.addRanges(
            std::move(OutputRanges), CachedRanges);
      } else if (!RangesOrError) {
        consumeError(RangesOrError.takeError());
      }
      updateDWARFObjectAddressRanges(DIE, RangesSectionOffset, DebugInfoPatcher,
                                     AbbrevWriter);
      break;
    }
    default: {
      // Handle any tag that can have DW_AT_location attribute.
      DWARFFormValue Value;
      uint64_t AttrOffset;
      if (Optional<AttrInfo> AttrVal =
              findAttributeInfo(DIE, dwarf::DW_AT_location)) {
        AttrOffset = AttrVal->Offset;
        Value = AttrVal->V;
        if (Value.isFormClass(DWARFFormValue::FC_Constant) ||
            Value.isFormClass(DWARFFormValue::FC_SectionOffset)) {
          uint64_t Offset = Value.isFormClass(DWARFFormValue::FC_Constant)
                                ? Value.getAsUnsignedConstant().getValue()
                                : Value.getAsSectionOffset().getValue();
          DebugLocationsVector InputLL;

          Optional<object::SectionedAddress> SectionAddress =
              Unit.getBaseAddress();
          uint64_t BaseAddress = 0;
          if (SectionAddress)
            BaseAddress = SectionAddress->Address;

          if (Unit.getVersion() >= 5) {
            Optional<uint64_t> LocOffset = Unit.getLoclistOffset(Offset);
            assert(LocOffset && "Location Offset is invalid.");
            Offset = *LocOffset;
          }

          Error E = Unit.getLocationTable().visitLocationList(
              &Offset, [&](const DWARFLocationEntry &Entry) {
                switch (Entry.Kind) {
                default:
                  llvm_unreachable("Unsupported DWARFLocationEntry Kind.");
                case dwarf::DW_LLE_end_of_list:
                  return false;
                case dwarf::DW_LLE_base_address: {
                  assert(Entry.SectionIndex == SectionedAddress::UndefSection &&
                         "absolute address expected");
                  BaseAddress = Entry.Value0;
                  break;
                }
                case dwarf::DW_LLE_offset_pair:
                  assert(
                      (Entry.SectionIndex == SectionedAddress::UndefSection &&
                       !Unit.isDWOUnit()) &&
                      "absolute address expected");
                  InputLL.emplace_back(DebugLocationEntry{
                      BaseAddress + Entry.Value0, BaseAddress + Entry.Value1,
                      Entry.Loc});
                  break;
                case dwarf::DW_RLE_start_length:
                  InputLL.emplace_back(DebugLocationEntry{
                      Entry.Value0, Entry.Value0 + Entry.Value1, Entry.Loc});
                  break;
                case dwarf::DW_LLE_base_addressx: {
                  Optional<object::SectionedAddress> EntryAddress =
                      Unit.getAddrOffsetSectionItem(Entry.Value0);
                  assert(EntryAddress && "base Address not found.");
                  BaseAddress = EntryAddress->Address;
                  break;
                }
                case dwarf::DW_LLE_startx_length: {
                  Optional<object::SectionedAddress> EntryAddress =
                      Unit.getAddrOffsetSectionItem(Entry.Value0);
                  assert(EntryAddress && "Address does not exist.");
                  InputLL.emplace_back(DebugLocationEntry{
                      EntryAddress->Address,
                      EntryAddress->Address + Entry.Value1, Entry.Loc});
                  break;
                }
                case dwarf::DW_LLE_startx_endx: {
                  Optional<object::SectionedAddress> StartAddress =
                      Unit.getAddrOffsetSectionItem(Entry.Value0);
                  assert(StartAddress && "Start Address does not exist.");
                  Optional<object::SectionedAddress> EndAddress =
                      Unit.getAddrOffsetSectionItem(Entry.Value1);
                  assert(EndAddress && "Start Address does not exist.");
                  InputLL.emplace_back(DebugLocationEntry{
                      StartAddress->Address, EndAddress->Address, Entry.Loc});
                  break;
                }
                }
                return true;
              });

          if (E || InputLL.empty()) {
            consumeError(std::move(E));
            errs() << "BOLT-WARNING: empty location list detected at 0x"
                   << Twine::utohexstr(Offset) << " for DIE at 0x"
                   << Twine::utohexstr(DIE.getOffset()) << " in CU at 0x"
                   << Twine::utohexstr(Unit.getOffset()) << '\n';
          } else {
            const uint64_t Address = InputLL.front().LowPC;
            DebugLocationsVector OutputLL;
            if (const BinaryFunction *Function =
                    BC.getBinaryFunctionContainingAddress(Address)) {
              OutputLL = Function->translateInputToOutputLocationList(InputLL);
              LLVM_DEBUG(if (OutputLL.empty()) {
                dbgs() << "BOLT-DEBUG: location list translated to an empty "
                          "one at 0x"
                       << Twine::utohexstr(DIE.getOffset()) << " in CU at 0x"
                       << Twine::utohexstr(Unit.getOffset()) << '\n';
              });
            } else {
              // It's possible for a subprogram to be removed and to have
              // address of 0. Adding this entry to output to preserve debug
              // information.
              OutputLL = InputLL;
            }
            uint32_t LocListIndex = 0;
            dwarf::Form Form = Value.getForm();
            if (Form == dwarf::DW_FORM_sec_offset ||
                Form == dwarf::DW_FORM_data4) {
              // For DWARF5 we can access location list entry either using
              // index, or offset. If it's offset, then it's from begnning of
              // the file. This implementation was before we could add entries
              // to the DIE. For DWARF4 this is no-op.
              // TODO: For DWARF5 convert all the offset based entries to index
              // based, and insert loclist_base if necessary.
              LocListIndex = DebugLoclistWriter::InvalidIndex;
            } else if (Form == dwarf::DW_FORM_loclistx) {
              LocListIndex = Value.getRawUValue();
            } else {
              llvm_unreachable("Unsupported LocList access Form.");
            }
            DebugLocWriter.addList(AttrOffset, LocListIndex,
                                   std::move(OutputLL));
          }
        } else {
          assert((Value.isFormClass(DWARFFormValue::FC_Exprloc) ||
                  Value.isFormClass(DWARFFormValue::FC_Block)) &&
                 "unexpected DW_AT_location form");
          if (Unit.isDWOUnit() || Unit.getVersion() >= 5) {
            ArrayRef<uint8_t> Expr = *Value.getAsBlock();
            DataExtractor Data(
                StringRef((const char *)Expr.data(), Expr.size()),
                Unit.getContext().isLittleEndian(), 0);
            DWARFExpression LocExpr(Data, Unit.getAddressByteSize(),
                                    Unit.getFormParams().Format);
            uint32_t PrevOffset = 0;
            constexpr uint32_t SizeOfOpcode = 1;
            constexpr uint32_t SizeOfForm = 1;
            for (auto &Expr : LocExpr) {
              if (!(Expr.getCode() == dwarf::DW_OP_GNU_addr_index ||
                    Expr.getCode() == dwarf::DW_OP_addrx))
                continue;

              const uint64_t Index = Expr.getRawOperand(0);
              Optional<object::SectionedAddress> EntryAddress =
                  Unit.getAddrOffsetSectionItem(Index);
              assert(EntryAddress && "Address is not found.");
              assert(Index <= std::numeric_limits<uint32_t>::max() &&
                     "Invalid Operand Index.");
              if (Expr.getCode() == dwarf::DW_OP_addrx) {
                const uint32_t EncodingSize =
                    Expr.getOperandEndOffset(0) - PrevOffset - SizeOfOpcode;
                const uint32_t Index = AddrWriter->getIndexFromAddress(
                    EntryAddress->Address, getCUId(Unit));
                // Encoding new size.
                SmallString<8> Tmp;
                raw_svector_ostream OSE(Tmp);
                encodeULEB128(Index, OSE);
                DebugInfoPatcher.addUDataPatch(AttrOffset, Tmp.size() + 1, 1);
                DebugInfoPatcher.addUDataPatch(AttrOffset + PrevOffset +
                                                   SizeOfOpcode + SizeOfForm,
                                               Index, EncodingSize);
              } else {
                // TODO: Re-do this as DWARF5.
                AddrWriter->addIndexAddress(EntryAddress->Address,
                                            static_cast<uint32_t>(Index),
                                            getCUId(Unit));
              }
              if (Expr.getDescription().Op[1] ==
                  DWARFExpression::Operation::SizeNA)
                PrevOffset = Expr.getOperandEndOffset(0);
              else
                PrevOffset = Expr.getOperandEndOffset(1);
            }
          }
        }
      } else if (Optional<AttrInfo> AttrVal =
                     findAttributeInfo(DIE, dwarf::DW_AT_low_pc)) {
        AttrOffset = AttrVal->Offset;
        Value = AttrVal->V;
        const Optional<uint64_t> Result = Value.getAsAddress();
        if (Result.hasValue()) {
          const uint64_t Address = Result.getValue();
          uint64_t NewAddress = 0;
          if (const BinaryFunction *Function =
                  BC.getBinaryFunctionContainingAddress(Address)) {
            NewAddress = Function->translateInputToOutputAddress(Address);
            LLVM_DEBUG(dbgs()
                       << "BOLT-DEBUG: Fixing low_pc 0x"
                       << Twine::utohexstr(Address) << " for DIE with tag "
                       << DIE.getTag() << " to 0x"
                       << Twine::utohexstr(NewAddress) << '\n');
          }

          dwarf::Form Form = Value.getForm();
          assert(Form != dwarf::DW_FORM_LLVM_addrx_offset &&
                 "DW_FORM_LLVM_addrx_offset is not supported");
          std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
          if (Form == dwarf::DW_FORM_GNU_addr_index) {
            const uint64_t Index = Value.getRawUValue();
            // If there is no new address, storing old address.
            // Re-using Index to make implementation easier.
            // DW_FORM_GNU_addr_index is variable lenght encoding
            // so we either have to create indices of same sizes, or use same
            // index.
            // TODO: We can now re-write .debug_info. This can be simplified to
            // just getting a new index and creating a patch.
            AddrWriter->addIndexAddress(NewAddress ? NewAddress : Address,
                                        Index, getCUId(Unit));
          } else if (Form == dwarf::DW_FORM_addrx) {
            const uint32_t Index = AddrWriter->getIndexFromAddress(
                NewAddress ? NewAddress : Address, getCUId(Unit));
            DebugInfoPatcher.addUDataPatch(AttrOffset, Index, AttrVal->Size);
          } else {
            DebugInfoPatcher.addLE64Patch(AttrOffset, NewAddress);
          }
        } else if (opts::Verbosity >= 1) {
          errs() << "BOLT-WARNING: unexpected form value for attribute at 0x"
                 << Twine::utohexstr(AttrOffset);
        }
      }
    }
    }

    // Handling references.
    assert(DIE.isValid() && "Invalid DIE.");
    const DWARFAbbreviationDeclaration *AbbrevDecl =
        DIE.getAbbreviationDeclarationPtr();
    if (!AbbrevDecl)
      continue;
    uint32_t Index = 0;
    for (const DWARFAbbreviationDeclaration::AttributeSpec &Decl :
         AbbrevDecl->attributes()) {
      switch (Decl.Form) {
      default:
        break;
      case dwarf::DW_FORM_ref1:
      case dwarf::DW_FORM_ref2:
      case dwarf::DW_FORM_ref4:
      case dwarf::DW_FORM_ref8:
      case dwarf::DW_FORM_ref_udata:
      case dwarf::DW_FORM_ref_addr: {
        Optional<AttrInfo> AttrVal = findAttributeInfo(DIE, AbbrevDecl, Index);
        uint32_t DestinationAddress =
            AttrVal->V.getRawUValue() +
            (Decl.Form == dwarf::DW_FORM_ref_addr ? 0 : Unit.getOffset());
        DebugInfoPatcher.addReferenceToPatch(
            AttrVal->Offset, DestinationAddress, AttrVal->Size, Decl.Form);
        // We can have only one reference, and it can be backward one.
        DebugInfoPatcher.addDestinationReferenceLabel(DestinationAddress);
        break;
      }
      }
      ++Index;
    }
  }
  if (DIEOffset > NextCUOffset)
    errs() << "BOLT-WARNING: corrupt DWARF detected at 0x"
           << Twine::utohexstr(Unit.getOffset()) << '\n';
}

void DWARFRewriter::updateDWARFObjectAddressRanges(
    const DWARFDie DIE, uint64_t DebugRangesOffset,
    SimpleBinaryPatcher &DebugInfoPatcher, DebugAbbrevWriter &AbbrevWriter,
    Optional<uint64_t> RangesBase) {

  // Some objects don't have an associated DIE and cannot be updated (such as
  // compiler-generated functions).
  if (!DIE)
    return;

  const DWARFAbbreviationDeclaration *AbbreviationDecl =
      DIE.getAbbreviationDeclarationPtr();
  if (!AbbreviationDecl) {
    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: object's DIE doesn't have an abbreviation: "
             << "skipping update. DIE at offset 0x"
             << Twine::utohexstr(DIE.getOffset()) << '\n';
    return;
  }

  if (RangesBase) {
    // If DW_AT_GNU_ranges_base is present, update it. No further modifications
    // are needed for ranges base.
    Optional<AttrInfo> RangesBaseAttrInfo =
        findAttributeInfo(DIE, dwarf::DW_AT_GNU_ranges_base);
    if (!RangesBaseAttrInfo)
      RangesBaseAttrInfo = findAttributeInfo(DIE, dwarf::DW_AT_rnglists_base);

    if (RangesBaseAttrInfo) {
      DebugInfoPatcher.addLE32Patch(RangesBaseAttrInfo->Offset,
                                    static_cast<uint32_t>(*RangesBase),
                                    RangesBaseAttrInfo->Size);
      RangesBase = None;
    }
  }

  Optional<AttrInfo> LowPCAttrInfo =
      findAttributeInfo(DIE, dwarf::DW_AT_low_pc);
  if (Optional<AttrInfo> AttrVal =
          findAttributeInfo(DIE, dwarf::DW_AT_ranges)) {
    // Case 1: The object was already non-contiguous and had DW_AT_ranges.
    // In this case we simply need to update the value of DW_AT_ranges
    // and introduce DW_AT_GNU_ranges_base if required.
    std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
    // For DWARF5 converting all of DW_AT_ranges into DW_FORM_rnglistx
    bool Converted = false;
    if (DIE.getDwarfUnit()->getVersion() >= 5 &&
        AttrVal->V.getForm() == dwarf::DW_FORM_sec_offset) {
      AbbrevWriter.addAttributePatch(*DIE.getDwarfUnit(), AbbreviationDecl,
                                     dwarf::DW_AT_ranges, dwarf::DW_AT_ranges,
                                     dwarf::DW_FORM_rnglistx);
      Converted = true;
    }
    if (Converted || AttrVal->V.getForm() == dwarf::DW_FORM_rnglistx)
      DebugInfoPatcher.addUDataPatch(AttrVal->Offset, DebugRangesOffset,
                                     AttrVal->Size);
    else
      DebugInfoPatcher.addLE32Patch(
          AttrVal->Offset, DebugRangesOffset - DebugInfoPatcher.getRangeBase(),
          AttrVal->Size);

    if (!RangesBase) {
      if (LowPCAttrInfo &&
          LowPCAttrInfo->V.getForm() != dwarf::DW_FORM_GNU_addr_index &&
          LowPCAttrInfo->V.getForm() != dwarf::DW_FORM_addrx)
        DebugInfoPatcher.addLE64Patch(LowPCAttrInfo->Offset, 0);
      return;
    }

    // Convert DW_AT_low_pc into DW_AT_GNU_ranges_base.
    if (!LowPCAttrInfo) {
      errs() << "BOLT-ERROR: skeleton CU at 0x"
             << Twine::utohexstr(DIE.getOffset())
             << " does not have DW_AT_GNU_ranges_base or DW_AT_low_pc to"
                " convert to update ranges base\n";
      return;
    }

    AbbrevWriter.addAttribute(*DIE.getDwarfUnit(), AbbreviationDecl,
                              dwarf::DW_AT_GNU_ranges_base,
                              dwarf::DW_FORM_sec_offset);
    reinterpret_cast<DebugInfoBinaryPatcher &>(DebugInfoPatcher)
        .insertNewEntry(DIE, *RangesBase);

    return;
  }

  // Case 2: The object has both DW_AT_low_pc and DW_AT_high_pc emitted back
  // to back. Replace with new attributes and patch the DIE.
  Optional<AttrInfo> HighPCAttrInfo =
      findAttributeInfo(DIE, dwarf::DW_AT_high_pc);
  if (LowPCAttrInfo && HighPCAttrInfo) {
    convertToRangesPatchAbbrev(*DIE.getDwarfUnit(), AbbreviationDecl,
                               AbbrevWriter, RangesBase);
    convertToRangesPatchDebugInfo(DIE, DebugRangesOffset, DebugInfoPatcher,
                                  RangesBase);
  } else {
    if (opts::Verbosity >= 1)
      errs() << "BOLT-ERROR: cannot update ranges for DIE at offset 0x"
             << Twine::utohexstr(DIE.getOffset()) << '\n';
  }
}

void DWARFRewriter::updateLineTableOffsets(const MCAsmLayout &Layout) {
  ErrorOr<BinarySection &> DbgInfoSection =
      BC.getUniqueSectionByName(".debug_info");
  ErrorOr<BinarySection &> TypeInfoSection =
      BC.getUniqueSectionByName(".debug_types");
  assert(((BC.DwCtx->getNumTypeUnits() > 0 && TypeInfoSection) ||
          BC.DwCtx->getNumTypeUnits() == 0) &&
         "Was not able to retrieve Debug Types section.");

  // We will be re-writing .debug_info so relocation mechanism doesn't work for
  // Debug Info Patcher.
  DebugInfoBinaryPatcher *DebugInfoPatcher = nullptr;
  if (BC.DwCtx->getNumCompileUnits()) {
    DbgInfoSection->registerPatcher(std::make_unique<DebugInfoBinaryPatcher>());
    DebugInfoPatcher =
        static_cast<DebugInfoBinaryPatcher *>(DbgInfoSection->getPatcher());
  }

  // There is no direct connection between CU and TU, but same offsets,
  // encoded in DW_AT_stmt_list, into .debug_line get modified.
  // We take advantage of that to map original CU line table offsets to new
  // ones.
  std::unordered_map<uint64_t, uint64_t> DebugLineOffsetMap;

  auto GetStatementListValue = [](DWARFUnit *Unit) {
    Optional<DWARFFormValue> StmtList =
        Unit->getUnitDIE().find(dwarf::DW_AT_stmt_list);
    Optional<uint64_t> Offset = dwarf::toSectionOffset(StmtList);
    assert(Offset && "Was not able to retreive value of DW_AT_stmt_list.");
    return *Offset;
  };

  const uint64_t Reloc32Type = BC.isAArch64()
                                   ? static_cast<uint64_t>(ELF::R_AARCH64_ABS32)
                                   : static_cast<uint64_t>(ELF::R_X86_64_32);

  for (const std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
    const unsigned CUID = CU->getOffset();
    MCSymbol *Label = BC.getDwarfLineTable(CUID).getLabel();
    if (!Label)
      continue;

    Optional<AttrInfo> AttrVal =
        findAttributeInfo(CU.get()->getUnitDIE(), dwarf::DW_AT_stmt_list);
    if (!AttrVal)
      continue;

    const uint64_t AttributeOffset = AttrVal->Offset;
    const uint64_t LineTableOffset = Layout.getSymbolOffset(*Label);
    DebugLineOffsetMap[GetStatementListValue(CU.get())] = LineTableOffset;
    assert(DbgInfoSection && ".debug_info section must exist");
    DebugInfoPatcher->addLE32Patch(AttributeOffset, LineTableOffset);
  }

  for (const std::unique_ptr<DWARFUnit> &TU : BC.DwCtx->types_section_units()) {
    DWARFUnit *Unit = TU.get();
    Optional<AttrInfo> AttrVal =
        findAttributeInfo(TU.get()->getUnitDIE(), dwarf::DW_AT_stmt_list);
    if (!AttrVal)
      continue;
    const uint64_t AttributeOffset = AttrVal->Offset;
    auto Iter = DebugLineOffsetMap.find(GetStatementListValue(Unit));
    assert(Iter != DebugLineOffsetMap.end() &&
           "Type Unit Updated Line Number Entry does not exist.");
    TypeInfoSection->addRelocation(AttributeOffset, nullptr, Reloc32Type,
                                   Iter->second, 0, /*Pending=*/true);
  }

  // Set .debug_info as finalized so it won't be skipped over when
  // we process sections while writing out the new binary. This ensures
  // that the pending relocations will be processed and not ignored.
  if (DbgInfoSection)
    DbgInfoSection->setIsFinalized();

  if (TypeInfoSection)
    TypeInfoSection->setIsFinalized();
}

CUOffsetMap
DWARFRewriter::finalizeDebugSections(DebugInfoBinaryPatcher &DebugInfoPatcher) {
  if (StrWriter->isInitialized()) {
    RewriteInstance::addToDebugSectionsToOverwrite(".debug_str");
    std::unique_ptr<DebugStrBufferVector> DebugStrSectionContents =
        StrWriter->finalize();
    BC.registerOrUpdateNoteSection(".debug_str",
                                   copyByteArray(*DebugStrSectionContents),
                                   DebugStrSectionContents->size());
  }

  std::unique_ptr<DebugBufferVector> RangesSectionContents =
      RangesSectionWriter->finalize();
  BC.registerOrUpdateNoteSection(
      llvm::isa<DebugRangeListsSectionWriter>(*RangesSectionWriter)
          ? ".debug_rnglists"
          : ".debug_ranges",
      copyByteArray(*RangesSectionContents), RangesSectionContents->size());

  if (BC.isDWARF5Used()) {
    std::unique_ptr<DebugBufferVector> LocationListSectionContents =
        makeFinalLocListsSection(DebugInfoPatcher, DWARFVersion::DWARF5);
    BC.registerOrUpdateNoteSection(".debug_loclists",
                                   copyByteArray(*LocationListSectionContents),
                                   LocationListSectionContents->size());
  }

  if (BC.isDWARFLegacyUsed()) {
    std::unique_ptr<DebugBufferVector> LocationListSectionContents =
        makeFinalLocListsSection(DebugInfoPatcher, DWARFVersion::DWARFLegacy);
    BC.registerOrUpdateNoteSection(".debug_loc",
                                   copyByteArray(*LocationListSectionContents),
                                   LocationListSectionContents->size());
  }

  // AddrWriter should be finalized after debug_loc since more addresses can be
  // added there.
  if (AddrWriter->isInitialized()) {
    AddressSectionBuffer AddressSectionContents = AddrWriter->finalize();
    BC.registerOrUpdateNoteSection(".debug_addr",
                                   copyByteArray(AddressSectionContents),
                                   AddressSectionContents.size());
    for (auto &CU : BC.DwCtx->compile_units()) {
      DWARFDie DIE = CU->getUnitDIE();
      uint64_t Offset = 0;
      uint64_t AttrOffset = 0;
      uint32_t Size = 0;
      Optional<AttrInfo> AttrValGnu =
          findAttributeInfo(DIE, dwarf::DW_AT_GNU_addr_base);
      Optional<AttrInfo> AttrVal =
          findAttributeInfo(DIE, dwarf::DW_AT_addr_base);
      Offset = AddrWriter->getOffset(*CU);

      if (AttrValGnu) {
        AttrOffset = AttrValGnu->Offset;
        Size = AttrValGnu->Size;
      }

      if (AttrVal) {
        AttrOffset = AttrVal->Offset;
        Size = AttrVal->Size;
      }

      if (AttrValGnu || AttrVal) {
        DebugInfoPatcher.addLE32Patch(AttrOffset, static_cast<int32_t>(Offset),
                                      Size);
      } else if (CU->getVersion() >= 5) {
        // A case where we were not using .debug_addr section, but after update
        // now using it.
        const DWARFAbbreviationDeclaration *Abbrev =
            DIE.getAbbreviationDeclarationPtr();
        AbbrevWriter->addAttribute(*CU, Abbrev, dwarf::DW_AT_addr_base,
                                   dwarf::DW_FORM_sec_offset);
        DebugInfoPatcher.insertNewEntry(DIE, static_cast<int32_t>(Offset));
      } else
        llvm_unreachable(
            "DWO CU uses .debug_address, but DW_AT_GNU_addr_base is missing.");
    }
  }

  std::unique_ptr<DebugBufferVector> AbbrevSectionContents =
      AbbrevWriter->finalize();
  BC.registerOrUpdateNoteSection(".debug_abbrev",
                                 copyByteArray(*AbbrevSectionContents),
                                 AbbrevSectionContents->size());

  // Update abbreviation offsets for CUs/TUs if they were changed.
  SimpleBinaryPatcher *DebugTypesPatcher = nullptr;
  for (auto &Unit : BC.DwCtx->normal_units()) {
    const uint64_t NewAbbrevOffset =
        AbbrevWriter->getAbbreviationsOffsetForUnit(*Unit);
    if (Unit->getAbbreviationsOffset() == NewAbbrevOffset)
      continue;

    // DWARFv4 or earlier
    // unit_length - 4 bytes
    // version - 2 bytes
    // So + 6 to patch debug_abbrev_offset
    constexpr uint64_t AbbrevFieldOffsetLegacy = 6;
    // DWARFv5
    // unit_length - 4 bytes
    // version - 2 bytes
    // unit_type - 1 byte
    // address_size - 1 byte
    // So + 8 to patch debug_abbrev_offset
    constexpr uint64_t AbbrevFieldOffsetV5 = 8;
    uint64_t AbbrevOffset =
        Unit->getVersion() >= 5 ? AbbrevFieldOffsetV5 : AbbrevFieldOffsetLegacy;
    if (!Unit->isTypeUnit() || Unit->getVersion() >= 5) {
      DebugInfoPatcher.addLE32Patch(Unit->getOffset() + AbbrevOffset,
                                    static_cast<uint32_t>(NewAbbrevOffset));
      continue;
    }

    if (!DebugTypesPatcher) {
      ErrorOr<BinarySection &> DebugTypes =
          BC.getUniqueSectionByName(".debug_types");
      DebugTypes->registerPatcher(std::make_unique<SimpleBinaryPatcher>());
      DebugTypesPatcher =
          static_cast<SimpleBinaryPatcher *>(DebugTypes->getPatcher());
    }
    DebugTypesPatcher->addLE32Patch(Unit->getOffset() + AbbrevOffset,
                                    static_cast<uint32_t>(NewAbbrevOffset));
  }

  // No more creating new DebugInfoPatches.
  CUOffsetMap CUMap =
      DebugInfoPatcher.computeNewOffsets(*BC.DwCtx.get(), false);

  // Skip .debug_aranges if we are re-generating .gdb_index.
  if (opts::KeepARanges || !BC.getGdbIndexSection()) {
    SmallVector<char, 16> ARangesBuffer;
    raw_svector_ostream OS(ARangesBuffer);

    auto MAB = std::unique_ptr<MCAsmBackend>(
        BC.TheTarget->createMCAsmBackend(*BC.STI, *BC.MRI, MCTargetOptions()));

    ARangesSectionWriter->writeARangesSection(OS, CUMap);
    const StringRef &ARangesContents = OS.str();

    BC.registerOrUpdateNoteSection(".debug_aranges",
                                   copyByteArray(ARangesContents),
                                   ARangesContents.size());
  }
  return CUMap;
}

// Creates all the data structures necessary for creating MCStreamer.
// They are passed by reference because they need to be kept around.
// Also creates known debug sections. These are sections handled by
// handleDebugDataPatching.
using KnownSectionsEntry = std::pair<MCSection *, DWARFSectionKind>;
namespace {

std::unique_ptr<BinaryContext>
createDwarfOnlyBC(const object::ObjectFile &File) {
  return cantFail(BinaryContext::createBinaryContext(
      &File, false,
      DWARFContext::create(File, DWARFContext::ProcessDebugRelocations::Ignore,
                           nullptr, "", WithColor::defaultErrorHandler,
                           WithColor::defaultWarningHandler)));
}

StringMap<KnownSectionsEntry>
createKnownSectionsMap(const MCObjectFileInfo &MCOFI) {
  StringMap<KnownSectionsEntry> KnownSectionsTemp = {
      {"debug_info.dwo", {MCOFI.getDwarfInfoDWOSection(), DW_SECT_INFO}},
      {"debug_types.dwo", {MCOFI.getDwarfTypesDWOSection(), DW_SECT_EXT_TYPES}},
      {"debug_str_offsets.dwo",
       {MCOFI.getDwarfStrOffDWOSection(), DW_SECT_STR_OFFSETS}},
      {"debug_str.dwo", {MCOFI.getDwarfStrDWOSection(), DW_SECT_EXT_unknown}},
      {"debug_loc.dwo", {MCOFI.getDwarfLocDWOSection(), DW_SECT_EXT_LOC}},
      {"debug_abbrev.dwo", {MCOFI.getDwarfAbbrevDWOSection(), DW_SECT_ABBREV}},
      {"debug_line.dwo", {MCOFI.getDwarfLineDWOSection(), DW_SECT_LINE}}};
  return KnownSectionsTemp;
}

StringRef getSectionName(const SectionRef &Section) {
  Expected<StringRef> SectionName = Section.getName();
  assert(SectionName && "Invalid section name.");
  StringRef Name = *SectionName;
  Name = Name.substr(Name.find_first_not_of("._"));
  return Name;
}

// Exctracts an appropriate slice if input is DWP.
// Applies patches or overwrites the section.
Optional<StringRef> updateDebugData(
    DWARFContext &DWCtx, std::string &Storage, const SectionRef &Section,
    const StringMap<KnownSectionsEntry> &KnownSections, MCStreamer &Streamer,
    DWARFRewriter &Writer, const DWARFUnitIndex::Entry *DWOEntry,
    uint64_t DWOId, std::unique_ptr<DebugBufferVector> &OutputBuffer) {
  auto applyPatch = [&](DebugInfoBinaryPatcher *Patcher,
                        StringRef Data) -> StringRef {
    Patcher->computeNewOffsets(DWCtx, true);
    Storage = Patcher->patchBinary(Data);
    return StringRef(Storage.c_str(), Storage.size());
  };

  using DWOSectionContribution =
      const DWARFUnitIndex::Entry::SectionContribution;
  auto getSliceData = [&](const DWARFUnitIndex::Entry *DWOEntry,
                          StringRef OutData, DWARFSectionKind Sec,
                          uint32_t &DWPOffset) -> StringRef {
    if (DWOEntry) {
      DWOSectionContribution *DWOContrubution = DWOEntry->getContribution(Sec);
      DWPOffset = DWOContrubution->Offset;
      OutData = OutData.substr(DWPOffset, DWOContrubution->Length);
    }
    return OutData;
  };

  StringRef Name = getSectionName(Section);
  auto SectionIter = KnownSections.find(Name);
  if (SectionIter == KnownSections.end())
    return None;
  Streamer.SwitchSection(SectionIter->second.first);
  Expected<StringRef> Contents = Section.getContents();
  assert(Contents && "Invalid contents.");
  StringRef OutData = *Contents;
  uint32_t DWPOffset = 0;

  switch (SectionIter->second.second) {
  default: {
    if (!Name.equals("debug_str.dwo"))
      errs() << "BOLT-WARNING: Unsupported Debug section: " << Name << "\n";
    return OutData;
  }
  case DWARFSectionKind::DW_SECT_INFO: {
    OutData = getSliceData(DWOEntry, OutData, DWARFSectionKind::DW_SECT_INFO,
                           DWPOffset);
    DebugInfoBinaryPatcher *Patcher = llvm::cast<DebugInfoBinaryPatcher>(
        Writer.getBinaryDWODebugInfoPatcher(DWOId));
    return applyPatch(Patcher, OutData);
  }
  case DWARFSectionKind::DW_SECT_EXT_TYPES: {
    return getSliceData(DWOEntry, OutData, DWARFSectionKind::DW_SECT_EXT_TYPES,
                        DWPOffset);
  }
  case DWARFSectionKind::DW_SECT_STR_OFFSETS: {
    return getSliceData(DWOEntry, OutData,
                        DWARFSectionKind::DW_SECT_STR_OFFSETS, DWPOffset);
  }
  case DWARFSectionKind::DW_SECT_ABBREV: {
    DebugAbbrevWriter *AbbrevWriter = Writer.getBinaryDWOAbbrevWriter(DWOId);
    OutputBuffer = AbbrevWriter->finalize();
    // Creating explicit StringRef here, otherwise
    // with impicit conversion it will take null byte as end of
    // string.
    return StringRef(reinterpret_cast<const char *>(OutputBuffer->data()),
                     OutputBuffer->size());
  }
  case DWARFSectionKind::DW_SECT_EXT_LOC: {
    DebugLocWriter *LocWriter = Writer.getDebugLocWriter(DWOId);
    OutputBuffer = LocWriter->getBuffer();
    // Creating explicit StringRef here, otherwise
    // with impicit conversion it will take null byte as end of
    // string.
    return StringRef(reinterpret_cast<const char *>(OutputBuffer->data()),
                     OutputBuffer->size());
  }
  case DWARFSectionKind::DW_SECT_LINE: {
    return getSliceData(DWOEntry, OutData, DWARFSectionKind::DW_SECT_LINE,
                        DWPOffset);
  }
  }
}

} // namespace

void DWARFRewriter::writeDWP(
    std::unordered_map<uint64_t, std::string> &DWOIdToName) {
  SmallString<0> OutputNameStr;
  StringRef OutputName;
  if (opts::DwarfOutputPath.empty()) {
    OutputName =
        Twine(opts::OutputFilename).concat(".dwp").toStringRef(OutputNameStr);
  } else {
    StringRef ExeFileName = llvm::sys::path::filename(opts::OutputFilename);
    OutputName = Twine(opts::DwarfOutputPath)
                     .concat("/")
                     .concat(ExeFileName)
                     .concat(".dwp")
                     .toStringRef(OutputNameStr);
    errs() << "BOLT-WARNING: dwarf-output-path is in effect and .dwp file will "
              "possibly be written to another location that is not the same as "
              "the executable\n";
  }
  std::error_code EC;
  std::unique_ptr<ToolOutputFile> Out =
      std::make_unique<ToolOutputFile>(OutputName, EC, sys::fs::OF_None);

  const object::ObjectFile *File = BC.DwCtx->getDWARFObj().getFile();
  std::unique_ptr<BinaryContext> TmpBC = createDwarfOnlyBC(*File);
  std::unique_ptr<MCStreamer> Streamer = TmpBC->createStreamer(Out->os());
  const MCObjectFileInfo &MCOFI = *Streamer->getContext().getObjectFileInfo();
  StringMap<KnownSectionsEntry> KnownSections = createKnownSectionsMap(MCOFI);
  MCSection *const StrSection = MCOFI.getDwarfStrDWOSection();
  MCSection *const StrOffsetSection = MCOFI.getDwarfStrOffDWOSection();

  // Data Structures for DWP book keeping
  // Size of array corresponds to the number of sections supported by DWO format
  // in DWARF4/5.
  uint32_t ContributionOffsets[8] = {};
  std::deque<SmallString<32>> UncompressedSections;
  DWPStringPool Strings(*Streamer, StrSection);
  MapVector<uint64_t, UnitIndexEntry> IndexEntries;
  constexpr uint32_t IndexVersion = 2;

  // Setup DWP code once.
  DWARFContext *DWOCtx = BC.getDWOContext();
  const DWARFUnitIndex *CUIndex = nullptr;
  bool IsDWP = false;
  if (DWOCtx) {
    CUIndex = &DWOCtx->getCUIndex();
    IsDWP = !CUIndex->getRows().empty();
  }

  for (const std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
    Optional<uint64_t> DWOId = CU->getDWOId();
    if (!DWOId)
      continue;

    // Skipping CUs that we failed to load.
    Optional<DWARFUnit *> DWOCU = BC.getDWOCU(*DWOId);
    if (!DWOCU)
      continue;

    assert(CU->getVersion() <= 4 && "For DWP output only DWARF4 is supported");
    UnitIndexEntry CurEntry = {};
    CurEntry.DWOName =
        dwarf::toString(CU->getUnitDIE().find(
                            {dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}),
                        "");
    const char *Name = CU->getUnitDIE().getShortName();
    if (Name)
      CurEntry.Name = Name;
    StringRef CurStrSection;
    StringRef CurStrOffsetSection;

    // This maps each section contained in this file to its length.
    // This information is later on used to calculate the contributions,
    // i.e. offset and length, of each compile/type unit to a section.
    std::vector<std::pair<DWARFSectionKind, uint32_t>> SectionLength;

    const DWARFUnitIndex::Entry *DWOEntry = nullptr;
    if (IsDWP)
      DWOEntry = CUIndex->getFromHash(*DWOId);

    bool StrSectionWrittenOut = false;
    const object::ObjectFile *DWOFile =
        (*DWOCU)->getContext().getDWARFObj().getFile();
    for (const SectionRef &Section : DWOFile->sections()) {
      std::string Storage = "";
      std::unique_ptr<DebugBufferVector> OutputData;
      Optional<StringRef> TOutData = updateDebugData(
          (*DWOCU)->getContext(), Storage, Section, KnownSections, *Streamer,
          *this, DWOEntry, *DWOId, OutputData);
      if (!TOutData)
        continue;

      StringRef OutData = *TOutData;
      StringRef Name = getSectionName(Section);
      if (Name.equals("debug_str.dwo")) {
        CurStrSection = OutData;
      } else {
        // Since handleDebugDataPatching returned true, we already know this is
        // a known section.
        auto SectionIter = KnownSections.find(Name);
        if (SectionIter->second.second == DWARFSectionKind::DW_SECT_STR_OFFSETS)
          CurStrOffsetSection = OutData;
        else
          Streamer->emitBytes(OutData);
        auto Index =
            getContributionIndex(SectionIter->second.second, IndexVersion);
        CurEntry.Contributions[Index].Offset = ContributionOffsets[Index];
        CurEntry.Contributions[Index].Length = OutData.size();
        ContributionOffsets[Index] += CurEntry.Contributions[Index].Length;
      }

      // Strings are combined in to a new string section, and de-duplicated
      // based on hash.
      if (!StrSectionWrittenOut && !CurStrOffsetSection.empty() &&
          !CurStrSection.empty()) {
        writeStringsAndOffsets(*Streamer.get(), Strings, StrOffsetSection,
                               CurStrSection, CurStrOffsetSection,
                               CU->getVersion());
        StrSectionWrittenOut = true;
      }
    }
    CompileUnitIdentifiers CUI{*DWOId, CurEntry.Name.c_str(),
                               CurEntry.DWOName.c_str()};
    auto P = IndexEntries.insert(std::make_pair(CUI.Signature, CurEntry));
    if (!P.second) {
      Error Err = buildDuplicateError(*P.first, CUI, "");
      errs() << "BOLT-ERROR: " << toString(std::move(Err)) << "\n";
      return;
    }
  }

  // Lie about the type contribution for DWARF < 5. In DWARFv5 the type
  // section does not exist, so no need to do anything about this.
  ContributionOffsets[getContributionIndex(DW_SECT_EXT_TYPES, 2)] = 0;
  writeIndex(*Streamer.get(), MCOFI.getDwarfCUIndexSection(),
             ContributionOffsets, IndexEntries, IndexVersion);

  Streamer->Finish();
  Out->keep();
}

void DWARFRewriter::writeDWOFiles(
    std::unordered_map<uint64_t, std::string> &DWOIdToName) {
  // Setup DWP code once.
  DWARFContext *DWOCtx = BC.getDWOContext();
  const DWARFUnitIndex *CUIndex = nullptr;
  bool IsDWP = false;
  if (DWOCtx) {
    CUIndex = &DWOCtx->getCUIndex();
    IsDWP = !CUIndex->getRows().empty();
  }

  for (const std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
    Optional<uint64_t> DWOId = CU->getDWOId();
    if (!DWOId)
      continue;

    // Skipping CUs that we failed to load.
    Optional<DWARFUnit *> DWOCU = BC.getDWOCU(*DWOId);
    if (!DWOCU)
      continue;

    std::string CompDir = opts::DwarfOutputPath.empty()
                              ? CU->getCompilationDir()
                              : opts::DwarfOutputPath.c_str();
    std::string ObjectName = getDWOName(*CU.get(), nullptr, DWOIdToName);
    auto FullPath = CompDir.append("/").append(ObjectName);

    std::error_code EC;
    std::unique_ptr<ToolOutputFile> TempOut =
        std::make_unique<ToolOutputFile>(FullPath, EC, sys::fs::OF_None);

    const DWARFUnitIndex::Entry *DWOEntry = nullptr;
    if (IsDWP)
      DWOEntry = CUIndex->getFromHash(*DWOId);

    const object::ObjectFile *File =
        (*DWOCU)->getContext().getDWARFObj().getFile();
    std::unique_ptr<BinaryContext> TmpBC = createDwarfOnlyBC(*File);
    std::unique_ptr<MCStreamer> Streamer = TmpBC->createStreamer(TempOut->os());
    StringMap<KnownSectionsEntry> KnownSections =
        createKnownSectionsMap(*Streamer->getContext().getObjectFileInfo());

    for (const SectionRef &Section : File->sections()) {
      std::string Storage = "";
      std::unique_ptr<DebugBufferVector> OutputData;
      if (Optional<StringRef> OutData = updateDebugData(
              (*DWOCU)->getContext(), Storage, Section, KnownSections,
              *Streamer, *this, DWOEntry, *DWOId, OutputData))
        Streamer->emitBytes(*OutData);
    }
    Streamer->Finish();
    TempOut->keep();
  }
}

void DWARFRewriter::updateGdbIndexSection(CUOffsetMap &CUMap) {
  if (!BC.getGdbIndexSection())
    return;

  // See https://sourceware.org/gdb/onlinedocs/gdb/Index-Section-Format.html
  // for .gdb_index section format.

  StringRef GdbIndexContents = BC.getGdbIndexSection()->getContents();

  const char *Data = GdbIndexContents.data();

  // Parse the header.
  const uint32_t Version = read32le(Data);
  if (Version != 7 && Version != 8) {
    errs() << "BOLT-ERROR: can only process .gdb_index versions 7 and 8\n";
    exit(1);
  }

  // Some .gdb_index generators use file offsets while others use section
  // offsets. Hence we can only rely on offsets relative to each other,
  // and ignore their absolute values.
  const uint32_t CUListOffset = read32le(Data + 4);
  const uint32_t CUTypesOffset = read32le(Data + 8);
  const uint32_t AddressTableOffset = read32le(Data + 12);
  const uint32_t SymbolTableOffset = read32le(Data + 16);
  const uint32_t ConstantPoolOffset = read32le(Data + 20);
  Data += 24;

  // Map CUs offsets to indices and verify existing index table.
  std::map<uint32_t, uint32_t> OffsetToIndexMap;
  const uint32_t CUListSize = CUTypesOffset - CUListOffset;
  const unsigned NumCUs = BC.DwCtx->getNumCompileUnits();
  if (CUListSize != NumCUs * 16) {
    errs() << "BOLT-ERROR: .gdb_index: CU count mismatch\n";
    exit(1);
  }
  for (unsigned Index = 0; Index < NumCUs; ++Index, Data += 16) {
    const DWARFUnit *CU = BC.DwCtx->getUnitAtIndex(Index);
    const uint64_t Offset = read64le(Data);
    if (CU->getOffset() != Offset) {
      errs() << "BOLT-ERROR: .gdb_index CU offset mismatch\n";
      exit(1);
    }

    OffsetToIndexMap[Offset] = Index;
  }

  // Ignore old address table.
  const uint32_t OldAddressTableSize = SymbolTableOffset - AddressTableOffset;
  // Move Data to the beginning of symbol table.
  Data += SymbolTableOffset - CUTypesOffset;

  // Calculate the size of the new address table.
  uint32_t NewAddressTableSize = 0;
  for (const auto &CURangesPair : ARangesSectionWriter->getCUAddressRanges()) {
    const SmallVector<DebugAddressRange, 2> &Ranges = CURangesPair.second;
    NewAddressTableSize += Ranges.size() * 20;
  }

  // Difference between old and new table (and section) sizes.
  // Could be negative.
  int32_t Delta = NewAddressTableSize - OldAddressTableSize;

  size_t NewGdbIndexSize = GdbIndexContents.size() + Delta;

  // Free'd by ExecutableFileMemoryManager.
  auto *NewGdbIndexContents = new uint8_t[NewGdbIndexSize];
  uint8_t *Buffer = NewGdbIndexContents;

  write32le(Buffer, Version);
  write32le(Buffer + 4, CUListOffset);
  write32le(Buffer + 8, CUTypesOffset);
  write32le(Buffer + 12, AddressTableOffset);
  write32le(Buffer + 16, SymbolTableOffset + Delta);
  write32le(Buffer + 20, ConstantPoolOffset + Delta);
  Buffer += 24;

  // Writing out CU List <Offset, Size>
  for (auto &CUInfo : CUMap) {
    write64le(Buffer, CUInfo.second.Offset);
    // Length encoded in CU doesn't contain first 4 bytes that encode length.
    write64le(Buffer + 8, CUInfo.second.Length + 4);
    Buffer += 16;
  }

  // Copy over types CU list
  // Spec says " triplet, the first value is the CU offset, the second value is
  // the type offset in the CU, and the third value is the type signature"
  // Looking at what is being generated by gdb-add-index. The first entry is TU
  // offset, second entry is offset from it, and third entry is the type
  // signature.
  memcpy(Buffer, GdbIndexContents.data() + CUTypesOffset,
         AddressTableOffset - CUTypesOffset);
  Buffer += AddressTableOffset - CUTypesOffset;

  // Generate new address table.
  for (const std::pair<const uint64_t, DebugAddressRangesVector> &CURangesPair :
       ARangesSectionWriter->getCUAddressRanges()) {
    const uint32_t CUIndex = OffsetToIndexMap[CURangesPair.first];
    const DebugAddressRangesVector &Ranges = CURangesPair.second;
    for (const DebugAddressRange &Range : Ranges) {
      write64le(Buffer, Range.LowPC);
      write64le(Buffer + 8, Range.HighPC);
      write32le(Buffer + 16, CUIndex);
      Buffer += 20;
    }
  }

  const size_t TrailingSize =
      GdbIndexContents.data() + GdbIndexContents.size() - Data;
  assert(Buffer + TrailingSize == NewGdbIndexContents + NewGdbIndexSize &&
         "size calculation error");

  // Copy over the rest of the original data.
  memcpy(Buffer, Data, TrailingSize);

  // Register the new section.
  BC.registerOrUpdateNoteSection(".gdb_index", NewGdbIndexContents,
                                 NewGdbIndexSize);
}

std::unique_ptr<DebugBufferVector>
DWARFRewriter::makeFinalLocListsSection(SimpleBinaryPatcher &DebugInfoPatcher,
                                        DWARFVersion Version) {
  auto LocBuffer = std::make_unique<DebugBufferVector>();
  auto LocStream = std::make_unique<raw_svector_ostream>(*LocBuffer);
  auto Writer =
      std::unique_ptr<MCObjectWriter>(BC.createObjectWriter(*LocStream));

  uint64_t SectionOffset = 0;
  // Add an empty list as the first entry;
  if (LocListWritersByCU.empty() ||
      LocListWritersByCU.begin()->second.get()->getDwarfVersion() < 5) {
    // Should be fine for both DWARF4 and DWARF5?
    const char Zeroes[16] = {0};
    *LocStream << StringRef(Zeroes, 16);
    SectionOffset += 2 * 8;
  }

  for (std::pair<const uint64_t, std::unique_ptr<DebugLocWriter>> &Loc :
       LocListWritersByCU) {
    DebugLocWriter *LocWriter = Loc.second.get();
    auto *LocListWriter = llvm::dyn_cast<DebugLoclistWriter>(LocWriter);

    if (Version == DWARFVersion::DWARF5 &&
        (!LocListWriter || LocListWriter->getDwarfVersion() <= 4))
      continue;

    if (Version == DWARFVersion::DWARFLegacy &&
        (LocListWriter && LocListWriter->getDwarfVersion() >= 5))
      continue;
    if (LocListWriter && (LocListWriter->getDwarfVersion() <= 4 ||
                          (LocListWriter->getDwarfVersion() >= 5 &&
                           LocListWriter->isSplitDwarf()))) {
      SimpleBinaryPatcher *Patcher =
          getBinaryDWODebugInfoPatcher(LocListWriter->getCUID());
      LocListWriter->finalize(0, *Patcher);
      continue;
    }
    LocWriter->finalize(SectionOffset, DebugInfoPatcher);
    std::unique_ptr<DebugBufferVector> CurrCULocationLists =
        LocWriter->getBuffer();
    *LocStream << *CurrCULocationLists;
    SectionOffset += CurrCULocationLists->size();
  }

  return LocBuffer;
}

namespace {

void getRangeAttrData(DWARFDie DIE, Optional<AttrInfo> &LowPCVal,
                      Optional<AttrInfo> &HighPCVal) {
  LowPCVal = findAttributeInfo(DIE, dwarf::DW_AT_low_pc);
  HighPCVal = findAttributeInfo(DIE, dwarf::DW_AT_high_pc);
  uint64_t LowPCOffset = LowPCVal->Offset;
  uint64_t HighPCOffset = HighPCVal->Offset;
  dwarf::Form LowPCForm = LowPCVal->V.getForm();
  dwarf::Form HighPCForm = HighPCVal->V.getForm();

  if (LowPCForm != dwarf::DW_FORM_addr &&
      LowPCForm != dwarf::DW_FORM_GNU_addr_index &&
      LowPCForm != dwarf::DW_FORM_addrx) {
    errs() << "BOLT-WARNING: unexpected low_pc form value. Cannot update DIE "
           << "at offset 0x" << Twine::utohexstr(DIE.getOffset()) << "\n";
    return;
  }
  if (HighPCForm != dwarf::DW_FORM_addr && HighPCForm != dwarf::DW_FORM_data8 &&
      HighPCForm != dwarf::DW_FORM_data4 &&
      HighPCForm != dwarf::DW_FORM_data2 &&
      HighPCForm != dwarf::DW_FORM_data1 &&
      HighPCForm != dwarf::DW_FORM_udata) {
    errs() << "BOLT-WARNING: unexpected high_pc form value. Cannot update DIE "
           << "at offset 0x" << Twine::utohexstr(DIE.getOffset()) << "\n";
    return;
  }
  if ((LowPCOffset == -1U || (LowPCOffset + 8 != HighPCOffset)) &&
      LowPCForm != dwarf::DW_FORM_GNU_addr_index &&
      LowPCForm != dwarf::DW_FORM_addrx) {
    errs() << "BOLT-WARNING: high_pc expected immediately after low_pc. "
           << "Cannot update DIE at offset 0x"
           << Twine::utohexstr(DIE.getOffset()) << '\n';
    return;
  }
}

} // namespace

void DWARFRewriter::convertToRangesPatchAbbrev(
    const DWARFUnit &Unit, const DWARFAbbreviationDeclaration *Abbrev,
    DebugAbbrevWriter &AbbrevWriter, Optional<uint64_t> RangesBase) {

  dwarf::Attribute RangeBaseAttribute = dwarf::DW_AT_GNU_ranges_base;
  dwarf::Form RangesForm = dwarf::DW_FORM_sec_offset;

  if (Unit.getVersion() >= 5) {
    RangeBaseAttribute = dwarf::DW_AT_rnglists_base;
    RangesForm = dwarf::DW_FORM_rnglistx;
  }
  // If we hit this point it means we converted subprogram DIEs from
  // low_pc/high_pc into ranges. The CU originally didn't have DW_AT_*_base, so
  // we are adding it here.
  if (RangesBase)
    AbbrevWriter.addAttribute(Unit, Abbrev, RangeBaseAttribute,
                              dwarf::DW_FORM_sec_offset);

  // Converting DW_AT_high_pc into DW_AT_ranges.
  // For DWARF4 it's DW_FORM_sec_offset.
  // For DWARF5 it can be either DW_FORM_sec_offset or DW_FORM_rnglistx.
  // For consistency for DWARF5 we always use DW_FORM_rnglistx.
  AbbrevWriter.addAttributePatch(Unit, Abbrev, dwarf::DW_AT_high_pc,
                                 dwarf::DW_AT_ranges, RangesForm);
}

void DWARFRewriter::convertToRangesPatchDebugInfo(
    DWARFDie DIE, uint64_t RangesSectionOffset,
    SimpleBinaryPatcher &DebugInfoPatcher, Optional<uint64_t> RangesBase) {
  Optional<AttrInfo> LowPCVal = None;
  Optional<AttrInfo> HighPCVal = None;
  getRangeAttrData(DIE, LowPCVal, HighPCVal);
  uint64_t LowPCOffset = LowPCVal->Offset;
  uint64_t HighPCOffset = HighPCVal->Offset;

  std::lock_guard<std::mutex> Lock(DebugInfoPatcherMutex);
  uint32_t BaseOffset = 0;
  dwarf::Form LowForm = LowPCVal->V.getForm();

  // In DWARF4 for DW_AT_low_pc in binary DW_FORM_addr is used. In the DWO
  // section DW_FORM_GNU_addr_index is used. So for if we are converting
  // DW_AT_low_pc/DW_AT_high_pc and see DW_FORM_GNU_addr_index. We are
  // converting in DWO section, and DW_AT_ranges [DW_FORM_sec_offset] is
  // relative to DW_AT_GNU_ranges_base.
  if (LowForm == dwarf::DW_FORM_GNU_addr_index) {
    // Use ULEB128 for the value.
    DebugInfoPatcher.addUDataPatch(LowPCOffset, 0, LowPCVal->Size);
    // Ranges are relative to DW_AT_GNU_ranges_base.
    BaseOffset = DebugInfoPatcher.getRangeBase();
  } else {
    // In DWARF 5 we can have DW_AT_low_pc either as DW_FORM_addr, or
    // DW_FORM_addrx. Former is when DW_AT_rnglists_base is present. Latter is
    // when it's absent.
    if (LowForm == dwarf::DW_FORM_addrx) {
      uint32_t Index =
          AddrWriter->getIndexFromAddress(0, DIE.getDwarfUnit()->getOffset());
      DebugInfoPatcher.addUDataPatch(LowPCOffset, Index, LowPCVal->Size);
    } else
      DebugInfoPatcher.addLE64Patch(LowPCOffset, 0);

    // Original CU didn't have DW_AT_*_base. We converted it's children (or
    // dwo), so need to insert it into CU.
    if (RangesBase)
      reinterpret_cast<DebugInfoBinaryPatcher &>(DebugInfoPatcher)
          .insertNewEntry(DIE, *RangesBase);
  }

  // HighPC was conveted into DW_AT_ranges.
  // For DWARF5 we only access ranges throught index.
  if (DIE.getDwarfUnit()->getVersion() >= 5)
    DebugInfoPatcher.addUDataPatch(HighPCOffset, RangesSectionOffset,
                                   HighPCVal->Size);
  else
    DebugInfoPatcher.addLE32Patch(
        HighPCOffset, RangesSectionOffset - BaseOffset, HighPCVal->Size);
}
