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
#include "bolt/Core/DIEBuilder.h"
#include "bolt/Core/DebugData.h"
#include "bolt/Core/DynoStats.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/DWARFLinker/DWARFStreamer.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFTypeUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
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
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt"

static void printDie(const DWARFDie &DIE) {
  DIDumpOptions DumpOpts;
  DumpOpts.ShowForm = true;
  DumpOpts.Verbose = true;
  DumpOpts.ChildRecurseDepth = 0;
  DumpOpts.ShowChildren = false;
  DIE.dump(dbgs(), 0, DumpOpts);
}

/// Lazily parse DWARF DIE and print it out.
LLVM_ATTRIBUTE_UNUSED
static void printDie(DWARFUnit &DU, uint64_t DIEOffset) {
  uint64_t OriginalOffsets = DIEOffset;
  uint64_t NextCUOffset = DU.getNextUnitOffset();
  DWARFDataExtractor DebugInfoData = DU.getDebugInfoExtractor();
  DWARFDebugInfoEntry DIEEntry;
  if (DIEEntry.extractFast(DU, &DIEOffset, DebugInfoData, NextCUOffset, 0)) {
    if (const DWARFAbbreviationDeclaration *AbbrDecl =
            DIEEntry.getAbbreviationDeclarationPtr()) {
      DWARFDie DDie(&DU, &DIEEntry);
      printDie(DDie);
    } else {
      dbgs() << "Failed to extract abbreviation for"
             << Twine::utohexstr(OriginalOffsets) << "\n";
    }
  } else {
    dbgs() << "Failed to extract DIE for " << Twine::utohexstr(OriginalOffsets)
           << " \n";
  }
}

using namespace bolt;

/// Take a set of DWARF address ranges corresponding to the input binary and
/// translate them to a set of address ranges in the output binary.
static DebugAddressRangesVector
translateInputToOutputRanges(const BinaryFunction &BF,
                             const DWARFAddressRangesVector &InputRanges) {
  DebugAddressRangesVector OutputRanges;

  // If the function hasn't changed return the same ranges.
  if (!BF.isEmitted()) {
    OutputRanges.resize(InputRanges.size());
    llvm::transform(InputRanges, OutputRanges.begin(),
                    [](const DWARFAddressRange &Range) {
                      return DebugAddressRange(Range.LowPC, Range.HighPC);
                    });
    return OutputRanges;
  }

  for (const DWARFAddressRange &Range : InputRanges)
    llvm::append_range(OutputRanges, BF.translateInputToOutputRange(
                                         {Range.LowPC, Range.HighPC}));

  // Post-processing pass to sort and merge ranges.
  llvm::sort(OutputRanges);
  DebugAddressRangesVector MergedRanges;
  uint64_t PrevHighPC = 0;
  for (const DebugAddressRange &Range : OutputRanges) {
    if (Range.LowPC <= PrevHighPC) {
      MergedRanges.back().HighPC =
          std::max(MergedRanges.back().HighPC, Range.HighPC);
    } else {
      MergedRanges.emplace_back(Range.LowPC, Range.HighPC);
    }
    PrevHighPC = MergedRanges.back().HighPC;
  }

  return MergedRanges;
}

/// Similar to translateInputToOutputRanges() but operates on location lists.
static DebugLocationsVector
translateInputToOutputLocationList(const BinaryFunction &BF,
                                   const DebugLocationsVector &InputLL) {
  DebugLocationsVector OutputLL;

  // If the function hasn't changed - there's nothing to update.
  if (!BF.isEmitted())
    return InputLL;

  for (const DebugLocationEntry &Entry : InputLL) {
    DebugAddressRangesVector OutRanges =
        BF.translateInputToOutputRange({Entry.LowPC, Entry.HighPC});
    if (!OutRanges.empty() && !OutputLL.empty()) {
      if (OutRanges.front().LowPC == OutputLL.back().HighPC &&
          Entry.Expr == OutputLL.back().Expr) {
        OutputLL.back().HighPC =
            std::max(OutputLL.back().HighPC, OutRanges.front().HighPC);
        OutRanges.erase(OutRanges.begin());
      }
    }
    llvm::transform(OutRanges, std::back_inserter(OutputLL),
                    [&Entry](const DebugAddressRange &R) {
                      return DebugLocationEntry{R.LowPC, R.HighPC, Entry.Expr};
                    });
  }

  // Sort and merge adjacent entries with identical locations.
  llvm::stable_sort(
      OutputLL, [](const DebugLocationEntry &A, const DebugLocationEntry &B) {
        return A.LowPC < B.LowPC;
      });
  DebugLocationsVector MergedLL;
  uint64_t PrevHighPC = 0;
  const SmallVectorImpl<uint8_t> *PrevExpr = nullptr;
  for (const DebugLocationEntry &Entry : OutputLL) {
    if (Entry.LowPC <= PrevHighPC && *PrevExpr == Entry.Expr) {
      MergedLL.back().HighPC = std::max(Entry.HighPC, MergedLL.back().HighPC);
    } else {
      const uint64_t Begin = std::max(Entry.LowPC, PrevHighPC);
      const uint64_t End = std::max(Begin, Entry.HighPC);
      MergedLL.emplace_back(DebugLocationEntry{Begin, End, Entry.Expr});
    }
    PrevHighPC = MergedLL.back().HighPC;
    PrevExpr = &MergedLL.back().Expr;
  }

  return MergedLL;
}

namespace llvm {
namespace bolt {
/// Emits debug information into .debug_info or .debug_types section.
class DIEStreamer : public DwarfStreamer {
  DIEBuilder *DIEBldr;
  DWARFRewriter &Rewriter;

private:
  /// Emit the compilation unit header for \p Unit in the debug_info
  /// section.
  ///
  /// A Dwarf 4 section header is encoded as:
  ///  uint32_t   Unit length (omitting this field)
  ///  uint16_t   Version
  ///  uint32_t   Abbreviation table offset
  ///  uint8_t    Address size
  /// Leading to a total of 11 bytes.
  ///
  /// A Dwarf 5 section header is encoded as:
  ///  uint32_t   Unit length (omitting this field)
  ///  uint16_t   Version
  ///  uint8_t    Unit type
  ///  uint8_t    Address size
  ///  uint32_t   Abbreviation table offset
  /// Leading to a total of 12 bytes.
  void emitCompileUnitHeader(DWARFUnit &Unit, DIE &UnitDIE,
                             unsigned DwarfVersion) {

    AsmPrinter &Asm = getAsmPrinter();
    switchToDebugInfoSection(DwarfVersion);

    emitCommonHeader(Unit, UnitDIE, DwarfVersion);

    if (DwarfVersion >= 5 &&
        Unit.getUnitType() != dwarf::UnitType::DW_UT_compile) {
      std::optional<uint64_t> DWOId = Unit.getDWOId();
      assert(DWOId &&
             "DWOId does not exist and this is not a DW_UT_compile Unit");
      Asm.emitInt64(*DWOId);
    }
  }

  void emitCommonHeader(DWARFUnit &Unit, DIE &UnitDIE, uint16_t Version) {
    dwarf::UnitType UT = dwarf::UnitType(Unit.getUnitType());
    llvm::AsmPrinter &Asm = getAsmPrinter();

    // Emit size of content not including length itself
    Asm.emitInt32(Unit.getHeaderSize() + UnitDIE.getSize() - 4);
    Asm.emitInt16(Version);

    // DWARF v5 reorders the address size and adds a unit type.
    if (Version >= 5) {
      Asm.emitInt8(UT);
      Asm.emitInt8(Asm.MAI->getCodePointerSize());
    }

    Asm.emitInt32(0);
    if (Version <= 4) {
      Asm.emitInt8(Asm.MAI->getCodePointerSize());
    }
  }

  void emitTypeUnitHeader(DWARFUnit &Unit, DIE &UnitDIE,
                          unsigned DwarfVersion) {
    AsmPrinter &Asm = getAsmPrinter();
    const uint64_t TypeSignature = cast<DWARFTypeUnit>(Unit).getTypeHash();
    DIE *TypeDIE = DIEBldr->getTypeDIE(Unit);
    const DIEBuilder::DWARFUnitInfo &UI = DIEBldr->getUnitInfoByDwarfUnit(Unit);
    Rewriter.addGDBTypeUnitEntry(
        {UI.UnitOffset, TypeSignature, TypeDIE->getOffset()});
    if (Unit.getVersion() < 5) {
      // Switch the section to .debug_types section.
      std::unique_ptr<MCStreamer> &MS = Asm.OutStreamer;
      llvm::MCContext &MC = Asm.OutContext;
      const llvm::MCObjectFileInfo *MOFI = MC.getObjectFileInfo();

      MS->switchSection(MOFI->getDwarfTypesSection(0));
      MC.setDwarfVersion(DwarfVersion);
    } else
      switchToDebugInfoSection(DwarfVersion);

    emitCommonHeader(Unit, UnitDIE, DwarfVersion);
    Asm.OutStreamer->emitIntValue(TypeSignature, sizeof(TypeSignature));
    Asm.emitDwarfLengthOrOffset(TypeDIE ? TypeDIE->getOffset() : 0);
  }

  void emitUnitHeader(DWARFUnit &Unit, DIE &UnitDIE) {
    if (Unit.isTypeUnit())
      emitTypeUnitHeader(Unit, UnitDIE, Unit.getVersion());
    else
      emitCompileUnitHeader(Unit, UnitDIE, Unit.getVersion());
  }

  void emitDIE(DIE &Die) override {
    AsmPrinter &Asm = getAsmPrinter();
    Asm.emitDwarfDIE(Die);
  }

public:
  DIEStreamer(DIEBuilder *DIEBldr, DWARFRewriter &Rewriter,
              DWARFLinker::OutputFileType OutFileType,
              raw_pwrite_stream &OutFile,
              std::function<StringRef(StringRef Input)> Translator,
              DWARFLinker::messageHandler Warning)
      : DwarfStreamer(OutFileType, OutFile, Translator, Warning),
        DIEBldr(DIEBldr), Rewriter(Rewriter){};

  using DwarfStreamer::emitCompileUnitHeader;

  void emitUnit(DWARFUnit &Unit, DIE &UnitDIE) {
    emitUnitHeader(Unit, UnitDIE);
    emitDIE(UnitDIE);
  }
};

/// Finds attributes FormValue and Offset.
///
/// \param DIE die to look up in.
/// \param Attrs finds the first attribute that matches and extracts it.
/// \return an optional AttrInfo with DWARFFormValue and Offset.
std::optional<AttrInfo> findAttributeInfo(const DWARFDie DIE,
                                          std::vector<dwarf::Attribute> Attrs) {
  for (dwarf::Attribute &Attr : Attrs)
    if (std::optional<AttrInfo> Info = findAttributeInfo(DIE, Attr))
      return Info;
  return std::nullopt;
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

static cl::opt<bool> KeepARanges(
    "keep-aranges",
    cl::desc(
        "keep or generate .debug_aranges section if .gdb_index is written"),
    cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool>
DeterministicDebugInfo("deterministic-debuginfo",
  cl::desc("disables parallel execution of tasks that may produce "
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

static cl::opt<unsigned> BatchSize(
    "cu-processing-batch-size",
    cl::desc(
        "Specifies the size of batches for processing CUs. Higher number has "
        "better performance, but more memory usage. Default value is 1."),
    cl::Hidden, cl::init(1), cl::cat(BoltCategory));

static cl::opt<bool> AlwaysConvertToRanges(
    "always-convert-to-ranges",
    cl::desc("This option is for testing purposes only. It forces BOLT to "
             "convert low_pc/high_pc to ranges always."),
    cl::ReallyHidden, cl::init(false), cl::cat(BoltCategory));
} // namespace opts

static bool getLowAndHighPC(const DIE &Die, const DWARFUnit &DU,
                            uint64_t &LowPC, uint64_t &HighPC,
                            uint64_t &SectionIndex) {
  DIEValue DvalLowPc = Die.findAttribute(dwarf::DW_AT_low_pc);
  DIEValue DvalHighPc = Die.findAttribute(dwarf::DW_AT_high_pc);
  if (!DvalLowPc || !DvalHighPc)
    return false;

  dwarf::Form Form = DvalLowPc.getForm();
  bool AddrOffset = Form == dwarf::DW_FORM_LLVM_addrx_offset;
  uint64_t LowPcValue = DvalLowPc.getDIEInteger().getValue();
  if (Form == dwarf::DW_FORM_GNU_addr_index || Form == dwarf::DW_FORM_addrx ||
      AddrOffset) {

    uint32_t Index = AddrOffset ? (LowPcValue >> 32) : LowPcValue;
    std::optional<object::SectionedAddress> SA =
        DU.getAddrOffsetSectionItem(Index);
    if (!SA)
      return false;
    if (AddrOffset)
      SA->Address += (LowPcValue & 0xffffffff);

    LowPC = SA->Address;
    SectionIndex = SA->SectionIndex;
  } else {
    LowPC = LowPcValue;
    SectionIndex = 0;
  }
  if (DvalHighPc.getForm() == dwarf::DW_FORM_addr)
    HighPC = DvalHighPc.getDIEInteger().getValue();
  else
    HighPC = LowPC + DvalHighPc.getDIEInteger().getValue();

  return true;
}

static Expected<llvm::DWARFAddressRangesVector>
getDIEAddressRanges(const DIE &Die, DWARFUnit &DU) {
  uint64_t LowPC, HighPC, Index;
  if (getLowAndHighPC(Die, DU, LowPC, HighPC, Index))
    return DWARFAddressRangesVector{{LowPC, HighPC, Index}};
  if (DIEValue Dval = Die.findAttribute(dwarf::DW_AT_ranges)) {
    if (Dval.getForm() == dwarf::DW_FORM_rnglistx)
      return DU.findRnglistFromIndex(Dval.getDIEInteger().getValue());

    return DU.findRnglistFromOffset(Dval.getDIEInteger().getValue());
  }

  return DWARFAddressRangesVector();
}

static std::optional<uint64_t> getAsAddress(const DWARFUnit &DU,
                                            const DIEValue &AttrVal) {
  DWARFFormValue::ValueType Value(AttrVal.getDIEInteger().getValue());
  if (std::optional<object::SectionedAddress> SA =
          DWARFFormValue::getAsSectionedAddress(Value, AttrVal.getForm(), &DU))
    return SA->Address;
  return std::nullopt;
}

/// Returns DWO Name to be used. Handles case where user specifies output DWO
/// directory, and there are duplicate names. Assumes DWO ID is unique.
static std::string
getDWOName(llvm::DWARFUnit &CU,
           std::unordered_map<std::string, uint32_t> &NameToIndexMap) {
  std::optional<uint64_t> DWOId = CU.getDWOId();
  assert(DWOId && "DWO ID not found.");
  (void)DWOId;

  std::string DWOName = dwarf::toString(
      CU.getUnitDIE().find({dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}),
      "");
  assert(!DWOName.empty() &&
         "DW_AT_dwo_name/DW_AT_GNU_dwo_name does not exists.");
  if (!opts::DwarfOutputPath.empty()) {
    DWOName = std::string(sys::path::filename(DWOName));
    auto Iter = NameToIndexMap.find(DWOName);
    if (Iter == NameToIndexMap.end())
      Iter = NameToIndexMap.insert({DWOName, 0}).first;
    DWOName.append(std::to_string(Iter->second));
    ++Iter->second;
  }
  DWOName.append(".dwo");
  return DWOName;
}

static std::unique_ptr<DIEStreamer>
createDIEStreamer(const Triple &TheTriple, raw_pwrite_stream &OutFile,
                  StringRef Swift5ReflectionSegmentName, DIEBuilder &DIEBldr,
                  DWARFRewriter &Rewriter) {

  std::unique_ptr<DIEStreamer> Streamer = std::make_unique<DIEStreamer>(
      &DIEBldr, Rewriter, llvm::DWARFLinker::OutputFileType::Object, OutFile,
      [](StringRef Input) -> StringRef { return Input; },
      [&](const Twine &Warning, StringRef Context, const DWARFDie *) {});
  Error Err = Streamer->init(TheTriple, Swift5ReflectionSegmentName);
  if (Err)
    errs()
        << "BOLT-WARNING: [internal-dwarf-error]: Could not init DIEStreamer!"
        << toString(std::move(Err)) << "\n";
  return Streamer;
}

static DWARFRewriter::UnitMeta
emitUnit(DIEBuilder &DIEBldr, DIEStreamer &Streamer, DWARFUnit &Unit) {
  DIE *UnitDIE = DIEBldr.getUnitDIEbyUnit(Unit);
  const DIEBuilder::DWARFUnitInfo &U = DIEBldr.getUnitInfoByDwarfUnit(Unit);
  Streamer.emitUnit(Unit, *UnitDIE);
  uint64_t TypeHash = 0;
  if (DWARFTypeUnit *DTU = dyn_cast_or_null<DWARFTypeUnit>(&Unit))
    TypeHash = DTU->getTypeHash();
  return {U.UnitOffset, U.UnitLength, TypeHash};
}

static void emitDWOBuilder(const std::string &DWOName,
                           DIEBuilder &DWODIEBuilder, DWARFRewriter &Rewriter,
                           DWARFUnit &SplitCU, DWARFUnit &CU,
                           DWARFRewriter::DWPState &State,
                           DebugLocWriter &LocWriter) {
  // Populate debug_info and debug_abbrev for current dwo into StringRef.
  DWODIEBuilder.generateAbbrevs();
  DWODIEBuilder.finish();

  SmallVector<char, 20> OutBuffer;
  std::shared_ptr<raw_svector_ostream> ObjOS =
      std::make_shared<raw_svector_ostream>(OutBuffer);
  const object::ObjectFile *File = SplitCU.getContext().getDWARFObj().getFile();
  auto TheTriple = std::make_unique<Triple>(File->makeTriple());
  std::unique_ptr<DIEStreamer> Streamer = createDIEStreamer(
      *TheTriple, *ObjOS, "DwoStreamerInitAug2", DWODIEBuilder, Rewriter);
  DWARFRewriter::UnitMetaVectorType TUMetaVector;
  DWARFRewriter::UnitMeta CUMI = {0, 0, 0};
  if (SplitCU.getContext().getMaxDWOVersion() >= 5) {
    for (std::unique_ptr<llvm::DWARFUnit> &CU :
         SplitCU.getContext().dwo_info_section_units()) {
      if (!CU->isTypeUnit())
        continue;
      DWARFRewriter::UnitMeta MI =
          emitUnit(DWODIEBuilder, *Streamer, *CU.get());
      TUMetaVector.emplace_back(MI);
    }
    CUMI = emitUnit(DWODIEBuilder, *Streamer, SplitCU);
  } else {
    for (std::unique_ptr<llvm::DWARFUnit> &CU :
         SplitCU.getContext().dwo_compile_units())
      emitUnit(DWODIEBuilder, *Streamer, *CU.get());

    // emit debug_types sections for dwarf4
    for (DWARFUnit *CU : DWODIEBuilder.getDWARF4TUVector()) {
      DWARFRewriter::UnitMeta MI = emitUnit(DWODIEBuilder, *Streamer, *CU);
      TUMetaVector.emplace_back(MI);
    }
  }

  Streamer->emitAbbrevs(DWODIEBuilder.getAbbrevs(),
                        SplitCU.getContext().getMaxVersion());
  Streamer->finish();

  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(ObjOS->str(), "in-memory object file", false);
  std::unique_ptr<object::ObjectFile> Obj = cantFail(
      object::ObjectFile::createObjectFile(ObjectMemBuffer->getMemBufferRef()),
      "error creating in-memory object");

  DWARFRewriter::OverriddenSectionsMap OverriddenSections;
  for (const SectionRef &Secs : Obj->sections()) {
    StringRef Contents = cantFail(Secs.getContents());
    StringRef Name = cantFail(Secs.getName());
    DWARFSectionKind Kind =
        StringSwitch<DWARFSectionKind>(Name)
            .Case(".debug_abbrev", DWARFSectionKind::DW_SECT_ABBREV)
            .Case(".debug_info", DWARFSectionKind::DW_SECT_INFO)
            .Case(".debug_types", DWARFSectionKind::DW_SECT_EXT_TYPES)
            .Default(DWARFSectionKind::DW_SECT_EXT_unknown);
    if (Kind == DWARFSectionKind::DW_SECT_EXT_unknown)
      continue;
    OverriddenSections[Kind] = Contents;
  }
  if (opts::WriteDWP)
    Rewriter.updateDWP(CU, OverriddenSections, CUMI, TUMetaVector, State,
                       LocWriter);
  else
    Rewriter.writeDWOFiles(CU, OverriddenSections, DWOName, LocWriter);
}

void DWARFRewriter::addStringHelper(DIEBuilder &DIEBldr, DIE &Die,
                                    const DWARFUnit &Unit,
                                    DIEValue &DIEAttrInfo, StringRef Str) {
  uint32_t NewOffset = StrWriter->addString(Str);
  if (Unit.getVersion() >= 5) {
    StrOffstsWriter->updateAddressMap(DIEAttrInfo.getDIEInteger().getValue(),
                                      NewOffset);
    return;
  }
  DIEBldr.replaceValue(&Die, DIEAttrInfo.getAttribute(), DIEAttrInfo.getForm(),
                       DIEInteger(NewOffset));
}

using DWARFUnitVec = std::vector<DWARFUnit *>;
using CUPartitionVector = std::vector<DWARFUnitVec>;
/// Partitions CUs in to buckets. Bucket size is controlled by
/// cu-processing-batch-size. All the CUs that have cross CU reference reference
/// as a source are put in to the same initial bucket.
static CUPartitionVector partitionCUs(DWARFContext &DwCtx) {
  CUPartitionVector Vec(2);
  unsigned Counter = 0;
  const DWARFDebugAbbrev *Abbr = DwCtx.getDebugAbbrev();
  for (std::unique_ptr<DWARFUnit> &CU : DwCtx.compile_units()) {
    Expected<const DWARFAbbreviationDeclarationSet *> AbbrDeclSet =
        Abbr->getAbbreviationDeclarationSet(CU->getAbbreviationsOffset());
    if (!AbbrDeclSet) {
      consumeError(AbbrDeclSet.takeError());
      return Vec;
    }
    bool CrossCURefFound = false;
    for (const DWARFAbbreviationDeclaration &Decl : *AbbrDeclSet.get()) {
      for (const DWARFAbbreviationDeclaration::AttributeSpec &Attr :
           Decl.attributes()) {
        if (Attr.Form == dwarf::DW_FORM_ref_addr) {
          CrossCURefFound = true;
          break;
        }
      }
      if (CrossCURefFound)
        break;
    }
    if (CrossCURefFound) {
      Vec[0].push_back(CU.get());
    } else {
      ++Counter;
      Vec.back().push_back(CU.get());
    }
    if (Counter % opts::BatchSize == 0 && !Vec.back().empty())
      Vec.push_back({});
  }
  return Vec;
}

void DWARFRewriter::updateDebugInfo() {
  ErrorOr<BinarySection &> DebugInfo = BC.getUniqueSectionByName(".debug_info");
  if (!DebugInfo)
    return;

  ARangesSectionWriter = std::make_unique<DebugARangesSectionWriter>();
  StrWriter = std::make_unique<DebugStrWriter>(BC);

  StrOffstsWriter = std::make_unique<DebugStrOffsetsWriter>();

  if (!opts::DeterministicDebugInfo) {
    opts::DeterministicDebugInfo = true;
    errs() << "BOLT-WARNING: --deterministic-debuginfo is being deprecated\n";
  }

  if (BC.isDWARF5Used()) {
    AddrWriter = std::make_unique<DebugAddrWriterDwarf5>(&BC);
    RangeListsSectionWriter = std::make_unique<DebugRangeListsSectionWriter>();
    DebugRangeListsSectionWriter::setAddressWriter(AddrWriter.get());
  } else {
    AddrWriter = std::make_unique<DebugAddrWriter>(&BC);
  }

  if (BC.isDWARFLegacyUsed())
    LegacyRangesSectionWriter = std::make_unique<DebugRangesSectionWriter>();

  DebugLoclistWriter::setAddressWriter(AddrWriter.get());

  uint32_t CUIndex = 0;
  std::mutex AccessMutex;
  // Needs to be invoked in the same order as CUs are processed.
  auto createRangeLocList = [&](DWARFUnit &CU) -> DebugLocWriter * {
    std::lock_guard<std::mutex> Lock(AccessMutex);
    const uint16_t DwarfVersion = CU.getVersion();
    if (DwarfVersion >= 5) {
      LocListWritersByCU[CUIndex] =
          std::make_unique<DebugLoclistWriter>(CU, DwarfVersion, false);

      if (std::optional<uint64_t> DWOId = CU.getDWOId()) {
        assert(RangeListsWritersByCU.count(*DWOId) == 0 &&
               "RangeLists writer for DWO unit already exists.");
        auto RangeListsSectionWriter =
            std::make_unique<DebugRangeListsSectionWriter>();
        RangeListsSectionWriter->initSection(CU);
        RangeListsWritersByCU[*DWOId] = std::move(RangeListsSectionWriter);
      }

    } else {
      LocListWritersByCU[CUIndex] = std::make_unique<DebugLocWriter>();
    }
    return LocListWritersByCU[CUIndex++].get();
  };

  // Unordered maps to handle name collision if output DWO directory is
  // specified.
  std::unordered_map<std::string, uint32_t> NameToIndexMap;

  auto updateDWONameCompDir = [&](DWARFUnit &Unit, DIEBuilder &DIEBldr,
                                  DIE &UnitDIE) -> std::string {
    DIEValue DWONameAttrInfo = UnitDIE.findAttribute(dwarf::DW_AT_dwo_name);
    if (!DWONameAttrInfo)
      DWONameAttrInfo = UnitDIE.findAttribute(dwarf::DW_AT_GNU_dwo_name);
    assert(DWONameAttrInfo && "DW_AT_dwo_name is not in Skeleton CU.");
    std::string ObjectName;

    {
      std::lock_guard<std::mutex> Lock(AccessMutex);
      ObjectName = getDWOName(Unit, NameToIndexMap);
    }
    addStringHelper(DIEBldr, UnitDIE, Unit, DWONameAttrInfo,
                    ObjectName.c_str());

    DIEValue CompDirAttrInfo = UnitDIE.findAttribute(dwarf::DW_AT_comp_dir);
    assert(CompDirAttrInfo && "DW_AT_comp_dir is not in Skeleton CU.");

    if (!opts::DwarfOutputPath.empty()) {
      if (!sys::fs::exists(opts::DwarfOutputPath))
        sys::fs::create_directory(opts::DwarfOutputPath);
      addStringHelper(DIEBldr, UnitDIE, Unit, CompDirAttrInfo,
                      opts::DwarfOutputPath.c_str());
    }
    return ObjectName;
  };

  DWPState State;
  if (opts::WriteDWP)
    initDWPState(State);
  auto processUnitDIE = [&](DWARFUnit *Unit, DIEBuilder *DIEBlder) {
    // Check if the unit is a skeleton and we need special updates for it and
    // its matching split/DWO CU.
    std::optional<DWARFUnit *> SplitCU;
    std::optional<uint64_t> RangesBase;
    std::optional<uint64_t> DWOId = Unit->getDWOId();
    if (DWOId)
      SplitCU = BC.getDWOCU(*DWOId);
    DebugLocWriter *DebugLocWriter = createRangeLocList(*Unit);
    DebugRangesSectionWriter *RangesSectionWriter =
        Unit->getVersion() >= 5 ? RangeListsSectionWriter.get()
                                : LegacyRangesSectionWriter.get();
    // Skipping CUs that failed to load.
    if (SplitCU) {
      DIEBuilder DWODIEBuilder(&(*SplitCU)->getContext(), true);
      DWODIEBuilder.buildDWOUnit(**SplitCU);
      std::string DWOName = updateDWONameCompDir(
          *Unit, *DIEBlder, *DIEBlder->getUnitDIEbyUnit(*Unit));

      DebugLoclistWriter DebugLocDWoWriter(*Unit, Unit->getVersion(), true);
      DebugRangesSectionWriter *TempRangesSectionWriter = RangesSectionWriter;
      if (Unit->getVersion() >= 5) {
        TempRangesSectionWriter = RangeListsWritersByCU[*DWOId].get();
      } else {
        RangesBase = RangesSectionWriter->getSectionOffset();
        // For DWARF5 there is now .debug_rnglists.dwo, so don't need to
        // update rnglists base.
        if (RangesBase) {
          DwoRangesBase[*DWOId] = *RangesBase;
          setDwoRangesBase(*DWOId, *RangesBase);
        }
      }

      updateUnitDebugInfo(*(*SplitCU), DWODIEBuilder, DebugLocDWoWriter,
                          *TempRangesSectionWriter);
      DebugLocDWoWriter.finalize(DWODIEBuilder,
                                 *DWODIEBuilder.getUnitDIEbyUnit(**SplitCU));
      if (Unit->getVersion() >= 5)
        TempRangesSectionWriter->finalizeSection();

      emitDWOBuilder(DWOName, DWODIEBuilder, *this, **SplitCU, *Unit, State,
                     DebugLocDWoWriter);
    }

    if (Unit->getVersion() >= 5) {
      RangesBase = RangesSectionWriter->getSectionOffset() +
                   getDWARF5RngListLocListHeaderSize();
      RangesSectionWriter->initSection(*Unit);
      StrOffstsWriter->finalizeSection(*Unit, *DIEBlder);
    }

    updateUnitDebugInfo(*Unit, *DIEBlder, *DebugLocWriter, *RangesSectionWriter,
                        RangesBase);
    DebugLocWriter->finalize(*DIEBlder, *DIEBlder->getUnitDIEbyUnit(*Unit));
    if (Unit->getVersion() >= 5)
      RangesSectionWriter->finalizeSection();
    AddrWriter->update(*DIEBlder, *Unit);
  };

  DIEBuilder DIEBlder(BC.DwCtx.get());
  DIEBlder.buildTypeUnits(StrOffstsWriter.get());
  SmallVector<char, 20> OutBuffer;
  std::unique_ptr<raw_svector_ostream> ObjOS =
      std::make_unique<raw_svector_ostream>(OutBuffer);
  const object::ObjectFile *File = BC.DwCtx->getDWARFObj().getFile();
  auto TheTriple = std::make_unique<Triple>(File->makeTriple());
  std::unique_ptr<DIEStreamer> Streamer =
      createDIEStreamer(*TheTriple, *ObjOS, "TypeStreamer", DIEBlder, *this);
  CUOffsetMap OffsetMap = finalizeTypeSections(DIEBlder, *Streamer);

  const bool SingleThreadedMode =
      opts::NoThreads || opts::DeterministicDebugInfo;
  if (!SingleThreadedMode)
    DIEBlder.buildCompileUnits();
  if (SingleThreadedMode) {
    CUPartitionVector PartVec = partitionCUs(*BC.DwCtx);
    for (std::vector<DWARFUnit *> &Vec : PartVec) {
      DIEBlder.buildCompileUnits(Vec);
      for (DWARFUnit *CU : DIEBlder.getProcessedCUs())
        processUnitDIE(CU, &DIEBlder);
      finalizeCompileUnits(DIEBlder, *Streamer, OffsetMap,
                           DIEBlder.getProcessedCUs());
    }
  } else {
    // Update unit debug info in parallel
    ThreadPool &ThreadPool = ParallelUtilities::getThreadPool();
    for (std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units())
      ThreadPool.async(processUnitDIE, CU.get(), &DIEBlder);
    ThreadPool.wait();
  }

  if (opts::WriteDWP)
    finalizeDWP(State);

  finalizeDebugSections(DIEBlder, *Streamer, *ObjOS, OffsetMap);
  updateGdbIndexSection(OffsetMap, CUIndex);
}

void DWARFRewriter::updateUnitDebugInfo(
    DWARFUnit &Unit, DIEBuilder &DIEBldr, DebugLocWriter &DebugLocWriter,
    DebugRangesSectionWriter &RangesSectionWriter,
    std::optional<uint64_t> RangesBase) {
  // Cache debug ranges so that the offset for identical ranges could be reused.
  std::map<DebugAddressRangesVector, uint64_t> CachedRanges;

  uint64_t DIEOffset = Unit.getOffset() + Unit.getHeaderSize();
  uint64_t NextCUOffset = Unit.getNextUnitOffset();
  const std::vector<std::unique_ptr<DIEBuilder::DIEInfo>> &DIs =
      DIEBldr.getDIEsByUnit(Unit);

  // Either updates or normalizes DW_AT_range to DW_AT_low_pc and DW_AT_high_pc.
  auto updateLowPCHighPC = [&](DIE *Die, const DIEValue &LowPCVal,
                               const DIEValue &HighPCVal, uint64_t LowPC,
                               const uint64_t HighPC) {
    dwarf::Attribute AttrLowPC = dwarf::DW_AT_low_pc;
    dwarf::Form FormLowPC = dwarf::DW_FORM_addr;
    dwarf::Attribute AttrHighPC = dwarf::DW_AT_high_pc;
    dwarf::Form FormHighPC = dwarf::DW_FORM_data4;
    const uint32_t Size = HighPC - LowPC;
    // Whatever was generated is not low_pc/high_pc, so will reset to
    // default for size 1.
    if (!LowPCVal || !HighPCVal) {
      if (Unit.getVersion() >= 5)
        FormLowPC = dwarf::DW_FORM_addrx;
      else if (Unit.isDWOUnit())
        FormLowPC = dwarf::DW_FORM_GNU_addr_index;
    } else {
      AttrLowPC = LowPCVal.getAttribute();
      FormLowPC = LowPCVal.getForm();
      AttrHighPC = HighPCVal.getAttribute();
      FormHighPC = HighPCVal.getForm();
    }

    if (FormLowPC == dwarf::DW_FORM_addrx ||
        FormLowPC == dwarf::DW_FORM_GNU_addr_index)
      LowPC = AddrWriter->getIndexFromAddress(LowPC, Unit);

    if (LowPCVal)
      DIEBldr.replaceValue(Die, AttrLowPC, FormLowPC, DIEInteger(LowPC));
    else
      DIEBldr.addValue(Die, AttrLowPC, FormLowPC, DIEInteger(LowPC));
    if (HighPCVal) {
      DIEBldr.replaceValue(Die, AttrHighPC, FormHighPC, DIEInteger(Size));
    } else {
      DIEBldr.deleteValue(Die, dwarf::DW_AT_ranges);
      DIEBldr.addValue(Die, AttrHighPC, FormHighPC, DIEInteger(Size));
    }
  };

  for (const std::unique_ptr<DIEBuilder::DIEInfo> &DI : DIs) {
    DIE *Die = DI->Die;
    switch (Die->getTag()) {
    case dwarf::DW_TAG_compile_unit:
    case dwarf::DW_TAG_skeleton_unit: {
      // For dwarf5 section 3.1.3
      // The following attributes are not part of a split full compilation unit
      // entry but instead are inherited (if present) from the corresponding
      // skeleton compilation unit: DW_AT_low_pc, DW_AT_high_pc, DW_AT_ranges,
      // DW_AT_stmt_list, DW_AT_comp_dir, DW_AT_str_offsets_base,
      // DW_AT_addr_base and DW_AT_rnglists_base.
      if (Unit.getVersion() == 5 && Unit.isDWOUnit())
        continue;
      auto ModuleRangesOrError = getDIEAddressRanges(*Die, Unit);
      if (!ModuleRangesOrError) {
        consumeError(ModuleRangesOrError.takeError());
        break;
      }
      DWARFAddressRangesVector &ModuleRanges = *ModuleRangesOrError;
      DebugAddressRangesVector OutputRanges =
          BC.translateModuleAddressRanges(ModuleRanges);
      DIEValue LowPCAttrInfo = Die->findAttribute(dwarf::DW_AT_low_pc);
      // For a case where LLD GCs only function used in the CU.
      // If CU doesn't have DW_AT_low_pc we are not going to convert,
      // so don't need to do anything.
      if (OutputRanges.empty() && !Unit.isDWOUnit() && LowPCAttrInfo)
        OutputRanges.push_back({0, 0});
      const uint64_t RangesSectionOffset =
          RangesSectionWriter.addRanges(OutputRanges);
      if (!Unit.isDWOUnit())
        ARangesSectionWriter->addCURanges(Unit.getOffset(),
                                          std::move(OutputRanges));
      updateDWARFObjectAddressRanges(Unit, DIEBldr, *Die, RangesSectionOffset,
                                     RangesBase);
      DIEValue StmtListAttrVal = Die->findAttribute(dwarf::DW_AT_stmt_list);
      if (LineTablePatchMap.count(&Unit))
        DIEBldr.replaceValue(Die, dwarf::DW_AT_stmt_list,
                             StmtListAttrVal.getForm(),
                             DIEInteger(LineTablePatchMap[&Unit]));
      break;
    }

    case dwarf::DW_TAG_subprogram: {
      // Get function address either from ranges or [LowPC, HighPC) pair.
      uint64_t Address = UINT64_MAX;
      uint64_t SectionIndex, HighPC;
      DebugAddressRangesVector FunctionRanges;
      if (!getLowAndHighPC(*Die, Unit, Address, HighPC, SectionIndex)) {
        Expected<DWARFAddressRangesVector> RangesOrError =
            getDIEAddressRanges(*Die, Unit);
        if (!RangesOrError) {
          consumeError(RangesOrError.takeError());
          break;
        }
        DWARFAddressRangesVector Ranges = *RangesOrError;
        // Not a function definition.
        if (Ranges.empty())
          break;

        for (const DWARFAddressRange &Range : Ranges) {
          if (const BinaryFunction *Function =
                  BC.getBinaryFunctionAtAddress(Range.LowPC))
            FunctionRanges.append(Function->getOutputAddressRanges());
        }
      } else {
        if (const BinaryFunction *Function =
                BC.getBinaryFunctionAtAddress(Address))
          FunctionRanges = Function->getOutputAddressRanges();
      }

      // Clear cached ranges as the new function will have its own set.
      CachedRanges.clear();
      DIEValue LowPCVal = Die->findAttribute(dwarf::DW_AT_low_pc);
      DIEValue HighPCVal = Die->findAttribute(dwarf::DW_AT_high_pc);
      if (FunctionRanges.empty()) {
        if (LowPCVal && HighPCVal) {
          FunctionRanges.push_back({0, HighPCVal.getDIEInteger().getValue()});
        } else {
          // I haven't seen this case, but who knows what other compilers
          // generate.
          FunctionRanges.push_back({0, 1});
          errs() << "BOLT-WARNING: [internal-dwarf-error]: subprogram got GCed "
                    "by the linker, DW_AT_ranges is used\n";
        }
      }

      if (FunctionRanges.size() == 1 && !opts::AlwaysConvertToRanges) {
        updateLowPCHighPC(Die, LowPCVal, HighPCVal, FunctionRanges.back().LowPC,
                          FunctionRanges.back().HighPC);
        break;
      }

      updateDWARFObjectAddressRanges(
          Unit, DIEBldr, *Die, RangesSectionWriter.addRanges(FunctionRanges));

      break;
    }
    case dwarf::DW_TAG_lexical_block:
    case dwarf::DW_TAG_inlined_subroutine:
    case dwarf::DW_TAG_try_block:
    case dwarf::DW_TAG_catch_block: {
      uint64_t RangesSectionOffset = 0;
      Expected<DWARFAddressRangesVector> RangesOrError =
          getDIEAddressRanges(*Die, Unit);
      const BinaryFunction *Function =
          RangesOrError && !RangesOrError->empty()
              ? BC.getBinaryFunctionContainingAddress(
                    RangesOrError->front().LowPC)
              : nullptr;
      DebugAddressRangesVector OutputRanges;
      if (Function) {
        OutputRanges = translateInputToOutputRanges(*Function, *RangesOrError);
        LLVM_DEBUG(if (OutputRanges.empty() != RangesOrError->empty()) {
          dbgs() << "BOLT-DEBUG: problem with DIE at 0x"
                 << Twine::utohexstr(Die->getOffset()) << " in CU at 0x"
                 << Twine::utohexstr(Unit.getOffset()) << '\n';
        });
        if (opts::AlwaysConvertToRanges || OutputRanges.size() > 1) {
          RangesSectionOffset = RangesSectionWriter.addRanges(
              std::move(OutputRanges), CachedRanges);
          OutputRanges.clear();
        } else if (OutputRanges.empty()) {
          OutputRanges.push_back({0, RangesOrError.get().front().HighPC});
        }
      } else if (!RangesOrError) {
        consumeError(RangesOrError.takeError());
      } else {
        OutputRanges.push_back({0, !RangesOrError->empty()
                                       ? RangesOrError.get().front().HighPC
                                       : 0});
      }
      DIEValue LowPCVal = Die->findAttribute(dwarf::DW_AT_low_pc);
      DIEValue HighPCVal = Die->findAttribute(dwarf::DW_AT_high_pc);
      if (OutputRanges.size() == 1) {
        updateLowPCHighPC(Die, LowPCVal, HighPCVal, OutputRanges.back().LowPC,
                          OutputRanges.back().HighPC);
        break;
      }
      updateDWARFObjectAddressRanges(Unit, DIEBldr, *Die, RangesSectionOffset);
      break;
    }
    case dwarf::DW_TAG_call_site: {
      auto patchPC = [&](DIE *Die, DIEValue &AttrVal, StringRef Entry) -> void {
        std::optional<uint64_t> Address = getAsAddress(Unit, AttrVal);
        const BinaryFunction *Function =
            BC.getBinaryFunctionContainingAddress(*Address);
        uint64_t UpdatedAddress = *Address;
        if (Function)
          UpdatedAddress =
              Function->translateInputToOutputAddress(UpdatedAddress);

        if (AttrVal.getForm() == dwarf::DW_FORM_addrx) {
          const uint32_t Index =
              AddrWriter->getIndexFromAddress(UpdatedAddress, Unit);
          DIEBldr.replaceValue(Die, AttrVal.getAttribute(), AttrVal.getForm(),
                               DIEInteger(Index));
        } else if (AttrVal.getForm() == dwarf::DW_FORM_addr) {
          DIEBldr.replaceValue(Die, AttrVal.getAttribute(), AttrVal.getForm(),
                               DIEInteger(UpdatedAddress));
        } else {
          errs() << "BOLT-ERROR: unsupported form for " << Entry << "\n";
        }
      };
      DIEValue CallPcAttrVal = Die->findAttribute(dwarf::DW_AT_call_pc);
      if (CallPcAttrVal)
        patchPC(Die, CallPcAttrVal, "DW_AT_call_pc");

      DIEValue CallRetPcAttrVal =
          Die->findAttribute(dwarf::DW_AT_call_return_pc);
      if (CallRetPcAttrVal)
        patchPC(Die, CallRetPcAttrVal, "DW_AT_call_return_pc");

      break;
    }
    default: {
      // Handle any tag that can have DW_AT_location attribute.
      DIEValue LocAttrInfo = Die->findAttribute(dwarf::DW_AT_location);
      DIEValue LowPCAttrInfo = Die->findAttribute(dwarf::DW_AT_low_pc);
      if (LocAttrInfo) {
        if (doesFormBelongToClass(LocAttrInfo.getForm(),
                                  DWARFFormValue::FC_Constant,
                                  Unit.getVersion()) ||
            doesFormBelongToClass(LocAttrInfo.getForm(),
                                  DWARFFormValue::FC_SectionOffset,
                                  Unit.getVersion())) {
          uint64_t Offset = LocAttrInfo.getForm() == dwarf::DW_FORM_loclistx
                                ? LocAttrInfo.getDIELocList().getValue()
                                : LocAttrInfo.getDIEInteger().getValue();
          DebugLocationsVector InputLL;

          std::optional<object::SectionedAddress> SectionAddress =
              Unit.getBaseAddress();
          uint64_t BaseAddress = 0;
          if (SectionAddress)
            BaseAddress = SectionAddress->Address;

          if (Unit.getVersion() >= 5 &&
              LocAttrInfo.getForm() == dwarf::DW_FORM_loclistx) {
            std::optional<uint64_t> LocOffset = Unit.getLoclistOffset(Offset);
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
                       (!Unit.isDWOUnit() || Unit.getVersion() == 5)) &&
                      "absolute address expected");
                  InputLL.emplace_back(DebugLocationEntry{
                      BaseAddress + Entry.Value0, BaseAddress + Entry.Value1,
                      Entry.Loc});
                  break;
                case dwarf::DW_LLE_start_length:
                  InputLL.emplace_back(DebugLocationEntry{
                      Entry.Value0, Entry.Value0 + Entry.Value1, Entry.Loc});
                  break;
                case dwarf::DW_LLE_base_addressx: {
                  std::optional<object::SectionedAddress> EntryAddress =
                      Unit.getAddrOffsetSectionItem(Entry.Value0);
                  assert(EntryAddress && "base Address not found.");
                  BaseAddress = EntryAddress->Address;
                  break;
                }
                case dwarf::DW_LLE_startx_length: {
                  std::optional<object::SectionedAddress> EntryAddress =
                      Unit.getAddrOffsetSectionItem(Entry.Value0);
                  assert(EntryAddress && "Address does not exist.");
                  InputLL.emplace_back(DebugLocationEntry{
                      EntryAddress->Address,
                      EntryAddress->Address + Entry.Value1, Entry.Loc});
                  break;
                }
                case dwarf::DW_LLE_startx_endx: {
                  std::optional<object::SectionedAddress> StartAddress =
                      Unit.getAddrOffsetSectionItem(Entry.Value0);
                  assert(StartAddress && "Start Address does not exist.");
                  std::optional<object::SectionedAddress> EndAddress =
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
                   << Twine::utohexstr(Offset) << " for DIE at 0x" << Die
                   << " in CU at 0x" << Twine::utohexstr(Unit.getOffset())
                   << '\n';
          } else {
            const uint64_t Address = InputLL.front().LowPC;
            DebugLocationsVector OutputLL;
            if (const BinaryFunction *Function =
                    BC.getBinaryFunctionContainingAddress(Address)) {
              OutputLL = translateInputToOutputLocationList(*Function, InputLL);
              LLVM_DEBUG(if (OutputLL.empty()) {
                dbgs() << "BOLT-DEBUG: location list translated to an empty "
                          "one at 0x"
                       << Die << " in CU at 0x"
                       << Twine::utohexstr(Unit.getOffset()) << '\n';
              });
            } else {
              // It's possible for a subprogram to be removed and to have
              // address of 0. Adding this entry to output to preserve debug
              // information.
              OutputLL = InputLL;
            }
            DebugLocWriter.addList(DIEBldr, *Die, LocAttrInfo, OutputLL);
          }
        } else {
          assert((doesFormBelongToClass(LocAttrInfo.getForm(),
                                        DWARFFormValue::FC_Exprloc,
                                        Unit.getVersion()) ||
                  doesFormBelongToClass(LocAttrInfo.getForm(),
                                        DWARFFormValue::FC_Block,
                                        Unit.getVersion())) &&
                 "unexpected DW_AT_location form");
          if (Unit.isDWOUnit() || Unit.getVersion() >= 5) {
            std::vector<uint8_t> Sblock;
            DIEValueList *AttrLocValList;
            if (doesFormBelongToClass(LocAttrInfo.getForm(),
                                      DWARFFormValue::FC_Exprloc,
                                      Unit.getVersion())) {
              for (const DIEValue &Val : LocAttrInfo.getDIELoc().values()) {
                Sblock.push_back(Val.getDIEInteger().getValue());
              }
              DIELoc *LocAttr = const_cast<DIELoc *>(&LocAttrInfo.getDIELoc());
              AttrLocValList = static_cast<DIEValueList *>(LocAttr);
            } else {
              for (const DIEValue &Val : LocAttrInfo.getDIEBlock().values()) {
                Sblock.push_back(Val.getDIEInteger().getValue());
              }
              DIEBlock *BlockAttr =
                  const_cast<DIEBlock *>(&LocAttrInfo.getDIEBlock());
              AttrLocValList = static_cast<DIEValueList *>(BlockAttr);
            }
            ArrayRef<uint8_t> Expr = ArrayRef<uint8_t>(Sblock);
            DataExtractor Data(
                StringRef((const char *)Expr.data(), Expr.size()),
                Unit.getContext().isLittleEndian(), 0);
            DWARFExpression LocExpr(Data, Unit.getAddressByteSize(),
                                    Unit.getFormParams().Format);
            uint32_t PrevOffset = 0;
            DIEValueList *NewAttr;
            DIEValue Value;
            uint32_t NewExprSize = 0;
            DIELoc *Loc = nullptr;
            DIEBlock *Block = nullptr;
            if (LocAttrInfo.getForm() == dwarf::DW_FORM_exprloc) {
              Loc = DIEBldr.allocateDIEValue<DIELoc>();
              NewAttr = Loc;
              Value = DIEValue(LocAttrInfo.getAttribute(),
                               LocAttrInfo.getForm(), Loc);
            } else if (doesFormBelongToClass(LocAttrInfo.getForm(),
                                             DWARFFormValue::FC_Block,
                                             Unit.getVersion())) {
              Block = DIEBldr.allocateDIEValue<DIEBlock>();
              NewAttr = Block;
              Value = DIEValue(LocAttrInfo.getAttribute(),
                               LocAttrInfo.getForm(), Block);
            } else {
              errs() << "BOLT-WARNING: Unexpected Form value in Updating "
                        "DW_AT_Location\n";
              continue;
            }

            for (const DWARFExpression::Operation &Expr : LocExpr) {
              uint32_t CurEndOffset = PrevOffset + 1;
              if (Expr.getDescription().Op.size() == 1)
                CurEndOffset = Expr.getOperandEndOffset(0);
              if (Expr.getDescription().Op.size() == 2)
                CurEndOffset = Expr.getOperandEndOffset(1);
              if (Expr.getDescription().Op.size() > 2)
                errs() << "BOLT-WARNING: [internal-dwarf-error]: Unsupported "
                          "number of operands.\n";
              // not addr index, just copy.
              if (!(Expr.getCode() == dwarf::DW_OP_GNU_addr_index ||
                    Expr.getCode() == dwarf::DW_OP_addrx)) {
                auto Itr = AttrLocValList->values().begin();
                std::advance(Itr, PrevOffset);
                uint32_t CopyNum = CurEndOffset - PrevOffset;
                NewExprSize += CopyNum;
                while (CopyNum--) {
                  DIEBldr.addValue(NewAttr, *Itr);
                  std::advance(Itr, 1);
                }
              } else {
                const uint64_t Index = Expr.getRawOperand(0);
                std::optional<object::SectionedAddress> EntryAddress =
                    Unit.getAddrOffsetSectionItem(Index);
                assert(EntryAddress && "Address is not found.");
                assert(Index <= std::numeric_limits<uint32_t>::max() &&
                       "Invalid Operand Index.");
                const uint32_t AddrIndex = AddrWriter->getIndexFromAddress(
                    EntryAddress->Address, Unit);
                // update Index into .debug_address section for DW_AT_location.
                // The Size field is not stored in IR, we need to minus 1 in
                // offset for each expr.
                SmallString<8> Tmp;
                raw_svector_ostream OSE(Tmp);
                encodeULEB128(AddrIndex, OSE);

                DIEBldr.addValue(NewAttr, static_cast<dwarf::Attribute>(0),
                                 dwarf::DW_FORM_data1,
                                 DIEInteger(Expr.getCode()));
                NewExprSize += 1;
                for (uint8_t Byte : Tmp) {
                  DIEBldr.addValue(NewAttr, static_cast<dwarf::Attribute>(0),
                                   dwarf::DW_FORM_data1, DIEInteger(Byte));
                  NewExprSize += 1;
                }
              }
              PrevOffset = CurEndOffset;
            }

            // update the size since the index might be changed
            if (Loc)
              Loc->setSize(NewExprSize);
            else
              Block->setSize(NewExprSize);
            DIEBldr.replaceValue(Die, LocAttrInfo.getAttribute(),
                                 LocAttrInfo.getForm(), Value);
          }
        }
      } else if (LowPCAttrInfo) {
        const std::optional<uint64_t> Result =
            LowPCAttrInfo.getDIEInteger().getValue();
        if (Result.has_value()) {
          const uint64_t Address = Result.value();
          uint64_t NewAddress = 0;
          if (const BinaryFunction *Function =
                  BC.getBinaryFunctionContainingAddress(Address)) {
            NewAddress = Function->translateInputToOutputAddress(Address);
            LLVM_DEBUG(dbgs()
                       << "BOLT-DEBUG: Fixing low_pc 0x"
                       << Twine::utohexstr(Address) << " for DIE with tag "
                       << Die->getTag() << " to 0x"
                       << Twine::utohexstr(NewAddress) << '\n');
          }

          dwarf::Form Form = LowPCAttrInfo.getForm();
          assert(Form != dwarf::DW_FORM_LLVM_addrx_offset &&
                 "DW_FORM_LLVM_addrx_offset is not supported");
          std::lock_guard<std::mutex> Lock(DWARFRewriterMutex);
          if (Form == dwarf::DW_FORM_addrx ||
              Form == dwarf::DW_FORM_GNU_addr_index) {
            const uint32_t Index = AddrWriter->getIndexFromAddress(
                NewAddress ? NewAddress : Address, Unit);
            DIEBldr.replaceValue(Die, LowPCAttrInfo.getAttribute(),
                                 LowPCAttrInfo.getForm(), DIEInteger(Index));
          } else {
            DIEBldr.replaceValue(Die, LowPCAttrInfo.getAttribute(),
                                 LowPCAttrInfo.getForm(),
                                 DIEInteger(NewAddress));
          }
        } else if (opts::Verbosity >= 1) {
          errs() << "BOLT-WARNING: unexpected form value for attribute "
                    "LowPCAttrInfo\n";
        }
      }
    }
    }
  }
  if (DIEOffset > NextCUOffset)
    errs() << "BOLT-WARNING: corrupt DWARF detected at 0x"
           << Twine::utohexstr(Unit.getOffset()) << '\n';
}

void DWARFRewriter::updateDWARFObjectAddressRanges(
    DWARFUnit &Unit, DIEBuilder &DIEBldr, DIE &Die, uint64_t DebugRangesOffset,
    std::optional<uint64_t> RangesBase) {

  if (RangesBase) {
    // If DW_AT_GNU_ranges_base is present, update it. No further modifications
    // are needed for ranges base.

    DIEValue RangesBaseInfo = Die.findAttribute(dwarf::DW_AT_GNU_ranges_base);
    if (!RangesBaseInfo) {
      RangesBaseInfo = Die.findAttribute(dwarf::DW_AT_rnglists_base);
    }

    if (RangesBaseInfo) {
      DIEBldr.replaceValue(&Die, RangesBaseInfo.getAttribute(),
                           RangesBaseInfo.getForm(),
                           DIEInteger(static_cast<uint32_t>(*RangesBase)));
      RangesBase = std::nullopt;
    }
  }

  DIEValue LowPCAttrInfo = Die.findAttribute(dwarf::DW_AT_low_pc);
  DIEValue RangesAttrInfo = Die.findAttribute(dwarf::DW_AT_ranges);
  if (RangesAttrInfo) {
    // Case 1: The object was already non-contiguous and had DW_AT_ranges.
    // In this case we simply need to update the value of DW_AT_ranges
    // and introduce DW_AT_GNU_ranges_base if required.
    // For DWARF5 converting all of DW_AT_ranges into DW_FORM_rnglistx
    bool NeedConverted = false;

    if (Unit.getVersion() >= 5 &&
        RangesAttrInfo.getForm() == dwarf::DW_FORM_sec_offset)
      NeedConverted = true;

    uint64_t CurRangeBase = 0;
    if (Unit.isDWOUnit()) {
      if (std::optional<uint64_t> DWOId = Unit.getDWOId())
        CurRangeBase = getDwoRangesBase(*DWOId);
      else
        errs() << "BOLT-WARNING: [internal-dwarf-error]: DWOId is not found "
                  "for DWO Unit.";
    }
    if (NeedConverted || RangesAttrInfo.getForm() == dwarf::DW_FORM_rnglistx)
      DIEBldr.replaceValue(&Die, dwarf::DW_AT_ranges, dwarf::DW_FORM_rnglistx,
                           DIEInteger(DebugRangesOffset));
    else
      DIEBldr.replaceValue(&Die, dwarf::DW_AT_ranges, RangesAttrInfo.getForm(),
                           DIEInteger(DebugRangesOffset - CurRangeBase));

    if (!RangesBase) {
      if (LowPCAttrInfo &&
          LowPCAttrInfo.getForm() != dwarf::DW_FORM_GNU_addr_index &&
          LowPCAttrInfo.getForm() != dwarf::DW_FORM_addrx)
        DIEBldr.replaceValue(&Die, dwarf::DW_AT_low_pc, LowPCAttrInfo.getForm(),
                             DIEInteger(0));
      return;
    }

    if (!(Die.getTag() == dwarf::DW_TAG_compile_unit ||
          Die.getTag() == dwarf::DW_TAG_skeleton_unit))
      return;

    // If we are at this point we are in the CU/Skeleton CU, and
    // DW_AT_GNU_ranges_base or DW_AT_rnglists_base doesn't exist.
    if (Unit.getVersion() <= 4)
      DIEBldr.addValue(&Die, dwarf::DW_AT_GNU_ranges_base, dwarf::DW_FORM_data4,
                       DIEInteger(*RangesBase));
    else if (Unit.getVersion() == 5)
      DIEBldr.addValue(&Die, dwarf::DW_AT_rnglists_base,
                       dwarf::DW_FORM_sec_offset, DIEInteger(*RangesBase));
    else
      DIEBldr.addValue(&Die, dwarf::DW_AT_rnglists_base,
                       dwarf::DW_FORM_sec_offset, DIEInteger(*RangesBase));
    return;
  }

  // Case 2: The object has both DW_AT_low_pc and DW_AT_high_pc emitted back
  // to back. Replace with new attributes and patch the DIE.
  DIEValue HighPCAttrInfo = Die.findAttribute(dwarf::DW_AT_high_pc);
  if (LowPCAttrInfo && HighPCAttrInfo) {

    convertToRangesPatchDebugInfo(Unit, DIEBldr, Die, DebugRangesOffset,
                                  LowPCAttrInfo, HighPCAttrInfo, RangesBase);
  } else if (!(Unit.isDWOUnit() &&
               Die.getTag() == dwarf::DW_TAG_compile_unit)) {
    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: cannot update ranges for DIE in Unit offset 0x"
             << Unit.getOffset() << '\n';
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

  // There is no direct connection between CU and TU, but same offsets,
  // encoded in DW_AT_stmt_list, into .debug_line get modified.
  // We take advantage of that to map original CU line table offsets to new
  // ones.
  std::unordered_map<uint64_t, uint64_t> DebugLineOffsetMap;

  auto GetStatementListValue = [](DWARFUnit *Unit) {
    std::optional<DWARFFormValue> StmtList =
        Unit->getUnitDIE().find(dwarf::DW_AT_stmt_list);
    std::optional<uint64_t> Offset = dwarf::toSectionOffset(StmtList);
    assert(Offset && "Was not able to retrieve value of DW_AT_stmt_list.");
    return *Offset;
  };

  for (const std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
    const unsigned CUID = CU->getOffset();
    MCSymbol *Label = BC.getDwarfLineTable(CUID).getLabel();
    if (!Label)
      continue;

    std::optional<AttrInfo> AttrVal =
        findAttributeInfo(CU.get()->getUnitDIE(), dwarf::DW_AT_stmt_list);
    if (!AttrVal)
      continue;

    const uint64_t LineTableOffset = Layout.getSymbolOffset(*Label);
    DebugLineOffsetMap[GetStatementListValue(CU.get())] = LineTableOffset;
    assert(DbgInfoSection && ".debug_info section must exist");
    LineTablePatchMap[CU.get()] = LineTableOffset;
  }

  for (const std::unique_ptr<DWARFUnit> &TU : BC.DwCtx->types_section_units()) {
    DWARFUnit *Unit = TU.get();
    std::optional<AttrInfo> AttrVal =
        findAttributeInfo(TU.get()->getUnitDIE(), dwarf::DW_AT_stmt_list);
    if (!AttrVal)
      continue;
    auto Iter = DebugLineOffsetMap.find(GetStatementListValue(Unit));
    assert(Iter != DebugLineOffsetMap.end() &&
           "Type Unit Updated Line Number Entry does not exist.");
    TypeUnitRelocMap[Unit] = Iter->second;
  }

  // Set .debug_info as finalized so it won't be skipped over when
  // we process sections while writing out the new binary. This ensures
  // that the pending relocations will be processed and not ignored.
  if (DbgInfoSection)
    DbgInfoSection->setIsFinalized();

  if (TypeInfoSection)
    TypeInfoSection->setIsFinalized();
}

CUOffsetMap DWARFRewriter::finalizeTypeSections(DIEBuilder &DIEBlder,
                                                DIEStreamer &Streamer) {
  // update TypeUnit DW_AT_stmt_list with new .debug_line information.
  for (const std::unique_ptr<DWARFUnit> &TU : BC.DwCtx->types_section_units()) {
    DIE *UnitDIE = DIEBlder.getUnitDIEbyUnit(*TU.get());
    DIEValue StmtAttrInfo = UnitDIE->findAttribute(dwarf::DW_AT_stmt_list);
    if (!StmtAttrInfo || !TypeUnitRelocMap.count(TU.get()))
      continue;
    DIEBlder.replaceValue(UnitDIE, dwarf::DW_AT_stmt_list,
                          StmtAttrInfo.getForm(),
                          DIEInteger(TypeUnitRelocMap[TU.get()]));
  }

  // generate and populate abbrevs here
  DIEBlder.generateAbbrevs();
  DIEBlder.finish();
  SmallVector<char, 20> OutBuffer;
  std::shared_ptr<raw_svector_ostream> ObjOS =
      std::make_shared<raw_svector_ostream>(OutBuffer);
  const object::ObjectFile *File = BC.DwCtx->getDWARFObj().getFile();
  auto TheTriple = std::make_unique<Triple>(File->makeTriple());
  std::unique_ptr<DIEStreamer> TypeStreamer =
      createDIEStreamer(*TheTriple, *ObjOS, "TypeStreamer", DIEBlder, *this);

  // generate debug_info and CUMap
  CUOffsetMap CUMap;
  for (std::unique_ptr<llvm::DWARFUnit> &CU : BC.DwCtx->info_section_units()) {
    if (!CU->isTypeUnit())
      continue;
    emitUnit(DIEBlder, Streamer, *CU.get());
    uint32_t StartOffset = CUOffset;
    DIE *UnitDIE = DIEBlder.getUnitDIEbyUnit(*CU.get());
    CUOffset += CU.get()->getHeaderSize();
    CUOffset += UnitDIE->getSize();
    CUMap[CU.get()->getOffset()] = {StartOffset, CUOffset - StartOffset - 4};
  }

  // Emit Type Unit of DWARF 4 to .debug_type section
  for (DWARFUnit *TU : DIEBlder.getDWARF4TUVector())
    emitUnit(DIEBlder, *TypeStreamer, *TU);

  TypeStreamer->finish();

  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(ObjOS->str(), "in-memory object file", false);
  std::unique_ptr<object::ObjectFile> Obj = cantFail(
      object::ObjectFile::createObjectFile(ObjectMemBuffer->getMemBufferRef()),
      "error creating in-memory object");

  for (const SectionRef &Section : Obj->sections()) {
    StringRef Contents = cantFail(Section.getContents());
    StringRef Name = cantFail(Section.getName());
    if (Name.equals(".debug_types"))
      BC.registerOrUpdateNoteSection(".debug_types", copyByteArray(Contents),
                                     Contents.size());
  }
  return CUMap;
}

void DWARFRewriter::finalizeDebugSections(DIEBuilder &DIEBlder,
                                          DIEStreamer &Streamer,
                                          raw_svector_ostream &ObjOS,
                                          CUOffsetMap &CUMap) {
  if (StrWriter->isInitialized()) {
    RewriteInstance::addToDebugSectionsToOverwrite(".debug_str");
    std::unique_ptr<DebugStrBufferVector> DebugStrSectionContents =
        StrWriter->releaseBuffer();
    BC.registerOrUpdateNoteSection(".debug_str",
                                   copyByteArray(*DebugStrSectionContents),
                                   DebugStrSectionContents->size());
  }

  if (StrOffstsWriter->isFinalized()) {
    RewriteInstance::addToDebugSectionsToOverwrite(".debug_str_offsets");
    std::unique_ptr<DebugStrOffsetsBufferVector>
        DebugStrOffsetsSectionContents = StrOffstsWriter->releaseBuffer();
    BC.registerOrUpdateNoteSection(
        ".debug_str_offsets", copyByteArray(*DebugStrOffsetsSectionContents),
        DebugStrOffsetsSectionContents->size());
  }

  if (BC.isDWARFLegacyUsed()) {
    std::unique_ptr<DebugBufferVector> RangesSectionContents =
        LegacyRangesSectionWriter->releaseBuffer();
    BC.registerOrUpdateNoteSection(".debug_ranges",
                                   copyByteArray(*RangesSectionContents),
                                   RangesSectionContents->size());
  }

  if (BC.isDWARF5Used()) {
    std::unique_ptr<DebugBufferVector> RangesSectionContents =
        RangeListsSectionWriter->releaseBuffer();
    BC.registerOrUpdateNoteSection(".debug_rnglists",
                                   copyByteArray(*RangesSectionContents),
                                   RangesSectionContents->size());
  }

  if (BC.isDWARF5Used()) {
    std::unique_ptr<DebugBufferVector> LocationListSectionContents =
        makeFinalLocListsSection(DWARFVersion::DWARF5);
    if (!LocationListSectionContents->empty())
      BC.registerOrUpdateNoteSection(
          ".debug_loclists", copyByteArray(*LocationListSectionContents),
          LocationListSectionContents->size());
  }

  if (BC.isDWARFLegacyUsed()) {
    std::unique_ptr<DebugBufferVector> LocationListSectionContents =
        makeFinalLocListsSection(DWARFVersion::DWARFLegacy);
    if (!LocationListSectionContents->empty())
      BC.registerOrUpdateNoteSection(
          ".debug_loc", copyByteArray(*LocationListSectionContents),
          LocationListSectionContents->size());
  }

  // AddrWriter should be finalized after debug_loc since more addresses can be
  // added there.
  if (AddrWriter->isInitialized()) {
    AddressSectionBuffer AddressSectionContents = AddrWriter->finalize();
    BC.registerOrUpdateNoteSection(".debug_addr",
                                   copyByteArray(AddressSectionContents),
                                   AddressSectionContents.size());
  }

  Streamer.emitAbbrevs(DIEBlder.getAbbrevs(), BC.DwCtx->getMaxVersion());
  Streamer.finish();

  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(ObjOS.str(), "in-memory object file", false);
  std::unique_ptr<object::ObjectFile> Obj = cantFail(
      object::ObjectFile::createObjectFile(ObjectMemBuffer->getMemBufferRef()),
      "error creating in-memory object");

  for (const SectionRef &Secs : Obj->sections()) {
    StringRef Contents = cantFail(Secs.getContents());
    StringRef Name = cantFail(Secs.getName());
    if (Name.equals(".debug_abbrev")) {
      BC.registerOrUpdateNoteSection(".debug_abbrev", copyByteArray(Contents),
                                     Contents.size());
    } else if (Name.equals(".debug_info")) {
      BC.registerOrUpdateNoteSection(".debug_info", copyByteArray(Contents),
                                     Contents.size());
    }
  }

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
}

void DWARFRewriter::finalizeCompileUnits(DIEBuilder &DIEBlder,
                                         DIEStreamer &Streamer,
                                         CUOffsetMap &CUMap,
                                         const std::list<DWARFUnit *> &CUs) {
  DIEBlder.generateAbbrevs();
  DIEBlder.finish();
  // generate debug_info and CUMap
  for (DWARFUnit *CU : CUs) {
    emitUnit(DIEBlder, Streamer, *CU);
    const uint32_t StartOffset = CUOffset;
    DIE *UnitDIE = DIEBlder.getUnitDIEbyUnit(*CU);
    CUOffset += CU->getHeaderSize();
    CUOffset += UnitDIE->getSize();
    CUMap[CU->getOffset()] = {StartOffset, CUOffset - StartOffset - 4};
  }
}

// Creates all the data structures necessary for creating MCStreamer.
// They are passed by reference because they need to be kept around.
// Also creates known debug sections. These are sections handled by
// handleDebugDataPatching.
namespace {

std::unique_ptr<BinaryContext>
createDwarfOnlyBC(const object::ObjectFile &File) {
  return cantFail(BinaryContext::createBinaryContext(
      &File, false,
      DWARFContext::create(File, DWARFContext::ProcessDebugRelocations::Ignore,
                           nullptr, "", WithColor::defaultErrorHandler,
                           WithColor::defaultWarningHandler)));
}

StringMap<DWARFRewriter::KnownSectionsEntry>
createKnownSectionsMap(const MCObjectFileInfo &MCOFI) {
  StringMap<DWARFRewriter::KnownSectionsEntry> KnownSectionsTemp = {
      {"debug_info.dwo", {MCOFI.getDwarfInfoDWOSection(), DW_SECT_INFO}},
      {"debug_types.dwo", {MCOFI.getDwarfTypesDWOSection(), DW_SECT_EXT_TYPES}},
      {"debug_str_offsets.dwo",
       {MCOFI.getDwarfStrOffDWOSection(), DW_SECT_STR_OFFSETS}},
      {"debug_str.dwo", {MCOFI.getDwarfStrDWOSection(), DW_SECT_EXT_unknown}},
      {"debug_loc.dwo", {MCOFI.getDwarfLocDWOSection(), DW_SECT_EXT_LOC}},
      {"debug_abbrev.dwo", {MCOFI.getDwarfAbbrevDWOSection(), DW_SECT_ABBREV}},
      {"debug_line.dwo", {MCOFI.getDwarfLineDWOSection(), DW_SECT_LINE}},
      {"debug_loclists.dwo",
       {MCOFI.getDwarfLoclistsDWOSection(), DW_SECT_LOCLISTS}},
      {"debug_rnglists.dwo",
       {MCOFI.getDwarfRnglistsDWOSection(), DW_SECT_RNGLISTS}}};
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
std::optional<StringRef> updateDebugData(
    DWARFContext &DWCtx, StringRef SectionName, StringRef SectionContents,
    const StringMap<DWARFRewriter::KnownSectionsEntry> &KnownSections,
    MCStreamer &Streamer, DWARFRewriter &Writer,
    const DWARFUnitIndex::Entry *CUDWOEntry, uint64_t DWOId,
    std::unique_ptr<DebugBufferVector> &OutputBuffer,
    DebugRangeListsSectionWriter *RangeListsWriter, DebugLocWriter &LocWriter,
    const llvm::bolt::DWARFRewriter::OverriddenSectionsMap &OverridenSections) {

  using DWOSectionContribution =
      const DWARFUnitIndex::Entry::SectionContribution;
  auto getSliceData = [&](const DWARFUnitIndex::Entry *DWOEntry,
                          StringRef OutData, DWARFSectionKind Sec,
                          uint64_t &DWPOffset) -> StringRef {
    if (DWOEntry) {
      DWOSectionContribution *DWOContrubution = DWOEntry->getContribution(Sec);
      DWPOffset = DWOContrubution->getOffset();
      OutData = OutData.substr(DWPOffset, DWOContrubution->getLength());
    }
    return OutData;
  };

  auto SectionIter = KnownSections.find(SectionName);
  if (SectionIter == KnownSections.end())
    return std::nullopt;
  Streamer.switchSection(SectionIter->second.first);
  uint64_t DWPOffset = 0;

  auto getOverridenSection =
      [&](DWARFSectionKind Kind) -> std::optional<StringRef> {
    auto Iter = OverridenSections.find(Kind);
    if (Iter == OverridenSections.end()) {
      errs()
          << "BOLT-WARNING: [internal-dwarf-error]: Could not find overriden "
             "section for: "
          << Twine::utohexstr(DWOId) << ".\n";
      return std::nullopt;
    }
    return Iter->second;
  };
  switch (SectionIter->second.second) {
  default: {
    if (!SectionName.equals("debug_str.dwo"))
      errs() << "BOLT-WARNING: unsupported debug section: " << SectionName
             << "\n";
    return SectionContents;
  }
  case DWARFSectionKind::DW_SECT_INFO: {
    return getOverridenSection(DWARFSectionKind::DW_SECT_INFO);
  }
  case DWARFSectionKind::DW_SECT_EXT_TYPES: {
    return getOverridenSection(DWARFSectionKind::DW_SECT_EXT_TYPES);
  }
  case DWARFSectionKind::DW_SECT_STR_OFFSETS: {
    return getSliceData(CUDWOEntry, SectionContents,
                        DWARFSectionKind::DW_SECT_STR_OFFSETS, DWPOffset);
  }
  case DWARFSectionKind::DW_SECT_ABBREV: {
    return getOverridenSection(DWARFSectionKind::DW_SECT_ABBREV);
  }
  case DWARFSectionKind::DW_SECT_EXT_LOC:
  case DWARFSectionKind::DW_SECT_LOCLISTS: {
    OutputBuffer = LocWriter.getBuffer();
    // Creating explicit StringRef here, otherwise
    // with implicit conversion it will take null byte as end of
    // string.
    return StringRef(reinterpret_cast<const char *>(OutputBuffer->data()),
                     OutputBuffer->size());
  }
  case DWARFSectionKind::DW_SECT_LINE: {
    return getSliceData(CUDWOEntry, SectionContents,
                        DWARFSectionKind::DW_SECT_LINE, DWPOffset);
  }
  case DWARFSectionKind::DW_SECT_RNGLISTS: {
    assert(RangeListsWriter && "RangeListsWriter was not created.");
    OutputBuffer = RangeListsWriter->releaseBuffer();
    return StringRef(reinterpret_cast<const char *>(OutputBuffer->data()),
                     OutputBuffer->size());
  }
  }
}

} // namespace

void DWARFRewriter::initDWPState(DWPState &State) {
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
  State.Out =
      std::make_unique<ToolOutputFile>(OutputName, EC, sys::fs::OF_None);
  const object::ObjectFile *File = BC.DwCtx->getDWARFObj().getFile();
  State.TmpBC = createDwarfOnlyBC(*File);
  State.Streamer = State.TmpBC->createStreamer(State.Out->os());
  State.MCOFI = State.Streamer->getContext().getObjectFileInfo();
  State.KnownSections = createKnownSectionsMap(*State.MCOFI);
  MCSection *const StrSection = State.MCOFI->getDwarfStrDWOSection();

  // Data Structures for DWP book keeping
  // Size of array corresponds to the number of sections supported by DWO format
  // in DWARF4/5.

  State.Strings = std::make_unique<DWPStringPool>(*State.Streamer, StrSection);

  // Setup DWP code once.
  DWARFContext *DWOCtx = BC.getDWOContext();

  if (DWOCtx) {
    State.CUIndex = &DWOCtx->getCUIndex();
    State.IsDWP = !State.CUIndex->getRows().empty();
  }
}

void DWARFRewriter::finalizeDWP(DWPState &State) {
  if (State.Version < 5) {
    // Lie about there being no info contributions so the TU index only includes
    // the type unit contribution for DWARF < 5. In DWARFv5 the TU index has a
    // contribution to the info section, so we do not want to lie about it.
    State.ContributionOffsets[0] = 0;
  }
  writeIndex(*State.Streamer.get(), State.MCOFI->getDwarfTUIndexSection(),
             State.ContributionOffsets, State.TypeIndexEntries,
             State.IndexVersion);

  if (State.Version < 5) {
    // Lie about the type contribution for DWARF < 5. In DWARFv5 the type
    // section does not exist, so no need to do anything about this.
    State.ContributionOffsets[getContributionIndex(DW_SECT_EXT_TYPES, 2)] = 0;
    // Unlie about the info contribution
    State.ContributionOffsets[0] = 1;
  }
  writeIndex(*State.Streamer.get(), State.MCOFI->getDwarfCUIndexSection(),
             State.ContributionOffsets, State.IndexEntries, State.IndexVersion);

  State.Streamer->finish();
  State.Out->keep();
}

void DWARFRewriter::updateDWP(DWARFUnit &CU,
                              const OverriddenSectionsMap &OverridenSections,
                              const DWARFRewriter::UnitMeta &CUMI,
                              DWARFRewriter::UnitMetaVectorType &TUMetaVector,
                              DWPState &State, DebugLocWriter &LocWriter) {
  const uint64_t DWOId = *CU.getDWOId();
  MCSection *const StrOffsetSection = State.MCOFI->getDwarfStrOffDWOSection();
  assert(StrOffsetSection && "StrOffsetSection does not exist.");
  // Skipping CUs that we failed to load.
  std::optional<DWARFUnit *> DWOCU = BC.getDWOCU(DWOId);
  if (!DWOCU)
    return;

  if (State.Version == 0) {
    State.Version = CU.getVersion();
    State.IndexVersion = State.Version < 5 ? 2 : 5;
  } else if (State.Version != CU.getVersion()) {
    errs() << "BOLT-ERROR: incompatible DWARF compile unit versions\n";
    exit(1);
  }

  UnitIndexEntry CurEntry = {};
  CurEntry.DWOName = dwarf::toString(
      CU.getUnitDIE().find({dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}),
      "");
  const char *Name = CU.getUnitDIE().getShortName();
  if (Name)
    CurEntry.Name = Name;
  StringRef CurStrSection;
  StringRef CurStrOffsetSection;

  // This maps each section contained in this file to its length.
  // This information is later on used to calculate the contributions,
  // i.e. offset and length, of each compile/type unit to a section.
  std::vector<std::pair<DWARFSectionKind, uint32_t>> SectionLength;

  const DWARFUnitIndex::Entry *CUDWOEntry = nullptr;
  if (State.IsDWP)
    CUDWOEntry = State.CUIndex->getFromHash(DWOId);

  bool StrSectionWrittenOut = false;
  const object::ObjectFile *DWOFile =
      (*DWOCU)->getContext().getDWARFObj().getFile();

  DebugRangeListsSectionWriter *RangeListssWriter = nullptr;
  if (CU.getVersion() == 5) {
    assert(RangeListsWritersByCU.count(DWOId) != 0 &&
           "No RangeListsWriter for DWO ID.");
    RangeListssWriter = RangeListsWritersByCU[DWOId].get();
  }
  auto AddType = [&](unsigned int Index, uint32_t IndexVersion, uint64_t Offset,
                     uint64_t Length, uint64_t Hash) -> void {
    UnitIndexEntry TUEntry = CurEntry;
    if (IndexVersion < 5)
      TUEntry.Contributions[0] = {};
    TUEntry.Contributions[Index].setOffset(Offset);
    TUEntry.Contributions[Index].setLength(Length);
    State.ContributionOffsets[Index] +=
        TUEntry.Contributions[Index].getLength32();
    State.TypeIndexEntries.insert(std::make_pair(Hash, TUEntry));
  };
  for (const SectionRef &Section : DWOFile->sections()) {
    std::unique_ptr<DebugBufferVector> OutputData;
    StringRef SectionName = getSectionName(Section);
    Expected<StringRef> ContentsExp = Section.getContents();
    assert(ContentsExp && "Invalid contents.");
    std::optional<StringRef> TOutData = updateDebugData(
        (*DWOCU)->getContext(), SectionName, *ContentsExp, State.KnownSections,
        *State.Streamer, *this, CUDWOEntry, DWOId, OutputData,
        RangeListssWriter, LocWriter, OverridenSections);
    if (!TOutData)
      continue;

    StringRef OutData = *TOutData;
    if (SectionName == "debug_types.dwo") {
      State.Streamer->emitBytes(OutData);
      continue;
    }

    if (SectionName.equals("debug_str.dwo")) {
      CurStrSection = OutData;
    } else {
      // Since handleDebugDataPatching returned true, we already know this is
      // a known section.
      auto SectionIter = State.KnownSections.find(SectionName);
      if (SectionIter->second.second == DWARFSectionKind::DW_SECT_STR_OFFSETS)
        CurStrOffsetSection = OutData;
      else
        State.Streamer->emitBytes(OutData);
      unsigned int Index =
          getContributionIndex(SectionIter->second.second, State.IndexVersion);
      uint64_t Offset = State.ContributionOffsets[Index];
      uint64_t Length = OutData.size();
      if (CU.getVersion() >= 5 &&
          SectionIter->second.second == DWARFSectionKind::DW_SECT_INFO) {
        for (UnitMeta &MI : TUMetaVector)
          MI.Offset += State.DebugInfoSize;

        Offset = State.DebugInfoSize + CUMI.Offset;
        Length = CUMI.Length;
        State.DebugInfoSize += OutData.size();
      }
      CurEntry.Contributions[Index].setOffset(Offset);
      CurEntry.Contributions[Index].setLength(Length);
      State.ContributionOffsets[Index] +=
          CurEntry.Contributions[Index].getLength32();
    }

    // Strings are combined in to a new string section, and de-duplicated
    // based on hash.
    if (!StrSectionWrittenOut && !CurStrOffsetSection.empty() &&
        !CurStrSection.empty()) {
      writeStringsAndOffsets(*State.Streamer.get(), *State.Strings.get(),
                             StrOffsetSection, CurStrSection,
                             CurStrOffsetSection, CU.getVersion());
      StrSectionWrittenOut = true;
    }
  }
  CompileUnitIdentifiers CUI{DWOId, CurEntry.Name.c_str(),
                             CurEntry.DWOName.c_str()};
  auto P = State.IndexEntries.insert(std::make_pair(CUI.Signature, CurEntry));
  if (!P.second) {
    Error Err = buildDuplicateError(*P.first, CUI, "");
    errs() << "BOLT-ERROR: " << toString(std::move(Err)) << "\n";
    return;
  }

  // Handling TU
  const unsigned Index = getContributionIndex(
      State.IndexVersion < 5 ? DW_SECT_EXT_TYPES : DW_SECT_INFO,
      State.IndexVersion);
  for (UnitMeta &MI : TUMetaVector)
    AddType(Index, State.IndexVersion, MI.Offset, MI.Length, MI.TUHash);
}

void DWARFRewriter::writeDWOFiles(
    DWARFUnit &CU, const OverriddenSectionsMap &OverridenSections,
    const std::string &DWOName, DebugLocWriter &LocWriter) {
  // Setup DWP code once.
  DWARFContext *DWOCtx = BC.getDWOContext();
  const uint64_t DWOId = *CU.getDWOId();
  const DWARFUnitIndex *CUIndex = nullptr;
  bool IsDWP = false;
  if (DWOCtx) {
    CUIndex = &DWOCtx->getCUIndex();
    IsDWP = !CUIndex->getRows().empty();
  }

  // Skipping CUs that we failed to load.
  std::optional<DWARFUnit *> DWOCU = BC.getDWOCU(DWOId);
  if (!DWOCU) {
    errs() << "BOLT-WARNING: [internal-dwarf-error]: CU for DWO_ID "
           << Twine::utohexstr(DWOId) << " is not found.\n";
    return;
  }

  std::string CompDir = opts::DwarfOutputPath.empty()
                            ? CU.getCompilationDir()
                            : opts::DwarfOutputPath.c_str();
  auto FullPath = CompDir.append("/").append(DWOName);

  std::error_code EC;
  std::unique_ptr<ToolOutputFile> TempOut =
      std::make_unique<ToolOutputFile>(FullPath, EC, sys::fs::OF_None);

  const DWARFUnitIndex::Entry *CUDWOEntry = nullptr;
  if (IsDWP)
    CUDWOEntry = CUIndex->getFromHash(DWOId);

  const object::ObjectFile *File =
      (*DWOCU)->getContext().getDWARFObj().getFile();
  std::unique_ptr<BinaryContext> TmpBC = createDwarfOnlyBC(*File);
  std::unique_ptr<MCStreamer> Streamer = TmpBC->createStreamer(TempOut->os());
  const MCObjectFileInfo &MCOFI = *Streamer->getContext().getObjectFileInfo();
  StringMap<KnownSectionsEntry> KnownSections = createKnownSectionsMap(MCOFI);

  DebugRangeListsSectionWriter *RangeListssWriter = nullptr;
  if (CU.getVersion() == 5) {
    assert(RangeListsWritersByCU.count(DWOId) != 0 &&
           "No RangeListsWriter for DWO ID.");
    RangeListssWriter = RangeListsWritersByCU[DWOId].get();

    // Handling .debug_rnglists.dwo separately. The original .o/.dwo might not
    // have .debug_rnglists so won't be part of the loop below.
    if (!RangeListssWriter->empty()) {
      std::unique_ptr<DebugBufferVector> OutputData;
      if (std::optional<StringRef> OutData = updateDebugData(
              (*DWOCU)->getContext(), "debug_rnglists.dwo", "", KnownSections,
              *Streamer, *this, CUDWOEntry, DWOId, OutputData,
              RangeListssWriter, LocWriter, OverridenSections))
        Streamer->emitBytes(*OutData);
    }
  }

  for (const SectionRef &Section : File->sections()) {
    std::unique_ptr<DebugBufferVector> OutputData;
    StringRef SectionName = getSectionName(Section);
    if (SectionName == "debug_rnglists.dwo")
      continue;
    Expected<StringRef> ContentsExp = Section.getContents();
    assert(ContentsExp && "Invalid contents.");
    if (std::optional<StringRef> OutData = updateDebugData(
            (*DWOCU)->getContext(), SectionName, *ContentsExp, KnownSections,
            *Streamer, *this, CUDWOEntry, DWOId, OutputData, RangeListssWriter,
            LocWriter, OverridenSections))
      Streamer->emitBytes(*OutData);
  }
  Streamer->finish();
  TempOut->keep();
}

void DWARFRewriter::addGDBTypeUnitEntry(const GDBIndexTUEntry &&Entry) {
  std::lock_guard<std::mutex> Lock(DWARFRewriterMutex);
  if (!BC.getGdbIndexSection())
    return;
  GDBIndexTUEntryVector.emplace_back(Entry);
}

void DWARFRewriter::updateGdbIndexSection(CUOffsetMap &CUMap, uint32_t NumCUs) {
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
  const uint32_t TUListSize = AddressTableOffset - CUTypesOffset;
  const unsigned NUmCUsEncoded = CUListSize / 16;
  unsigned MaxDWARFVersion = BC.DwCtx->getMaxVersion();
  unsigned NumDWARF5TUs =
      getGDBIndexTUEntryVector().size() - BC.DwCtx->getNumTypeUnits();
  bool SkipTypeUnits = false;
  // For DWARF5 Types are in .debug_info.
  // LLD doesn't generate Types CU List, and in CU list offset
  // only includes CUs.
  // GDB 11+ includes only CUs in CU list and generates Types
  // list.
  // GDB 9 includes CUs and TUs in CU list and generates TYpes
  // list. The NumCUs is CUs + TUs, so need to modify the check.
  // For split-dwarf
  // GDB-11, DWARF5: TU units from dwo are not included.
  // GDB-11, DWARF4: TU units from dwo are included.
  if (MaxDWARFVersion >= 5)
    SkipTypeUnits = !TUListSize ? true
                                : ((NUmCUsEncoded + NumDWARF5TUs) ==
                                   BC.DwCtx->getNumCompileUnits());

  if (!((CUListSize == NumCUs * 16) ||
        (CUListSize == (NumCUs + NumDWARF5TUs) * 16))) {
    errs() << "BOLT-ERROR: .gdb_index: CU count mismatch\n";
    exit(1);
  }
  DenseSet<uint64_t> OriginalOffsets;
  for (unsigned Index = 0, Units = BC.DwCtx->getNumCompileUnits();
       Index < Units; ++Index) {
    const DWARFUnit *CU = BC.DwCtx->getUnitAtIndex(Index);
    if (SkipTypeUnits && CU->isTypeUnit())
      continue;
    const uint64_t Offset = read64le(Data);
    Data += 16;
    if (CU->getOffset() != Offset) {
      errs() << "BOLT-ERROR: .gdb_index CU offset mismatch\n";
      exit(1);
    }

    OriginalOffsets.insert(Offset);
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

  using MapEntry = std::pair<uint32_t, CUInfo>;
  std::vector<MapEntry> CUVector(CUMap.begin(), CUMap.end());
  // Need to sort since we write out all of TUs in .debug_info before CUs.
  std::sort(CUVector.begin(), CUVector.end(),
            [](const MapEntry &E1, const MapEntry &E2) -> bool {
              return E1.second.Offset < E2.second.Offset;
            });
  // Writing out CU List <Offset, Size>
  for (auto &CUInfo : CUVector) {
    // Skipping TU for DWARF5 when they are not included in CU list.
    if (!OriginalOffsets.count(CUInfo.first))
      continue;
    write64le(Buffer, CUInfo.second.Offset);
    // Length encoded in CU doesn't contain first 4 bytes that encode length.
    write64le(Buffer + 8, CUInfo.second.Length + 4);
    Buffer += 16;
  }

  // Rewrite TU CU List, since abbrevs can be different.
  // Entry example:
  // 0: offset = 0x00000000, type_offset = 0x0000001e, type_signature =
  // 0x418503b8111e9a7b Spec says " triplet, the first value is the CU offset,
  // the second value is the type offset in the CU, and the third value is the
  // type signature" Looking at what is being generated by gdb-add-index. The
  // first entry is TU offset, second entry is offset from it, and third entry
  // is the type signature.
  if (TUListSize)
    for (const GDBIndexTUEntry &Entry : getGDBIndexTUEntryVector()) {
      write64le(Buffer, Entry.UnitOffset);
      write64le(Buffer + 8, Entry.TypeDIERelativeOffset);
      write64le(Buffer + 16, Entry.TypeHash);
      Buffer += sizeof(GDBIndexTUEntry);
    }

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
DWARFRewriter::makeFinalLocListsSection(DWARFVersion Version) {
  auto LocBuffer = std::make_unique<DebugBufferVector>();
  auto LocStream = std::make_unique<raw_svector_ostream>(*LocBuffer);
  auto Writer =
      std::unique_ptr<MCObjectWriter>(BC.createObjectWriter(*LocStream));

  for (std::pair<const uint64_t, std::unique_ptr<DebugLocWriter>> &Loc :
       LocListWritersByCU) {
    DebugLocWriter *LocWriter = Loc.second.get();
    auto *LocListWriter = llvm::dyn_cast<DebugLoclistWriter>(LocWriter);

    // Filter out DWARF4, writing out DWARF5
    if (Version == DWARFVersion::DWARF5 &&
        (!LocListWriter || LocListWriter->getDwarfVersion() <= 4))
      continue;

    // Filter out DWARF5, writing out DWARF4
    if (Version == DWARFVersion::DWARFLegacy &&
        (LocListWriter && LocListWriter->getDwarfVersion() >= 5))
      continue;

    // Skipping DWARF4/5 split dwarf.
    if (LocListWriter && LocListWriter->getDwarfVersion() <= 4)
      continue;
    std::unique_ptr<DebugBufferVector> CurrCULocationLists =
        LocWriter->getBuffer();
    *LocStream << *CurrCULocationLists;
  }

  return LocBuffer;
}

void DWARFRewriter::convertToRangesPatchDebugInfo(
    DWARFUnit &Unit, DIEBuilder &DIEBldr, DIE &Die,
    uint64_t RangesSectionOffset, DIEValue &LowPCAttrInfo,
    DIEValue &HighPCAttrInfo, std::optional<uint64_t> RangesBase) {
  uint32_t BaseOffset = 0;
  dwarf::Form LowForm = LowPCAttrInfo.getForm();
  dwarf::Attribute RangeBaseAttribute = dwarf::DW_AT_GNU_ranges_base;
  dwarf::Form RangesForm = dwarf::DW_FORM_sec_offset;

  if (Unit.getVersion() >= 5) {
    RangeBaseAttribute = dwarf::DW_AT_rnglists_base;
    RangesForm = dwarf::DW_FORM_rnglistx;
  } else if (Unit.getVersion() < 4) {
    RangesForm = dwarf::DW_FORM_data4;
  }
  bool IsUnitDie = Die.getTag() == dwarf::DW_TAG_compile_unit ||
                   Die.getTag() == dwarf::DW_TAG_skeleton_unit;
  if (!IsUnitDie)
    DIEBldr.deleteValue(&Die, LowPCAttrInfo.getAttribute());
  // In DWARF4 for DW_AT_low_pc in binary DW_FORM_addr is used. In the DWO
  // section DW_FORM_GNU_addr_index is used. So for if we are converting
  // DW_AT_low_pc/DW_AT_high_pc and see DW_FORM_GNU_addr_index. We are
  // converting in DWO section, and DW_AT_ranges [DW_FORM_sec_offset] is
  // relative to DW_AT_GNU_ranges_base.
  if (LowForm == dwarf::DW_FORM_GNU_addr_index) {
    // Ranges are relative to DW_AT_GNU_ranges_base.
    uint64_t CurRangeBase = 0;
    if (std::optional<uint64_t> DWOId = Unit.getDWOId()) {
      CurRangeBase = getDwoRangesBase(*DWOId);
    }
    BaseOffset = CurRangeBase;
  } else {
    // In DWARF 5 we can have DW_AT_low_pc either as DW_FORM_addr, or
    // DW_FORM_addrx. Former is when DW_AT_rnglists_base is present. Latter is
    // when it's absent.
    if (IsUnitDie) {
      if (LowForm == dwarf::DW_FORM_addrx) {
        const uint32_t Index = AddrWriter->getIndexFromAddress(0, Unit);
        DIEBldr.replaceValue(&Die, LowPCAttrInfo.getAttribute(),
                             LowPCAttrInfo.getForm(), DIEInteger(Index));
      } else {
        DIEBldr.replaceValue(&Die, LowPCAttrInfo.getAttribute(),
                             LowPCAttrInfo.getForm(), DIEInteger(0));
      }
    }
    // Original CU didn't have DW_AT_*_base. We converted it's children (or
    // dwo), so need to insert it into CU.
    if (RangesBase)
      DIEBldr.addValue(&Die, RangeBaseAttribute, dwarf::DW_FORM_sec_offset,
                       DIEInteger(*RangesBase));
  }

  uint64_t RangeAttrVal = RangesSectionOffset - BaseOffset;
  if (Unit.getVersion() >= 5)
    RangeAttrVal = RangesSectionOffset;
  // HighPC was conveted into DW_AT_ranges.
  // For DWARF5 we only access ranges through index.

  DIEBldr.replaceValue(&Die, HighPCAttrInfo.getAttribute(), dwarf::DW_AT_ranges,
                       RangesForm, DIEInteger(RangeAttrVal));
}
