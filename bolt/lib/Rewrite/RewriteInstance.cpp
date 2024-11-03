//===- bolt/Rewrite/RewriteInstance.cpp - ELF rewriter --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/RewriteInstance.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryEmitter.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/DebugData.h"
#include "bolt/Core/Exceptions.h"
#include "bolt/Core/FunctionLayout.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Core/Relocation.h"
#include "bolt/Passes/CacheMetrics.h"
#include "bolt/Passes/ReorderFunctions.h"
#include "bolt/Profile/BoltAddressTranslation.h"
#include "bolt/Profile/DataAggregator.h"
#include "bolt/Profile/DataReader.h"
#include "bolt/Profile/YAMLProfileReader.h"
#include "bolt/Profile/YAMLProfileWriter.h"
#include "bolt/Rewrite/BinaryPassManager.h"
#include "bolt/Rewrite/DWARFRewriter.h"
#include "bolt/Rewrite/ExecutableFileMemoryManager.h"
#include "bolt/RuntimeLibs/HugifyRuntimeLibrary.h"
#include "bolt/RuntimeLibs/InstrumentationRuntimeLibrary.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <fstream>
#include <memory>
#include <optional>
#include <system_error>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace object;
using namespace bolt;

extern cl::opt<uint32_t> X86AlignBranchBoundary;
extern cl::opt<bool> X86AlignBranchWithin32BBoundaries;

namespace opts {

extern cl::opt<MacroFusionType> AlignMacroOpFusion;
extern cl::list<std::string> HotTextMoveSections;
extern cl::opt<bool> Hugify;
extern cl::opt<bool> Instrument;
extern cl::opt<JumpTableSupportLevel> JumpTables;
extern cl::list<std::string> ReorderData;
extern cl::opt<bolt::ReorderFunctions::ReorderType> ReorderFunctions;
extern cl::opt<bool> TimeBuild;

static cl::opt<bool> ForceToDataRelocations(
    "force-data-relocations",
    cl::desc("force relocations to data sections to always be processed"),

    cl::Hidden, cl::cat(BoltCategory));

cl::opt<std::string>
    BoltID("bolt-id",
           cl::desc("add any string to tag this execution in the "
                    "output binary via bolt info section"),
           cl::cat(BoltCategory));

cl::opt<bool> DumpDotAll(
    "dump-dot-all",
    cl::desc("dump function CFGs to graphviz format after each stage;"
             "enable '-print-loops' for color-coded blocks"),
    cl::Hidden, cl::cat(BoltCategory));

static cl::list<std::string>
ForceFunctionNames("funcs",
  cl::CommaSeparated,
  cl::desc("limit optimizations to functions from the list"),
  cl::value_desc("func1,func2,func3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<std::string>
FunctionNamesFile("funcs-file",
  cl::desc("file with list of functions to optimize"),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::list<std::string> ForceFunctionNamesNR(
    "funcs-no-regex", cl::CommaSeparated,
    cl::desc("limit optimizations to functions from the list (non-regex)"),
    cl::value_desc("func1,func2,func3,..."), cl::Hidden, cl::cat(BoltCategory));

static cl::opt<std::string> FunctionNamesFileNR(
    "funcs-file-no-regex",
    cl::desc("file with list of functions to optimize (non-regex)"), cl::Hidden,
    cl::cat(BoltCategory));

cl::opt<bool>
KeepTmp("keep-tmp",
  cl::desc("preserve intermediate .o file"),
  cl::Hidden,
  cl::cat(BoltCategory));

cl::opt<bool> Lite("lite", cl::desc("skip processing of cold functions"),
                   cl::cat(BoltCategory));

static cl::opt<unsigned>
LiteThresholdPct("lite-threshold-pct",
  cl::desc("threshold (in percent) for selecting functions to process in lite "
            "mode. Higher threshold means fewer functions to process. E.g "
            "threshold of 90 means only top 10 percent of functions with "
            "profile will be processed."),
  cl::init(0),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned> LiteThresholdCount(
    "lite-threshold-count",
    cl::desc("similar to '-lite-threshold-pct' but specify threshold using "
             "absolute function call count. I.e. limit processing to functions "
             "executed at least the specified number of times."),
    cl::init(0), cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<unsigned>
    MaxFunctions("max-funcs",
                 cl::desc("maximum number of functions to process"), cl::Hidden,
                 cl::cat(BoltCategory));

static cl::opt<unsigned> MaxDataRelocations(
    "max-data-relocations",
    cl::desc("maximum number of data relocations to process"), cl::Hidden,
    cl::cat(BoltCategory));

cl::opt<bool> PrintAll("print-all",
                       cl::desc("print functions after each stage"), cl::Hidden,
                       cl::cat(BoltCategory));

cl::opt<bool> PrintCFG("print-cfg",
                       cl::desc("print functions after CFG construction"),
                       cl::Hidden, cl::cat(BoltCategory));

cl::opt<bool> PrintDisasm("print-disasm",
                          cl::desc("print function after disassembly"),
                          cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool>
    PrintGlobals("print-globals",
                 cl::desc("print global symbols after disassembly"), cl::Hidden,
                 cl::cat(BoltCategory));

extern cl::opt<bool> PrintSections;

static cl::opt<bool> PrintLoopInfo("print-loops",
                                   cl::desc("print loop related information"),
                                   cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool> PrintSDTMarkers("print-sdt",
                                     cl::desc("print all SDT markers"),
                                     cl::Hidden, cl::cat(BoltCategory));

enum PrintPseudoProbesOptions {
  PPP_None = 0,
  PPP_Probes_Section_Decode = 0x1,
  PPP_Probes_Address_Conversion = 0x2,
  PPP_Encoded_Probes = 0x3,
  PPP_All = 0xf
};

cl::opt<PrintPseudoProbesOptions> PrintPseudoProbes(
    "print-pseudo-probes", cl::desc("print pseudo probe info"),
    cl::init(PPP_None),
    cl::values(clEnumValN(PPP_Probes_Section_Decode, "decode",
                          "decode probes section from binary"),
               clEnumValN(PPP_Probes_Address_Conversion, "address_conversion",
                          "update address2ProbesMap with output block address"),
               clEnumValN(PPP_Encoded_Probes, "encoded_probes",
                          "display the encoded probes in binary section"),
               clEnumValN(PPP_All, "all", "enable all debugging printout")),
    cl::ZeroOrMore, cl::Hidden, cl::cat(BoltCategory));

static cl::opt<cl::boolOrDefault> RelocationMode(
    "relocs", cl::desc("use relocations in the binary (default=autodetect)"),
    cl::cat(BoltCategory));

static cl::opt<std::string>
SaveProfile("w",
  cl::desc("save recorded profile to a file"),
  cl::cat(BoltOutputCategory));

static cl::list<std::string>
SkipFunctionNames("skip-funcs",
  cl::CommaSeparated,
  cl::desc("list of functions to skip"),
  cl::value_desc("func1,func2,func3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<std::string>
SkipFunctionNamesFile("skip-funcs-file",
  cl::desc("file with list of functions to skip"),
  cl::Hidden,
  cl::cat(BoltCategory));

cl::opt<bool>
TrapOldCode("trap-old-code",
  cl::desc("insert traps in old function bodies (relocation mode)"),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<std::string> DWPPathName("dwp",
                                        cl::desc("Path and name to DWP file."),
                                        cl::Hidden, cl::init(""),
                                        cl::cat(BoltCategory));

static cl::opt<bool>
UseGnuStack("use-gnu-stack",
  cl::desc("use GNU_STACK program header for new segment (workaround for "
           "issues with strip/objcopy)"),
  cl::ZeroOrMore,
  cl::cat(BoltCategory));

static cl::opt<bool>
    TimeRewrite("time-rewrite",
                cl::desc("print time spent in rewriting passes"), cl::Hidden,
                cl::cat(BoltCategory));

static cl::opt<bool>
SequentialDisassembly("sequential-disassembly",
  cl::desc("performs disassembly sequentially"),
  cl::init(false),
  cl::cat(BoltOptCategory));

static cl::opt<bool> WriteBoltInfoSection(
    "bolt-info", cl::desc("write bolt info section in the output binary"),
    cl::init(true), cl::Hidden, cl::cat(BoltOutputCategory));

} // namespace opts

constexpr const char *RewriteInstance::SectionsToOverwrite[];
std::vector<std::string> RewriteInstance::DebugSectionsToOverwrite = {
    ".debug_abbrev", ".debug_aranges",  ".debug_line",   ".debug_line_str",
    ".debug_loc",    ".debug_loclists", ".debug_ranges", ".debug_rnglists",
    ".gdb_index",    ".debug_addr"};

const char RewriteInstance::TimerGroupName[] = "rewrite";
const char RewriteInstance::TimerGroupDesc[] = "Rewrite passes";

namespace llvm {
namespace bolt {

extern const char *BoltRevision;

MCPlusBuilder *createMCPlusBuilder(const Triple::ArchType Arch,
                                   const MCInstrAnalysis *Analysis,
                                   const MCInstrInfo *Info,
                                   const MCRegisterInfo *RegInfo) {
#ifdef X86_AVAILABLE
  if (Arch == Triple::x86_64)
    return createX86MCPlusBuilder(Analysis, Info, RegInfo);
#endif

#ifdef AARCH64_AVAILABLE
  if (Arch == Triple::aarch64)
    return createAArch64MCPlusBuilder(Analysis, Info, RegInfo);
#endif

  llvm_unreachable("architecture unsupported by MCPlusBuilder");
}

} // namespace bolt
} // namespace llvm

namespace {

bool refersToReorderedSection(ErrorOr<BinarySection &> Section) {
  return llvm::any_of(opts::ReorderData, [&](const std::string &SectionName) {
    return Section && Section->getName() == SectionName;
  });
}

} // anonymous namespace

Expected<std::unique_ptr<RewriteInstance>>
RewriteInstance::createRewriteInstance(ELFObjectFileBase *File, const int Argc,
                                       const char *const *Argv,
                                       StringRef ToolPath) {
  Error Err = Error::success();
  auto RI = std::make_unique<RewriteInstance>(File, Argc, Argv, ToolPath, Err);
  if (Err)
    return std::move(Err);
  return std::move(RI);
}

RewriteInstance::RewriteInstance(ELFObjectFileBase *File, const int Argc,
                                 const char *const *Argv, StringRef ToolPath,
                                 Error &Err)
    : InputFile(File), Argc(Argc), Argv(Argv), ToolPath(ToolPath),
      SHStrTab(StringTableBuilder::ELF) {
  ErrorAsOutParameter EAO(&Err);
  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(InputFile);
  if (!ELF64LEFile) {
    Err = createStringError(errc::not_supported,
                            "Only 64-bit LE ELF binaries are supported");
    return;
  }

  bool IsPIC = false;
  const ELFFile<ELF64LE> &Obj = ELF64LEFile->getELFFile();
  if (Obj.getHeader().e_type != ELF::ET_EXEC) {
    outs() << "BOLT-INFO: shared object or position-independent executable "
              "detected\n";
    IsPIC = true;
  }

  auto BCOrErr = BinaryContext::createBinaryContext(
      File, IsPIC,
      DWARFContext::create(*File, DWARFContext::ProcessDebugRelocations::Ignore,
                           nullptr, opts::DWPPathName,
                           WithColor::defaultErrorHandler,
                           WithColor::defaultWarningHandler));
  if (Error E = BCOrErr.takeError()) {
    Err = std::move(E);
    return;
  }
  BC = std::move(BCOrErr.get());
  BC->initializeTarget(std::unique_ptr<MCPlusBuilder>(createMCPlusBuilder(
      BC->TheTriple->getArch(), BC->MIA.get(), BC->MII.get(), BC->MRI.get())));

  BAT = std::make_unique<BoltAddressTranslation>();

  if (opts::UpdateDebugSections)
    DebugInfoRewriter = std::make_unique<DWARFRewriter>(*BC);

  if (opts::Instrument)
    BC->setRuntimeLibrary(std::make_unique<InstrumentationRuntimeLibrary>());
  else if (opts::Hugify)
    BC->setRuntimeLibrary(std::make_unique<HugifyRuntimeLibrary>());
}

RewriteInstance::~RewriteInstance() {}

Error RewriteInstance::setProfile(StringRef Filename) {
  if (!sys::fs::exists(Filename))
    return errorCodeToError(make_error_code(errc::no_such_file_or_directory));

  if (ProfileReader) {
    // Already exists
    return make_error<StringError>(Twine("multiple profiles specified: ") +
                                       ProfileReader->getFilename() + " and " +
                                       Filename,
                                   inconvertibleErrorCode());
  }

  // Spawn a profile reader based on file contents.
  if (DataAggregator::checkPerfDataMagic(Filename))
    ProfileReader = std::make_unique<DataAggregator>(Filename);
  else if (YAMLProfileReader::isYAML(Filename))
    ProfileReader = std::make_unique<YAMLProfileReader>(Filename);
  else
    ProfileReader = std::make_unique<DataReader>(Filename);

  return Error::success();
}

/// Return true if the function \p BF should be disassembled.
static bool shouldDisassemble(const BinaryFunction &BF) {
  if (BF.isPseudo())
    return false;

  if (opts::processAllFunctions())
    return true;

  return !BF.isIgnored();
}

Error RewriteInstance::discoverStorage() {
  NamedRegionTimer T("discoverStorage", "discover storage", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);

  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(InputFile);
  const ELFFile<ELF64LE> &Obj = ELF64LEFile->getELFFile();

  BC->StartFunctionAddress = Obj.getHeader().e_entry;

  NextAvailableAddress = 0;
  uint64_t NextAvailableOffset = 0;
  Expected<ELF64LE::PhdrRange> PHsOrErr = Obj.program_headers();
  if (Error E = PHsOrErr.takeError())
    return E;

  ELF64LE::PhdrRange PHs = PHsOrErr.get();
  for (const ELF64LE::Phdr &Phdr : PHs) {
    switch (Phdr.p_type) {
    case ELF::PT_LOAD:
      BC->FirstAllocAddress = std::min(BC->FirstAllocAddress,
                                       static_cast<uint64_t>(Phdr.p_vaddr));
      NextAvailableAddress = std::max(NextAvailableAddress,
                                      Phdr.p_vaddr + Phdr.p_memsz);
      NextAvailableOffset = std::max(NextAvailableOffset,
                                     Phdr.p_offset + Phdr.p_filesz);

      BC->SegmentMapInfo[Phdr.p_vaddr] = SegmentInfo{Phdr.p_vaddr,
                                                     Phdr.p_memsz,
                                                     Phdr.p_offset,
                                                     Phdr.p_filesz,
                                                     Phdr.p_align};
      break;
    case ELF::PT_INTERP:
      BC->HasInterpHeader = true;
      break;
    }
  }

  for (const SectionRef &Section : InputFile->sections()) {
    Expected<StringRef> SectionNameOrErr = Section.getName();
    if (Error E = SectionNameOrErr.takeError())
      return E;
    StringRef SectionName = SectionNameOrErr.get();
    if (SectionName == ".text") {
      BC->OldTextSectionAddress = Section.getAddress();
      BC->OldTextSectionSize = Section.getSize();

      Expected<StringRef> SectionContentsOrErr = Section.getContents();
      if (Error E = SectionContentsOrErr.takeError())
        return E;
      StringRef SectionContents = SectionContentsOrErr.get();
      BC->OldTextSectionOffset =
          SectionContents.data() - InputFile->getData().data();
    }

    if (!opts::HeatmapMode &&
        !(opts::AggregateOnly && BAT->enabledFor(InputFile)) &&
        (SectionName.startswith(getOrgSecPrefix()) ||
         SectionName == getBOLTTextSectionName()))
      return createStringError(
          errc::function_not_supported,
          "BOLT-ERROR: input file was processed by BOLT. Cannot re-optimize");
  }

  if (!NextAvailableAddress || !NextAvailableOffset)
    return createStringError(errc::executable_format_error,
                             "no PT_LOAD pheader seen");

  outs() << "BOLT-INFO: first alloc address is 0x"
         << Twine::utohexstr(BC->FirstAllocAddress) << '\n';

  FirstNonAllocatableOffset = NextAvailableOffset;

  NextAvailableAddress = alignTo(NextAvailableAddress, BC->PageAlign);
  NextAvailableOffset = alignTo(NextAvailableOffset, BC->PageAlign);

  // Hugify: Additional huge page from left side due to
  // weird ASLR mapping addresses (4KB aligned)
  if (opts::Hugify && !BC->HasFixedLoadAddress)
    NextAvailableAddress += BC->PageAlign;

  if (!opts::UseGnuStack) {
    // This is where the black magic happens. Creating PHDR table in a segment
    // other than that containing ELF header is tricky. Some loaders and/or
    // parts of loaders will apply e_phoff from ELF header assuming both are in
    // the same segment, while others will do the proper calculation.
    // We create the new PHDR table in such a way that both of the methods
    // of loading and locating the table work. There's a slight file size
    // overhead because of that.
    //
    // NB: bfd's strip command cannot do the above and will corrupt the
    //     binary during the process of stripping non-allocatable sections.
    if (NextAvailableOffset <= NextAvailableAddress - BC->FirstAllocAddress)
      NextAvailableOffset = NextAvailableAddress - BC->FirstAllocAddress;
    else
      NextAvailableAddress = NextAvailableOffset + BC->FirstAllocAddress;

    assert(NextAvailableOffset ==
               NextAvailableAddress - BC->FirstAllocAddress &&
           "PHDR table address calculation error");

    outs() << "BOLT-INFO: creating new program header table at address 0x"
           << Twine::utohexstr(NextAvailableAddress) << ", offset 0x"
           << Twine::utohexstr(NextAvailableOffset) << '\n';

    PHDRTableAddress = NextAvailableAddress;
    PHDRTableOffset = NextAvailableOffset;

    // Reserve space for 3 extra pheaders.
    unsigned Phnum = Obj.getHeader().e_phnum;
    Phnum += 3;

    NextAvailableAddress += Phnum * sizeof(ELF64LEPhdrTy);
    NextAvailableOffset += Phnum * sizeof(ELF64LEPhdrTy);
  }

  // Align at cache line.
  NextAvailableAddress = alignTo(NextAvailableAddress, 64);
  NextAvailableOffset = alignTo(NextAvailableOffset, 64);

  NewTextSegmentAddress = NextAvailableAddress;
  NewTextSegmentOffset = NextAvailableOffset;
  BC->LayoutStartAddress = NextAvailableAddress;

  // Tools such as objcopy can strip section contents but leave header
  // entries. Check that at least .text is mapped in the file.
  if (!getFileOffsetForAddress(BC->OldTextSectionAddress))
    return createStringError(errc::executable_format_error,
                             "BOLT-ERROR: input binary is not a valid ELF "
                             "executable as its text section is not "
                             "mapped to a valid segment");
  return Error::success();
}

void RewriteInstance::parseSDTNotes() {
  if (!SDTSection)
    return;

  StringRef Buf = SDTSection->getContents();
  DataExtractor DE = DataExtractor(Buf, BC->AsmInfo->isLittleEndian(),
                                   BC->AsmInfo->getCodePointerSize());
  uint64_t Offset = 0;

  while (DE.isValidOffset(Offset)) {
    uint32_t NameSz = DE.getU32(&Offset);
    DE.getU32(&Offset); // skip over DescSz
    uint32_t Type = DE.getU32(&Offset);
    Offset = alignTo(Offset, 4);

    if (Type != 3)
      errs() << "BOLT-WARNING: SDT note type \"" << Type
             << "\" is not expected\n";

    if (NameSz == 0)
      errs() << "BOLT-WARNING: SDT note has empty name\n";

    StringRef Name = DE.getCStr(&Offset);

    if (!Name.equals("stapsdt"))
      errs() << "BOLT-WARNING: SDT note name \"" << Name
             << "\" is not expected\n";

    // Parse description
    SDTMarkerInfo Marker;
    Marker.PCOffset = Offset;
    Marker.PC = DE.getU64(&Offset);
    Marker.Base = DE.getU64(&Offset);
    Marker.Semaphore = DE.getU64(&Offset);
    Marker.Provider = DE.getCStr(&Offset);
    Marker.Name = DE.getCStr(&Offset);
    Marker.Args = DE.getCStr(&Offset);
    Offset = alignTo(Offset, 4);
    BC->SDTMarkers[Marker.PC] = Marker;
  }

  if (opts::PrintSDTMarkers)
    printSDTMarkers();
}

void RewriteInstance::parsePseudoProbe() {
  if (!PseudoProbeDescSection && !PseudoProbeSection) {
    // pesudo probe is not added to binary. It is normal and no warning needed.
    return;
  }

  // If only one section is found, it might mean the ELF is corrupted.
  if (!PseudoProbeDescSection) {
    errs() << "BOLT-WARNING: fail in reading .pseudo_probe_desc binary\n";
    return;
  } else if (!PseudoProbeSection) {
    errs() << "BOLT-WARNING: fail in reading .pseudo_probe binary\n";
    return;
  }

  StringRef Contents = PseudoProbeDescSection->getContents();
  if (!BC->ProbeDecoder.buildGUID2FuncDescMap(
          reinterpret_cast<const uint8_t *>(Contents.data()),
          Contents.size())) {
    errs() << "BOLT-WARNING: fail in building GUID2FuncDescMap\n";
    return;
  }

  MCPseudoProbeDecoder::Uint64Set GuidFilter;
  MCPseudoProbeDecoder::Uint64Map FuncStartAddrs;
  for (const BinaryFunction *F : BC->getAllBinaryFunctions()) {
    for (const MCSymbol *Sym : F->getSymbols()) {
      FuncStartAddrs[Function::getGUID(NameResolver::restore(Sym->getName()))] =
          F->getAddress();
    }
  }
  Contents = PseudoProbeSection->getContents();
  if (!BC->ProbeDecoder.buildAddress2ProbeMap(
          reinterpret_cast<const uint8_t *>(Contents.data()), Contents.size(),
          GuidFilter, FuncStartAddrs)) {
    BC->ProbeDecoder.getAddress2ProbesMap().clear();
    errs() << "BOLT-WARNING: fail in building Address2ProbeMap\n";
    return;
  }

  if (opts::PrintPseudoProbes == opts::PrintPseudoProbesOptions::PPP_All ||
      opts::PrintPseudoProbes ==
          opts::PrintPseudoProbesOptions::PPP_Probes_Section_Decode) {
    outs() << "Report of decoding input pseudo probe binaries \n";
    BC->ProbeDecoder.printGUID2FuncDescMap(outs());
    BC->ProbeDecoder.printProbesForAllAddresses(outs());
  }
}

void RewriteInstance::printSDTMarkers() {
  outs() << "BOLT-INFO: Number of SDT markers is " << BC->SDTMarkers.size()
         << "\n";
  for (auto It : BC->SDTMarkers) {
    SDTMarkerInfo &Marker = It.second;
    outs() << "BOLT-INFO: PC: " << utohexstr(Marker.PC)
           << ", Base: " << utohexstr(Marker.Base)
           << ", Semaphore: " << utohexstr(Marker.Semaphore)
           << ", Provider: " << Marker.Provider << ", Name: " << Marker.Name
           << ", Args: " << Marker.Args << "\n";
  }
}

void RewriteInstance::parseBuildID() {
  if (!BuildIDSection)
    return;

  StringRef Buf = BuildIDSection->getContents();

  // Reading notes section (see Portable Formats Specification, Version 1.1,
  // pg 2-5, section "Note Section").
  DataExtractor DE = DataExtractor(Buf, true, 8);
  uint64_t Offset = 0;
  if (!DE.isValidOffset(Offset))
    return;
  uint32_t NameSz = DE.getU32(&Offset);
  if (!DE.isValidOffset(Offset))
    return;
  uint32_t DescSz = DE.getU32(&Offset);
  if (!DE.isValidOffset(Offset))
    return;
  uint32_t Type = DE.getU32(&Offset);

  LLVM_DEBUG(dbgs() << "NameSz = " << NameSz << "; DescSz = " << DescSz
                    << "; Type = " << Type << "\n");

  // Type 3 is a GNU build-id note section
  if (Type != 3)
    return;

  StringRef Name = Buf.slice(Offset, Offset + NameSz);
  Offset = alignTo(Offset + NameSz, 4);
  if (Name.substr(0, 3) != "GNU")
    return;

  BuildID = Buf.slice(Offset, Offset + DescSz);
}

std::optional<std::string> RewriteInstance::getPrintableBuildID() const {
  if (BuildID.empty())
    return std::nullopt;

  std::string Str;
  raw_string_ostream OS(Str);
  const unsigned char *CharIter = BuildID.bytes_begin();
  while (CharIter != BuildID.bytes_end()) {
    if (*CharIter < 0x10)
      OS << "0";
    OS << Twine::utohexstr(*CharIter);
    ++CharIter;
  }
  return OS.str();
}

void RewriteInstance::patchBuildID() {
  raw_fd_ostream &OS = Out->os();

  if (BuildID.empty())
    return;

  size_t IDOffset = BuildIDSection->getContents().rfind(BuildID);
  assert(IDOffset != StringRef::npos && "failed to patch build-id");

  uint64_t FileOffset = getFileOffsetForAddress(BuildIDSection->getAddress());
  if (!FileOffset) {
    errs() << "BOLT-WARNING: Non-allocatable build-id will not be updated.\n";
    return;
  }

  char LastIDByte = BuildID[BuildID.size() - 1];
  LastIDByte ^= 1;
  OS.pwrite(&LastIDByte, 1, FileOffset + IDOffset + BuildID.size() - 1);

  outs() << "BOLT-INFO: patched build-id (flipped last bit)\n";
}

Error RewriteInstance::run() {
  assert(BC && "failed to create a binary context");

  outs() << "BOLT-INFO: Target architecture: "
         << Triple::getArchTypeName(
                (llvm::Triple::ArchType)InputFile->getArch())
         << "\n";
  outs() << "BOLT-INFO: BOLT version: " << BoltRevision << "\n";

  if (Error E = discoverStorage())
    return E;
  if (Error E = readSpecialSections())
    return E;
  adjustCommandLineOptions();
  discoverFileObjects();

  preprocessProfileData();

  // Skip disassembling if we have a translation table and we are running an
  // aggregation job.
  if (opts::AggregateOnly && BAT->enabledFor(InputFile)) {
    processProfileData();
    return Error::success();
  }

  selectFunctionsToProcess();

  readDebugInfo();

  disassembleFunctions();

  processProfileDataPreCFG();

  buildFunctionsCFG();

  processProfileData();

  postProcessFunctions();

  if (opts::DiffOnly)
    return Error::success();

  preregisterSections();

  runOptimizationPasses();

  emitAndLink();

  updateMetadata();

  if (opts::LinuxKernelMode) {
    errs() << "BOLT-WARNING: not writing the output file for Linux Kernel\n";
    return Error::success();
  } else if (opts::OutputFilename == "/dev/null") {
    outs() << "BOLT-INFO: skipping writing final binary to disk\n";
    return Error::success();
  }

  // Rewrite allocatable contents and copy non-allocatable parts with mods.
  rewriteFile();
  return Error::success();
}

void RewriteInstance::discoverFileObjects() {
  NamedRegionTimer T("discoverFileObjects", "discover file objects",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  FileSymRefs.clear();
  BC->getBinaryFunctions().clear();
  BC->clearBinaryData();

  // For local symbols we want to keep track of associated FILE symbol name for
  // disambiguation by combined name.
  StringRef FileSymbolName;
  bool SeenFileName = false;
  struct SymbolRefHash {
    size_t operator()(SymbolRef const &S) const {
      return std::hash<decltype(DataRefImpl::p)>{}(S.getRawDataRefImpl().p);
    }
  };
  std::unordered_map<SymbolRef, StringRef, SymbolRefHash> SymbolToFileName;
  for (const ELFSymbolRef &Symbol : InputFile->symbols()) {
    Expected<StringRef> NameOrError = Symbol.getName();
    if (NameOrError && NameOrError->startswith("__asan_init")) {
      errs() << "BOLT-ERROR: input file was compiled or linked with sanitizer "
                "support. Cannot optimize.\n";
      exit(1);
    }
    if (NameOrError && NameOrError->startswith("__llvm_coverage_mapping")) {
      errs() << "BOLT-ERROR: input file was compiled or linked with coverage "
                "support. Cannot optimize.\n";
      exit(1);
    }

    if (cantFail(Symbol.getFlags()) & SymbolRef::SF_Undefined)
      continue;

    if (cantFail(Symbol.getType()) == SymbolRef::ST_File) {
      StringRef Name =
          cantFail(std::move(NameOrError), "cannot get symbol name for file");
      // Ignore Clang LTO artificial FILE symbol as it is not always generated,
      // and this uncertainty is causing havoc in function name matching.
      if (Name == "ld-temp.o")
        continue;
      FileSymbolName = Name;
      SeenFileName = true;
      continue;
    }
    if (!FileSymbolName.empty() &&
        !(cantFail(Symbol.getFlags()) & SymbolRef::SF_Global))
      SymbolToFileName[Symbol] = FileSymbolName;
  }

  // Sort symbols in the file by value. Ignore symbols from non-allocatable
  // sections.
  auto isSymbolInMemory = [this](const SymbolRef &Sym) {
    if (cantFail(Sym.getType()) == SymbolRef::ST_File)
      return false;
    if (cantFail(Sym.getFlags()) & SymbolRef::SF_Absolute)
      return true;
    if (cantFail(Sym.getFlags()) & SymbolRef::SF_Undefined)
      return false;
    BinarySection Section(*BC, *cantFail(Sym.getSection()));
    return Section.isAllocatable();
  };
  std::vector<SymbolRef> SortedFileSymbols;
  llvm::copy_if(InputFile->symbols(), std::back_inserter(SortedFileSymbols),
                isSymbolInMemory);
  auto CompareSymbols = [this](const SymbolRef &A, const SymbolRef &B) {
    // Marker symbols have the highest precedence, while
    // SECTIONs have the lowest.
    auto AddressA = cantFail(A.getAddress());
    auto AddressB = cantFail(B.getAddress());
    if (AddressA != AddressB)
      return AddressA < AddressB;

    bool AMarker = BC->isMarker(A);
    bool BMarker = BC->isMarker(B);
    if (AMarker || BMarker) {
      return AMarker && !BMarker;
    }

    auto AType = cantFail(A.getType());
    auto BType = cantFail(B.getType());
    if (AType == SymbolRef::ST_Function && BType != SymbolRef::ST_Function)
      return true;
    if (BType == SymbolRef::ST_Debug && AType != SymbolRef::ST_Debug)
      return true;

    return false;
  };

  llvm::stable_sort(SortedFileSymbols, CompareSymbols);

  auto LastSymbol = SortedFileSymbols.end();
  if (!SortedFileSymbols.empty())
    --LastSymbol;

  // For aarch64, the ABI defines mapping symbols so we identify data in the
  // code section (see IHI0056B). $d identifies data contents.
  // Compilers usually merge multiple data objects in a single $d-$x interval,
  // but we need every data object to be marked with $d. Because of that we
  // create a vector of MarkerSyms with all locations of data objects.

  struct MarkerSym {
    uint64_t Address;
    MarkerSymType Type;
  };

  std::vector<MarkerSym> SortedMarkerSymbols;
  auto addExtraDataMarkerPerSymbol =
      [this](const std::vector<SymbolRef> &SortedFileSymbols,
             std::vector<MarkerSym> &SortedMarkerSymbols) {
        bool IsData = false;
        uint64_t LastAddr = 0;
        for (auto Sym = SortedFileSymbols.begin();
             Sym < SortedFileSymbols.end(); ++Sym) {
          uint64_t Address = cantFail(Sym->getAddress());
          if (LastAddr == Address) // don't repeat markers
            continue;

          MarkerSymType MarkerType = BC->getMarkerType(*Sym);
          if (MarkerType != MarkerSymType::NONE) {
            SortedMarkerSymbols.push_back(MarkerSym{Address, MarkerType});
            LastAddr = Address;
            IsData = MarkerType == MarkerSymType::DATA;
            continue;
          }

          if (IsData) {
            SortedMarkerSymbols.push_back(
                MarkerSym{cantFail(Sym->getAddress()), MarkerSymType::DATA});
            LastAddr = Address;
          }
        }
      };

  if (BC->isAArch64()) {
    addExtraDataMarkerPerSymbol(SortedFileSymbols, SortedMarkerSymbols);
    LastSymbol = std::stable_partition(
        SortedFileSymbols.begin(), SortedFileSymbols.end(),
        [this](const SymbolRef &Symbol) { return !BC->isMarker(Symbol); });
    if (!SortedFileSymbols.empty())
      --LastSymbol;
  }

  BinaryFunction *PreviousFunction = nullptr;
  unsigned AnonymousId = 0;

  const auto SortedSymbolsEnd = LastSymbol == SortedFileSymbols.end()
                                    ? LastSymbol
                                    : std::next(LastSymbol);
  for (auto ISym = SortedFileSymbols.begin(); ISym != SortedSymbolsEnd;
       ++ISym) {
    const SymbolRef &Symbol = *ISym;
    // Keep undefined symbols for pretty printing?
    if (cantFail(Symbol.getFlags()) & SymbolRef::SF_Undefined)
      continue;

    const SymbolRef::Type SymbolType = cantFail(Symbol.getType());

    if (SymbolType == SymbolRef::ST_File)
      continue;

    StringRef SymName = cantFail(Symbol.getName(), "cannot get symbol name");
    uint64_t Address =
        cantFail(Symbol.getAddress(), "cannot get symbol address");
    if (Address == 0) {
      if (opts::Verbosity >= 1 && SymbolType == SymbolRef::ST_Function)
        errs() << "BOLT-WARNING: function with 0 address seen\n";
      continue;
    }

    // Ignore input hot markers
    if (SymName == "__hot_start" || SymName == "__hot_end")
      continue;

    FileSymRefs[Address] = Symbol;

    // Skip section symbols that will be registered by disassemblePLT().
    if ((cantFail(Symbol.getType()) == SymbolRef::ST_Debug)) {
      ErrorOr<BinarySection &> BSection = BC->getSectionForAddress(Address);
      if (BSection && getPLTSectionInfo(BSection->getName()))
        continue;
    }

    /// It is possible we are seeing a globalized local. LLVM might treat it as
    /// a local if it has a "private global" prefix, e.g. ".L". Thus we have to
    /// change the prefix to enforce global scope of the symbol.
    std::string Name = SymName.startswith(BC->AsmInfo->getPrivateGlobalPrefix())
                           ? "PG" + std::string(SymName)
                           : std::string(SymName);

    // Disambiguate all local symbols before adding to symbol table.
    // Since we don't know if we will see a global with the same name,
    // always modify the local name.
    //
    // NOTE: the naming convention for local symbols should match
    //       the one we use for profile data.
    std::string UniqueName;
    std::string AlternativeName;
    if (Name.empty()) {
      UniqueName = "ANONYMOUS." + std::to_string(AnonymousId++);
    } else if (cantFail(Symbol.getFlags()) & SymbolRef::SF_Global) {
      if (const BinaryData *BD = BC->getBinaryDataByName(Name)) {
        if (BD->getSize() == ELFSymbolRef(Symbol).getSize() &&
            BD->getAddress() == Address) {
          if (opts::Verbosity > 1)
            errs() << "BOLT-WARNING: ignoring duplicate global symbol " << Name
                   << "\n";
          // Ignore duplicate entry - possibly a bug in the linker
          continue;
        }
        errs() << "BOLT-ERROR: bad input binary, global symbol \"" << Name
               << "\" is not unique\n";
        exit(1);
      }
      UniqueName = Name;
    } else {
      // If we have a local file name, we should create 2 variants for the
      // function name. The reason is that perf profile might have been
      // collected on a binary that did not have the local file name (e.g. as
      // a side effect of stripping debug info from the binary):
      //
      //   primary:     <function>/<id>
      //   alternative: <function>/<file>/<id2>
      //
      // The <id> field is used for disambiguation of local symbols since there
      // could be identical function names coming from identical file names
      // (e.g. from different directories).
      std::string AltPrefix;
      auto SFI = SymbolToFileName.find(Symbol);
      if (SymbolType == SymbolRef::ST_Function && SFI != SymbolToFileName.end())
        AltPrefix = Name + "/" + std::string(SFI->second);

      UniqueName = NR.uniquify(Name);
      if (!AltPrefix.empty())
        AlternativeName = NR.uniquify(AltPrefix);
    }

    uint64_t SymbolSize = ELFSymbolRef(Symbol).getSize();
    uint64_t SymbolAlignment = Symbol.getAlignment();
    unsigned SymbolFlags = cantFail(Symbol.getFlags());

    auto registerName = [&](uint64_t FinalSize) {
      // Register names even if it's not a function, e.g. for an entry point.
      BC->registerNameAtAddress(UniqueName, Address, FinalSize, SymbolAlignment,
                                SymbolFlags);
      if (!AlternativeName.empty())
        BC->registerNameAtAddress(AlternativeName, Address, FinalSize,
                                  SymbolAlignment, SymbolFlags);
    };

    section_iterator Section =
        cantFail(Symbol.getSection(), "cannot get symbol section");
    if (Section == InputFile->section_end()) {
      // Could be an absolute symbol. Could record for pretty printing.
      LLVM_DEBUG(if (opts::Verbosity > 1) {
        dbgs() << "BOLT-INFO: absolute sym " << UniqueName << "\n";
      });
      registerName(SymbolSize);
      continue;
    }

    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: considering symbol " << UniqueName
                      << " for function\n");

    if (!Section->isText()) {
      assert(SymbolType != SymbolRef::ST_Function &&
             "unexpected function inside non-code section");
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: rejecting as symbol is not in code\n");
      registerName(SymbolSize);
      continue;
    }

    // Assembly functions could be ST_NONE with 0 size. Check that the
    // corresponding section is a code section and they are not inside any
    // other known function to consider them.
    //
    // Sometimes assembly functions are not marked as functions and neither are
    // their local labels. The only way to tell them apart is to look at
    // symbol scope - global vs local.
    if (PreviousFunction && SymbolType != SymbolRef::ST_Function) {
      if (PreviousFunction->containsAddress(Address)) {
        if (PreviousFunction->isSymbolValidInScope(Symbol, SymbolSize)) {
          LLVM_DEBUG(dbgs()
                     << "BOLT-DEBUG: symbol is a function local symbol\n");
        } else if (Address == PreviousFunction->getAddress() && !SymbolSize) {
          LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring symbol as a marker\n");
        } else if (opts::Verbosity > 1) {
          errs() << "BOLT-WARNING: symbol " << UniqueName
                 << " seen in the middle of function " << *PreviousFunction
                 << ". Could be a new entry.\n";
        }
        registerName(SymbolSize);
        continue;
      } else if (PreviousFunction->getSize() == 0 &&
                 PreviousFunction->isSymbolValidInScope(Symbol, SymbolSize)) {
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: symbol is a function local symbol\n");
        registerName(SymbolSize);
        continue;
      }
    }

    if (PreviousFunction && PreviousFunction->containsAddress(Address) &&
        PreviousFunction->getAddress() != Address) {
      if (PreviousFunction->isSymbolValidInScope(Symbol, SymbolSize)) {
        if (opts::Verbosity >= 1)
          outs() << "BOLT-INFO: skipping possibly another entry for function "
                 << *PreviousFunction << " : " << UniqueName << '\n';
        registerName(SymbolSize);
      } else {
        outs() << "BOLT-INFO: using " << UniqueName << " as another entry to "
               << "function " << *PreviousFunction << '\n';

        registerName(0);

        PreviousFunction->addEntryPointAtOffset(Address -
                                                PreviousFunction->getAddress());

        // Remove the symbol from FileSymRefs so that we can skip it from
        // in the future.
        auto SI = FileSymRefs.find(Address);
        assert(SI != FileSymRefs.end() && "symbol expected to be present");
        assert(SI->second == Symbol && "wrong symbol found");
        FileSymRefs.erase(SI);
      }
      continue;
    }

    // Checkout for conflicts with function data from FDEs.
    bool IsSimple = true;
    auto FDEI = CFIRdWrt->getFDEs().lower_bound(Address);
    if (FDEI != CFIRdWrt->getFDEs().end()) {
      const dwarf::FDE &FDE = *FDEI->second;
      if (FDEI->first != Address) {
        // There's no matching starting address in FDE. Make sure the previous
        // FDE does not contain this address.
        if (FDEI != CFIRdWrt->getFDEs().begin()) {
          --FDEI;
          const dwarf::FDE &PrevFDE = *FDEI->second;
          uint64_t PrevStart = PrevFDE.getInitialLocation();
          uint64_t PrevLength = PrevFDE.getAddressRange();
          if (Address > PrevStart && Address < PrevStart + PrevLength) {
            errs() << "BOLT-ERROR: function " << UniqueName
                   << " is in conflict with FDE ["
                   << Twine::utohexstr(PrevStart) << ", "
                   << Twine::utohexstr(PrevStart + PrevLength)
                   << "). Skipping.\n";
            IsSimple = false;
          }
        }
      } else if (FDE.getAddressRange() != SymbolSize) {
        if (SymbolSize) {
          // Function addresses match but sizes differ.
          errs() << "BOLT-WARNING: sizes differ for function " << UniqueName
                 << ". FDE : " << FDE.getAddressRange()
                 << "; symbol table : " << SymbolSize << ". Using max size.\n";
        }
        SymbolSize = std::max(SymbolSize, FDE.getAddressRange());
        if (BC->getBinaryDataAtAddress(Address)) {
          BC->setBinaryDataSize(Address, SymbolSize);
        } else {
          LLVM_DEBUG(dbgs() << "BOLT-DEBUG: No BD @ 0x"
                            << Twine::utohexstr(Address) << "\n");
        }
      }
    }

    BinaryFunction *BF = nullptr;
    // Since function may not have yet obtained its real size, do a search
    // using the list of registered functions instead of calling
    // getBinaryFunctionAtAddress().
    auto BFI = BC->getBinaryFunctions().find(Address);
    if (BFI != BC->getBinaryFunctions().end()) {
      BF = &BFI->second;
      // Duplicate the function name. Make sure everything matches before we add
      // an alternative name.
      if (SymbolSize != BF->getSize()) {
        if (opts::Verbosity >= 1) {
          if (SymbolSize && BF->getSize())
            errs() << "BOLT-WARNING: size mismatch for duplicate entries "
                   << *BF << " and " << UniqueName << '\n';
          outs() << "BOLT-INFO: adjusting size of function " << *BF << " old "
                 << BF->getSize() << " new " << SymbolSize << "\n";
        }
        BF->setSize(std::max(SymbolSize, BF->getSize()));
        BC->setBinaryDataSize(Address, BF->getSize());
      }
      BF->addAlternativeName(UniqueName);
    } else {
      ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
      // Skip symbols from invalid sections
      if (!Section) {
        errs() << "BOLT-WARNING: " << UniqueName << " (0x"
               << Twine::utohexstr(Address) << ") does not have any section\n";
        continue;
      }

      // Skip symbols from zero-sized sections.
      if (!Section->getSize())
        continue;

      BF = BC->createBinaryFunction(UniqueName, *Section, Address, SymbolSize);
      if (!IsSimple)
        BF->setSimple(false);
    }
    if (!AlternativeName.empty())
      BF->addAlternativeName(AlternativeName);

    registerName(SymbolSize);
    PreviousFunction = BF;
  }

  // Read dynamic relocation first as their presence affects the way we process
  // static relocations. E.g. we will ignore a static relocation at an address
  // that is a subject to dynamic relocation processing.
  processDynamicRelocations();

  // Process PLT section.
  disassemblePLT();

  // See if we missed any functions marked by FDE.
  for (const auto &FDEI : CFIRdWrt->getFDEs()) {
    const uint64_t Address = FDEI.first;
    const dwarf::FDE *FDE = FDEI.second;
    const BinaryFunction *BF = BC->getBinaryFunctionAtAddress(Address);
    if (BF)
      continue;

    BF = BC->getBinaryFunctionContainingAddress(Address);
    if (BF) {
      errs() << "BOLT-WARNING: FDE [0x" << Twine::utohexstr(Address) << ", 0x"
             << Twine::utohexstr(Address + FDE->getAddressRange())
             << ") conflicts with function " << *BF << '\n';
      continue;
    }

    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: FDE [0x" << Twine::utohexstr(Address) << ", 0x"
             << Twine::utohexstr(Address + FDE->getAddressRange())
             << ") has no corresponding symbol table entry\n";

    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
    assert(Section && "cannot get section for address from FDE");
    std::string FunctionName =
        "__BOLT_FDE_FUNCat" + Twine::utohexstr(Address).str();
    BC->createBinaryFunction(FunctionName, *Section, Address,
                             FDE->getAddressRange());
  }

  BC->setHasSymbolsWithFileName(SeenFileName);

  // Now that all the functions were created - adjust their boundaries.
  adjustFunctionBoundaries();

  // Annotate functions with code/data markers in AArch64
  for (auto ISym = SortedMarkerSymbols.begin();
       ISym != SortedMarkerSymbols.end(); ++ISym) {

    auto *BF =
        BC->getBinaryFunctionContainingAddress(ISym->Address, true, true);

    if (!BF) {
      // Stray marker
      continue;
    }
    const auto EntryOffset = ISym->Address - BF->getAddress();
    if (ISym->Type == MarkerSymType::CODE) {
      BF->markCodeAtOffset(EntryOffset);
      continue;
    }
    if (ISym->Type == MarkerSymType::DATA) {
      BF->markDataAtOffset(EntryOffset);
      BC->AddressToConstantIslandMap[ISym->Address] = BF;
      continue;
    }
    llvm_unreachable("Unknown marker");
  }

  if (opts::LinuxKernelMode) {
    // Read all special linux kernel sections and their relocations
    processLKSections();
  } else {
    // Read all relocations now that we have binary functions mapped.
    processRelocations();
  }
}

void RewriteInstance::createPLTBinaryFunction(uint64_t TargetAddress,
                                              uint64_t EntryAddress,
                                              uint64_t EntrySize) {
  if (!TargetAddress)
    return;

  auto setPLTSymbol = [&](BinaryFunction *BF, StringRef Name) {
    const unsigned PtrSize = BC->AsmInfo->getCodePointerSize();
    MCSymbol *TargetSymbol = BC->registerNameAtAddress(
        Name.str() + "@GOT", TargetAddress, PtrSize, PtrSize);
    BF->setPLTSymbol(TargetSymbol);
  };

  BinaryFunction *BF = BC->getBinaryFunctionAtAddress(EntryAddress);
  if (BF && BC->isAArch64()) {
    // Handle IFUNC trampoline
    setPLTSymbol(BF, BF->getOneName());
    return;
  }

  const Relocation *Rel = BC->getDynamicRelocationAt(TargetAddress);
  if (!Rel || !Rel->Symbol)
    return;

  ErrorOr<BinarySection &> Section = BC->getSectionForAddress(EntryAddress);
  assert(Section && "cannot get section for address");
  if (!BF)
    BF = BC->createBinaryFunction(Rel->Symbol->getName().str() + "@PLT",
                                  *Section, EntryAddress, 0, EntrySize,
                                  Section->getAlignment());
  else
    BF->addAlternativeName(Rel->Symbol->getName().str() + "@PLT");
  setPLTSymbol(BF, Rel->Symbol->getName());
}

void RewriteInstance::disassemblePLTSectionAArch64(BinarySection &Section) {
  const uint64_t SectionAddress = Section.getAddress();
  const uint64_t SectionSize = Section.getSize();
  StringRef PLTContents = Section.getContents();
  ArrayRef<uint8_t> PLTData(
      reinterpret_cast<const uint8_t *>(PLTContents.data()), SectionSize);

  auto disassembleInstruction = [&](uint64_t InstrOffset, MCInst &Instruction,
                                    uint64_t &InstrSize) {
    const uint64_t InstrAddr = SectionAddress + InstrOffset;
    if (!BC->DisAsm->getInstruction(Instruction, InstrSize,
                                    PLTData.slice(InstrOffset), InstrAddr,
                                    nulls())) {
      errs() << "BOLT-ERROR: unable to disassemble instruction in PLT section "
             << Section.getName() << " at offset 0x"
             << Twine::utohexstr(InstrOffset) << '\n';
      exit(1);
    }
  };

  uint64_t InstrOffset = 0;
  // Locate new plt entry
  while (InstrOffset < SectionSize) {
    InstructionListType Instructions;
    MCInst Instruction;
    uint64_t EntryOffset = InstrOffset;
    uint64_t EntrySize = 0;
    uint64_t InstrSize;
    // Loop through entry instructions
    while (InstrOffset < SectionSize) {
      disassembleInstruction(InstrOffset, Instruction, InstrSize);
      EntrySize += InstrSize;
      if (!BC->MIB->isIndirectBranch(Instruction)) {
        Instructions.emplace_back(Instruction);
        InstrOffset += InstrSize;
        continue;
      }

      const uint64_t EntryAddress = SectionAddress + EntryOffset;
      const uint64_t TargetAddress = BC->MIB->analyzePLTEntry(
          Instruction, Instructions.begin(), Instructions.end(), EntryAddress);

      createPLTBinaryFunction(TargetAddress, EntryAddress, EntrySize);
      break;
    }

    // Branch instruction
    InstrOffset += InstrSize;

    // Skip nops if any
    while (InstrOffset < SectionSize) {
      disassembleInstruction(InstrOffset, Instruction, InstrSize);
      if (!BC->MIB->isNoop(Instruction))
        break;

      InstrOffset += InstrSize;
    }
  }
}

void RewriteInstance::disassemblePLTSectionX86(BinarySection &Section,
                                               uint64_t EntrySize) {
  const uint64_t SectionAddress = Section.getAddress();
  const uint64_t SectionSize = Section.getSize();
  StringRef PLTContents = Section.getContents();
  ArrayRef<uint8_t> PLTData(
      reinterpret_cast<const uint8_t *>(PLTContents.data()), SectionSize);

  auto disassembleInstruction = [&](uint64_t InstrOffset, MCInst &Instruction,
                                    uint64_t &InstrSize) {
    const uint64_t InstrAddr = SectionAddress + InstrOffset;
    if (!BC->DisAsm->getInstruction(Instruction, InstrSize,
                                    PLTData.slice(InstrOffset), InstrAddr,
                                    nulls())) {
      errs() << "BOLT-ERROR: unable to disassemble instruction in PLT section "
             << Section.getName() << " at offset 0x"
             << Twine::utohexstr(InstrOffset) << '\n';
      exit(1);
    }
  };

  for (uint64_t EntryOffset = 0; EntryOffset + EntrySize <= SectionSize;
       EntryOffset += EntrySize) {
    MCInst Instruction;
    uint64_t InstrSize, InstrOffset = EntryOffset;
    while (InstrOffset < EntryOffset + EntrySize) {
      disassembleInstruction(InstrOffset, Instruction, InstrSize);
      // Check if the entry size needs adjustment.
      if (EntryOffset == 0 && BC->MIB->isTerminateBranch(Instruction) &&
          EntrySize == 8)
        EntrySize = 16;

      if (BC->MIB->isIndirectBranch(Instruction))
        break;

      InstrOffset += InstrSize;
    }

    if (InstrOffset + InstrSize > EntryOffset + EntrySize)
      continue;

    uint64_t TargetAddress;
    if (!BC->MIB->evaluateMemOperandTarget(Instruction, TargetAddress,
                                           SectionAddress + InstrOffset,
                                           InstrSize)) {
      errs() << "BOLT-ERROR: error evaluating PLT instruction at offset 0x"
             << Twine::utohexstr(SectionAddress + InstrOffset) << '\n';
      exit(1);
    }

    createPLTBinaryFunction(TargetAddress, SectionAddress + EntryOffset,
                            EntrySize);
  }
}

void RewriteInstance::disassemblePLT() {
  auto analyzeOnePLTSection = [&](BinarySection &Section, uint64_t EntrySize) {
    if (BC->isAArch64())
      return disassemblePLTSectionAArch64(Section);
    return disassemblePLTSectionX86(Section, EntrySize);
  };

  for (BinarySection &Section : BC->allocatableSections()) {
    const PLTSectionInfo *PLTSI = getPLTSectionInfo(Section.getName());
    if (!PLTSI)
      continue;

    analyzeOnePLTSection(Section, PLTSI->EntrySize);

    BinaryFunction *PltBF;
    auto BFIter = BC->getBinaryFunctions().find(Section.getAddress());
    if (BFIter != BC->getBinaryFunctions().end()) {
      PltBF = &BFIter->second;
    } else {
      // If we did not register any function at the start of the section,
      // then it must be a general PLT entry. Add a function at the location.
      PltBF = BC->createBinaryFunction(
          "__BOLT_PSEUDO_" + Section.getName().str(), Section,
          Section.getAddress(), 0, PLTSI->EntrySize, Section.getAlignment());
    }
    PltBF->setPseudo(true);
  }
}

void RewriteInstance::adjustFunctionBoundaries() {
  for (auto BFI = BC->getBinaryFunctions().begin(),
            BFE = BC->getBinaryFunctions().end();
       BFI != BFE; ++BFI) {
    BinaryFunction &Function = BFI->second;
    const BinaryFunction *NextFunction = nullptr;
    if (std::next(BFI) != BFE)
      NextFunction = &std::next(BFI)->second;

    // Check if it's a fragment of a function.
    std::optional<StringRef> FragName =
        Function.hasRestoredNameRegex(".*\\.cold(\\.[0-9]+)?");
    if (FragName) {
      static bool PrintedWarning = false;
      if (!PrintedWarning) {
        PrintedWarning = true;
        errs() << "BOLT-WARNING: split function detected on input : "
               << *FragName;
        if (BC->HasRelocations)
          errs() << ". The support is limited in relocation mode";
        if (opts::Lite) {
          opts::Lite = false;
          errs() << "\nBOLT-WARNING: disabling lite mode (-lite) when split "
                 << "functions are present\n";
        }
      }
      Function.IsFragment = true;
    }

    // Check if there's a symbol or a function with a larger address in the
    // same section. If there is - it determines the maximum size for the
    // current function. Otherwise, it is the size of a containing section
    // the defines it.
    //
    // NOTE: ignore some symbols that could be tolerated inside the body
    //       of a function.
    auto NextSymRefI = FileSymRefs.upper_bound(Function.getAddress());
    while (NextSymRefI != FileSymRefs.end()) {
      SymbolRef &Symbol = NextSymRefI->second;
      const uint64_t SymbolAddress = NextSymRefI->first;
      const uint64_t SymbolSize = ELFSymbolRef(Symbol).getSize();

      if (NextFunction && SymbolAddress >= NextFunction->getAddress())
        break;

      if (!Function.isSymbolValidInScope(Symbol, SymbolSize))
        break;

      // This is potentially another entry point into the function.
      uint64_t EntryOffset = NextSymRefI->first - Function.getAddress();
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: adding entry point to function "
                        << Function << " at offset 0x"
                        << Twine::utohexstr(EntryOffset) << '\n');
      Function.addEntryPointAtOffset(EntryOffset);

      ++NextSymRefI;
    }

    // Function runs at most till the end of the containing section.
    uint64_t NextObjectAddress = Function.getOriginSection()->getEndAddress();
    // Or till the next object marked by a symbol.
    if (NextSymRefI != FileSymRefs.end())
      NextObjectAddress = std::min(NextSymRefI->first, NextObjectAddress);

    // Or till the next function not marked by a symbol.
    if (NextFunction)
      NextObjectAddress =
          std::min(NextFunction->getAddress(), NextObjectAddress);

    const uint64_t MaxSize = NextObjectAddress - Function.getAddress();
    if (MaxSize < Function.getSize()) {
      errs() << "BOLT-ERROR: symbol seen in the middle of the function "
             << Function << ". Skipping.\n";
      Function.setSimple(false);
      Function.setMaxSize(Function.getSize());
      continue;
    }
    Function.setMaxSize(MaxSize);
    if (!Function.getSize() && Function.isSimple()) {
      // Some assembly functions have their size set to 0, use the max
      // size as their real size.
      if (opts::Verbosity >= 1)
        outs() << "BOLT-INFO: setting size of function " << Function << " to "
               << Function.getMaxSize() << " (was 0)\n";
      Function.setSize(Function.getMaxSize());
    }
  }
}

void RewriteInstance::relocateEHFrameSection() {
  assert(EHFrameSection && "Non-empty .eh_frame section expected.");

  BinarySection *RelocatedEHFrameSection =
      getSection(".relocated" + getEHFrameSectionName());
  assert(RelocatedEHFrameSection &&
         "Relocated eh_frame section should be preregistered.");
  DWARFDataExtractor DE(EHFrameSection->getContents(),
                        BC->AsmInfo->isLittleEndian(),
                        BC->AsmInfo->getCodePointerSize());
  auto createReloc = [&](uint64_t Value, uint64_t Offset, uint64_t DwarfType) {
    if (DwarfType == dwarf::DW_EH_PE_omit)
      return;

    // Only fix references that are relative to other locations.
    if (!(DwarfType & dwarf::DW_EH_PE_pcrel) &&
        !(DwarfType & dwarf::DW_EH_PE_textrel) &&
        !(DwarfType & dwarf::DW_EH_PE_funcrel) &&
        !(DwarfType & dwarf::DW_EH_PE_datarel))
      return;

    if (!(DwarfType & dwarf::DW_EH_PE_sdata4))
      return;

    uint64_t RelType;
    switch (DwarfType & 0x0f) {
    default:
      llvm_unreachable("unsupported DWARF encoding type");
    case dwarf::DW_EH_PE_sdata4:
    case dwarf::DW_EH_PE_udata4:
      RelType = Relocation::getPC32();
      Offset -= 4;
      break;
    case dwarf::DW_EH_PE_sdata8:
    case dwarf::DW_EH_PE_udata8:
      RelType = Relocation::getPC64();
      Offset -= 8;
      break;
    }

    // Create a relocation against an absolute value since the goal is to
    // preserve the contents of the section independent of the new values
    // of referenced symbols.
    RelocatedEHFrameSection->addRelocation(Offset, nullptr, RelType, Value);
  };

  Error E = EHFrameParser::parse(DE, EHFrameSection->getAddress(), createReloc);
  check_error(std::move(E), "failed to patch EH frame");
}

ArrayRef<uint8_t> RewriteInstance::getLSDAData() {
  return ArrayRef<uint8_t>(LSDASection->getData(),
                           LSDASection->getContents().size());
}

uint64_t RewriteInstance::getLSDAAddress() { return LSDASection->getAddress(); }

Error RewriteInstance::readSpecialSections() {
  NamedRegionTimer T("readSpecialSections", "read special sections",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);

  bool HasTextRelocations = false;
  bool HasSymbolTable = false;
  bool HasDebugInfo = false;

  // Process special sections.
  for (const SectionRef &Section : InputFile->sections()) {
    Expected<StringRef> SectionNameOrErr = Section.getName();
    check_error(SectionNameOrErr.takeError(), "cannot get section name");
    StringRef SectionName = *SectionNameOrErr;

    if (Error E = Section.getContents().takeError())
      return E;
    BC->registerSection(Section);
    LLVM_DEBUG(
        dbgs() << "BOLT-DEBUG: registering section " << SectionName << " @ 0x"
               << Twine::utohexstr(Section.getAddress()) << ":0x"
               << Twine::utohexstr(Section.getAddress() + Section.getSize())
               << "\n");
    if (isDebugSection(SectionName))
      HasDebugInfo = true;
    if (isKSymtabSection(SectionName))
      opts::LinuxKernelMode = true;
  }

  if (HasDebugInfo && !opts::UpdateDebugSections && !opts::AggregateOnly) {
    errs() << "BOLT-WARNING: debug info will be stripped from the binary. "
              "Use -update-debug-sections to keep it.\n";
  }

  HasTextRelocations = (bool)BC->getUniqueSectionByName(".rela.text");
  HasSymbolTable = (bool)BC->getUniqueSectionByName(".symtab");
  LSDASection = BC->getUniqueSectionByName(".gcc_except_table");
  EHFrameSection = BC->getUniqueSectionByName(".eh_frame");
  GOTPLTSection = BC->getUniqueSectionByName(".got.plt");
  RelaPLTSection = BC->getUniqueSectionByName(".rela.plt");
  RelaDynSection = BC->getUniqueSectionByName(".rela.dyn");
  BuildIDSection = BC->getUniqueSectionByName(".note.gnu.build-id");
  SDTSection = BC->getUniqueSectionByName(".note.stapsdt");
  PseudoProbeDescSection = BC->getUniqueSectionByName(".pseudo_probe_desc");
  PseudoProbeSection = BC->getUniqueSectionByName(".pseudo_probe");

  if (ErrorOr<BinarySection &> BATSec =
          BC->getUniqueSectionByName(BoltAddressTranslation::SECTION_NAME)) {
    // Do not read BAT when plotting a heatmap
    if (!opts::HeatmapMode) {
      if (std::error_code EC = BAT->parse(BATSec->getContents())) {
        errs() << "BOLT-ERROR: failed to parse BOLT address translation "
                  "table.\n";
        exit(1);
      }
    }
  }

  if (opts::PrintSections) {
    outs() << "BOLT-INFO: Sections from original binary:\n";
    BC->printSections(outs());
  }

  if (opts::RelocationMode == cl::BOU_TRUE && !HasTextRelocations) {
    errs() << "BOLT-ERROR: relocations against code are missing from the input "
              "file. Cannot proceed in relocations mode (-relocs).\n";
    exit(1);
  }

  BC->HasRelocations =
      HasTextRelocations && (opts::RelocationMode != cl::BOU_FALSE);

  BC->IsStripped = !HasSymbolTable;

  // Force non-relocation mode for heatmap generation
  if (opts::HeatmapMode)
    BC->HasRelocations = false;

  if (BC->HasRelocations)
    outs() << "BOLT-INFO: enabling " << (opts::StrictMode ? "strict " : "")
           << "relocation mode\n";

  if (BC->IsStripped)
    outs() << "BOLT-INFO: input binary is stripped. The support is limited and "
           << "is considered experimental.\n";

  // Read EH frame for function boundaries info.
  Expected<const DWARFDebugFrame *> EHFrameOrError = BC->DwCtx->getEHFrame();
  if (!EHFrameOrError)
    report_error("expected valid eh_frame section", EHFrameOrError.takeError());
  CFIRdWrt.reset(new CFIReaderWriter(*EHFrameOrError.get()));

  // Parse build-id
  parseBuildID();
  if (std::optional<std::string> FileBuildID = getPrintableBuildID())
    BC->setFileBuildID(*FileBuildID);

  parseSDTNotes();

  // Read .dynamic/PT_DYNAMIC.
  return readELFDynamic();
}

void RewriteInstance::adjustCommandLineOptions() {
  if (BC->isAArch64() && !BC->HasRelocations)
    errs() << "BOLT-WARNING: non-relocation mode for AArch64 is not fully "
              "supported\n";

  if (RuntimeLibrary *RtLibrary = BC->getRuntimeLibrary())
    RtLibrary->adjustCommandLineOptions(*BC);

  if (opts::AlignMacroOpFusion != MFT_NONE && !BC->isX86()) {
    outs() << "BOLT-INFO: disabling -align-macro-fusion on non-x86 platform\n";
    opts::AlignMacroOpFusion = MFT_NONE;
  }

  if (BC->isX86() && BC->MAB->allowAutoPadding()) {
    if (!BC->HasRelocations) {
      errs() << "BOLT-ERROR: cannot apply mitigations for Intel JCC erratum in "
                "non-relocation mode\n";
      exit(1);
    }
    outs() << "BOLT-WARNING: using mitigation for Intel JCC erratum, layout "
              "may take several minutes\n";
    opts::AlignMacroOpFusion = MFT_NONE;
  }

  if (opts::AlignMacroOpFusion != MFT_NONE && !BC->HasRelocations) {
    outs() << "BOLT-INFO: disabling -align-macro-fusion in non-relocation "
              "mode\n";
    opts::AlignMacroOpFusion = MFT_NONE;
  }

  if (opts::SplitEH && !BC->HasRelocations) {
    errs() << "BOLT-WARNING: disabling -split-eh in non-relocation mode\n";
    opts::SplitEH = false;
  }

  if (opts::StrictMode && !BC->HasRelocations) {
    errs() << "BOLT-WARNING: disabling strict mode (-strict) in non-relocation "
              "mode\n";
    opts::StrictMode = false;
  }

  if (BC->HasRelocations && opts::AggregateOnly &&
      !opts::StrictMode.getNumOccurrences()) {
    outs() << "BOLT-INFO: enabling strict relocation mode for aggregation "
              "purposes\n";
    opts::StrictMode = true;
  }

  if (BC->isX86() && BC->HasRelocations &&
      opts::AlignMacroOpFusion == MFT_HOT && !ProfileReader) {
    outs() << "BOLT-INFO: enabling -align-macro-fusion=all since no profile "
              "was specified\n";
    opts::AlignMacroOpFusion = MFT_ALL;
  }

  if (!BC->HasRelocations &&
      opts::ReorderFunctions != ReorderFunctions::RT_NONE) {
    errs() << "BOLT-ERROR: function reordering only works when "
           << "relocations are enabled\n";
    exit(1);
  }

  if (opts::ReorderFunctions != ReorderFunctions::RT_NONE &&
      !opts::HotText.getNumOccurrences()) {
    opts::HotText = true;
  } else if (opts::HotText && !BC->HasRelocations) {
    errs() << "BOLT-WARNING: hot text is disabled in non-relocation mode\n";
    opts::HotText = false;
  }

  if (opts::HotText && opts::HotTextMoveSections.getNumOccurrences() == 0) {
    opts::HotTextMoveSections.addValue(".stub");
    opts::HotTextMoveSections.addValue(".mover");
    opts::HotTextMoveSections.addValue(".never_hugify");
  }

  if (opts::UseOldText && !BC->OldTextSectionAddress) {
    errs() << "BOLT-WARNING: cannot use old .text as the section was not found"
              "\n";
    opts::UseOldText = false;
  }
  if (opts::UseOldText && !BC->HasRelocations) {
    errs() << "BOLT-WARNING: cannot use old .text in non-relocation mode\n";
    opts::UseOldText = false;
  }

  if (!opts::AlignText.getNumOccurrences())
    opts::AlignText = BC->PageAlign;

  if (opts::AlignText < opts::AlignFunctions)
    opts::AlignText = (unsigned)opts::AlignFunctions;

  if (BC->isX86() && opts::Lite.getNumOccurrences() == 0 && !opts::StrictMode &&
      !opts::UseOldText)
    opts::Lite = true;

  if (opts::Lite && opts::UseOldText) {
    errs() << "BOLT-WARNING: cannot combine -lite with -use-old-text. "
              "Disabling -use-old-text.\n";
    opts::UseOldText = false;
  }

  if (opts::Lite && opts::StrictMode) {
    errs() << "BOLT-ERROR: -strict and -lite cannot be used at the same time\n";
    exit(1);
  }

  if (opts::Lite)
    outs() << "BOLT-INFO: enabling lite mode\n";

  if (!opts::SaveProfile.empty() && BAT->enabledFor(InputFile)) {
    errs() << "BOLT-ERROR: unable to save profile in YAML format for input "
              "file processed by BOLT. Please remove -w option and use branch "
              "profile.\n";
    exit(1);
  }
}

namespace {
template <typename ELFT>
int64_t getRelocationAddend(const ELFObjectFile<ELFT> *Obj,
                            const RelocationRef &RelRef) {
  using ELFShdrTy = typename ELFT::Shdr;
  using Elf_Rela = typename ELFT::Rela;
  int64_t Addend = 0;
  const ELFFile<ELFT> &EF = Obj->getELFFile();
  DataRefImpl Rel = RelRef.getRawDataRefImpl();
  const ELFShdrTy *RelocationSection = cantFail(EF.getSection(Rel.d.a));
  switch (RelocationSection->sh_type) {
  default:
    llvm_unreachable("unexpected relocation section type");
  case ELF::SHT_REL:
    break;
  case ELF::SHT_RELA: {
    const Elf_Rela *RelA = Obj->getRela(Rel);
    Addend = RelA->r_addend;
    break;
  }
  }

  return Addend;
}

int64_t getRelocationAddend(const ELFObjectFileBase *Obj,
                            const RelocationRef &Rel) {
  if (auto *ELF32LE = dyn_cast<ELF32LEObjectFile>(Obj))
    return getRelocationAddend(ELF32LE, Rel);
  if (auto *ELF64LE = dyn_cast<ELF64LEObjectFile>(Obj))
    return getRelocationAddend(ELF64LE, Rel);
  if (auto *ELF32BE = dyn_cast<ELF32BEObjectFile>(Obj))
    return getRelocationAddend(ELF32BE, Rel);
  auto *ELF64BE = cast<ELF64BEObjectFile>(Obj);
  return getRelocationAddend(ELF64BE, Rel);
}

template <typename ELFT>
uint32_t getRelocationSymbol(const ELFObjectFile<ELFT> *Obj,
                             const RelocationRef &RelRef) {
  using ELFShdrTy = typename ELFT::Shdr;
  uint32_t Symbol = 0;
  const ELFFile<ELFT> &EF = Obj->getELFFile();
  DataRefImpl Rel = RelRef.getRawDataRefImpl();
  const ELFShdrTy *RelocationSection = cantFail(EF.getSection(Rel.d.a));
  switch (RelocationSection->sh_type) {
  default:
    llvm_unreachable("unexpected relocation section type");
  case ELF::SHT_REL:
    Symbol = Obj->getRel(Rel)->getSymbol(EF.isMips64EL());
    break;
  case ELF::SHT_RELA:
    Symbol = Obj->getRela(Rel)->getSymbol(EF.isMips64EL());
    break;
  }

  return Symbol;
}

uint32_t getRelocationSymbol(const ELFObjectFileBase *Obj,
                             const RelocationRef &Rel) {
  if (auto *ELF32LE = dyn_cast<ELF32LEObjectFile>(Obj))
    return getRelocationSymbol(ELF32LE, Rel);
  if (auto *ELF64LE = dyn_cast<ELF64LEObjectFile>(Obj))
    return getRelocationSymbol(ELF64LE, Rel);
  if (auto *ELF32BE = dyn_cast<ELF32BEObjectFile>(Obj))
    return getRelocationSymbol(ELF32BE, Rel);
  auto *ELF64BE = cast<ELF64BEObjectFile>(Obj);
  return getRelocationSymbol(ELF64BE, Rel);
}
} // anonymous namespace

bool RewriteInstance::analyzeRelocation(
    const RelocationRef &Rel, uint64_t &RType, std::string &SymbolName,
    bool &IsSectionRelocation, uint64_t &SymbolAddress, int64_t &Addend,
    uint64_t &ExtractedValue, bool &Skip) const {
  Skip = false;
  if (!Relocation::isSupported(RType))
    return false;

  const bool IsAArch64 = BC->isAArch64();

  const size_t RelSize = Relocation::getSizeForType(RType);

  ErrorOr<uint64_t> Value =
      BC->getUnsignedValueAtAddress(Rel.getOffset(), RelSize);
  assert(Value && "failed to extract relocated value");
  if ((Skip = Relocation::skipRelocationProcess(RType, *Value)))
    return true;

  ExtractedValue = Relocation::extractValue(RType, *Value, Rel.getOffset());
  Addend = getRelocationAddend(InputFile, Rel);

  const bool IsPCRelative = Relocation::isPCRelative(RType);
  const uint64_t PCRelOffset = IsPCRelative && !IsAArch64 ? Rel.getOffset() : 0;
  bool SkipVerification = false;
  auto SymbolIter = Rel.getSymbol();
  if (SymbolIter == InputFile->symbol_end()) {
    SymbolAddress = ExtractedValue - Addend + PCRelOffset;
    MCSymbol *RelSymbol =
        BC->getOrCreateGlobalSymbol(SymbolAddress, "RELSYMat");
    SymbolName = std::string(RelSymbol->getName());
    IsSectionRelocation = false;
  } else {
    const SymbolRef &Symbol = *SymbolIter;
    SymbolName = std::string(cantFail(Symbol.getName()));
    SymbolAddress = cantFail(Symbol.getAddress());
    SkipVerification = (cantFail(Symbol.getType()) == SymbolRef::ST_Other);
    // Section symbols are marked as ST_Debug.
    IsSectionRelocation = (cantFail(Symbol.getType()) == SymbolRef::ST_Debug);
    // Check for PLT entry registered with symbol name
    if (!SymbolAddress && IsAArch64) {
      const BinaryData *BD = BC->getPLTBinaryDataByName(SymbolName);
      SymbolAddress = BD ? BD->getAddress() : 0;
    }
  }
  // For PIE or dynamic libs, the linker may choose not to put the relocation
  // result at the address if it is a X86_64_64 one because it will emit a
  // dynamic relocation (X86_RELATIVE) for the dynamic linker and loader to
  // resolve it at run time. The static relocation result goes as the addend
  // of the dynamic relocation in this case. We can't verify these cases.
  // FIXME: perhaps we can try to find if it really emitted a corresponding
  // RELATIVE relocation at this offset with the correct value as the addend.
  if (!BC->HasFixedLoadAddress && RelSize == 8)
    SkipVerification = true;

  if (IsSectionRelocation && !IsAArch64) {
    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(SymbolAddress);
    assert(Section && "section expected for section relocation");
    SymbolName = "section " + std::string(Section->getName());
    // Convert section symbol relocations to regular relocations inside
    // non-section symbols.
    if (Section->containsAddress(ExtractedValue) && !IsPCRelative) {
      SymbolAddress = ExtractedValue;
      Addend = 0;
    } else {
      Addend = ExtractedValue - (SymbolAddress - PCRelOffset);
    }
  }

  // If no symbol has been found or if it is a relocation requiring the
  // creation of a GOT entry, do not link against the symbol but against
  // whatever address was extracted from the instruction itself. We are
  // not creating a GOT entry as this was already processed by the linker.
  // For GOT relocs, do not subtract addend as the addend does not refer
  // to this instruction's target, but it refers to the target in the GOT
  // entry.
  if (Relocation::isGOT(RType)) {
    Addend = 0;
    SymbolAddress = ExtractedValue + PCRelOffset;
  } else if (Relocation::isTLS(RType)) {
    SkipVerification = true;
  } else if (!SymbolAddress) {
    assert(!IsSectionRelocation);
    if (ExtractedValue || Addend == 0 || IsPCRelative) {
      SymbolAddress =
          truncateToSize(ExtractedValue - Addend + PCRelOffset, RelSize);
    } else {
      // This is weird case.  The extracted value is zero but the addend is
      // non-zero and the relocation is not pc-rel.  Using the previous logic,
      // the SymbolAddress would end up as a huge number.  Seen in
      // exceptions_pic.test.
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: relocation @ 0x"
                        << Twine::utohexstr(Rel.getOffset())
                        << " value does not match addend for "
                        << "relocation to undefined symbol.\n");
      return true;
    }
  }

  auto verifyExtractedValue = [&]() {
    if (SkipVerification)
      return true;

    if (IsAArch64)
      return true;

    if (SymbolName == "__hot_start" || SymbolName == "__hot_end")
      return true;

    if (RType == ELF::R_X86_64_PLT32)
      return true;

    return truncateToSize(ExtractedValue, RelSize) ==
           truncateToSize(SymbolAddress + Addend - PCRelOffset, RelSize);
  };

  (void)verifyExtractedValue;
  assert(verifyExtractedValue() && "mismatched extracted relocation value");

  return true;
}

void RewriteInstance::processDynamicRelocations() {
  // Read relocations for PLT - DT_JMPREL.
  if (PLTRelocationsSize > 0) {
    ErrorOr<BinarySection &> PLTRelSectionOrErr =
        BC->getSectionForAddress(*PLTRelocationsAddress);
    if (!PLTRelSectionOrErr)
      report_error("unable to find section corresponding to DT_JMPREL",
                   PLTRelSectionOrErr.getError());
    if (PLTRelSectionOrErr->getSize() != PLTRelocationsSize)
      report_error("section size mismatch for DT_PLTRELSZ",
                   errc::executable_format_error);
    readDynamicRelocations(PLTRelSectionOrErr->getSectionRef(),
                           /*IsJmpRel*/ true);
  }

  // The rest of dynamic relocations - DT_RELA.
  if (DynamicRelocationsSize > 0) {
    ErrorOr<BinarySection &> DynamicRelSectionOrErr =
        BC->getSectionForAddress(*DynamicRelocationsAddress);
    if (!DynamicRelSectionOrErr)
      report_error("unable to find section corresponding to DT_RELA",
                   DynamicRelSectionOrErr.getError());
    if (DynamicRelSectionOrErr->getSize() != DynamicRelocationsSize)
      report_error("section size mismatch for DT_RELASZ",
                   errc::executable_format_error);
    readDynamicRelocations(DynamicRelSectionOrErr->getSectionRef(),
                           /*IsJmpRel*/ false);
  }
}

void RewriteInstance::processRelocations() {
  if (!BC->HasRelocations)
    return;

  for (const SectionRef &Section : InputFile->sections()) {
    if (cantFail(Section.getRelocatedSection()) != InputFile->section_end() &&
        !BinarySection(*BC, Section).isAllocatable())
      readRelocations(Section);
  }

  if (NumFailedRelocations)
    errs() << "BOLT-WARNING: Failed to analyze " << NumFailedRelocations
           << " relocations\n";
}

void RewriteInstance::insertLKMarker(uint64_t PC, uint64_t SectionOffset,
                                     int32_t PCRelativeOffset,
                                     bool IsPCRelative, StringRef SectionName) {
  BC->LKMarkers[PC].emplace_back(LKInstructionMarkerInfo{
      SectionOffset, PCRelativeOffset, IsPCRelative, SectionName});
}

void RewriteInstance::processLKSections() {
  assert(opts::LinuxKernelMode &&
         "process Linux Kernel special sections and their relocations only in "
         "linux kernel mode.\n");

  processLKExTable();
  processLKPCIFixup();
  processLKKSymtab();
  processLKKSymtab(true);
  processLKBugTable();
  processLKSMPLocks();
}

/// Process __ex_table section of Linux Kernel.
/// This section contains information regarding kernel level exception
/// handling (https://www.kernel.org/doc/html/latest/x86/exception-tables.html).
/// More documentation is in arch/x86/include/asm/extable.h.
///
/// The section is the list of the following structures:
///
///   struct exception_table_entry {
///     int insn;
///     int fixup;
///     int handler;
///   };
///
void RewriteInstance::processLKExTable() {
  ErrorOr<BinarySection &> SectionOrError =
      BC->getUniqueSectionByName("__ex_table");
  if (!SectionOrError)
    return;

  const uint64_t SectionSize = SectionOrError->getSize();
  const uint64_t SectionAddress = SectionOrError->getAddress();
  assert((SectionSize % 12) == 0 &&
         "The size of the __ex_table section should be a multiple of 12");
  for (uint64_t I = 0; I < SectionSize; I += 4) {
    const uint64_t EntryAddress = SectionAddress + I;
    ErrorOr<uint64_t> Offset = BC->getSignedValueAtAddress(EntryAddress, 4);
    assert(Offset && "failed reading PC-relative offset for __ex_table");
    int32_t SignedOffset = *Offset;
    const uint64_t RefAddress = EntryAddress + SignedOffset;

    BinaryFunction *ContainingBF =
        BC->getBinaryFunctionContainingAddress(RefAddress);
    if (!ContainingBF)
      continue;

    MCSymbol *ReferencedSymbol = ContainingBF->getSymbol();
    const uint64_t FunctionOffset = RefAddress - ContainingBF->getAddress();
    switch (I % 12) {
    default:
      llvm_unreachable("bad alignment of __ex_table");
      break;
    case 0:
      // insn
      insertLKMarker(RefAddress, I, SignedOffset, true, "__ex_table");
      break;
    case 4:
      // fixup
      if (FunctionOffset)
        ReferencedSymbol = ContainingBF->addEntryPointAtOffset(FunctionOffset);
      BC->addRelocation(EntryAddress, ReferencedSymbol, Relocation::getPC32(),
                        0, *Offset);
      break;
    case 8:
      // handler
      assert(!FunctionOffset &&
             "__ex_table handler entry should point to function start");
      BC->addRelocation(EntryAddress, ReferencedSymbol, Relocation::getPC32(),
                        0, *Offset);
      break;
    }
  }
}

/// Process .pci_fixup section of Linux Kernel.
/// This section contains a list of entries for different PCI devices and their
/// corresponding hook handler (code pointer where the fixup
/// code resides, usually on x86_64 it is an entry PC relative 32 bit offset).
/// Documentation is in include/linux/pci.h.
void RewriteInstance::processLKPCIFixup() {
  ErrorOr<BinarySection &> SectionOrError =
      BC->getUniqueSectionByName(".pci_fixup");
  assert(SectionOrError &&
         ".pci_fixup section not found in Linux Kernel binary");
  const uint64_t SectionSize = SectionOrError->getSize();
  const uint64_t SectionAddress = SectionOrError->getAddress();
  assert((SectionSize % 16) == 0 && ".pci_fixup size is not a multiple of 16");

  for (uint64_t I = 12; I + 4 <= SectionSize; I += 16) {
    const uint64_t PC = SectionAddress + I;
    ErrorOr<uint64_t> Offset = BC->getSignedValueAtAddress(PC, 4);
    assert(Offset && "cannot read value from .pci_fixup");
    const int32_t SignedOffset = *Offset;
    const uint64_t HookupAddress = PC + SignedOffset;
    BinaryFunction *HookupFunction =
        BC->getBinaryFunctionAtAddress(HookupAddress);
    assert(HookupFunction && "expected function for entry in .pci_fixup");
    BC->addRelocation(PC, HookupFunction->getSymbol(), Relocation::getPC32(), 0,
                      *Offset);
  }
}

/// Process __ksymtab[_gpl] sections of Linux Kernel.
/// This section lists all the vmlinux symbols that kernel modules can access.
///
/// All the entries are 4 bytes each and hence we can read them by one by one
/// and ignore the ones that are not pointing to the .text section. All pointers
/// are PC relative offsets. Always, points to the beginning of the function.
void RewriteInstance::processLKKSymtab(bool IsGPL) {
  StringRef SectionName = "__ksymtab";
  if (IsGPL)
    SectionName = "__ksymtab_gpl";
  ErrorOr<BinarySection &> SectionOrError =
      BC->getUniqueSectionByName(SectionName);
  assert(SectionOrError &&
         "__ksymtab[_gpl] section not found in Linux Kernel binary");
  const uint64_t SectionSize = SectionOrError->getSize();
  const uint64_t SectionAddress = SectionOrError->getAddress();
  assert((SectionSize % 4) == 0 &&
         "The size of the __ksymtab[_gpl] section should be a multiple of 4");

  for (uint64_t I = 0; I < SectionSize; I += 4) {
    const uint64_t EntryAddress = SectionAddress + I;
    ErrorOr<uint64_t> Offset = BC->getSignedValueAtAddress(EntryAddress, 4);
    assert(Offset && "Reading valid PC-relative offset for a ksymtab entry");
    const int32_t SignedOffset = *Offset;
    const uint64_t RefAddress = EntryAddress + SignedOffset;
    BinaryFunction *BF = BC->getBinaryFunctionAtAddress(RefAddress);
    if (!BF)
      continue;

    BC->addRelocation(EntryAddress, BF->getSymbol(), Relocation::getPC32(), 0,
                      *Offset);
  }
}

/// Process __bug_table section.
/// This section contains information useful for kernel debugging.
/// Each entry in the section is a struct bug_entry that contains a pointer to
/// the ud2 instruction corresponding to the bug, corresponding file name (both
/// pointers use PC relative offset addressing), line number, and flags.
/// The definition of the struct bug_entry can be found in
/// `include/asm-generic/bug.h`
void RewriteInstance::processLKBugTable() {
  ErrorOr<BinarySection &> SectionOrError =
      BC->getUniqueSectionByName("__bug_table");
  if (!SectionOrError)
    return;

  const uint64_t SectionSize = SectionOrError->getSize();
  const uint64_t SectionAddress = SectionOrError->getAddress();
  assert((SectionSize % 12) == 0 &&
         "The size of the __bug_table section should be a multiple of 12");
  for (uint64_t I = 0; I < SectionSize; I += 12) {
    const uint64_t EntryAddress = SectionAddress + I;
    ErrorOr<uint64_t> Offset = BC->getSignedValueAtAddress(EntryAddress, 4);
    assert(Offset &&
           "Reading valid PC-relative offset for a __bug_table entry");
    const int32_t SignedOffset = *Offset;
    const uint64_t RefAddress = EntryAddress + SignedOffset;
    assert(BC->getBinaryFunctionContainingAddress(RefAddress) &&
           "__bug_table entries should point to a function");

    insertLKMarker(RefAddress, I, SignedOffset, true, "__bug_table");
  }
}

/// .smp_locks section contains PC-relative references to instructions with LOCK
/// prefix. The prefix can be converted to NOP at boot time on non-SMP systems.
void RewriteInstance::processLKSMPLocks() {
  ErrorOr<BinarySection &> SectionOrError =
      BC->getUniqueSectionByName(".smp_locks");
  if (!SectionOrError)
    return;

  uint64_t SectionSize = SectionOrError->getSize();
  const uint64_t SectionAddress = SectionOrError->getAddress();
  assert((SectionSize % 4) == 0 &&
         "The size of the .smp_locks section should be a multiple of 4");

  for (uint64_t I = 0; I < SectionSize; I += 4) {
    const uint64_t EntryAddress = SectionAddress + I;
    ErrorOr<uint64_t> Offset = BC->getSignedValueAtAddress(EntryAddress, 4);
    assert(Offset && "Reading valid PC-relative offset for a .smp_locks entry");
    int32_t SignedOffset = *Offset;
    uint64_t RefAddress = EntryAddress + SignedOffset;

    BinaryFunction *ContainingBF =
        BC->getBinaryFunctionContainingAddress(RefAddress);
    if (!ContainingBF)
      continue;

    insertLKMarker(RefAddress, I, SignedOffset, true, ".smp_locks");
  }
}

void RewriteInstance::readDynamicRelocations(const SectionRef &Section,
                                             bool IsJmpRel) {
  assert(BinarySection(*BC, Section).isAllocatable() && "allocatable expected");

  LLVM_DEBUG({
    StringRef SectionName = cantFail(Section.getName());
    dbgs() << "BOLT-DEBUG: reading relocations for section " << SectionName
           << ":\n";
  });

  for (const RelocationRef &Rel : Section.relocations()) {
    const uint64_t RType = Rel.getType();
    if (Relocation::isNone(RType))
      continue;

    StringRef SymbolName = "<none>";
    MCSymbol *Symbol = nullptr;
    uint64_t SymbolAddress = 0;
    const uint64_t Addend = getRelocationAddend(InputFile, Rel);

    symbol_iterator SymbolIter = Rel.getSymbol();
    if (SymbolIter != InputFile->symbol_end()) {
      SymbolName = cantFail(SymbolIter->getName());
      BinaryData *BD = BC->getBinaryDataByName(SymbolName);
      Symbol = BD ? BD->getSymbol()
                  : BC->getOrCreateUndefinedGlobalSymbol(SymbolName);
      SymbolAddress = cantFail(SymbolIter->getAddress());
      (void)SymbolAddress;
    }

    LLVM_DEBUG(
      SmallString<16> TypeName;
      Rel.getTypeName(TypeName);
      dbgs() << "BOLT-DEBUG: dynamic relocation at 0x"
             << Twine::utohexstr(Rel.getOffset()) << " : " << TypeName
             << " : " << SymbolName << " : " <<  Twine::utohexstr(SymbolAddress)
             << " : + 0x" << Twine::utohexstr(Addend) << '\n'
    );

    if (IsJmpRel)
      IsJmpRelocation[RType] = true;

    if (Symbol)
      SymbolIndex[Symbol] = getRelocationSymbol(InputFile, Rel);

    BC->addDynamicRelocation(Rel.getOffset(), Symbol, RType, Addend);
  }
}

void RewriteInstance::printRelocationInfo(const RelocationRef &Rel,
                                          StringRef SymbolName,
                                          uint64_t SymbolAddress,
                                          uint64_t Addend,
                                          uint64_t ExtractedValue) const {
  SmallString<16> TypeName;
  Rel.getTypeName(TypeName);
  const uint64_t Address = SymbolAddress + Addend;
  const uint64_t Offset = Rel.getOffset();
  ErrorOr<BinarySection &> Section = BC->getSectionForAddress(SymbolAddress);
  BinaryFunction *Func =
      BC->getBinaryFunctionContainingAddress(Offset, false, BC->isAArch64());
  dbgs() << formatv("Relocation: offset = {0:x}; type = {1}; value = {2:x}; ",
                    Offset, TypeName, ExtractedValue)
         << formatv("symbol = {0} ({1}); symbol address = {2:x}; ", SymbolName,
                    Section ? Section->getName() : "", SymbolAddress)
         << formatv("addend = {0:x}; address = {1:x}; in = ", Addend, Address);
  if (Func)
    dbgs() << Func->getPrintName();
  else
    dbgs() << BC->getSectionForAddress(Rel.getOffset())->getName();
  dbgs() << '\n';
}

void RewriteInstance::readRelocations(const SectionRef &Section) {
  LLVM_DEBUG({
    StringRef SectionName = cantFail(Section.getName());
    dbgs() << "BOLT-DEBUG: reading relocations for section " << SectionName
           << ":\n";
  });
  if (BinarySection(*BC, Section).isAllocatable()) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring runtime relocations\n");
    return;
  }
  section_iterator SecIter = cantFail(Section.getRelocatedSection());
  assert(SecIter != InputFile->section_end() && "relocated section expected");
  SectionRef RelocatedSection = *SecIter;

  StringRef RelocatedSectionName = cantFail(RelocatedSection.getName());
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: relocated section is "
                    << RelocatedSectionName << '\n');

  if (!BinarySection(*BC, RelocatedSection).isAllocatable()) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring relocations against "
                      << "non-allocatable section\n");
    return;
  }
  const bool SkipRelocs = StringSwitch<bool>(RelocatedSectionName)
                              .Cases(".plt", ".rela.plt", ".got.plt",
                                     ".eh_frame", ".gcc_except_table", true)
                              .Default(false);
  if (SkipRelocs) {
    LLVM_DEBUG(
        dbgs() << "BOLT-DEBUG: ignoring relocations against known section\n");
    return;
  }

  for (const RelocationRef &Rel : Section.relocations())
    handleRelocation(RelocatedSection, Rel);
}

void RewriteInstance::handleRelocation(const SectionRef &RelocatedSection,
                                       const RelocationRef &Rel) {
  const bool IsAArch64 = BC->isAArch64();
  const bool IsFromCode = RelocatedSection.isText();

  SmallString<16> TypeName;
  Rel.getTypeName(TypeName);
  uint64_t RType = Rel.getType();
  if (Relocation::skipRelocationType(RType))
    return;

  // Adjust the relocation type as the linker might have skewed it.
  if (BC->isX86() && (RType & ELF::R_X86_64_converted_reloc_bit)) {
    if (opts::Verbosity >= 1)
      dbgs() << "BOLT-WARNING: ignoring R_X86_64_converted_reloc_bit\n";
    RType &= ~ELF::R_X86_64_converted_reloc_bit;
  }

  if (Relocation::isTLS(RType)) {
    // No special handling required for TLS relocations on X86.
    if (BC->isX86())
      return;

    // The non-got related TLS relocations on AArch64 also could be skipped.
    if (!Relocation::isGOT(RType))
      return;
  }

  if (!IsAArch64 && BC->getDynamicRelocationAt(Rel.getOffset())) {
    LLVM_DEBUG({
      dbgs() << formatv("BOLT-DEBUG: address {0:x} has a ", Rel.getOffset())
             << "dynamic relocation against it. Ignoring static relocation.\n";
    });
    return;
  }

  std::string SymbolName;
  uint64_t SymbolAddress;
  int64_t Addend;
  uint64_t ExtractedValue;
  bool IsSectionRelocation;
  bool Skip;
  if (!analyzeRelocation(Rel, RType, SymbolName, IsSectionRelocation,
                         SymbolAddress, Addend, ExtractedValue, Skip)) {
    LLVM_DEBUG({
      dbgs() << "BOLT-WARNING: failed to analyze relocation @ offset = "
             << formatv("{0:x}; type name = {1}\n", Rel.getOffset(), TypeName);
    });
    ++NumFailedRelocations;
    return;
  }

  if (Skip) {
    LLVM_DEBUG({
      dbgs() << "BOLT-DEBUG: skipping relocation @ offset = "
             << formatv("{0:x}; type name = {1}\n", Rel.getOffset(), TypeName);
    });
    return;
  }

  const uint64_t Address = SymbolAddress + Addend;

  LLVM_DEBUG({
    dbgs() << "BOLT-DEBUG: ";
    printRelocationInfo(Rel, SymbolName, SymbolAddress, Addend, ExtractedValue);
  });

  BinaryFunction *ContainingBF = nullptr;
  if (IsFromCode) {
    ContainingBF =
        BC->getBinaryFunctionContainingAddress(Rel.getOffset(),
                                               /*CheckPastEnd*/ false,
                                               /*UseMaxSize*/ true);
    assert(ContainingBF && "cannot find function for address in code");
    if (!IsAArch64 && !ContainingBF->containsAddress(Rel.getOffset())) {
      if (opts::Verbosity >= 1)
        outs() << formatv("BOLT-INFO: {0} has relocations in padding area\n",
                          *ContainingBF);
      ContainingBF->setSize(ContainingBF->getMaxSize());
      ContainingBF->setSimple(false);
      return;
    }
  }

  MCSymbol *ReferencedSymbol = nullptr;
  if (!IsSectionRelocation)
    if (BinaryData *BD = BC->getBinaryDataByName(SymbolName))
      ReferencedSymbol = BD->getSymbol();

  ErrorOr<BinarySection &> ReferencedSection =
      BC->getSectionForAddress(SymbolAddress);

  const bool IsToCode = ReferencedSection && ReferencedSection->isText();

  // Special handling of PC-relative relocations.
  if (!IsAArch64 && Relocation::isPCRelative(RType)) {
    if (!IsFromCode && IsToCode) {
      // PC-relative relocations from data to code are tricky since the
      // original information is typically lost after linking, even with
      // '--emit-relocs'. Such relocations are normally used by PIC-style
      // jump tables and they reference both the jump table and jump
      // targets by computing the difference between the two. If we blindly
      // apply the relocation, it will appear that it references an arbitrary
      // location in the code, possibly in a different function from the one
      // containing the jump table.
      //
      // For that reason, we only register the fact that there is a
      // PC-relative relocation at a given address against the code.
      // The actual referenced label/address will be determined during jump
      // table analysis.
      BC->addPCRelativeDataRelocation(Rel.getOffset());
    } else if (ContainingBF && !IsSectionRelocation && ReferencedSymbol) {
      // If we know the referenced symbol, register the relocation from
      // the code. It's required  to properly handle cases where
      // "symbol + addend" references an object different from "symbol".
      ContainingBF->addRelocation(Rel.getOffset(), ReferencedSymbol, RType,
                                  Addend, ExtractedValue);
    } else {
      LLVM_DEBUG({
        dbgs() << "BOLT-DEBUG: not creating PC-relative relocation at"
               << formatv("{0:x} for {1}\n", Rel.getOffset(), SymbolName);
      });
    }

    return;
  }

  bool ForceRelocation = BC->forceSymbolRelocations(SymbolName);
  if (BC->isAArch64() && Relocation::isGOT(RType))
    ForceRelocation = true;

  if (!ReferencedSection && !ForceRelocation) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: cannot determine referenced section.\n");
    return;
  }

  // Occasionally we may see a reference past the last byte of the function
  // typically as a result of __builtin_unreachable(). Check it here.
  BinaryFunction *ReferencedBF = BC->getBinaryFunctionContainingAddress(
      Address, /*CheckPastEnd*/ true, /*UseMaxSize*/ IsAArch64);

  if (!IsSectionRelocation) {
    if (BinaryFunction *BF =
            BC->getBinaryFunctionContainingAddress(SymbolAddress)) {
      if (BF != ReferencedBF) {
        // It's possible we are referencing a function without referencing any
        // code, e.g. when taking a bitmask action on a function address.
        errs() << "BOLT-WARNING: non-standard function reference (e.g. bitmask)"
               << formatv(" detected against function {0} from ", *BF);
        if (IsFromCode)
          errs() << formatv("function {0}\n", *ContainingBF);
        else
          errs() << formatv("data section at {0:x}\n", Rel.getOffset());
        LLVM_DEBUG(printRelocationInfo(Rel, SymbolName, SymbolAddress, Addend,
                                       ExtractedValue));
        ReferencedBF = BF;
      }
    }
  } else if (ReferencedBF) {
    assert(ReferencedSection && "section expected for section relocation");
    if (*ReferencedBF->getOriginSection() != *ReferencedSection) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring false function reference\n");
      ReferencedBF = nullptr;
    }
  }

  // Workaround for a member function pointer de-virtualization bug. We check
  // if a non-pc-relative relocation in the code is pointing to (fptr - 1).
  if (IsToCode && ContainingBF && !Relocation::isPCRelative(RType) &&
      (!ReferencedBF || (ReferencedBF->getAddress() != Address))) {
    if (const BinaryFunction *RogueBF =
            BC->getBinaryFunctionAtAddress(Address + 1)) {
      // Do an extra check that the function was referenced previously.
      // It's a linear search, but it should rarely happen.
      auto CheckReloc = [&](const Relocation &Rel) {
        return Rel.Symbol == RogueBF->getSymbol() &&
               !Relocation::isPCRelative(Rel.Type);
      };
      bool Found = llvm::any_of(
          llvm::make_second_range(ContainingBF->Relocations), CheckReloc);

      if (Found) {
        errs() << "BOLT-WARNING: detected possible compiler de-virtualization "
                  "bug: -1 addend used with non-pc-relative relocation against "
               << formatv("function {0} in function {1}\n", *RogueBF,
                          *ContainingBF);
        return;
      }
    }
  }

  if (ForceRelocation) {
    std::string Name =
        Relocation::isGOT(RType) ? "__BOLT_got_zero" : SymbolName;
    ReferencedSymbol = BC->registerNameAtAddress(Name, 0, 0, 0);
    SymbolAddress = 0;
    if (Relocation::isGOT(RType))
      Addend = Address;
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: forcing relocation against symbol "
                      << SymbolName << " with addend " << Addend << '\n');
  } else if (ReferencedBF) {
    ReferencedSymbol = ReferencedBF->getSymbol();
    uint64_t RefFunctionOffset = 0;

    // Adjust the point of reference to a code location inside a function.
    if (ReferencedBF->containsAddress(Address, /*UseMaxSize = */ true)) {
      RefFunctionOffset = Address - ReferencedBF->getAddress();
      if (RefFunctionOffset) {
        if (ContainingBF && ContainingBF != ReferencedBF) {
          ReferencedSymbol =
              ReferencedBF->addEntryPointAtOffset(RefFunctionOffset);
        } else {
          ReferencedSymbol =
              ReferencedBF->getOrCreateLocalLabel(Address,
                                                  /*CreatePastEnd =*/true);
          ReferencedBF->registerReferencedOffset(RefFunctionOffset);
        }
        if (opts::Verbosity > 1 &&
            BinarySection(*BC, RelocatedSection).isWritable())
          errs() << "BOLT-WARNING: writable reference into the middle of the "
                 << formatv("function {0} detected at address {1:x}\n",
                            *ReferencedBF, Rel.getOffset());
      }
      SymbolAddress = Address;
      Addend = 0;
    }
    LLVM_DEBUG({
      dbgs() << "  referenced function " << *ReferencedBF;
      if (Address != ReferencedBF->getAddress())
        dbgs() << formatv(" at offset {0:x}", RefFunctionOffset);
      dbgs() << '\n';
    });
  } else {
    if (IsToCode && SymbolAddress) {
      // This can happen e.g. with PIC-style jump tables.
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: no corresponding function for "
                           "relocation against code\n");
    }

    // In AArch64 there are zero reasons to keep a reference to the
    // "original" symbol plus addend. The original symbol is probably just a
    // section symbol. If we are here, this means we are probably accessing
    // data, so it is imperative to keep the original address.
    if (IsAArch64) {
      SymbolName = formatv("SYMBOLat{0:x}", Address);
      SymbolAddress = Address;
      Addend = 0;
    }

    if (BinaryData *BD = BC->getBinaryDataContainingAddress(SymbolAddress)) {
      // Note: this assertion is trying to check sanity of BinaryData objects
      // but AArch64 has inferred and incomplete object locations coming from
      // GOT/TLS or any other non-trivial relocation (that requires creation
      // of sections and whose symbol address is not really what should be
      // encoded in the instruction). So we essentially disabled this check
      // for AArch64 and live with bogus names for objects.
      assert((IsAArch64 || IsSectionRelocation ||
              BD->nameStartsWith(SymbolName) ||
              BD->nameStartsWith("PG" + SymbolName) ||
              (BD->nameStartsWith("ANONYMOUS") &&
               (BD->getSectionName().startswith(".plt") ||
                BD->getSectionName().endswith(".plt")))) &&
             "BOLT symbol names of all non-section relocations must match up "
             "with symbol names referenced in the relocation");

      if (IsSectionRelocation)
        BC->markAmbiguousRelocations(*BD, Address);

      ReferencedSymbol = BD->getSymbol();
      Addend += (SymbolAddress - BD->getAddress());
      SymbolAddress = BD->getAddress();
      assert(Address == SymbolAddress + Addend);
    } else {
      // These are mostly local data symbols but undefined symbols
      // in relocation sections can get through here too, from .plt.
      assert(
          (IsAArch64 || IsSectionRelocation ||
           BC->getSectionNameForAddress(SymbolAddress)->startswith(".plt")) &&
          "known symbols should not resolve to anonymous locals");

      if (IsSectionRelocation) {
        ReferencedSymbol =
            BC->getOrCreateGlobalSymbol(SymbolAddress, "SYMBOLat");
      } else {
        SymbolRef Symbol = *Rel.getSymbol();
        const uint64_t SymbolSize =
            IsAArch64 ? 0 : ELFSymbolRef(Symbol).getSize();
        const uint64_t SymbolAlignment = IsAArch64 ? 1 : Symbol.getAlignment();
        const uint32_t SymbolFlags = cantFail(Symbol.getFlags());
        std::string Name;
        if (SymbolFlags & SymbolRef::SF_Global) {
          Name = SymbolName;
        } else {
          if (StringRef(SymbolName)
                  .startswith(BC->AsmInfo->getPrivateGlobalPrefix()))
            Name = NR.uniquify("PG" + SymbolName);
          else
            Name = NR.uniquify(SymbolName);
        }
        ReferencedSymbol = BC->registerNameAtAddress(
            Name, SymbolAddress, SymbolSize, SymbolAlignment, SymbolFlags);
      }

      if (IsSectionRelocation) {
        BinaryData *BD = BC->getBinaryDataByName(ReferencedSymbol->getName());
        BC->markAmbiguousRelocations(*BD, Address);
      }
    }
  }

  auto checkMaxDataRelocations = [&]() {
    ++NumDataRelocations;
    LLVM_DEBUG(if (opts::MaxDataRelocations &&
                   NumDataRelocations + 1 == opts::MaxDataRelocations) {
      dbgs() << "BOLT-DEBUG: processing ending on data relocation "
             << NumDataRelocations << ": ";
      printRelocationInfo(Rel, ReferencedSymbol->getName(), SymbolAddress,
                          Addend, ExtractedValue);
    });

    return (!opts::MaxDataRelocations ||
            NumDataRelocations < opts::MaxDataRelocations);
  };

  if ((ReferencedSection && refersToReorderedSection(ReferencedSection)) ||
      (opts::ForceToDataRelocations && checkMaxDataRelocations()))
    ForceRelocation = true;

  if (IsFromCode) {
    ContainingBF->addRelocation(Rel.getOffset(), ReferencedSymbol, RType,
                                Addend, ExtractedValue);
  } else if (IsToCode || ForceRelocation) {
    BC->addRelocation(Rel.getOffset(), ReferencedSymbol, RType, Addend,
                      ExtractedValue);
  } else {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring relocation from data to data\n");
  }
}

void RewriteInstance::selectFunctionsToProcess() {
  // Extend the list of functions to process or skip from a file.
  auto populateFunctionNames = [](cl::opt<std::string> &FunctionNamesFile,
                                  cl::list<std::string> &FunctionNames) {
    if (FunctionNamesFile.empty())
      return;
    std::ifstream FuncsFile(FunctionNamesFile, std::ios::in);
    std::string FuncName;
    while (std::getline(FuncsFile, FuncName))
      FunctionNames.push_back(FuncName);
  };
  populateFunctionNames(opts::FunctionNamesFile, opts::ForceFunctionNames);
  populateFunctionNames(opts::SkipFunctionNamesFile, opts::SkipFunctionNames);
  populateFunctionNames(opts::FunctionNamesFileNR, opts::ForceFunctionNamesNR);

  // Make a set of functions to process to speed up lookups.
  std::unordered_set<std::string> ForceFunctionsNR(
      opts::ForceFunctionNamesNR.begin(), opts::ForceFunctionNamesNR.end());

  if ((!opts::ForceFunctionNames.empty() ||
       !opts::ForceFunctionNamesNR.empty()) &&
      !opts::SkipFunctionNames.empty()) {
    errs() << "BOLT-ERROR: cannot select functions to process and skip at the "
              "same time. Please use only one type of selection.\n";
    exit(1);
  }

  uint64_t LiteThresholdExecCount = 0;
  if (opts::LiteThresholdPct) {
    if (opts::LiteThresholdPct > 100)
      opts::LiteThresholdPct = 100;

    std::vector<const BinaryFunction *> TopFunctions;
    for (auto &BFI : BC->getBinaryFunctions()) {
      const BinaryFunction &Function = BFI.second;
      if (ProfileReader->mayHaveProfileData(Function))
        TopFunctions.push_back(&Function);
    }
    llvm::sort(
        TopFunctions, [](const BinaryFunction *A, const BinaryFunction *B) {
          return A->getKnownExecutionCount() < B->getKnownExecutionCount();
        });

    size_t Index = TopFunctions.size() * opts::LiteThresholdPct / 100;
    if (Index)
      --Index;
    LiteThresholdExecCount = TopFunctions[Index]->getKnownExecutionCount();
    outs() << "BOLT-INFO: limiting processing to functions with at least "
           << LiteThresholdExecCount << " invocations\n";
  }
  LiteThresholdExecCount = std::max(
      LiteThresholdExecCount, static_cast<uint64_t>(opts::LiteThresholdCount));

  StringSet<> ReorderFunctionsUserSet;
  if (opts::ReorderFunctions == ReorderFunctions::RT_USER) {
    for (const std::string &Function :
         ReorderFunctions::readFunctionOrderFile())
      ReorderFunctionsUserSet.insert(Function);
  }

  uint64_t NumFunctionsToProcess = 0;
  auto shouldProcess = [&](const BinaryFunction &Function) {
    if (opts::MaxFunctions && NumFunctionsToProcess > opts::MaxFunctions)
      return false;

    // If the list is not empty, only process functions from the list.
    if (!opts::ForceFunctionNames.empty() || !ForceFunctionsNR.empty()) {
      // Regex check (-funcs and -funcs-file options).
      for (std::string &Name : opts::ForceFunctionNames)
        if (Function.hasNameRegex(Name))
          return true;

      // Non-regex check (-funcs-no-regex and -funcs-file-no-regex).
      std::optional<StringRef> Match =
          Function.forEachName([&ForceFunctionsNR](StringRef Name) {
            return ForceFunctionsNR.count(Name.str());
          });
      return Match.has_value();
    }

    for (std::string &Name : opts::SkipFunctionNames)
      if (Function.hasNameRegex(Name))
        return false;

    if (opts::Lite) {
      // Forcibly include functions specified in the -function-order file.
      if (opts::ReorderFunctions == ReorderFunctions::RT_USER) {
        std::optional<StringRef> Match =
            Function.forEachName([&](StringRef Name) {
              return ReorderFunctionsUserSet.contains(Name);
            });
        if (Match.has_value())
          return true;
      }

      if (ProfileReader && !ProfileReader->mayHaveProfileData(Function))
        return false;

      if (Function.getKnownExecutionCount() < LiteThresholdExecCount)
        return false;
    }

    return true;
  };

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    // Pseudo functions are explicitly marked by us not to be processed.
    if (Function.isPseudo()) {
      Function.IsIgnored = true;
      Function.HasExternalRefRelocations = true;
      continue;
    }

    if (!shouldProcess(Function)) {
      LLVM_DEBUG(dbgs() << "BOLT-INFO: skipping processing of function "
                        << Function << " per user request\n");
      Function.setIgnored();
    } else {
      ++NumFunctionsToProcess;
      if (opts::MaxFunctions && NumFunctionsToProcess == opts::MaxFunctions)
        outs() << "BOLT-INFO: processing ending on " << Function << '\n';
    }
  }
}

void RewriteInstance::readDebugInfo() {
  NamedRegionTimer T("readDebugInfo", "read debug info", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);
  if (!opts::UpdateDebugSections)
    return;

  BC->preprocessDebugInfo();
}

void RewriteInstance::preprocessProfileData() {
  if (!ProfileReader)
    return;

  NamedRegionTimer T("preprocessprofile", "pre-process profile data",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);

  outs() << "BOLT-INFO: pre-processing profile using "
         << ProfileReader->getReaderName() << '\n';

  if (BAT->enabledFor(InputFile)) {
    outs() << "BOLT-INFO: profile collection done on a binary already "
              "processed by BOLT\n";
    ProfileReader->setBAT(&*BAT);
  }

  if (Error E = ProfileReader->preprocessProfile(*BC.get()))
    report_error("cannot pre-process profile", std::move(E));

  if (!BC->hasSymbolsWithFileName() && ProfileReader->hasLocalsWithFileName()) {
    errs() << "BOLT-ERROR: input binary does not have local file symbols "
              "but profile data includes function names with embedded file "
              "names. It appears that the input binary was stripped while a "
              "profiled binary was not\n";
    exit(1);
  }
}

void RewriteInstance::processProfileDataPreCFG() {
  if (!ProfileReader)
    return;

  NamedRegionTimer T("processprofile-precfg", "process profile data pre-CFG",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);

  if (Error E = ProfileReader->readProfilePreCFG(*BC.get()))
    report_error("cannot read profile pre-CFG", std::move(E));
}

void RewriteInstance::processProfileData() {
  if (!ProfileReader)
    return;

  NamedRegionTimer T("processprofile", "process profile data", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);

  if (Error E = ProfileReader->readProfile(*BC.get()))
    report_error("cannot read profile", std::move(E));

  if (!opts::SaveProfile.empty()) {
    YAMLProfileWriter PW(opts::SaveProfile);
    PW.writeProfile(*this);
  }
  if (opts::AggregateOnly &&
      opts::ProfileFormat == opts::ProfileFormatKind::PF_YAML) {
    YAMLProfileWriter PW(opts::OutputFilename);
    PW.writeProfile(*this);
  }

  // Release memory used by profile reader.
  ProfileReader.reset();

  if (opts::AggregateOnly)
    exit(0);
}

void RewriteInstance::disassembleFunctions() {
  NamedRegionTimer T("disassembleFunctions", "disassemble functions",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    ErrorOr<ArrayRef<uint8_t>> FunctionData = Function.getData();
    if (!FunctionData) {
      errs() << "BOLT-ERROR: corresponding section is non-executable or "
             << "empty for function " << Function << '\n';
      exit(1);
    }

    // Treat zero-sized functions as non-simple ones.
    if (Function.getSize() == 0) {
      Function.setSimple(false);
      continue;
    }

    // Offset of the function in the file.
    const auto *FileBegin =
        reinterpret_cast<const uint8_t *>(InputFile->getData().data());
    Function.setFileOffset(FunctionData->begin() - FileBegin);

    if (!shouldDisassemble(Function)) {
      NamedRegionTimer T("scan", "scan functions", "buildfuncs",
                         "Scan Binary Functions", opts::TimeBuild);
      Function.scanExternalRefs();
      Function.setSimple(false);
      continue;
    }

    if (!Function.disassemble()) {
      if (opts::processAllFunctions())
        BC->exitWithBugReport("function cannot be properly disassembled. "
                              "Unable to continue in relocation mode.",
                              Function);
      if (opts::Verbosity >= 1)
        outs() << "BOLT-INFO: could not disassemble function " << Function
               << ". Will ignore.\n";
      // Forcefully ignore the function.
      Function.setIgnored();
      continue;
    }

    if (opts::PrintAll || opts::PrintDisasm)
      Function.print(outs(), "after disassembly");
  }

  BC->processInterproceduralReferences();
  BC->populateJumpTables();

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    if (!shouldDisassemble(Function))
      continue;

    Function.postProcessEntryPoints();
    Function.postProcessJumpTables();
  }

  BC->clearJumpTableTempData();
  BC->adjustCodePadding();

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    if (!shouldDisassemble(Function))
      continue;

    if (!Function.isSimple()) {
      assert((!BC->HasRelocations || Function.getSize() == 0 ||
              Function.hasIndirectTargetToSplitFragment()) &&
             "unexpected non-simple function in relocation mode");
      continue;
    }

    // Fill in CFI information for this function
    if (!Function.trapsOnEntry() && !CFIRdWrt->fillCFIInfoFor(Function)) {
      if (BC->HasRelocations) {
        BC->exitWithBugReport("unable to fill CFI.", Function);
      } else {
        errs() << "BOLT-WARNING: unable to fill CFI for function " << Function
               << ". Skipping.\n";
        Function.setSimple(false);
        continue;
      }
    }

    // Parse LSDA.
    if (Function.getLSDAAddress() != 0 &&
        !BC->getFragmentsToSkip().count(&Function))
      Function.parseLSDA(getLSDAData(), getLSDAAddress());
  }
}

void RewriteInstance::buildFunctionsCFG() {
  NamedRegionTimer T("buildCFG", "buildCFG", "buildfuncs",
                     "Build Binary Functions", opts::TimeBuild);

  // Create annotation indices to allow lock-free execution
  BC->MIB->getOrCreateAnnotationIndex("JTIndexReg");
  BC->MIB->getOrCreateAnnotationIndex("NOP");
  BC->MIB->getOrCreateAnnotationIndex("Size");

  ParallelUtilities::WorkFuncWithAllocTy WorkFun =
      [&](BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocId) {
        if (!BF.buildCFG(AllocId))
          return;

        if (opts::PrintAll) {
          auto L = BC->scopeLock();
          BF.print(outs(), "while building cfg");
        }
      };

  ParallelUtilities::PredicateTy SkipPredicate = [&](const BinaryFunction &BF) {
    return !shouldDisassemble(BF) || !BF.isSimple();
  };

  ParallelUtilities::runOnEachFunctionWithUniqueAllocId(
      *BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, WorkFun,
      SkipPredicate, "disassembleFunctions-buildCFG",
      /*ForceSequential*/ opts::SequentialDisassembly || opts::PrintAll);

  BC->postProcessSymbolTable();
}

void RewriteInstance::postProcessFunctions() {
  // We mark fragments as non-simple here, not during disassembly,
  // So we can build their CFGs.
  BC->skipMarkedFragments();
  BC->clearFragmentsToSkip();

  BC->TotalScore = 0;
  BC->SumExecutionCount = 0;
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    if (Function.empty())
      continue;

    Function.postProcessCFG();

    if (opts::PrintAll || opts::PrintCFG)
      Function.print(outs(), "after building cfg");

    if (opts::DumpDotAll)
      Function.dumpGraphForPass("00_build-cfg");

    if (opts::PrintLoopInfo) {
      Function.calculateLoopInfo();
      Function.printLoopInfo(outs());
    }

    BC->TotalScore += Function.getFunctionScore();
    BC->SumExecutionCount += Function.getKnownExecutionCount();
  }

  if (opts::PrintGlobals) {
    outs() << "BOLT-INFO: Global symbols:\n";
    BC->printGlobalSymbols(outs());
  }
}

void RewriteInstance::runOptimizationPasses() {
  NamedRegionTimer T("runOptimizationPasses", "run optimization passes",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  BinaryFunctionPassManager::runAllPasses(*BC);
}

namespace {

class BOLTSymbolResolver : public JITSymbolResolver {
  BinaryContext &BC;

public:
  BOLTSymbolResolver(BinaryContext &BC) : BC(BC) {}

  // We are responsible for all symbols
  Expected<LookupSet> getResponsibilitySet(const LookupSet &Symbols) override {
    return Symbols;
  }

  // Some of our symbols may resolve to zero and this should not be an error
  bool allowsZeroSymbols() override { return true; }

  /// Resolves the address of each symbol requested
  void lookup(const LookupSet &Symbols,
              OnResolvedFunction OnResolved) override {
    JITSymbolResolver::LookupResult AllResults;

    if (BC.EFMM->ObjectsLoaded) {
      for (const StringRef &Symbol : Symbols) {
        std::string SymName = Symbol.str();
        LLVM_DEBUG(dbgs() << "BOLT: looking for " << SymName << "\n");
        // Resolve to a PLT entry if possible
        if (const BinaryData *I = BC.getPLTBinaryDataByName(SymName)) {
          AllResults[Symbol] =
              JITEvaluatedSymbol(I->getAddress(), JITSymbolFlags());
          continue;
        }
        OnResolved(make_error<StringError>(
            "Symbol not found required by runtime: " + Symbol,
            inconvertibleErrorCode()));
        return;
      }
      OnResolved(std::move(AllResults));
      return;
    }

    for (const StringRef &Symbol : Symbols) {
      std::string SymName = Symbol.str();
      LLVM_DEBUG(dbgs() << "BOLT: looking for " << SymName << "\n");

      if (BinaryData *I = BC.getBinaryDataByName(SymName)) {
        uint64_t Address = I->isMoved() && !I->isJumpTable()
                               ? I->getOutputAddress()
                               : I->getAddress();
        LLVM_DEBUG(dbgs() << "Resolved to address 0x"
                          << Twine::utohexstr(Address) << "\n");
        AllResults[Symbol] = JITEvaluatedSymbol(Address, JITSymbolFlags());
        continue;
      }
      LLVM_DEBUG(dbgs() << "Resolved to address 0x0\n");
      AllResults[Symbol] = JITEvaluatedSymbol(0, JITSymbolFlags());
    }

    OnResolved(std::move(AllResults));
  }
};

} // anonymous namespace

void RewriteInstance::preregisterSections() {
  // Preregister sections before emission to set their order in the output.
  const unsigned ROFlags = BinarySection::getFlags(/*IsReadOnly*/ true,
                                                   /*IsText*/ false,
                                                   /*IsAllocatable*/ true);
  if (BinarySection *EHFrameSection = getSection(getEHFrameSectionName())) {
    // New .eh_frame.
    BC->registerOrUpdateSection(getNewSecPrefix() + getEHFrameSectionName(),
                                ELF::SHT_PROGBITS, ROFlags);
    // Fully register a relocatable copy of the original .eh_frame.
    BC->registerSection(".relocated.eh_frame", *EHFrameSection);
  }
  BC->registerOrUpdateSection(getNewSecPrefix() + ".gcc_except_table",
                              ELF::SHT_PROGBITS, ROFlags);
  BC->registerOrUpdateSection(getNewSecPrefix() + ".rodata", ELF::SHT_PROGBITS,
                              ROFlags);
  BC->registerOrUpdateSection(getNewSecPrefix() + ".rodata.cold",
                              ELF::SHT_PROGBITS, ROFlags);
}

void RewriteInstance::emitAndLink() {
  NamedRegionTimer T("emitAndLink", "emit and link", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);
  std::error_code EC;

  // This is an object file, which we keep for debugging purposes.
  // Once we decide it's useless, we should create it in memory.
  SmallString<128> OutObjectPath;
  sys::fs::getPotentiallyUniqueTempFileName("output", "o", OutObjectPath);
  std::unique_ptr<ToolOutputFile> TempOut =
      std::make_unique<ToolOutputFile>(OutObjectPath, EC, sys::fs::OF_None);
  check_error(EC, "cannot create output object file");

  std::unique_ptr<buffer_ostream> BOS =
      std::make_unique<buffer_ostream>(TempOut->os());
  raw_pwrite_stream *OS = BOS.get();

  // Implicitly MCObjectStreamer takes ownership of MCAsmBackend (MAB)
  // and MCCodeEmitter (MCE). ~MCObjectStreamer() will delete these
  // two instances.
  std::unique_ptr<MCStreamer> Streamer = BC->createStreamer(*OS);

  if (EHFrameSection) {
    if (opts::UseOldText || opts::StrictMode) {
      // The section is going to be regenerated from scratch.
      // Empty the contents, but keep the section reference.
      EHFrameSection->clearContents();
    } else {
      // Make .eh_frame relocatable.
      relocateEHFrameSection();
    }
  }

  emitBinaryContext(*Streamer, *BC, getOrgSecPrefix());

  Streamer->finish();
  if (Streamer->getContext().hadError()) {
    errs() << "BOLT-ERROR: Emission failed.\n";
    exit(1);
  }

  ErrorOr<BinarySection &> TextSection =
      BC->getUniqueSectionByName(BC->getMainCodeSectionName());
  if (BC->HasRelocations && TextSection)
    BC->renameSection(*TextSection, getOrgSecPrefix() + ".text");

  //////////////////////////////////////////////////////////////////////////////
  // Assign addresses to new sections.
  //////////////////////////////////////////////////////////////////////////////

  // Get output object as ObjectFile.
  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(BOS->str(), "in-memory object file", false);
  std::unique_ptr<object::ObjectFile> Obj = cantFail(
      object::ObjectFile::createObjectFile(ObjectMemBuffer->getMemBufferRef()),
      "error creating in-memory object");

  BOLTSymbolResolver Resolver = BOLTSymbolResolver(*BC);

  MCAsmLayout FinalLayout(
      static_cast<MCObjectStreamer *>(Streamer.get())->getAssembler());

  // Disable stubs because RuntimeDyld may try to increase the size of
  // sections accounting for stubs. We need those sections to match the
  // same size seen in the input binary, in case this section is a copy
  // of the original one seen in the binary.
  BC->EFMM.reset(new ExecutableFileMemoryManager(*BC, /*AllowStubs=*/false));
  BC->EFMM->setNewSecPrefix(getNewSecPrefix());
  BC->EFMM->setOrgSecPrefix(getOrgSecPrefix());

  RTDyld.reset(new decltype(RTDyld)::element_type(*BC->EFMM, Resolver));
  RTDyld->setProcessAllSections(false);
  RTDyld->loadObject(*Obj);

  // Assign addresses to all sections. If key corresponds to the object
  // created by ourselves, call our regular mapping function. If we are
  // loading additional objects as part of runtime libraries for
  // instrumentation, treat them as extra sections.
  mapFileSections(*RTDyld);

  RTDyld->finalizeWithMemoryManagerLocking();
  if (RTDyld->hasError()) {
    errs() << "BOLT-ERROR: RTDyld failed: " << RTDyld->getErrorString() << "\n";
    exit(1);
  }

  // Update output addresses based on the new section map and
  // layout. Only do this for the object created by ourselves.
  updateOutputValues(FinalLayout);

  if (opts::UpdateDebugSections)
    DebugInfoRewriter->updateLineTableOffsets(FinalLayout);

  if (RuntimeLibrary *RtLibrary = BC->getRuntimeLibrary())
    RtLibrary->link(*BC, ToolPath, *RTDyld, [this](RuntimeDyld &R) {
      // Map newly registered sections.
      this->mapAllocatableSections(*RTDyld);
    });

  // Once the code is emitted, we can rename function sections to actual
  // output sections and de-register sections used for emission.
  for (BinaryFunction *Function : BC->getAllBinaryFunctions()) {
    ErrorOr<BinarySection &> Section = Function->getCodeSection();
    if (Section &&
        (Function->getImageAddress() == 0 || Function->getImageSize() == 0))
      continue;

    // Restore origin section for functions that were emitted or supposed to
    // be emitted to patch sections.
    if (Section)
      BC->deregisterSection(*Section);
    assert(Function->getOriginSectionName() && "expected origin section");
    Function->CodeSectionName = Function->getOriginSectionName()->str();
    for (const FunctionFragment &FF :
         Function->getLayout().getSplitFragments()) {
      if (ErrorOr<BinarySection &> ColdSection =
              Function->getCodeSection(FF.getFragmentNum()))
        BC->deregisterSection(*ColdSection);
    }
    if (Function->getLayout().isSplit())
      Function->setColdCodeSectionName(getBOLTTextSectionName());
  }

  if (opts::PrintCacheMetrics) {
    outs() << "BOLT-INFO: cache metrics after emitting functions:\n";
    CacheMetrics::printAll(BC->getSortedFunctions());
  }

  if (opts::KeepTmp) {
    TempOut->keep();
    outs() << "BOLT-INFO: intermediary output object file saved for debugging "
              "purposes: "
           << OutObjectPath << "\n";
  }
}

void RewriteInstance::updateMetadata() {
  updateSDTMarkers();
  updateLKMarkers();
  parsePseudoProbe();
  updatePseudoProbes();

  if (opts::UpdateDebugSections) {
    NamedRegionTimer T("updateDebugInfo", "update debug info", TimerGroupName,
                       TimerGroupDesc, opts::TimeRewrite);
    DebugInfoRewriter->updateDebugInfo();
  }

  if (opts::WriteBoltInfoSection)
    addBoltInfoSection();
}

void RewriteInstance::updatePseudoProbes() {
  // check if there is pseudo probe section decoded
  if (BC->ProbeDecoder.getAddress2ProbesMap().empty())
    return;
  // input address converted to output
  AddressProbesMap &Address2ProbesMap = BC->ProbeDecoder.getAddress2ProbesMap();
  const GUIDProbeFunctionMap &GUID2Func =
      BC->ProbeDecoder.getGUID2FuncDescMap();

  for (auto &AP : Address2ProbesMap) {
    BinaryFunction *F = BC->getBinaryFunctionContainingAddress(AP.first);
    // If F is removed, eliminate all probes inside it from inline tree
    // Setting probes' addresses as INT64_MAX means elimination
    if (!F) {
      for (MCDecodedPseudoProbe &Probe : AP.second)
        Probe.setAddress(INT64_MAX);
      continue;
    }
    // If F is not emitted, the function will remain in the same address as its
    // input
    if (!F->isEmitted())
      continue;

    uint64_t Offset = AP.first - F->getAddress();
    const BinaryBasicBlock *BB = F->getBasicBlockContainingOffset(Offset);
    uint64_t BlkOutputAddress = BB->getOutputAddressRange().first;
    // Check if block output address is defined.
    // If not, such block is removed from binary. Then remove the probes from
    // inline tree
    if (BlkOutputAddress == 0) {
      for (MCDecodedPseudoProbe &Probe : AP.second)
        Probe.setAddress(INT64_MAX);
      continue;
    }

    unsigned ProbeTrack = AP.second.size();
    std::list<MCDecodedPseudoProbe>::iterator Probe = AP.second.begin();
    while (ProbeTrack != 0) {
      if (Probe->isBlock()) {
        Probe->setAddress(BlkOutputAddress);
      } else if (Probe->isCall()) {
        // A call probe may be duplicated due to ICP
        // Go through output of InputOffsetToAddressMap to collect all related
        // probes
        const InputOffsetToAddressMapTy &Offset2Addr =
            F->getInputOffsetToAddressMap();
        auto CallOutputAddresses = Offset2Addr.equal_range(Offset);
        auto CallOutputAddress = CallOutputAddresses.first;
        if (CallOutputAddress == CallOutputAddresses.second) {
          Probe->setAddress(INT64_MAX);
        } else {
          Probe->setAddress(CallOutputAddress->second);
          CallOutputAddress = std::next(CallOutputAddress);
        }

        while (CallOutputAddress != CallOutputAddresses.second) {
          AP.second.push_back(*Probe);
          AP.second.back().setAddress(CallOutputAddress->second);
          Probe->getInlineTreeNode()->addProbes(&(AP.second.back()));
          CallOutputAddress = std::next(CallOutputAddress);
        }
      }
      Probe = std::next(Probe);
      ProbeTrack--;
    }
  }

  if (opts::PrintPseudoProbes == opts::PrintPseudoProbesOptions::PPP_All ||
      opts::PrintPseudoProbes ==
          opts::PrintPseudoProbesOptions::PPP_Probes_Address_Conversion) {
    outs() << "Pseudo Probe Address Conversion results:\n";
    // table that correlates address to block
    std::unordered_map<uint64_t, StringRef> Addr2BlockNames;
    for (auto &F : BC->getBinaryFunctions())
      for (BinaryBasicBlock &BinaryBlock : F.second)
        Addr2BlockNames[BinaryBlock.getOutputAddressRange().first] =
            BinaryBlock.getName();

    // scan all addresses -> correlate probe to block when print out
    std::vector<uint64_t> Addresses;
    for (auto &Entry : Address2ProbesMap)
      Addresses.push_back(Entry.first);
    llvm::sort(Addresses);
    for (uint64_t Key : Addresses) {
      for (MCDecodedPseudoProbe &Probe : Address2ProbesMap[Key]) {
        if (Probe.getAddress() == INT64_MAX)
          outs() << "Deleted Probe: ";
        else
          outs() << "Address: " << format_hex(Probe.getAddress(), 8) << " ";
        Probe.print(outs(), GUID2Func, true);
        // print block name only if the probe is block type and undeleted.
        if (Probe.isBlock() && Probe.getAddress() != INT64_MAX)
          outs() << format_hex(Probe.getAddress(), 8) << " Probe is in "
                 << Addr2BlockNames[Probe.getAddress()] << "\n";
      }
    }
    outs() << "=======================================\n";
  }

  // encode pseudo probes with updated addresses
  encodePseudoProbes();
}

template <typename F>
static void emitLEB128IntValue(F encode, uint64_t Value,
                               SmallString<8> &Contents) {
  SmallString<128> Tmp;
  raw_svector_ostream OSE(Tmp);
  encode(Value, OSE);
  Contents.append(OSE.str().begin(), OSE.str().end());
}

void RewriteInstance::encodePseudoProbes() {
  // Buffer for new pseudo probes section
  SmallString<8> Contents;
  MCDecodedPseudoProbe *LastProbe = nullptr;

  auto EmitInt = [&](uint64_t Value, uint32_t Size) {
    const bool IsLittleEndian = BC->AsmInfo->isLittleEndian();
    uint64_t Swapped = support::endian::byte_swap(
        Value, IsLittleEndian ? support::little : support::big);
    unsigned Index = IsLittleEndian ? 0 : 8 - Size;
    auto Entry = StringRef(reinterpret_cast<char *>(&Swapped) + Index, Size);
    Contents.append(Entry.begin(), Entry.end());
  };

  auto EmitULEB128IntValue = [&](uint64_t Value) {
    SmallString<128> Tmp;
    raw_svector_ostream OSE(Tmp);
    encodeULEB128(Value, OSE, 0);
    Contents.append(OSE.str().begin(), OSE.str().end());
  };

  auto EmitSLEB128IntValue = [&](int64_t Value) {
    SmallString<128> Tmp;
    raw_svector_ostream OSE(Tmp);
    encodeSLEB128(Value, OSE);
    Contents.append(OSE.str().begin(), OSE.str().end());
  };

  // Emit indiviual pseudo probes in a inline tree node
  // Probe index, type, attribute, address type and address are encoded
  // Address of the first probe is absolute.
  // Other probes' address are represented by delta
  auto EmitDecodedPseudoProbe = [&](MCDecodedPseudoProbe *&CurProbe) {
    assert(!isSentinelProbe(CurProbe->getAttributes()) &&
           "Sentinel probes should not be emitted");
    EmitULEB128IntValue(CurProbe->getIndex());
    uint8_t PackedType = CurProbe->getType() | (CurProbe->getAttributes() << 4);
    uint8_t Flag =
        LastProbe ? ((int8_t)MCPseudoProbeFlag::AddressDelta << 7) : 0;
    EmitInt(Flag | PackedType, 1);
    if (LastProbe) {
      // Emit the delta between the address label and LastProbe.
      int64_t Delta = CurProbe->getAddress() - LastProbe->getAddress();
      EmitSLEB128IntValue(Delta);
    } else {
      // Emit absolute address for encoding the first pseudo probe.
      uint32_t AddrSize = BC->AsmInfo->getCodePointerSize();
      EmitInt(CurProbe->getAddress(), AddrSize);
    }
  };

  std::map<InlineSite, MCDecodedPseudoProbeInlineTree *,
           std::greater<InlineSite>>
      Inlinees;

  // DFS of inline tree to emit pseudo probes in all tree node
  // Inline site index of a probe is emitted first.
  // Then tree node Guid, size of pseudo probes and children nodes, and detail
  // of contained probes are emitted Deleted probes are skipped Root node is not
  // encoded to binaries. It's a "wrapper" of inline trees of each function.
  std::list<std::pair<uint64_t, MCDecodedPseudoProbeInlineTree *>> NextNodes;
  const MCDecodedPseudoProbeInlineTree &Root =
      BC->ProbeDecoder.getDummyInlineRoot();
  for (auto Child = Root.getChildren().begin();
       Child != Root.getChildren().end(); ++Child)
    Inlinees[Child->first] = Child->second.get();

  for (auto Inlinee : Inlinees)
    // INT64_MAX is "placeholder" of unused callsite index field in the pair
    NextNodes.push_back({INT64_MAX, Inlinee.second});

  Inlinees.clear();

  while (!NextNodes.empty()) {
    uint64_t ProbeIndex = NextNodes.back().first;
    MCDecodedPseudoProbeInlineTree *Cur = NextNodes.back().second;
    NextNodes.pop_back();

    if (Cur->Parent && !Cur->Parent->isRoot())
      // Emit probe inline site
      EmitULEB128IntValue(ProbeIndex);

    // Emit probes grouped by GUID.
    LLVM_DEBUG({
      dbgs().indent(MCPseudoProbeTable::DdgPrintIndent);
      dbgs() << "GUID: " << Cur->Guid << "\n";
    });
    // Emit Guid
    EmitInt(Cur->Guid, 8);
    // Emit number of probes in this node
    uint64_t Deleted = 0;
    for (MCDecodedPseudoProbe *&Probe : Cur->getProbes())
      if (Probe->getAddress() == INT64_MAX)
        Deleted++;
    LLVM_DEBUG(dbgs() << "Deleted Probes:" << Deleted << "\n");
    uint64_t ProbesSize = Cur->getProbes().size() - Deleted;
    EmitULEB128IntValue(ProbesSize);
    // Emit number of direct inlinees
    EmitULEB128IntValue(Cur->getChildren().size());
    // Emit probes in this group
    for (MCDecodedPseudoProbe *&Probe : Cur->getProbes()) {
      if (Probe->getAddress() == INT64_MAX)
        continue;
      EmitDecodedPseudoProbe(Probe);
      LastProbe = Probe;
    }

    for (auto Child = Cur->getChildren().begin();
         Child != Cur->getChildren().end(); ++Child)
      Inlinees[Child->first] = Child->second.get();
    for (const auto &Inlinee : Inlinees) {
      assert(Cur->Guid != 0 && "non root tree node must have nonzero Guid");
      NextNodes.push_back({std::get<1>(Inlinee.first), Inlinee.second});
      LLVM_DEBUG({
        dbgs().indent(MCPseudoProbeTable::DdgPrintIndent);
        dbgs() << "InlineSite: " << std::get<1>(Inlinee.first) << "\n";
      });
    }
    Inlinees.clear();
  }

  // Create buffer for new contents for the section
  // Freed when parent section is destroyed
  uint8_t *Output = new uint8_t[Contents.str().size()];
  memcpy(Output, Contents.str().data(), Contents.str().size());
  addToDebugSectionsToOverwrite(".pseudo_probe");
  BC->registerOrUpdateSection(".pseudo_probe", PseudoProbeSection->getELFType(),
                              PseudoProbeSection->getELFFlags(), Output,
                              Contents.str().size(), 1);
  if (opts::PrintPseudoProbes == opts::PrintPseudoProbesOptions::PPP_All ||
      opts::PrintPseudoProbes ==
          opts::PrintPseudoProbesOptions::PPP_Encoded_Probes) {
    // create a dummy decoder;
    MCPseudoProbeDecoder DummyDecoder;
    StringRef DescContents = PseudoProbeDescSection->getContents();
    DummyDecoder.buildGUID2FuncDescMap(
        reinterpret_cast<const uint8_t *>(DescContents.data()),
        DescContents.size());
    StringRef ProbeContents = PseudoProbeSection->getOutputContents();
    MCPseudoProbeDecoder::Uint64Set GuidFilter;
    MCPseudoProbeDecoder::Uint64Map FuncStartAddrs;
    for (const BinaryFunction *F : BC->getAllBinaryFunctions()) {
      const uint64_t Addr =
          F->isEmitted() ? F->getOutputAddress() : F->getAddress();
      FuncStartAddrs[Function::getGUID(
          NameResolver::restore(F->getOneName()))] = Addr;
    }
    DummyDecoder.buildAddress2ProbeMap(
        reinterpret_cast<const uint8_t *>(ProbeContents.data()),
        ProbeContents.size(), GuidFilter, FuncStartAddrs);
    DummyDecoder.printProbesForAllAddresses(outs());
  }
}

void RewriteInstance::updateSDTMarkers() {
  NamedRegionTimer T("updateSDTMarkers", "update SDT markers", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);

  if (!SDTSection)
    return;
  SDTSection->registerPatcher(std::make_unique<SimpleBinaryPatcher>());

  SimpleBinaryPatcher *SDTNotePatcher =
      static_cast<SimpleBinaryPatcher *>(SDTSection->getPatcher());
  for (auto &SDTInfoKV : BC->SDTMarkers) {
    const uint64_t OriginalAddress = SDTInfoKV.first;
    SDTMarkerInfo &SDTInfo = SDTInfoKV.second;
    const BinaryFunction *F =
        BC->getBinaryFunctionContainingAddress(OriginalAddress);
    if (!F)
      continue;
    const uint64_t NewAddress =
        F->translateInputToOutputAddress(OriginalAddress);
    SDTNotePatcher->addLE64Patch(SDTInfo.PCOffset, NewAddress);
  }
}

void RewriteInstance::updateLKMarkers() {
  if (BC->LKMarkers.size() == 0)
    return;

  NamedRegionTimer T("updateLKMarkers", "update LK markers", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);

  std::unordered_map<std::string, uint64_t> PatchCounts;
  for (std::pair<const uint64_t, std::vector<LKInstructionMarkerInfo>>
           &LKMarkerInfoKV : BC->LKMarkers) {
    const uint64_t OriginalAddress = LKMarkerInfoKV.first;
    const BinaryFunction *BF =
        BC->getBinaryFunctionContainingAddress(OriginalAddress, false, true);
    if (!BF)
      continue;

    uint64_t NewAddress = BF->translateInputToOutputAddress(OriginalAddress);
    if (NewAddress == 0)
      continue;

    // Apply base address.
    if (OriginalAddress >= 0xffffffff00000000 && NewAddress < 0xffffffff)
      NewAddress = NewAddress + 0xffffffff00000000;

    if (OriginalAddress == NewAddress)
      continue;

    for (LKInstructionMarkerInfo &LKMarkerInfo : LKMarkerInfoKV.second) {
      StringRef SectionName = LKMarkerInfo.SectionName;
      SimpleBinaryPatcher *LKPatcher;
      ErrorOr<BinarySection &> BSec = BC->getUniqueSectionByName(SectionName);
      assert(BSec && "missing section info for kernel section");
      if (!BSec->getPatcher())
        BSec->registerPatcher(std::make_unique<SimpleBinaryPatcher>());
      LKPatcher = static_cast<SimpleBinaryPatcher *>(BSec->getPatcher());
      PatchCounts[std::string(SectionName)]++;
      if (LKMarkerInfo.IsPCRelative)
        LKPatcher->addLE32Patch(LKMarkerInfo.SectionOffset,
                                NewAddress - OriginalAddress +
                                    LKMarkerInfo.PCRelativeOffset);
      else
        LKPatcher->addLE64Patch(LKMarkerInfo.SectionOffset, NewAddress);
    }
  }
  outs() << "BOLT-INFO: patching linux kernel sections. Total patches per "
            "section are as follows:\n";
  for (const std::pair<const std::string, uint64_t> &KV : PatchCounts)
    outs() << "  Section: " << KV.first << ", patch-counts: " << KV.second
           << '\n';
}

void RewriteInstance::mapFileSections(RuntimeDyld &RTDyld) {
  BC->deregisterUnusedSections();

  // If no new .eh_frame was written, remove relocated original .eh_frame.
  BinarySection *RelocatedEHFrameSection =
      getSection(".relocated" + getEHFrameSectionName());
  if (RelocatedEHFrameSection && RelocatedEHFrameSection->hasValidSectionID()) {
    BinarySection *NewEHFrameSection =
        getSection(getNewSecPrefix() + getEHFrameSectionName());
    if (!NewEHFrameSection || !NewEHFrameSection->isFinalized()) {
      // RTDyld will still have to process relocations for the section, hence
      // we need to assign it the address that wouldn't result in relocation
      // processing failure.
      RTDyld.reassignSectionAddress(RelocatedEHFrameSection->getSectionID(),
                                    NextAvailableAddress);
      BC->deregisterSection(*RelocatedEHFrameSection);
    }
  }

  mapCodeSections(RTDyld);

  // Map the rest of the sections.
  mapAllocatableSections(RTDyld);
}

std::vector<BinarySection *> RewriteInstance::getCodeSections() {
  std::vector<BinarySection *> CodeSections;
  for (BinarySection &Section : BC->textSections())
    if (Section.hasValidSectionID())
      CodeSections.emplace_back(&Section);

  auto compareSections = [&](const BinarySection *A, const BinarySection *B) {
    // Place movers before anything else.
    if (A->getName() == BC->getHotTextMoverSectionName())
      return true;
    if (B->getName() == BC->getHotTextMoverSectionName())
      return false;

    // Depending on the option, put main text at the beginning or at the end.
    if (opts::HotFunctionsAtEnd)
      return B->getName() == BC->getMainCodeSectionName();
    else
      return A->getName() == BC->getMainCodeSectionName();
  };

  // Determine the order of sections.
  llvm::stable_sort(CodeSections, compareSections);

  return CodeSections;
}

void RewriteInstance::mapCodeSections(RuntimeDyld &RTDyld) {
  if (BC->HasRelocations) {
    // Map sections for functions with pre-assigned addresses.
    for (BinaryFunction *InjectedFunction : BC->getInjectedBinaryFunctions()) {
      const uint64_t OutputAddress = InjectedFunction->getOutputAddress();
      if (!OutputAddress)
        continue;

      ErrorOr<BinarySection &> FunctionSection =
          InjectedFunction->getCodeSection();
      assert(FunctionSection && "function should have section");
      FunctionSection->setOutputAddress(OutputAddress);
      RTDyld.reassignSectionAddress(FunctionSection->getSectionID(),
                                    OutputAddress);
      InjectedFunction->setImageAddress(FunctionSection->getAllocAddress());
      InjectedFunction->setImageSize(FunctionSection->getOutputSize());
    }

    // Populate the list of sections to be allocated.
    std::vector<BinarySection *> CodeSections = getCodeSections();

    // Remove sections that were pre-allocated (patch sections).
    llvm::erase_if(CodeSections, [](BinarySection *Section) {
      return Section->getOutputAddress();
    });
    LLVM_DEBUG(dbgs() << "Code sections in the order of output:\n";
      for (const BinarySection *Section : CodeSections)
        dbgs() << Section->getName() << '\n';
    );

    uint64_t PaddingSize = 0; // size of padding required at the end

    // Allocate sections starting at a given Address.
    auto allocateAt = [&](uint64_t Address) {
      for (BinarySection *Section : CodeSections) {
        Address = alignTo(Address, Section->getAlignment());
        Section->setOutputAddress(Address);
        Address += Section->getOutputSize();

        // Hugify: Additional huge page from right side due to
        // weird ASLR mapping addresses (4KB aligned)
        if (opts::Hugify && !BC->HasFixedLoadAddress &&
            Section->getName() == BC->getMainCodeSectionName())
          Address = alignTo(Address, Section->getAlignment());
      }

      // Make sure we allocate enough space for huge pages.
      ErrorOr<BinarySection &> TextSection =
          BC->getUniqueSectionByName(BC->getMainCodeSectionName());
      if (opts::HotText && TextSection && TextSection->hasValidSectionID()) {
        uint64_t HotTextEnd =
            TextSection->getOutputAddress() + TextSection->getOutputSize();
        HotTextEnd = alignTo(HotTextEnd, BC->PageAlign);
        if (HotTextEnd > Address) {
          PaddingSize = HotTextEnd - Address;
          Address = HotTextEnd;
        }
      }
      return Address;
    };

    // Check if we can fit code in the original .text
    bool AllocationDone = false;
    if (opts::UseOldText) {
      const uint64_t CodeSize =
          allocateAt(BC->OldTextSectionAddress) - BC->OldTextSectionAddress;

      if (CodeSize <= BC->OldTextSectionSize) {
        outs() << "BOLT-INFO: using original .text for new code with 0x"
               << Twine::utohexstr(opts::AlignText) << " alignment\n";
        AllocationDone = true;
      } else {
        errs() << "BOLT-WARNING: original .text too small to fit the new code"
               << " using 0x" << Twine::utohexstr(opts::AlignText)
               << " alignment. " << CodeSize << " bytes needed, have "
               << BC->OldTextSectionSize << " bytes available.\n";
        opts::UseOldText = false;
      }
    }

    if (!AllocationDone)
      NextAvailableAddress = allocateAt(NextAvailableAddress);

    // Do the mapping for ORC layer based on the allocation.
    for (BinarySection *Section : CodeSections) {
      LLVM_DEBUG(
          dbgs() << "BOLT: mapping " << Section->getName() << " at 0x"
                 << Twine::utohexstr(Section->getAllocAddress()) << " to 0x"
                 << Twine::utohexstr(Section->getOutputAddress()) << '\n');
      RTDyld.reassignSectionAddress(Section->getSectionID(),
                                    Section->getOutputAddress());
      Section->setOutputFileOffset(
          getFileOffsetForAddress(Section->getOutputAddress()));
    }

    // Check if we need to insert a padding section for hot text.
    if (PaddingSize && !opts::UseOldText)
      outs() << "BOLT-INFO: padding code to 0x"
             << Twine::utohexstr(NextAvailableAddress)
             << " to accommodate hot text\n";

    return;
  }

  // Processing in non-relocation mode.
  uint64_t NewTextSectionStartAddress = NextAvailableAddress;

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isEmitted())
      continue;

    bool TooLarge = false;
    ErrorOr<BinarySection &> FuncSection = Function.getCodeSection();
    assert(FuncSection && "cannot find section for function");
    FuncSection->setOutputAddress(Function.getAddress());
    LLVM_DEBUG(dbgs() << "BOLT: mapping 0x"
                      << Twine::utohexstr(FuncSection->getAllocAddress())
                      << " to 0x" << Twine::utohexstr(Function.getAddress())
                      << '\n');
    RTDyld.reassignSectionAddress(FuncSection->getSectionID(),
                                  Function.getAddress());
    Function.setImageAddress(FuncSection->getAllocAddress());
    Function.setImageSize(FuncSection->getOutputSize());
    if (Function.getImageSize() > Function.getMaxSize()) {
      TooLarge = true;
      FailedAddresses.emplace_back(Function.getAddress());
    }

    // Map jump tables if updating in-place.
    if (opts::JumpTables == JTS_BASIC) {
      for (auto &JTI : Function.JumpTables) {
        JumpTable *JT = JTI.second;
        BinarySection &Section = JT->getOutputSection();
        Section.setOutputAddress(JT->getAddress());
        Section.setOutputFileOffset(getFileOffsetForAddress(JT->getAddress()));
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: mapping JT " << Section.getName()
                          << " to 0x" << Twine::utohexstr(JT->getAddress())
                          << '\n');
        RTDyld.reassignSectionAddress(Section.getSectionID(), JT->getAddress());
      }
    }

    if (!Function.isSplit())
      continue;

    assert(Function.getLayout().isHotColdSplit() &&
           "Cannot allocate more than two fragments per function in "
           "non-relocation mode.");

    FunctionFragment &FF =
        Function.getLayout().getFragment(FragmentNum::cold());
    ErrorOr<BinarySection &> ColdSection =
        Function.getCodeSection(FF.getFragmentNum());
    assert(ColdSection && "cannot find section for cold part");
    // Cold fragments are aligned at 16 bytes.
    NextAvailableAddress = alignTo(NextAvailableAddress, 16);
    if (TooLarge) {
      // The corresponding FDE will refer to address 0.
      FF.setAddress(0);
      FF.setImageAddress(0);
      FF.setImageSize(0);
      FF.setFileOffset(0);
      BC->deregisterSection(*ColdSection);
    } else {
      FF.setAddress(NextAvailableAddress);
      FF.setImageAddress(ColdSection->getAllocAddress());
      FF.setImageSize(ColdSection->getOutputSize());
      FF.setFileOffset(getFileOffsetForAddress(NextAvailableAddress));
      ColdSection->setOutputAddress(FF.getAddress());
    }

    LLVM_DEBUG(
        dbgs() << formatv(
            "BOLT: mapping cold fragment {0:x+} to {1:x+} with size {2:x+}\n",
            FF.getImageAddress(), FF.getAddress(), FF.getImageSize()));
    RTDyld.reassignSectionAddress(ColdSection->getSectionID(), FF.getAddress());

    NextAvailableAddress += FF.getImageSize();
  }

  // Add the new text section aggregating all existing code sections.
  // This is pseudo-section that serves a purpose of creating a corresponding
  // entry in section header table.
  int64_t NewTextSectionSize =
      NextAvailableAddress - NewTextSectionStartAddress;
  if (NewTextSectionSize) {
    const unsigned Flags = BinarySection::getFlags(/*IsReadOnly=*/true,
                                                   /*IsText=*/true,
                                                   /*IsAllocatable=*/true);
    BinarySection &Section =
      BC->registerOrUpdateSection(getBOLTTextSectionName(),
                                  ELF::SHT_PROGBITS,
                                  Flags,
                                  /*Data=*/nullptr,
                                  NewTextSectionSize,
                                  16);
    Section.setOutputAddress(NewTextSectionStartAddress);
    Section.setOutputFileOffset(
        getFileOffsetForAddress(NewTextSectionStartAddress));
  }
}

void RewriteInstance::mapAllocatableSections(RuntimeDyld &RTDyld) {
  // Allocate read-only sections first, then writable sections.
  enum : uint8_t { ST_READONLY, ST_READWRITE };
  for (uint8_t SType = ST_READONLY; SType <= ST_READWRITE; ++SType) {
    for (BinarySection &Section : BC->allocatableSections()) {
      if (!Section.hasValidSectionID())
        continue;

      if (Section.isWritable() == (SType == ST_READONLY))
        continue;

      if (Section.getOutputAddress()) {
        LLVM_DEBUG({
          dbgs() << "BOLT-DEBUG: section " << Section.getName()
                 << " is already mapped at 0x"
                 << Twine::utohexstr(Section.getOutputAddress()) << '\n';
        });
        continue;
      }

      if (Section.hasSectionRef()) {
        LLVM_DEBUG({
          dbgs() << "BOLT-DEBUG: mapping original section " << Section.getName()
                 << " to 0x" << Twine::utohexstr(Section.getAddress()) << '\n';
        });
        Section.setOutputAddress(Section.getAddress());
        Section.setOutputFileOffset(Section.getInputFileOffset());
        RTDyld.reassignSectionAddress(Section.getSectionID(),
                                      Section.getAddress());
      } else {
        NextAvailableAddress =
            alignTo(NextAvailableAddress, Section.getAlignment());
        LLVM_DEBUG({
          dbgs() << "BOLT: mapping section " << Section.getName() << " (0x"
                 << Twine::utohexstr(Section.getAllocAddress()) << ") to 0x"
                 << Twine::utohexstr(NextAvailableAddress) << ":0x"
                 << Twine::utohexstr(NextAvailableAddress +
                                     Section.getOutputSize())
                 << '\n';
        });

        RTDyld.reassignSectionAddress(Section.getSectionID(),
                                      NextAvailableAddress);
        Section.setOutputAddress(NextAvailableAddress);
        Section.setOutputFileOffset(
            getFileOffsetForAddress(NextAvailableAddress));

        NextAvailableAddress += Section.getOutputSize();
      }
    }
  }
}

void RewriteInstance::updateOutputValues(const MCAsmLayout &Layout) {
  for (BinaryFunction *Function : BC->getAllBinaryFunctions())
    Function->updateOutputValues(Layout);
}

void RewriteInstance::patchELFPHDRTable() {
  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(InputFile);
  if (!ELF64LEFile) {
    errs() << "BOLT-ERROR: only 64-bit LE ELF binaries are supported\n";
    exit(1);
  }
  const ELFFile<ELF64LE> &Obj = ELF64LEFile->getELFFile();
  raw_fd_ostream &OS = Out->os();

  // Write/re-write program headers.
  Phnum = Obj.getHeader().e_phnum;
  if (PHDRTableOffset) {
    // Writing new pheader table.
    Phnum += 1; // only adding one new segment
    // Segment size includes the size of the PHDR area.
    NewTextSegmentSize = NextAvailableAddress - PHDRTableAddress;
  } else {
    assert(!PHDRTableAddress && "unexpected address for program header table");
    // Update existing table.
    PHDRTableOffset = Obj.getHeader().e_phoff;
    NewTextSegmentSize = NextAvailableAddress - NewTextSegmentAddress;
  }
  OS.seek(PHDRTableOffset);

  bool ModdedGnuStack = false;
  (void)ModdedGnuStack;
  bool AddedSegment = false;
  (void)AddedSegment;

  auto createNewTextPhdr = [&]() {
    ELF64LEPhdrTy NewPhdr;
    NewPhdr.p_type = ELF::PT_LOAD;
    if (PHDRTableAddress) {
      NewPhdr.p_offset = PHDRTableOffset;
      NewPhdr.p_vaddr = PHDRTableAddress;
      NewPhdr.p_paddr = PHDRTableAddress;
    } else {
      NewPhdr.p_offset = NewTextSegmentOffset;
      NewPhdr.p_vaddr = NewTextSegmentAddress;
      NewPhdr.p_paddr = NewTextSegmentAddress;
    }
    NewPhdr.p_filesz = NewTextSegmentSize;
    NewPhdr.p_memsz = NewTextSegmentSize;
    NewPhdr.p_flags = ELF::PF_X | ELF::PF_R;
    // FIXME: Currently instrumentation is experimental and the runtime data
    // is emitted with code, thus everything needs to be writable
    if (opts::Instrument)
      NewPhdr.p_flags |= ELF::PF_W;
    NewPhdr.p_align = BC->PageAlign;

    return NewPhdr;
  };

  // Copy existing program headers with modifications.
  for (const ELF64LE::Phdr &Phdr : cantFail(Obj.program_headers())) {
    ELF64LE::Phdr NewPhdr = Phdr;
    if (PHDRTableAddress && Phdr.p_type == ELF::PT_PHDR) {
      NewPhdr.p_offset = PHDRTableOffset;
      NewPhdr.p_vaddr = PHDRTableAddress;
      NewPhdr.p_paddr = PHDRTableAddress;
      NewPhdr.p_filesz = sizeof(NewPhdr) * Phnum;
      NewPhdr.p_memsz = sizeof(NewPhdr) * Phnum;
    } else if (Phdr.p_type == ELF::PT_GNU_EH_FRAME) {
      ErrorOr<BinarySection &> EHFrameHdrSec =
          BC->getUniqueSectionByName(getNewSecPrefix() + ".eh_frame_hdr");
      if (EHFrameHdrSec && EHFrameHdrSec->isAllocatable() &&
          EHFrameHdrSec->isFinalized()) {
        NewPhdr.p_offset = EHFrameHdrSec->getOutputFileOffset();
        NewPhdr.p_vaddr = EHFrameHdrSec->getOutputAddress();
        NewPhdr.p_paddr = EHFrameHdrSec->getOutputAddress();
        NewPhdr.p_filesz = EHFrameHdrSec->getOutputSize();
        NewPhdr.p_memsz = EHFrameHdrSec->getOutputSize();
      }
    } else if (opts::UseGnuStack && Phdr.p_type == ELF::PT_GNU_STACK) {
      NewPhdr = createNewTextPhdr();
      ModdedGnuStack = true;
    } else if (!opts::UseGnuStack && Phdr.p_type == ELF::PT_DYNAMIC) {
      // Insert the new header before DYNAMIC.
      ELF64LE::Phdr NewTextPhdr = createNewTextPhdr();
      OS.write(reinterpret_cast<const char *>(&NewTextPhdr),
               sizeof(NewTextPhdr));
      AddedSegment = true;
    }
    OS.write(reinterpret_cast<const char *>(&NewPhdr), sizeof(NewPhdr));
  }

  if (!opts::UseGnuStack && !AddedSegment) {
    // Append the new header to the end of the table.
    ELF64LE::Phdr NewTextPhdr = createNewTextPhdr();
    OS.write(reinterpret_cast<const char *>(&NewTextPhdr), sizeof(NewTextPhdr));
  }

  assert((!opts::UseGnuStack || ModdedGnuStack) &&
         "could not find GNU_STACK program header to modify");
}

namespace {

/// Write padding to \p OS such that its current \p Offset becomes aligned
/// at \p Alignment. Return new (aligned) offset.
uint64_t appendPadding(raw_pwrite_stream &OS, uint64_t Offset,
                       uint64_t Alignment) {
  if (!Alignment)
    return Offset;

  const uint64_t PaddingSize =
      offsetToAlignment(Offset, llvm::Align(Alignment));
  for (unsigned I = 0; I < PaddingSize; ++I)
    OS.write((unsigned char)0);
  return Offset + PaddingSize;
}

}

void RewriteInstance::rewriteNoteSections() {
  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(InputFile);
  if (!ELF64LEFile) {
    errs() << "BOLT-ERROR: only 64-bit LE ELF binaries are supported\n";
    exit(1);
  }
  const ELFFile<ELF64LE> &Obj = ELF64LEFile->getELFFile();
  raw_fd_ostream &OS = Out->os();

  uint64_t NextAvailableOffset = getFileOffsetForAddress(NextAvailableAddress);
  assert(NextAvailableOffset >= FirstNonAllocatableOffset &&
         "next available offset calculation failure");
  OS.seek(NextAvailableOffset);

  // Copy over non-allocatable section contents and update file offsets.
  for (const ELF64LE::Shdr &Section : cantFail(Obj.sections())) {
    if (Section.sh_type == ELF::SHT_NULL)
      continue;
    if (Section.sh_flags & ELF::SHF_ALLOC)
      continue;

    SectionRef SecRef = ELF64LEFile->toSectionRef(&Section);
    BinarySection *BSec = BC->getSectionForSectionRef(SecRef);
    assert(BSec && !BSec->isAllocatable() &&
           "Matching non-allocatable BinarySection should exist.");

    StringRef SectionName =
        cantFail(Obj.getSectionName(Section), "cannot get section name");
    if (shouldStrip(Section, SectionName))
      continue;

    // Insert padding as needed.
    NextAvailableOffset =
        appendPadding(OS, NextAvailableOffset, Section.sh_addralign);

    // New section size.
    uint64_t Size = 0;
    bool DataWritten = false;
    uint8_t *SectionData = nullptr;
    // Copy over section contents unless it's one of the sections we overwrite.
    if (!willOverwriteSection(SectionName)) {
      Size = Section.sh_size;
      StringRef Dataref = InputFile->getData().substr(Section.sh_offset, Size);
      std::string Data;
      if (BSec->getPatcher()) {
        Data = BSec->getPatcher()->patchBinary(Dataref);
        Dataref = StringRef(Data);
      }

      // Section was expanded, so need to treat it as overwrite.
      if (Size != Dataref.size()) {
        BSec = &BC->registerOrUpdateNoteSection(
            SectionName, copyByteArray(Dataref), Dataref.size());
        Size = 0;
      } else {
        OS << Dataref;
        DataWritten = true;

        // Add padding as the section extension might rely on the alignment.
        Size = appendPadding(OS, Size, Section.sh_addralign);
      }
    }

    // Perform section post-processing.
    assert(BSec->getAlignment() <= Section.sh_addralign &&
           "alignment exceeds value in file");

    if (BSec->getAllocAddress()) {
      assert(!DataWritten && "Writing section twice.");
      (void)DataWritten;
      SectionData = BSec->getOutputData();

      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: " << (Size ? "appending" : "writing")
                        << " contents to section " << SectionName << '\n');
      OS.write(reinterpret_cast<char *>(SectionData), BSec->getOutputSize());
      Size += BSec->getOutputSize();
    }

    BSec->setOutputFileOffset(NextAvailableOffset);
    BSec->flushPendingRelocations(OS, [this](const MCSymbol *S) {
      return getNewValueForSymbol(S->getName());
    });

    // Set/modify section info.
    BinarySection &NewSection = BC->registerOrUpdateNoteSection(
        SectionName, SectionData, Size, Section.sh_addralign,
        !BSec->isWritable(), BSec->getELFType());
    NewSection.setOutputAddress(0);
    NewSection.setOutputFileOffset(NextAvailableOffset);

    NextAvailableOffset += Size;
  }

  // Write new note sections.
  for (BinarySection &Section : BC->nonAllocatableSections()) {
    if (Section.getOutputFileOffset() || !Section.getAllocAddress())
      continue;

    assert(!Section.hasPendingRelocations() && "cannot have pending relocs");

    NextAvailableOffset =
        appendPadding(OS, NextAvailableOffset, Section.getAlignment());
    Section.setOutputFileOffset(NextAvailableOffset);

    LLVM_DEBUG(
        dbgs() << "BOLT-DEBUG: writing out new section " << Section.getName()
               << " of size " << Section.getOutputSize() << " at offset 0x"
               << Twine::utohexstr(Section.getOutputFileOffset()) << '\n');

    OS.write(Section.getOutputContents().data(), Section.getOutputSize());
    NextAvailableOffset += Section.getOutputSize();
  }
}

template <typename ELFT>
void RewriteInstance::finalizeSectionStringTable(ELFObjectFile<ELFT> *File) {
  // Pre-populate section header string table.
  for (const BinarySection &Section : BC->sections())
    if (!Section.isAnonymous())
      SHStrTab.add(Section.getOutputName());
  SHStrTab.finalize();

  const size_t SHStrTabSize = SHStrTab.getSize();
  uint8_t *DataCopy = new uint8_t[SHStrTabSize];
  memset(DataCopy, 0, SHStrTabSize);
  SHStrTab.write(DataCopy);
  BC->registerOrUpdateNoteSection(".shstrtab",
                                  DataCopy,
                                  SHStrTabSize,
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true,
                                  ELF::SHT_STRTAB);
}

void RewriteInstance::addBoltInfoSection() {
  std::string DescStr;
  raw_string_ostream DescOS(DescStr);

  DescOS << "BOLT revision: " << BoltRevision << ", "
         << "command line:";
  for (int I = 0; I < Argc; ++I)
    DescOS << " " << Argv[I];
  DescOS.flush();

  // Encode as GNU GOLD VERSION so it is easily printable by 'readelf -n'
  const std::string BoltInfo =
      BinarySection::encodeELFNote("GNU", DescStr, 4 /*NT_GNU_GOLD_VERSION*/);
  BC->registerOrUpdateNoteSection(".note.bolt_info", copyByteArray(BoltInfo),
                                  BoltInfo.size(),
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true, ELF::SHT_NOTE);
}

void RewriteInstance::addBATSection() {
  BC->registerOrUpdateNoteSection(BoltAddressTranslation::SECTION_NAME, nullptr,
                                  0,
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true, ELF::SHT_NOTE);
}

void RewriteInstance::encodeBATSection() {
  std::string DescStr;
  raw_string_ostream DescOS(DescStr);

  BAT->write(*BC, DescOS);
  DescOS.flush();

  const std::string BoltInfo =
      BinarySection::encodeELFNote("BOLT", DescStr, BinarySection::NT_BOLT_BAT);
  BC->registerOrUpdateNoteSection(BoltAddressTranslation::SECTION_NAME,
                                  copyByteArray(BoltInfo), BoltInfo.size(),
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true, ELF::SHT_NOTE);
}

template <typename ELFShdrTy>
bool RewriteInstance::shouldStrip(const ELFShdrTy &Section,
                                  StringRef SectionName) {
  // Strip non-allocatable relocation sections.
  if (!(Section.sh_flags & ELF::SHF_ALLOC) && Section.sh_type == ELF::SHT_RELA)
    return true;

  // Strip debug sections if not updating them.
  if (isDebugSection(SectionName) && !opts::UpdateDebugSections)
    return true;

  // Strip symtab section if needed
  if (opts::RemoveSymtab && Section.sh_type == ELF::SHT_SYMTAB)
    return true;

  return false;
}

template <typename ELFT>
std::vector<typename object::ELFObjectFile<ELFT>::Elf_Shdr>
RewriteInstance::getOutputSections(ELFObjectFile<ELFT> *File,
                                   std::vector<uint32_t> &NewSectionIndex) {
  using ELFShdrTy = typename ELFObjectFile<ELFT>::Elf_Shdr;
  const ELFFile<ELFT> &Obj = File->getELFFile();
  typename ELFT::ShdrRange Sections = cantFail(Obj.sections());

  // Keep track of section header entries attached to the corresponding section.
  std::vector<std::pair<BinarySection *, ELFShdrTy>> OutputSections;
  auto addSection = [&](const ELFShdrTy &Section, BinarySection *BinSec) {
    ELFShdrTy NewSection = Section;
    NewSection.sh_name = SHStrTab.getOffset(BinSec->getOutputName());
    OutputSections.emplace_back(BinSec, std::move(NewSection));
  };

  // Copy over entries for original allocatable sections using modified name.
  for (const ELFShdrTy &Section : Sections) {
    // Always ignore this section.
    if (Section.sh_type == ELF::SHT_NULL) {
      OutputSections.emplace_back(nullptr, Section);
      continue;
    }

    if (!(Section.sh_flags & ELF::SHF_ALLOC))
      continue;

    SectionRef SecRef = File->toSectionRef(&Section);
    BinarySection *BinSec = BC->getSectionForSectionRef(SecRef);
    assert(BinSec && "Matching BinarySection should exist.");

    addSection(Section, BinSec);
  }

  for (BinarySection &Section : BC->allocatableSections()) {
    if (!Section.isFinalized())
      continue;

    if (Section.hasSectionRef() || Section.isAnonymous()) {
      if (opts::Verbosity)
        outs() << "BOLT-INFO: not writing section header for section "
               << Section.getOutputName() << '\n';
      continue;
    }

    if (opts::Verbosity >= 1)
      outs() << "BOLT-INFO: writing section header for "
             << Section.getOutputName() << '\n';
    ELFShdrTy NewSection;
    NewSection.sh_type = ELF::SHT_PROGBITS;
    NewSection.sh_addr = Section.getOutputAddress();
    NewSection.sh_offset = Section.getOutputFileOffset();
    NewSection.sh_size = Section.getOutputSize();
    NewSection.sh_entsize = 0;
    NewSection.sh_flags = Section.getELFFlags();
    NewSection.sh_link = 0;
    NewSection.sh_info = 0;
    NewSection.sh_addralign = Section.getAlignment();
    addSection(NewSection, &Section);
  }

  // Sort all allocatable sections by their offset.
  llvm::stable_sort(OutputSections, [](const auto &A, const auto &B) {
    return A.second.sh_offset < B.second.sh_offset;
  });

  // Fix section sizes to prevent overlapping.
  ELFShdrTy *PrevSection = nullptr;
  BinarySection *PrevBinSec = nullptr;
  for (auto &SectionKV : OutputSections) {
    ELFShdrTy &Section = SectionKV.second;

    // TBSS section does not take file or memory space. Ignore it for layout
    // purposes.
    if (Section.sh_type == ELF::SHT_NOBITS && (Section.sh_flags & ELF::SHF_TLS))
      continue;

    if (PrevSection &&
        PrevSection->sh_addr + PrevSection->sh_size > Section.sh_addr) {
      if (opts::Verbosity > 1)
        outs() << "BOLT-INFO: adjusting size for section "
               << PrevBinSec->getOutputName() << '\n';
      PrevSection->sh_size = Section.sh_addr > PrevSection->sh_addr
                                 ? Section.sh_addr - PrevSection->sh_addr
                                 : 0;
    }

    PrevSection = &Section;
    PrevBinSec = SectionKV.first;
  }

  uint64_t LastFileOffset = 0;

  // Copy over entries for non-allocatable sections performing necessary
  // adjustments.
  for (const ELFShdrTy &Section : Sections) {
    if (Section.sh_type == ELF::SHT_NULL)
      continue;
    if (Section.sh_flags & ELF::SHF_ALLOC)
      continue;

    StringRef SectionName =
        cantFail(Obj.getSectionName(Section), "cannot get section name");

    if (shouldStrip(Section, SectionName))
      continue;

    SectionRef SecRef = File->toSectionRef(&Section);
    BinarySection *BinSec = BC->getSectionForSectionRef(SecRef);
    assert(BinSec && "Matching BinarySection should exist.");

    ELFShdrTy NewSection = Section;
    NewSection.sh_offset = BinSec->getOutputFileOffset();
    NewSection.sh_size = BinSec->getOutputSize();

    if (NewSection.sh_type == ELF::SHT_SYMTAB)
      NewSection.sh_info = NumLocalSymbols;

    addSection(NewSection, BinSec);

    LastFileOffset = BinSec->getOutputFileOffset();
  }

  // Create entries for new non-allocatable sections.
  for (BinarySection &Section : BC->nonAllocatableSections()) {
    if (Section.getOutputFileOffset() <= LastFileOffset)
      continue;

    if (opts::Verbosity >= 1)
      outs() << "BOLT-INFO: writing section header for "
             << Section.getOutputName() << '\n';

    ELFShdrTy NewSection;
    NewSection.sh_type = Section.getELFType();
    NewSection.sh_addr = 0;
    NewSection.sh_offset = Section.getOutputFileOffset();
    NewSection.sh_size = Section.getOutputSize();
    NewSection.sh_entsize = 0;
    NewSection.sh_flags = Section.getELFFlags();
    NewSection.sh_link = 0;
    NewSection.sh_info = 0;
    NewSection.sh_addralign = Section.getAlignment();

    addSection(NewSection, &Section);
  }

  // Assign indices to sections.
  std::unordered_map<std::string, uint64_t> NameToIndex;
  for (uint32_t Index = 1; Index < OutputSections.size(); ++Index)
    OutputSections[Index].first->setIndex(Index);

  // Update section index mapping
  NewSectionIndex.clear();
  NewSectionIndex.resize(Sections.size(), 0);
  for (const ELFShdrTy &Section : Sections) {
    if (Section.sh_type == ELF::SHT_NULL)
      continue;

    size_t OrgIndex = std::distance(Sections.begin(), &Section);

    SectionRef SecRef = File->toSectionRef(&Section);
    BinarySection *BinSec = BC->getSectionForSectionRef(SecRef);
    assert(BinSec && "BinarySection should exist for an input section.");

    // Some sections are stripped
    if (!BinSec->hasValidIndex())
      continue;

    NewSectionIndex[OrgIndex] = BinSec->getIndex();
  }

  std::vector<ELFShdrTy> SectionsOnly(OutputSections.size());
  llvm::transform(OutputSections, SectionsOnly.begin(),
                  [](auto &SectionInfo) { return SectionInfo.second; });

  return SectionsOnly;
}

// Rewrite section header table inserting new entries as needed. The sections
// header table size itself may affect the offsets of other sections,
// so we are placing it at the end of the binary.
//
// As we rewrite entries we need to track how many sections were inserted
// as it changes the sh_link value. We map old indices to new ones for
// existing sections.
template <typename ELFT>
void RewriteInstance::patchELFSectionHeaderTable(ELFObjectFile<ELFT> *File) {
  using ELFShdrTy = typename ELFObjectFile<ELFT>::Elf_Shdr;
  using ELFEhdrTy = typename ELFObjectFile<ELFT>::Elf_Ehdr;
  raw_fd_ostream &OS = Out->os();
  const ELFFile<ELFT> &Obj = File->getELFFile();

  std::vector<uint32_t> NewSectionIndex;
  std::vector<ELFShdrTy> OutputSections =
      getOutputSections(File, NewSectionIndex);
  LLVM_DEBUG(
    dbgs() << "BOLT-DEBUG: old to new section index mapping:\n";
    for (uint64_t I = 0; I < NewSectionIndex.size(); ++I)
      dbgs() << "  " << I << " -> " << NewSectionIndex[I] << '\n';
  );

  // Align starting address for section header table.
  uint64_t SHTOffset = OS.tell();
  SHTOffset = appendPadding(OS, SHTOffset, sizeof(ELFShdrTy));

  // Write all section header entries while patching section references.
  for (ELFShdrTy &Section : OutputSections) {
    Section.sh_link = NewSectionIndex[Section.sh_link];
    if (Section.sh_type == ELF::SHT_REL || Section.sh_type == ELF::SHT_RELA) {
      if (Section.sh_info)
        Section.sh_info = NewSectionIndex[Section.sh_info];
    }
    OS.write(reinterpret_cast<const char *>(&Section), sizeof(Section));
  }

  // Fix ELF header.
  ELFEhdrTy NewEhdr = Obj.getHeader();

  if (BC->HasRelocations) {
    if (RuntimeLibrary *RtLibrary = BC->getRuntimeLibrary())
      NewEhdr.e_entry = RtLibrary->getRuntimeStartAddress();
    else
      NewEhdr.e_entry = getNewFunctionAddress(NewEhdr.e_entry);
    assert((NewEhdr.e_entry || !Obj.getHeader().e_entry) &&
           "cannot find new address for entry point");
  }
  NewEhdr.e_phoff = PHDRTableOffset;
  NewEhdr.e_phnum = Phnum;
  NewEhdr.e_shoff = SHTOffset;
  NewEhdr.e_shnum = OutputSections.size();
  NewEhdr.e_shstrndx = NewSectionIndex[NewEhdr.e_shstrndx];
  OS.pwrite(reinterpret_cast<const char *>(&NewEhdr), sizeof(NewEhdr), 0);
}

template <typename ELFT, typename WriteFuncTy, typename StrTabFuncTy>
void RewriteInstance::updateELFSymbolTable(
    ELFObjectFile<ELFT> *File, bool IsDynSym,
    const typename object::ELFObjectFile<ELFT>::Elf_Shdr &SymTabSection,
    const std::vector<uint32_t> &NewSectionIndex, WriteFuncTy Write,
    StrTabFuncTy AddToStrTab) {
  const ELFFile<ELFT> &Obj = File->getELFFile();
  using ELFSymTy = typename ELFObjectFile<ELFT>::Elf_Sym;

  StringRef StringSection =
      cantFail(Obj.getStringTableForSymtab(SymTabSection));

  unsigned NumHotTextSymsUpdated = 0;
  unsigned NumHotDataSymsUpdated = 0;

  std::map<const BinaryFunction *, uint64_t> IslandSizes;
  auto getConstantIslandSize = [&IslandSizes](const BinaryFunction &BF) {
    auto Itr = IslandSizes.find(&BF);
    if (Itr != IslandSizes.end())
      return Itr->second;
    return IslandSizes[&BF] = BF.estimateConstantIslandSize();
  };

  // Symbols for the new symbol table.
  std::vector<ELFSymTy> Symbols;

  auto getNewSectionIndex = [&](uint32_t OldIndex) {
    // For dynamic symbol table, the section index could be wrong on the input,
    // and its value is ignored by the runtime if it's different from
    // SHN_UNDEF and SHN_ABS.
    // However, we still need to update dynamic symbol table, so return a
    // section index, even though the index is broken.
    if (IsDynSym && OldIndex >= NewSectionIndex.size())
      return OldIndex;

    assert(OldIndex < NewSectionIndex.size() && "section index out of bounds");
    const uint32_t NewIndex = NewSectionIndex[OldIndex];

    // We may have stripped the section that dynsym was referencing due to
    // the linker bug. In that case return the old index avoiding marking
    // the symbol as undefined.
    if (IsDynSym && NewIndex != OldIndex && NewIndex == ELF::SHN_UNDEF)
      return OldIndex;
    return NewIndex;
  };

  // Add extra symbols for the function.
  //
  // Note that addExtraSymbols() could be called multiple times for the same
  // function with different FunctionSymbol matching the main function entry
  // point.
  auto addExtraSymbols = [&](const BinaryFunction &Function,
                             const ELFSymTy &FunctionSymbol) {
    if (Function.isFolded()) {
      BinaryFunction *ICFParent = Function.getFoldedIntoFunction();
      while (ICFParent->isFolded())
        ICFParent = ICFParent->getFoldedIntoFunction();
      ELFSymTy ICFSymbol = FunctionSymbol;
      SmallVector<char, 256> Buf;
      ICFSymbol.st_name =
          AddToStrTab(Twine(cantFail(FunctionSymbol.getName(StringSection)))
                          .concat(".icf.0")
                          .toStringRef(Buf));
      ICFSymbol.st_value = ICFParent->getOutputAddress();
      ICFSymbol.st_size = ICFParent->getOutputSize();
      ICFSymbol.st_shndx = ICFParent->getCodeSection()->getIndex();
      Symbols.emplace_back(ICFSymbol);
    }
    if (Function.isSplit()) {
      for (const FunctionFragment &FF :
           Function.getLayout().getSplitFragments()) {
        if (FF.getAddress()) {
          ELFSymTy NewColdSym = FunctionSymbol;
          const SmallString<256> SymbolName = formatv(
              "{0}.cold.{1}", cantFail(FunctionSymbol.getName(StringSection)),
              FF.getFragmentNum().get() - 1);
          NewColdSym.st_name = AddToStrTab(SymbolName);
          NewColdSym.st_shndx =
              Function.getCodeSection(FF.getFragmentNum())->getIndex();
          NewColdSym.st_value = FF.getAddress();
          NewColdSym.st_size = FF.getImageSize();
          NewColdSym.setBindingAndType(ELF::STB_LOCAL, ELF::STT_FUNC);
          Symbols.emplace_back(NewColdSym);
        }
      }
    }
    if (Function.hasConstantIsland()) {
      uint64_t DataMark = Function.getOutputDataAddress();
      uint64_t CISize = getConstantIslandSize(Function);
      uint64_t CodeMark = DataMark + CISize;
      ELFSymTy DataMarkSym = FunctionSymbol;
      DataMarkSym.st_name = AddToStrTab("$d");
      DataMarkSym.st_value = DataMark;
      DataMarkSym.st_size = 0;
      DataMarkSym.setType(ELF::STT_NOTYPE);
      DataMarkSym.setBinding(ELF::STB_LOCAL);
      ELFSymTy CodeMarkSym = DataMarkSym;
      CodeMarkSym.st_name = AddToStrTab("$x");
      CodeMarkSym.st_value = CodeMark;
      Symbols.emplace_back(DataMarkSym);
      Symbols.emplace_back(CodeMarkSym);
    }
    if (Function.hasConstantIsland() && Function.isSplit()) {
      uint64_t DataMark = Function.getOutputColdDataAddress();
      uint64_t CISize = getConstantIslandSize(Function);
      uint64_t CodeMark = DataMark + CISize;
      ELFSymTy DataMarkSym = FunctionSymbol;
      DataMarkSym.st_name = AddToStrTab("$d");
      DataMarkSym.st_value = DataMark;
      DataMarkSym.st_size = 0;
      DataMarkSym.setType(ELF::STT_NOTYPE);
      DataMarkSym.setBinding(ELF::STB_LOCAL);
      ELFSymTy CodeMarkSym = DataMarkSym;
      CodeMarkSym.st_name = AddToStrTab("$x");
      CodeMarkSym.st_value = CodeMark;
      Symbols.emplace_back(DataMarkSym);
      Symbols.emplace_back(CodeMarkSym);
    }
  };

  // For regular (non-dynamic) symbol table, exclude symbols referring
  // to non-allocatable sections.
  auto shouldStrip = [&](const ELFSymTy &Symbol) {
    if (Symbol.isAbsolute() || !Symbol.isDefined())
      return false;

    // If we cannot link the symbol to a section, leave it as is.
    Expected<const typename ELFT::Shdr *> Section =
        Obj.getSection(Symbol.st_shndx);
    if (!Section)
      return false;

    // Remove the section symbol iif the corresponding section was stripped.
    if (Symbol.getType() == ELF::STT_SECTION) {
      if (!getNewSectionIndex(Symbol.st_shndx))
        return true;
      return false;
    }

    // Symbols in non-allocatable sections are typically remnants of relocations
    // emitted under "-emit-relocs" linker option. Delete those as we delete
    // relocations against non-allocatable sections.
    if (!((*Section)->sh_flags & ELF::SHF_ALLOC))
      return true;

    return false;
  };

  for (const ELFSymTy &Symbol : cantFail(Obj.symbols(&SymTabSection))) {
    // For regular (non-dynamic) symbol table strip unneeded symbols.
    if (!IsDynSym && shouldStrip(Symbol))
      continue;

    const BinaryFunction *Function =
        BC->getBinaryFunctionAtAddress(Symbol.st_value);
    // Ignore false function references, e.g. when the section address matches
    // the address of the function.
    if (Function && Symbol.getType() == ELF::STT_SECTION)
      Function = nullptr;

    // For non-dynamic symtab, make sure the symbol section matches that of
    // the function. It can mismatch e.g. if the symbol is a section marker
    // in which case we treat the symbol separately from the function.
    // For dynamic symbol table, the section index could be wrong on the input,
    // and its value is ignored by the runtime if it's different from
    // SHN_UNDEF and SHN_ABS.
    if (!IsDynSym && Function &&
        Symbol.st_shndx !=
            Function->getOriginSection()->getSectionRef().getIndex())
      Function = nullptr;

    // Create a new symbol based on the existing symbol.
    ELFSymTy NewSymbol = Symbol;

    if (Function) {
      // If the symbol matched a function that was not emitted, update the
      // corresponding section index but otherwise leave it unchanged.
      if (Function->isEmitted()) {
        NewSymbol.st_value = Function->getOutputAddress();
        NewSymbol.st_size = Function->getOutputSize();
        NewSymbol.st_shndx = Function->getCodeSection()->getIndex();
      } else if (Symbol.st_shndx < ELF::SHN_LORESERVE) {
        NewSymbol.st_shndx = getNewSectionIndex(Symbol.st_shndx);
      }

      // Add new symbols to the symbol table if necessary.
      if (!IsDynSym)
        addExtraSymbols(*Function, NewSymbol);
    } else {
      // Check if the function symbol matches address inside a function, i.e.
      // it marks a secondary entry point.
      Function =
          (Symbol.getType() == ELF::STT_FUNC)
              ? BC->getBinaryFunctionContainingAddress(Symbol.st_value,
                                                       /*CheckPastEnd=*/false,
                                                       /*UseMaxSize=*/true)
              : nullptr;

      if (Function && Function->isEmitted()) {
        assert(Function->getLayout().isHotColdSplit() &&
               "Adding symbols based on cold fragment when there are more than "
               "2 fragments");
        const uint64_t OutputAddress =
            Function->translateInputToOutputAddress(Symbol.st_value);

        NewSymbol.st_value = OutputAddress;
        // Force secondary entry points to have zero size.
        NewSymbol.st_size = 0;

        // Find fragment containing entrypoint
        FunctionLayout::fragment_const_iterator FF = llvm::find_if(
            Function->getLayout().fragments(), [&](const FunctionFragment &FF) {
              uint64_t Lo = FF.getAddress();
              uint64_t Hi = Lo + FF.getImageSize();
              return Lo <= OutputAddress && OutputAddress < Hi;
            });

        if (FF == Function->getLayout().fragment_end()) {
          assert(
              OutputAddress >= Function->getCodeSection()->getOutputAddress() &&
              OutputAddress < (Function->getCodeSection()->getOutputAddress() +
                               Function->getCodeSection()->getOutputSize()) &&
              "Cannot locate fragment containg secondary entrypoint");
          FF = Function->getLayout().fragment_begin();
        }

        NewSymbol.st_shndx =
            Function->getCodeSection(FF->getFragmentNum())->getIndex();
      } else {
        // Check if the symbol belongs to moved data object and update it.
        BinaryData *BD = opts::ReorderData.empty()
                             ? nullptr
                             : BC->getBinaryDataAtAddress(Symbol.st_value);
        if (BD && BD->isMoved() && !BD->isJumpTable()) {
          assert((!BD->getSize() || !Symbol.st_size ||
                  Symbol.st_size == BD->getSize()) &&
                 "sizes must match");

          BinarySection &OutputSection = BD->getOutputSection();
          assert(OutputSection.getIndex());
          LLVM_DEBUG(dbgs()
                     << "BOLT-DEBUG: moving " << BD->getName() << " from "
                     << *BC->getSectionNameForAddress(Symbol.st_value) << " ("
                     << Symbol.st_shndx << ") to " << OutputSection.getName()
                     << " (" << OutputSection.getIndex() << ")\n");
          NewSymbol.st_shndx = OutputSection.getIndex();
          NewSymbol.st_value = BD->getOutputAddress();
        } else {
          // Otherwise just update the section for the symbol.
          if (Symbol.st_shndx < ELF::SHN_LORESERVE)
            NewSymbol.st_shndx = getNewSectionIndex(Symbol.st_shndx);
        }

        // Detect local syms in the text section that we didn't update
        // and that were preserved by the linker to support relocations against
        // .text. Remove them from the symtab.
        if (Symbol.getType() == ELF::STT_NOTYPE &&
            Symbol.getBinding() == ELF::STB_LOCAL && Symbol.st_size == 0) {
          if (BC->getBinaryFunctionContainingAddress(Symbol.st_value,
                                                     /*CheckPastEnd=*/false,
                                                     /*UseMaxSize=*/true)) {
            // Can only delete the symbol if not patching. Such symbols should
            // not exist in the dynamic symbol table.
            assert(!IsDynSym && "cannot delete symbol");
            continue;
          }
        }
      }
    }

    // Handle special symbols based on their name.
    Expected<StringRef> SymbolName = Symbol.getName(StringSection);
    assert(SymbolName && "cannot get symbol name");

    auto updateSymbolValue = [&](const StringRef Name,
                                 std::optional<uint64_t> Value = std::nullopt) {
      NewSymbol.st_value = Value ? *Value : getNewValueForSymbol(Name);
      NewSymbol.st_shndx = ELF::SHN_ABS;
      outs() << "BOLT-INFO: setting " << Name << " to 0x"
             << Twine::utohexstr(NewSymbol.st_value) << '\n';
    };

    if (opts::HotText &&
        (*SymbolName == "__hot_start" || *SymbolName == "__hot_end")) {
      updateSymbolValue(*SymbolName);
      ++NumHotTextSymsUpdated;
    }

    if (opts::HotData && (*SymbolName == "__hot_data_start" ||
                          *SymbolName == "__hot_data_end")) {
      updateSymbolValue(*SymbolName);
      ++NumHotDataSymsUpdated;
    }

    if (*SymbolName == "_end")
      updateSymbolValue(*SymbolName, NextAvailableAddress);

    if (IsDynSym)
      Write((&Symbol - cantFail(Obj.symbols(&SymTabSection)).begin()) *
                sizeof(ELFSymTy),
            NewSymbol);
    else
      Symbols.emplace_back(NewSymbol);
  }

  if (IsDynSym) {
    assert(Symbols.empty());
    return;
  }

  // Add symbols of injected functions
  for (BinaryFunction *Function : BC->getInjectedBinaryFunctions()) {
    ELFSymTy NewSymbol;
    BinarySection *OriginSection = Function->getOriginSection();
    NewSymbol.st_shndx =
        OriginSection
            ? getNewSectionIndex(OriginSection->getSectionRef().getIndex())
            : Function->getCodeSection()->getIndex();
    NewSymbol.st_value = Function->getOutputAddress();
    NewSymbol.st_name = AddToStrTab(Function->getOneName());
    NewSymbol.st_size = Function->getOutputSize();
    NewSymbol.st_other = 0;
    NewSymbol.setBindingAndType(ELF::STB_LOCAL, ELF::STT_FUNC);
    Symbols.emplace_back(NewSymbol);

    if (Function->isSplit()) {
      assert(Function->getLayout().isHotColdSplit() &&
             "Adding symbols based on cold fragment when there are more than "
             "2 fragments");
      ELFSymTy NewColdSym = NewSymbol;
      NewColdSym.setType(ELF::STT_NOTYPE);
      SmallVector<char, 256> Buf;
      NewColdSym.st_name = AddToStrTab(
          Twine(Function->getPrintName()).concat(".cold.0").toStringRef(Buf));
      const FunctionFragment &ColdFF =
          Function->getLayout().getFragment(FragmentNum::cold());
      NewColdSym.st_value = ColdFF.getAddress();
      NewColdSym.st_size = ColdFF.getImageSize();
      Symbols.emplace_back(NewColdSym);
    }
  }

  assert((!NumHotTextSymsUpdated || NumHotTextSymsUpdated == 2) &&
         "either none or both __hot_start/__hot_end symbols were expected");
  assert((!NumHotDataSymsUpdated || NumHotDataSymsUpdated == 2) &&
         "either none or both __hot_data_start/__hot_data_end symbols were "
         "expected");

  auto addSymbol = [&](const std::string &Name) {
    ELFSymTy Symbol;
    Symbol.st_value = getNewValueForSymbol(Name);
    Symbol.st_shndx = ELF::SHN_ABS;
    Symbol.st_name = AddToStrTab(Name);
    Symbol.st_size = 0;
    Symbol.st_other = 0;
    Symbol.setBindingAndType(ELF::STB_WEAK, ELF::STT_NOTYPE);

    outs() << "BOLT-INFO: setting " << Name << " to 0x"
           << Twine::utohexstr(Symbol.st_value) << '\n';

    Symbols.emplace_back(Symbol);
  };

  if (opts::HotText && !NumHotTextSymsUpdated) {
    addSymbol("__hot_start");
    addSymbol("__hot_end");
  }

  if (opts::HotData && !NumHotDataSymsUpdated) {
    addSymbol("__hot_data_start");
    addSymbol("__hot_data_end");
  }

  // Put local symbols at the beginning.
  llvm::stable_sort(Symbols, [](const ELFSymTy &A, const ELFSymTy &B) {
    if (A.getBinding() == ELF::STB_LOCAL && B.getBinding() != ELF::STB_LOCAL)
      return true;
    return false;
  });

  for (const ELFSymTy &Symbol : Symbols)
    Write(0, Symbol);
}

template <typename ELFT>
void RewriteInstance::patchELFSymTabs(ELFObjectFile<ELFT> *File) {
  const ELFFile<ELFT> &Obj = File->getELFFile();
  using ELFShdrTy = typename ELFObjectFile<ELFT>::Elf_Shdr;
  using ELFSymTy = typename ELFObjectFile<ELFT>::Elf_Sym;

  // Compute a preview of how section indices will change after rewriting, so
  // we can properly update the symbol table based on new section indices.
  std::vector<uint32_t> NewSectionIndex;
  getOutputSections(File, NewSectionIndex);

  // Set pointer at the end of the output file, so we can pwrite old symbol
  // tables if we need to.
  uint64_t NextAvailableOffset = getFileOffsetForAddress(NextAvailableAddress);
  assert(NextAvailableOffset >= FirstNonAllocatableOffset &&
         "next available offset calculation failure");
  Out->os().seek(NextAvailableOffset);

  // Update dynamic symbol table.
  const ELFShdrTy *DynSymSection = nullptr;
  for (const ELFShdrTy &Section : cantFail(Obj.sections())) {
    if (Section.sh_type == ELF::SHT_DYNSYM) {
      DynSymSection = &Section;
      break;
    }
  }
  assert((DynSymSection || BC->IsStaticExecutable) &&
         "dynamic symbol table expected");
  if (DynSymSection) {
    updateELFSymbolTable(
        File,
        /*IsDynSym=*/true,
        *DynSymSection,
        NewSectionIndex,
        [&](size_t Offset, const ELFSymTy &Sym) {
          Out->os().pwrite(reinterpret_cast<const char *>(&Sym),
                           sizeof(ELFSymTy),
                           DynSymSection->sh_offset + Offset);
        },
        [](StringRef) -> size_t { return 0; });
  }

  if (opts::RemoveSymtab)
    return;

  // (re)create regular symbol table.
  const ELFShdrTy *SymTabSection = nullptr;
  for (const ELFShdrTy &Section : cantFail(Obj.sections())) {
    if (Section.sh_type == ELF::SHT_SYMTAB) {
      SymTabSection = &Section;
      break;
    }
  }
  if (!SymTabSection) {
    errs() << "BOLT-WARNING: no symbol table found\n";
    return;
  }

  const ELFShdrTy *StrTabSection =
      cantFail(Obj.getSection(SymTabSection->sh_link));
  std::string NewContents;
  std::string NewStrTab = std::string(
      File->getData().substr(StrTabSection->sh_offset, StrTabSection->sh_size));
  StringRef SecName = cantFail(Obj.getSectionName(*SymTabSection));
  StringRef StrSecName = cantFail(Obj.getSectionName(*StrTabSection));

  NumLocalSymbols = 0;
  updateELFSymbolTable(
      File,
      /*IsDynSym=*/false,
      *SymTabSection,
      NewSectionIndex,
      [&](size_t Offset, const ELFSymTy &Sym) {
        if (Sym.getBinding() == ELF::STB_LOCAL)
          ++NumLocalSymbols;
        NewContents.append(reinterpret_cast<const char *>(&Sym),
                           sizeof(ELFSymTy));
      },
      [&](StringRef Str) {
        size_t Idx = NewStrTab.size();
        NewStrTab.append(NameResolver::restore(Str).str());
        NewStrTab.append(1, '\0');
        return Idx;
      });

  BC->registerOrUpdateNoteSection(SecName,
                                  copyByteArray(NewContents),
                                  NewContents.size(),
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true,
                                  ELF::SHT_SYMTAB);

  BC->registerOrUpdateNoteSection(StrSecName,
                                  copyByteArray(NewStrTab),
                                  NewStrTab.size(),
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true,
                                  ELF::SHT_STRTAB);
}

template <typename ELFT>
void
RewriteInstance::patchELFAllocatableRelaSections(ELFObjectFile<ELFT> *File) {
  using Elf_Rela = typename ELFT::Rela;
  raw_fd_ostream &OS = Out->os();
  const ELFFile<ELFT> &EF = File->getELFFile();

  uint64_t RelDynOffset = 0, RelDynEndOffset = 0;
  uint64_t RelPltOffset = 0, RelPltEndOffset = 0;

  auto setSectionFileOffsets = [&](uint64_t Address, uint64_t &Start,
                                   uint64_t &End) {
    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
    Start = Section->getInputFileOffset();
    End = Start + Section->getSize();
  };

  if (!DynamicRelocationsAddress && !PLTRelocationsAddress)
    return;

  if (DynamicRelocationsAddress)
    setSectionFileOffsets(*DynamicRelocationsAddress, RelDynOffset,
                          RelDynEndOffset);

  if (PLTRelocationsAddress)
    setSectionFileOffsets(*PLTRelocationsAddress, RelPltOffset,
                          RelPltEndOffset);

  DynamicRelativeRelocationsCount = 0;

  auto writeRela = [&OS](const Elf_Rela *RelA, uint64_t &Offset) {
    OS.pwrite(reinterpret_cast<const char *>(RelA), sizeof(*RelA), Offset);
    Offset += sizeof(*RelA);
  };

  auto writeRelocations = [&](bool PatchRelative) {
    for (BinarySection &Section : BC->allocatableSections()) {
      for (const Relocation &Rel : Section.dynamicRelocations()) {
        const bool IsRelative = Rel.isRelative();
        if (PatchRelative != IsRelative)
          continue;

        if (IsRelative)
          ++DynamicRelativeRelocationsCount;

        Elf_Rela NewRelA;
        uint64_t SectionAddress = Section.getOutputAddress();
        SectionAddress =
            SectionAddress == 0 ? Section.getAddress() : SectionAddress;
        MCSymbol *Symbol = Rel.Symbol;
        uint32_t SymbolIdx = 0;
        uint64_t Addend = Rel.Addend;

        if (Rel.Symbol) {
          SymbolIdx = getOutputDynamicSymbolIndex(Symbol);
        } else {
          // Usually this case is used for R_*_(I)RELATIVE relocations
          const uint64_t Address = getNewFunctionOrDataAddress(Addend);
          if (Address)
            Addend = Address;
        }

        NewRelA.setSymbolAndType(SymbolIdx, Rel.Type, EF.isMips64EL());
        NewRelA.r_offset = SectionAddress + Rel.Offset;
        NewRelA.r_addend = Addend;

        const bool IsJmpRel =
            !!(IsJmpRelocation.find(Rel.Type) != IsJmpRelocation.end());
        uint64_t &Offset = IsJmpRel ? RelPltOffset : RelDynOffset;
        const uint64_t &EndOffset =
            IsJmpRel ? RelPltEndOffset : RelDynEndOffset;
        if (!Offset || !EndOffset) {
          errs() << "BOLT-ERROR: Invalid offsets for dynamic relocation\n";
          exit(1);
        }

        if (Offset + sizeof(NewRelA) > EndOffset) {
          errs() << "BOLT-ERROR: Offset overflow for dynamic relocation\n";
          exit(1);
        }

        writeRela(&NewRelA, Offset);
      }
    }
  };

  // The dynamic linker expects R_*_RELATIVE relocations to be emitted first
  writeRelocations(/* PatchRelative */ true);
  writeRelocations(/* PatchRelative */ false);

  auto fillNone = [&](uint64_t &Offset, uint64_t EndOffset) {
    if (!Offset)
      return;

    typename ELFObjectFile<ELFT>::Elf_Rela RelA;
    RelA.setSymbolAndType(0, Relocation::getNone(), EF.isMips64EL());
    RelA.r_offset = 0;
    RelA.r_addend = 0;
    while (Offset < EndOffset)
      writeRela(&RelA, Offset);

    assert(Offset == EndOffset && "Unexpected section overflow");
  };

  // Fill the rest of the sections with R_*_NONE relocations
  fillNone(RelDynOffset, RelDynEndOffset);
  fillNone(RelPltOffset, RelPltEndOffset);
}

template <typename ELFT>
void RewriteInstance::patchELFGOT(ELFObjectFile<ELFT> *File) {
  raw_fd_ostream &OS = Out->os();

  SectionRef GOTSection;
  for (const SectionRef &Section : File->sections()) {
    StringRef SectionName = cantFail(Section.getName());
    if (SectionName == ".got") {
      GOTSection = Section;
      break;
    }
  }
  if (!GOTSection.getObject()) {
    if (!BC->IsStaticExecutable)
      errs() << "BOLT-INFO: no .got section found\n";
    return;
  }

  StringRef GOTContents = cantFail(GOTSection.getContents());
  for (const uint64_t *GOTEntry =
           reinterpret_cast<const uint64_t *>(GOTContents.data());
       GOTEntry < reinterpret_cast<const uint64_t *>(GOTContents.data() +
                                                     GOTContents.size());
       ++GOTEntry) {
    if (uint64_t NewAddress = getNewFunctionAddress(*GOTEntry)) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: patching GOT entry 0x"
                        << Twine::utohexstr(*GOTEntry) << " with 0x"
                        << Twine::utohexstr(NewAddress) << '\n');
      OS.pwrite(reinterpret_cast<const char *>(&NewAddress), sizeof(NewAddress),
                reinterpret_cast<const char *>(GOTEntry) -
                    File->getData().data());
    }
  }
}

template <typename ELFT>
void RewriteInstance::patchELFDynamic(ELFObjectFile<ELFT> *File) {
  if (BC->IsStaticExecutable)
    return;

  const ELFFile<ELFT> &Obj = File->getELFFile();
  raw_fd_ostream &OS = Out->os();

  using Elf_Phdr = typename ELFFile<ELFT>::Elf_Phdr;
  using Elf_Dyn = typename ELFFile<ELFT>::Elf_Dyn;

  // Locate DYNAMIC by looking through program headers.
  uint64_t DynamicOffset = 0;
  const Elf_Phdr *DynamicPhdr = nullptr;
  for (const Elf_Phdr &Phdr : cantFail(Obj.program_headers())) {
    if (Phdr.p_type == ELF::PT_DYNAMIC) {
      DynamicOffset = Phdr.p_offset;
      DynamicPhdr = &Phdr;
      assert(Phdr.p_memsz == Phdr.p_filesz && "dynamic sizes should match");
      break;
    }
  }
  assert(DynamicPhdr && "missing dynamic in ELF binary");

  bool ZNowSet = false;

  // Go through all dynamic entries and patch functions addresses with
  // new ones.
  typename ELFT::DynRange DynamicEntries =
      cantFail(Obj.dynamicEntries(), "error accessing dynamic table");
  auto DTB = DynamicEntries.begin();
  for (const Elf_Dyn &Dyn : DynamicEntries) {
    Elf_Dyn NewDE = Dyn;
    bool ShouldPatch = true;
    switch (Dyn.d_tag) {
    default:
      ShouldPatch = false;
      break;
    case ELF::DT_RELACOUNT:
      NewDE.d_un.d_val = DynamicRelativeRelocationsCount;
      break;
    case ELF::DT_INIT:
    case ELF::DT_FINI: {
      if (BC->HasRelocations) {
        if (uint64_t NewAddress = getNewFunctionAddress(Dyn.getPtr())) {
          LLVM_DEBUG(dbgs() << "BOLT-DEBUG: patching dynamic entry of type "
                            << Dyn.getTag() << '\n');
          NewDE.d_un.d_ptr = NewAddress;
        }
      }
      RuntimeLibrary *RtLibrary = BC->getRuntimeLibrary();
      if (RtLibrary && Dyn.getTag() == ELF::DT_FINI) {
        if (uint64_t Addr = RtLibrary->getRuntimeFiniAddress())
          NewDE.d_un.d_ptr = Addr;
      }
      if (RtLibrary && Dyn.getTag() == ELF::DT_INIT && !BC->HasInterpHeader) {
        if (auto Addr = RtLibrary->getRuntimeStartAddress()) {
          LLVM_DEBUG(dbgs() << "BOLT-DEBUG: Set DT_INIT to 0x"
                            << Twine::utohexstr(Addr) << '\n');
          NewDE.d_un.d_ptr = Addr;
        }
      }
      break;
    }
    case ELF::DT_FLAGS:
      if (BC->RequiresZNow) {
        NewDE.d_un.d_val |= ELF::DF_BIND_NOW;
        ZNowSet = true;
      }
      break;
    case ELF::DT_FLAGS_1:
      if (BC->RequiresZNow) {
        NewDE.d_un.d_val |= ELF::DF_1_NOW;
        ZNowSet = true;
      }
      break;
    }
    if (ShouldPatch)
      OS.pwrite(reinterpret_cast<const char *>(&NewDE), sizeof(NewDE),
                DynamicOffset + (&Dyn - DTB) * sizeof(Dyn));
  }

  if (BC->RequiresZNow && !ZNowSet) {
    errs() << "BOLT-ERROR: output binary requires immediate relocation "
              "processing which depends on DT_FLAGS or DT_FLAGS_1 presence in "
              ".dynamic. Please re-link the binary with -znow.\n";
    exit(1);
  }
}

template <typename ELFT>
Error RewriteInstance::readELFDynamic(ELFObjectFile<ELFT> *File) {
  const ELFFile<ELFT> &Obj = File->getELFFile();

  using Elf_Phdr = typename ELFFile<ELFT>::Elf_Phdr;
  using Elf_Dyn = typename ELFFile<ELFT>::Elf_Dyn;

  // Locate DYNAMIC by looking through program headers.
  const Elf_Phdr *DynamicPhdr = nullptr;
  for (const Elf_Phdr &Phdr : cantFail(Obj.program_headers())) {
    if (Phdr.p_type == ELF::PT_DYNAMIC) {
      DynamicPhdr = &Phdr;
      break;
    }
  }

  if (!DynamicPhdr) {
    outs() << "BOLT-INFO: static input executable detected\n";
    // TODO: static PIE executable might have dynamic header
    BC->IsStaticExecutable = true;
    return Error::success();
  }

  if (DynamicPhdr->p_memsz != DynamicPhdr->p_filesz)
    return createStringError(errc::executable_format_error,
                             "dynamic section sizes should match");

  // Go through all dynamic entries to locate entries of interest.
  auto DynamicEntriesOrErr = Obj.dynamicEntries();
  if (!DynamicEntriesOrErr)
    return DynamicEntriesOrErr.takeError();
  typename ELFT::DynRange DynamicEntries = DynamicEntriesOrErr.get();

  for (const Elf_Dyn &Dyn : DynamicEntries) {
    switch (Dyn.d_tag) {
    case ELF::DT_INIT:
      if (!BC->HasInterpHeader) {
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: Set start function address\n");
        BC->StartFunctionAddress = Dyn.getPtr();
      }
      break;
    case ELF::DT_FINI:
      BC->FiniFunctionAddress = Dyn.getPtr();
      break;
    case ELF::DT_RELA:
      DynamicRelocationsAddress = Dyn.getPtr();
      break;
    case ELF::DT_RELASZ:
      DynamicRelocationsSize = Dyn.getVal();
      break;
    case ELF::DT_JMPREL:
      PLTRelocationsAddress = Dyn.getPtr();
      break;
    case ELF::DT_PLTRELSZ:
      PLTRelocationsSize = Dyn.getVal();
      break;
    case ELF::DT_RELACOUNT:
      DynamicRelativeRelocationsCount = Dyn.getVal();
      break;
    }
  }

  if (!DynamicRelocationsAddress || !DynamicRelocationsSize) {
    DynamicRelocationsAddress.reset();
    DynamicRelocationsSize = 0;
  }

  if (!PLTRelocationsAddress || !PLTRelocationsSize) {
    PLTRelocationsAddress.reset();
    PLTRelocationsSize = 0;
  }
  return Error::success();
}

uint64_t RewriteInstance::getNewFunctionAddress(uint64_t OldAddress) {
  const BinaryFunction *Function = BC->getBinaryFunctionAtAddress(OldAddress);
  if (!Function)
    return 0;

  return Function->getOutputAddress();
}

uint64_t RewriteInstance::getNewFunctionOrDataAddress(uint64_t OldAddress) {
  if (uint64_t Function = getNewFunctionAddress(OldAddress))
    return Function;

  const BinaryData *BD = BC->getBinaryDataAtAddress(OldAddress);
  if (BD && BD->isMoved())
    return BD->getOutputAddress();

  return 0;
}

void RewriteInstance::rewriteFile() {
  std::error_code EC;
  Out = std::make_unique<ToolOutputFile>(opts::OutputFilename, EC,
                                         sys::fs::OF_None);
  check_error(EC, "cannot create output executable file");

  raw_fd_ostream &OS = Out->os();

  // Copy allocatable part of the input.
  OS << InputFile->getData().substr(0, FirstNonAllocatableOffset);

  // We obtain an asm-specific writer so that we can emit nops in an
  // architecture-specific way at the end of the function.
  std::unique_ptr<MCAsmBackend> MAB(
      BC->TheTarget->createMCAsmBackend(*BC->STI, *BC->MRI, MCTargetOptions()));
  auto Streamer = BC->createStreamer(OS);
  // Make sure output stream has enough reserved space, otherwise
  // pwrite() will fail.
  uint64_t Offset = OS.seek(getFileOffsetForAddress(NextAvailableAddress));
  (void)Offset;
  assert(Offset == getFileOffsetForAddress(NextAvailableAddress) &&
         "error resizing output file");

  // Overwrite functions with fixed output address. This is mostly used by
  // non-relocation mode, with one exception: injected functions are covered
  // here in both modes.
  uint64_t CountOverwrittenFunctions = 0;
  uint64_t OverwrittenScore = 0;
  for (BinaryFunction *Function : BC->getAllBinaryFunctions()) {
    if (Function->getImageAddress() == 0 || Function->getImageSize() == 0)
      continue;

    if (Function->getImageSize() > Function->getMaxSize()) {
      if (opts::Verbosity >= 1)
        errs() << "BOLT-WARNING: new function size (0x"
               << Twine::utohexstr(Function->getImageSize())
               << ") is larger than maximum allowed size (0x"
               << Twine::utohexstr(Function->getMaxSize()) << ") for function "
               << *Function << '\n';

      // Remove jump table sections that this function owns in non-reloc mode
      // because we don't want to write them anymore.
      if (!BC->HasRelocations && opts::JumpTables == JTS_BASIC) {
        for (auto &JTI : Function->JumpTables) {
          JumpTable *JT = JTI.second;
          BinarySection &Section = JT->getOutputSection();
          BC->deregisterSection(Section);
        }
      }
      continue;
    }

    const auto HasAddress = [](const FunctionFragment &FF) {
      return FF.empty() ||
             (FF.getImageAddress() != 0 && FF.getImageSize() != 0);
    };
    const bool SplitFragmentsHaveAddress =
        llvm::all_of(Function->getLayout().getSplitFragments(), HasAddress);
    if (Function->isSplit() && !SplitFragmentsHaveAddress) {
      const auto HasNoAddress = [](const FunctionFragment &FF) {
        return FF.getImageAddress() == 0 && FF.getImageSize() == 0;
      };
      assert(llvm::all_of(Function->getLayout().getSplitFragments(),
                          HasNoAddress) &&
             "Some split fragments have an address while others do not");
      (void)HasNoAddress;
      continue;
    }

    OverwrittenScore += Function->getFunctionScore();
    // Overwrite function in the output file.
    if (opts::Verbosity >= 2)
      outs() << "BOLT: rewriting function \"" << *Function << "\"\n";

    OS.pwrite(reinterpret_cast<char *>(Function->getImageAddress()),
              Function->getImageSize(), Function->getFileOffset());

    // Write nops at the end of the function.
    if (Function->getMaxSize() != std::numeric_limits<uint64_t>::max()) {
      uint64_t Pos = OS.tell();
      OS.seek(Function->getFileOffset() + Function->getImageSize());
      MAB->writeNopData(OS, Function->getMaxSize() - Function->getImageSize(),
                        &*BC->STI);

      OS.seek(Pos);
    }

    if (!Function->isSplit()) {
      ++CountOverwrittenFunctions;
      if (opts::MaxFunctions &&
          CountOverwrittenFunctions == opts::MaxFunctions) {
        outs() << "BOLT: maximum number of functions reached\n";
        break;
      }
      continue;
    }

    // Write cold part
    if (opts::Verbosity >= 2)
      outs() << formatv("BOLT: rewriting function \"{0}\" (split parts)\n",
                        *Function);

    for (const FunctionFragment &FF :
         Function->getLayout().getSplitFragments()) {
      OS.pwrite(reinterpret_cast<char *>(FF.getImageAddress()),
                FF.getImageSize(), FF.getFileOffset());
    }

    ++CountOverwrittenFunctions;
    if (opts::MaxFunctions && CountOverwrittenFunctions == opts::MaxFunctions) {
      outs() << "BOLT: maximum number of functions reached\n";
      break;
    }
  }

  // Print function statistics for non-relocation mode.
  if (!BC->HasRelocations) {
    outs() << "BOLT: " << CountOverwrittenFunctions << " out of "
           << BC->getBinaryFunctions().size()
           << " functions were overwritten.\n";
    if (BC->TotalScore != 0) {
      double Coverage = OverwrittenScore / (double)BC->TotalScore * 100.0;
      outs() << format("BOLT-INFO: rewritten functions cover %.2lf", Coverage)
             << "% of the execution count of simple functions of "
                "this binary\n";
    }
  }

  if (BC->HasRelocations && opts::TrapOldCode) {
    uint64_t SavedPos = OS.tell();
    // Overwrite function body to make sure we never execute these instructions.
    for (auto &BFI : BC->getBinaryFunctions()) {
      BinaryFunction &BF = BFI.second;
      if (!BF.getFileOffset() || !BF.isEmitted())
        continue;
      OS.seek(BF.getFileOffset());
      for (unsigned I = 0; I < BF.getMaxSize(); ++I)
        OS.write((unsigned char)BC->MIB->getTrapFillValue());
    }
    OS.seek(SavedPos);
  }

  // Write all allocatable sections - reloc-mode text is written here as well
  for (BinarySection &Section : BC->allocatableSections()) {
    if (!Section.isFinalized() || !Section.getOutputData())
      continue;

    if (opts::Verbosity >= 1)
      outs() << "BOLT: writing new section " << Section.getName()
             << "\n data at 0x" << Twine::utohexstr(Section.getAllocAddress())
             << "\n of size " << Section.getOutputSize() << "\n at offset "
             << Section.getOutputFileOffset() << '\n';
    OS.pwrite(reinterpret_cast<const char *>(Section.getOutputData()),
              Section.getOutputSize(), Section.getOutputFileOffset());
  }

  for (BinarySection &Section : BC->allocatableSections())
    Section.flushPendingRelocations(OS, [this](const MCSymbol *S) {
      return getNewValueForSymbol(S->getName());
    });

  // If .eh_frame is present create .eh_frame_hdr.
  if (EHFrameSection)
    writeEHFrameHeader();

  // Add BOLT Addresses Translation maps to allow profile collection to
  // happen in the output binary
  if (opts::EnableBAT)
    addBATSection();

  // Patch program header table.
  patchELFPHDRTable();

  // Finalize memory image of section string table.
  finalizeSectionStringTable();

  // Update symbol tables.
  patchELFSymTabs();

  patchBuildID();

  if (opts::EnableBAT)
    encodeBATSection();

  // Copy non-allocatable sections once allocatable part is finished.
  rewriteNoteSections();

  if (BC->HasRelocations) {
    patchELFAllocatableRelaSections();
    patchELFGOT();
  }

  // Patch dynamic section/segment.
  patchELFDynamic();

  // Update ELF book-keeping info.
  patchELFSectionHeaderTable();

  if (opts::PrintSections) {
    outs() << "BOLT-INFO: Sections after processing:\n";
    BC->printSections(outs());
  }

  Out->keep();
  EC = sys::fs::setPermissions(opts::OutputFilename, sys::fs::perms::all_all);
  check_error(EC, "cannot set permissions of output file");
}

void RewriteInstance::writeEHFrameHeader() {
  BinarySection *NewEHFrameSection =
      getSection(getNewSecPrefix() + getEHFrameSectionName());

  // No need to update the header if no new .eh_frame was created.
  if (!NewEHFrameSection)
    return;

  DWARFDebugFrame NewEHFrame(BC->TheTriple->getArch(), true,
                             NewEHFrameSection->getOutputAddress());
  Error E = NewEHFrame.parse(DWARFDataExtractor(
      NewEHFrameSection->getOutputContents(), BC->AsmInfo->isLittleEndian(),
      BC->AsmInfo->getCodePointerSize()));
  check_error(std::move(E), "failed to parse EH frame");

  uint64_t RelocatedEHFrameAddress = 0;
  StringRef RelocatedEHFrameContents;
  BinarySection *RelocatedEHFrameSection =
      getSection(".relocated" + getEHFrameSectionName());
  if (RelocatedEHFrameSection) {
    RelocatedEHFrameAddress = RelocatedEHFrameSection->getOutputAddress();
    RelocatedEHFrameContents = RelocatedEHFrameSection->getOutputContents();
  }
  DWARFDebugFrame RelocatedEHFrame(BC->TheTriple->getArch(), true,
                                   RelocatedEHFrameAddress);
  Error Er = RelocatedEHFrame.parse(DWARFDataExtractor(
      RelocatedEHFrameContents, BC->AsmInfo->isLittleEndian(),
      BC->AsmInfo->getCodePointerSize()));
  check_error(std::move(Er), "failed to parse EH frame");

  LLVM_DEBUG(dbgs() << "BOLT: writing a new .eh_frame_hdr\n");

  NextAvailableAddress =
      appendPadding(Out->os(), NextAvailableAddress, EHFrameHdrAlign);

  const uint64_t EHFrameHdrOutputAddress = NextAvailableAddress;
  const uint64_t EHFrameHdrFileOffset =
      getFileOffsetForAddress(NextAvailableAddress);

  std::vector<char> NewEHFrameHdr = CFIRdWrt->generateEHFrameHeader(
      RelocatedEHFrame, NewEHFrame, EHFrameHdrOutputAddress, FailedAddresses);

  assert(Out->os().tell() == EHFrameHdrFileOffset && "offset mismatch");
  Out->os().write(NewEHFrameHdr.data(), NewEHFrameHdr.size());

  const unsigned Flags = BinarySection::getFlags(/*IsReadOnly=*/true,
                                                 /*IsText=*/false,
                                                 /*IsAllocatable=*/true);
  BinarySection *OldEHFrameHdrSection = getSection(".eh_frame_hdr");
  if (OldEHFrameHdrSection)
    OldEHFrameHdrSection->setOutputName(getOrgSecPrefix() + ".eh_frame_hdr");

  BinarySection &EHFrameHdrSec = BC->registerOrUpdateSection(
      getNewSecPrefix() + ".eh_frame_hdr", ELF::SHT_PROGBITS, Flags, nullptr,
      NewEHFrameHdr.size(), /*Alignment=*/1);
  EHFrameHdrSec.setOutputFileOffset(EHFrameHdrFileOffset);
  EHFrameHdrSec.setOutputAddress(EHFrameHdrOutputAddress);
  EHFrameHdrSec.setOutputName(".eh_frame_hdr");

  NextAvailableAddress += EHFrameHdrSec.getOutputSize();

  // Merge new .eh_frame with the relocated original so that gdb can locate all
  // FDEs.
  if (RelocatedEHFrameSection) {
    const uint64_t NewEHFrameSectionSize =
        RelocatedEHFrameSection->getOutputAddress() +
        RelocatedEHFrameSection->getOutputSize() -
        NewEHFrameSection->getOutputAddress();
    NewEHFrameSection->updateContents(NewEHFrameSection->getOutputData(),
                                      NewEHFrameSectionSize);
    BC->deregisterSection(*RelocatedEHFrameSection);
  }

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: size of .eh_frame after merge is "
                    << NewEHFrameSection->getOutputSize() << '\n');
}

uint64_t RewriteInstance::getNewValueForSymbol(const StringRef Name) {
  uint64_t Value = RTDyld->getSymbol(Name).getAddress();
  if (Value != 0)
    return Value;

  // Return the original value if we haven't emitted the symbol.
  BinaryData *BD = BC->getBinaryDataByName(Name);
  if (!BD)
    return 0;

  return BD->getAddress();
}

uint64_t RewriteInstance::getFileOffsetForAddress(uint64_t Address) const {
  // Check if it's possibly part of the new segment.
  if (Address >= NewTextSegmentAddress)
    return Address - NewTextSegmentAddress + NewTextSegmentOffset;

  // Find an existing segment that matches the address.
  const auto SegmentInfoI = BC->SegmentMapInfo.upper_bound(Address);
  if (SegmentInfoI == BC->SegmentMapInfo.begin())
    return 0;

  const SegmentInfo &SegmentInfo = std::prev(SegmentInfoI)->second;
  if (Address < SegmentInfo.Address ||
      Address >= SegmentInfo.Address + SegmentInfo.FileSize)
    return 0;

  return SegmentInfo.FileOffset + Address - SegmentInfo.Address;
}

bool RewriteInstance::willOverwriteSection(StringRef SectionName) {
  for (const char *const &OverwriteName : SectionsToOverwrite)
    if (SectionName == OverwriteName)
      return true;
  for (std::string &OverwriteName : DebugSectionsToOverwrite)
    if (SectionName == OverwriteName)
      return true;

  ErrorOr<BinarySection &> Section = BC->getUniqueSectionByName(SectionName);
  return Section && Section->isAllocatable() && Section->isFinalized();
}

bool RewriteInstance::isDebugSection(StringRef SectionName) {
  if (SectionName.startswith(".debug_") || SectionName.startswith(".zdebug_") ||
      SectionName == ".gdb_index" || SectionName == ".stab" ||
      SectionName == ".stabstr")
    return true;

  return false;
}

bool RewriteInstance::isKSymtabSection(StringRef SectionName) {
  if (SectionName.startswith("__ksymtab"))
    return true;

  return false;
}
