//===- bolt/Rewrite/RewriteInstance.cpp - ELF rewriter --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/RewriteInstance.h"
#include "bolt/Core/AddressMap.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryEmitter.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/DebugData.h"
#include "bolt/Core/Exceptions.h"
#include "bolt/Core/FunctionLayout.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Core/Relocation.h"
#include "bolt/Passes/BinaryPasses.h"
#include "bolt/Passes/CacheMetrics.h"
#include "bolt/Passes/IdenticalCodeFolding.h"
#include "bolt/Passes/PAuthGadgetScanner.h"
#include "bolt/Passes/ReorderFunctions.h"
#include "bolt/Profile/BoltAddressTranslation.h"
#include "bolt/Profile/DataAggregator.h"
#include "bolt/Profile/DataReader.h"
#include "bolt/Profile/YAMLProfileReader.h"
#include "bolt/Profile/YAMLProfileWriter.h"
#include "bolt/Rewrite/BinaryPassManager.h"
#include "bolt/Rewrite/DWARFRewriter.h"
#include "bolt/Rewrite/ExecutableFileMemoryManager.h"
#include "bolt/Rewrite/JITLinkLinker.h"
#include "bolt/Rewrite/MetadataRewriters.h"
#include "bolt/RuntimeLibs/HugifyRuntimeLibrary.h"
#include "bolt/RuntimeLibs/InstrumentationRuntimeLibrary.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/AddressRanges.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
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

extern cl::list<std::string> HotTextMoveSections;
extern cl::opt<bool> Hugify;
extern cl::opt<bool> Instrument;
extern cl::opt<bool> KeepNops;
extern cl::opt<bool> Lite;
extern cl::list<std::string> ReorderData;
extern cl::opt<bolt::ReorderFunctions::ReorderType> ReorderFunctions;
extern cl::opt<bool> TerminalHLT;
extern cl::opt<bool> TerminalTrap;
extern cl::opt<bool> TimeBuild;
extern cl::opt<bool> TimeRewrite;
extern cl::opt<bolt::IdenticalCodeFolding::ICFLevel, false,
               llvm::bolt::DeprecatedICFNumericOptionParser>
    ICF;

static cl::opt<bool>
    AllowStripped("allow-stripped",
                  cl::desc("allow processing of stripped binaries"), cl::Hidden,
                  cl::cat(BoltCategory));

static cl::opt<bool> ForceToDataRelocations(
    "force-data-relocations",
    cl::desc("force relocations to data sections to always be processed"),

    cl::Hidden, cl::cat(BoltCategory));

static cl::opt<std::string>
    BoltID("bolt-id",
           cl::desc("add any string to tag this execution in the "
                    "output binary via bolt info section"),
           cl::cat(BoltCategory));

cl::opt<bool> DumpDotAll(
    "dump-dot-all",
    cl::desc("dump function CFGs to graphviz format after each stage;"
             "enable '-print-loops' for color-coded blocks"),
    cl::Hidden, cl::cat(BoltCategory));

cl::list<std::string> DumpDotFunc(
    "dump-dot-func", cl::CommaSeparated,
    cl::desc(
        "dump function CFGs to graphviz format for specified functions only;"
        "takes function name patterns (regex supported)"),
    cl::value_desc("func1,func2,func3,..."), cl::Hidden, cl::cat(BoltCategory));

bool shouldDumpDot(const bolt::BinaryFunction &Function) {
  // If dump-dot-all is enabled, dump all functions
  if (DumpDotAll)
    return !Function.isIgnored();

  // If no specific functions specified in dump-dot-func, don't dump any
  if (DumpDotFunc.empty())
    return false;

  if (Function.isIgnored())
    return false;

  // Check if function matches any of the specified patterns
  for (const std::string &Name : DumpDotFunc) {
    if (Function.hasNameRegex(Name)) {
      return true;
    }
  }

  return false;
}

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

static cl::opt<bool>
    PrintProfile("print-profile",
                 cl::desc("print functions after attaching profile"),
                 cl::Hidden, cl::cat(BoltCategory));

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

static cl::opt<cl::boolOrDefault> RelocationMode(
    "relocs", cl::desc("use relocations in the binary (default=autodetect)"),
    cl::cat(BoltCategory));

extern cl::opt<std::string> SaveProfile;

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

static cl::opt<bool> TrapOldCode(
    "trap-old-code",
    cl::desc("insert traps in old function bodies (relocation mode)"),
    cl::Hidden, cl::cat(BoltCategory));

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

static cl::opt<uint64_t> CustomAllocationVMA(
    "custom-allocation-vma",
    cl::desc("use a custom address at which new code will be put, "
             "bypassing BOLT's logic to detect where to put code"),
    cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool>
SequentialDisassembly("sequential-disassembly",
  cl::desc("performs disassembly sequentially"),
  cl::init(false),
  cl::cat(BoltOptCategory));

static cl::opt<bool> WriteBoltInfoSection(
    "bolt-info", cl::desc("write bolt info section in the output binary"),
    cl::init(true), cl::Hidden, cl::cat(BoltOutputCategory));

cl::bits<GadgetScannerKind> GadgetScannersToRun(
    "scanners", cl::desc("which gadget scanners to run"),
    cl::values(
        clEnumValN(GS_PACRET, "pacret",
                   "pac-ret: return address protection (subset of \"pauth\")"),
        clEnumValN(GS_PAUTH, "pauth", "All Pointer Authentication scanners"),
        clEnumValN(GS_ALL, "all", "All implemented scanners")),
    cl::ZeroOrMore, cl::CommaSeparated, cl::cat(BinaryAnalysisCategory));

} // namespace opts

// FIXME: implement a better way to mark sections for replacement.
constexpr const char *RewriteInstance::SectionsToOverwrite[];
std::vector<std::string> RewriteInstance::DebugSectionsToOverwrite = {
    ".debug_abbrev", ".debug_aranges",  ".debug_line",   ".debug_line_str",
    ".debug_loc",    ".debug_loclists", ".debug_ranges", ".debug_rnglists",
    ".gdb_index",    ".debug_addr",     ".debug_abbrev", ".debug_info",
    ".debug_types",  ".pseudo_probe"};

const char RewriteInstance::TimerGroupName[] = "rewrite";
const char RewriteInstance::TimerGroupDesc[] = "Rewrite passes";

namespace llvm {
namespace bolt {

extern const char *BoltRevision;

// Weird location for createMCPlusBuilder, but this is here to avoid a
// cyclic dependency of libCore (its natural place) and libTarget. libRewrite
// can depend on libTarget, but not libCore. Since libRewrite is the only
// user of this function, we define it here.
MCPlusBuilder *createMCPlusBuilder(const Triple::ArchType Arch,
                                   const MCInstrAnalysis *Analysis,
                                   const MCInstrInfo *Info,
                                   const MCRegisterInfo *RegInfo,
                                   const MCSubtargetInfo *STI) {
#ifdef X86_AVAILABLE
  if (Arch == Triple::x86_64)
    return createX86MCPlusBuilder(Analysis, Info, RegInfo, STI);
#endif

#ifdef AARCH64_AVAILABLE
  if (Arch == Triple::aarch64)
    return createAArch64MCPlusBuilder(Analysis, Info, RegInfo, STI);
#endif

#ifdef RISCV_AVAILABLE
  if (Arch == Triple::riscv64)
    return createRISCVMCPlusBuilder(Analysis, Info, RegInfo, STI);
#endif

  llvm_unreachable("architecture unsupported by MCPlusBuilder");
}

} // namespace bolt
} // namespace llvm

using ELF64LEPhdrTy = ELF64LEFile::Elf_Phdr;

namespace {

bool refersToReorderedSection(ErrorOr<BinarySection &> Section) {
  return llvm::any_of(opts::ReorderData, [&](const std::string &SectionName) {
    return Section && Section->getName() == SectionName;
  });
}

} // anonymous namespace

Expected<std::unique_ptr<RewriteInstance>>
RewriteInstance::create(ELFObjectFileBase *File, const int Argc,
                        const char *const *Argv, StringRef ToolPath,
                        raw_ostream &Stdout, raw_ostream &Stderr) {
  Error Err = Error::success();
  auto RI = std::make_unique<RewriteInstance>(File, Argc, Argv, ToolPath,
                                              Stdout, Stderr, Err);
  if (Err)
    return std::move(Err);
  return std::move(RI);
}

RewriteInstance::RewriteInstance(ELFObjectFileBase *File, const int Argc,
                                 const char *const *Argv, StringRef ToolPath,
                                 raw_ostream &Stdout, raw_ostream &Stderr,
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
    Stdout << "BOLT-INFO: shared object or position-independent executable "
              "detected\n";
    IsPIC = true;
  }

  // Make sure we don't miss any output on core dumps.
  Stdout.SetUnbuffered();
  Stderr.SetUnbuffered();
  LLVM_DEBUG(dbgs().SetUnbuffered());

  // Read RISCV subtarget features from input file
  std::unique_ptr<SubtargetFeatures> Features;
  Triple TheTriple = File->makeTriple();
  if (TheTriple.getArch() == llvm::Triple::riscv64) {
    Expected<SubtargetFeatures> FeaturesOrErr = File->getFeatures();
    if (auto E = FeaturesOrErr.takeError()) {
      Err = std::move(E);
      return;
    } else {
      Features.reset(new SubtargetFeatures(*FeaturesOrErr));
    }
  }

  Relocation::Arch = TheTriple.getArch();
  auto BCOrErr = BinaryContext::createBinaryContext(
      TheTriple, std::make_shared<orc::SymbolStringPool>(), File->getFileName(),
      Features.get(), IsPIC,
      DWARFContext::create(*File, DWARFContext::ProcessDebugRelocations::Ignore,
                           nullptr, opts::DWPPathName,
                           WithColor::defaultErrorHandler,
                           WithColor::defaultWarningHandler),
      JournalingStreams{Stdout, Stderr});
  if (Error E = BCOrErr.takeError()) {
    Err = std::move(E);
    return;
  }
  BC = std::move(BCOrErr.get());
  BC->initializeTarget(std::unique_ptr<MCPlusBuilder>(
      createMCPlusBuilder(BC->TheTriple->getArch(), BC->MIA.get(),
                          BC->MII.get(), BC->MRI.get(), BC->STI.get())));

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

// Return if a section stored in the image falls into a segment address space.
// If not, Set \p Overlap to true if there's a partial overlap.
template <class ELFT>
static bool checkOffsets(const typename ELFT::Phdr &Phdr,
                         const typename ELFT::Shdr &Sec, bool &Overlap) {
  // SHT_NOBITS sections don't need to have an offset inside the segment.
  if (Sec.sh_type == ELF::SHT_NOBITS)
    return true;

  // Only non-empty sections can be at the end of a segment.
  uint64_t SectionSize = Sec.sh_size ? Sec.sh_size : 1ull;
  AddressRange SectionAddressRange((uint64_t)Sec.sh_offset,
                                   Sec.sh_offset + SectionSize);
  AddressRange SegmentAddressRange(Phdr.p_offset,
                                   Phdr.p_offset + Phdr.p_filesz);
  if (SegmentAddressRange.contains(SectionAddressRange))
    return true;

  Overlap = SegmentAddressRange.intersects(SectionAddressRange);
  return false;
}

// Check that an allocatable section belongs to a virtual address
// space of a segment.
template <class ELFT>
static bool checkVMA(const typename ELFT::Phdr &Phdr,
                     const typename ELFT::Shdr &Sec, bool &Overlap) {
  // Only non-empty sections can be at the end of a segment.
  uint64_t SectionSize = Sec.sh_size ? Sec.sh_size : 1ull;
  AddressRange SectionAddressRange((uint64_t)Sec.sh_addr,
                                   Sec.sh_addr + SectionSize);
  AddressRange SegmentAddressRange(Phdr.p_vaddr, Phdr.p_vaddr + Phdr.p_memsz);

  if (SegmentAddressRange.contains(SectionAddressRange))
    return true;
  Overlap = SegmentAddressRange.intersects(SectionAddressRange);
  return false;
}

void RewriteInstance::markGnuRelroSections() {
  using ELFT = ELF64LE;
  using ELFShdrTy = typename ELFObjectFile<ELFT>::Elf_Shdr;
  auto ELF64LEFile = cast<ELF64LEObjectFile>(InputFile);
  const ELFFile<ELFT> &Obj = ELF64LEFile->getELFFile();

  auto handleSection = [&](const ELFT::Phdr &Phdr, SectionRef SecRef) {
    BinarySection *BinarySection = BC->getSectionForSectionRef(SecRef);
    // If the section is non-allocatable, ignore it for GNU_RELRO purposes:
    // it can't be made read-only after runtime relocations processing.
    if (!BinarySection || !BinarySection->isAllocatable())
      return;
    const ELFShdrTy *Sec = cantFail(Obj.getSection(SecRef.getIndex()));
    bool ImageOverlap{false}, VMAOverlap{false};
    bool ImageContains = checkOffsets<ELFT>(Phdr, *Sec, ImageOverlap);
    bool VMAContains = checkVMA<ELFT>(Phdr, *Sec, VMAOverlap);
    if (ImageOverlap) {
      if (opts::Verbosity >= 1)
        BC->errs() << "BOLT-WARNING: GNU_RELRO segment has partial file offset "
                   << "overlap with section " << BinarySection->getName()
                   << '\n';
      return;
    }
    if (VMAOverlap) {
      if (opts::Verbosity >= 1)
        BC->errs() << "BOLT-WARNING: GNU_RELRO segment has partial VMA overlap "
                   << "with section " << BinarySection->getName() << '\n';
      return;
    }
    if (!ImageContains || !VMAContains)
      return;
    BinarySection->setRelro();
    if (opts::Verbosity >= 1)
      BC->outs() << "BOLT-INFO: marking " << BinarySection->getName()
                 << " as GNU_RELRO\n";
  };

  for (const ELFT::Phdr &Phdr : cantFail(Obj.program_headers()))
    if (Phdr.p_type == ELF::PT_GNU_RELRO)
      for (SectionRef SecRef : InputFile->sections())
        handleSection(Phdr, SecRef);
}

Error RewriteInstance::discoverStorage() {
  NamedRegionTimer T("discoverStorage", "discover storage", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);

  auto ELF64LEFile = cast<ELF64LEObjectFile>(InputFile);
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

      BC->SegmentMapInfo[Phdr.p_vaddr] =
          SegmentInfo{Phdr.p_vaddr,
                      Phdr.p_memsz,
                      Phdr.p_offset,
                      Phdr.p_filesz,
                      Phdr.p_align,
                      (Phdr.p_flags & ELF::PF_X) != 0,
                      (Phdr.p_flags & ELF::PF_W) != 0};
      if (BC->TheTriple->getArch() == llvm::Triple::x86_64 &&
          Phdr.p_vaddr >= BinaryContext::KernelStartX86_64)
        BC->IsLinuxKernel = true;
      break;
    case ELF::PT_INTERP:
      BC->HasInterpHeader = true;
      break;
    }
  }

  if (BC->IsLinuxKernel)
    BC->outs() << "BOLT-INFO: Linux kernel binary detected\n";

  for (const SectionRef &Section : InputFile->sections()) {
    Expected<StringRef> SectionNameOrErr = Section.getName();
    if (Error E = SectionNameOrErr.takeError())
      return E;
    StringRef SectionName = SectionNameOrErr.get();
    if (SectionName == BC->getMainCodeSectionName()) {
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
        (SectionName.starts_with(getOrgSecPrefix()) ||
         SectionName == getBOLTTextSectionName()))
      return createStringError(
          errc::function_not_supported,
          "BOLT-ERROR: input file was processed by BOLT. Cannot re-optimize");
  }

  if (!NextAvailableAddress || !NextAvailableOffset)
    return createStringError(errc::executable_format_error,
                             "no PT_LOAD pheader seen");

  BC->outs() << "BOLT-INFO: first alloc address is 0x"
             << Twine::utohexstr(BC->FirstAllocAddress) << '\n';

  FirstNonAllocatableOffset = NextAvailableOffset;

  if (opts::CustomAllocationVMA) {
    // If user specified a custom address where we should start writing new
    // data, honor that.
    NextAvailableAddress = opts::CustomAllocationVMA;
    // Sanity check the user-supplied address and emit warnings if something
    // seems off.
    for (const ELF64LE::Phdr &Phdr : PHs) {
      switch (Phdr.p_type) {
      case ELF::PT_LOAD:
        if (NextAvailableAddress >= Phdr.p_vaddr &&
            NextAvailableAddress < Phdr.p_vaddr + Phdr.p_memsz) {
          BC->errs() << "BOLT-WARNING: user-supplied allocation vma 0x"
                     << Twine::utohexstr(NextAvailableAddress)
                     << " conflicts with ELF segment at 0x"
                     << Twine::utohexstr(Phdr.p_vaddr) << "\n";
        }
      }
    }
  }
  NextAvailableAddress = alignTo(NextAvailableAddress, BC->PageAlign);
  NextAvailableOffset = alignTo(NextAvailableOffset, BC->PageAlign);

  // Hugify: Additional huge page from left side due to
  // weird ASLR mapping addresses (4KB aligned)
  if (opts::Hugify && !BC->HasFixedLoadAddress) {
    NextAvailableAddress += BC->PageAlign;
  }

  NewTextSegmentAddress = NextAvailableAddress;
  NewTextSegmentOffset = NextAvailableOffset;

  if (!opts::UseGnuStack && !BC->IsLinuxKernel) {
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

    BC->outs() << "BOLT-INFO: creating new program header table at address 0x"
               << Twine::utohexstr(NextAvailableAddress) << ", offset 0x"
               << Twine::utohexstr(NextAvailableOffset) << '\n';

    PHDRTableAddress = NextAvailableAddress;
    PHDRTableOffset = NextAvailableOffset;
    NewTextSegmentAddress = NextAvailableAddress;
    NewTextSegmentOffset = NextAvailableOffset;

    // Reserve space for 3 extra pheaders.
    unsigned Phnum = Obj.getHeader().e_phnum;
    Phnum += 3;

    // Reserve two more pheaders to avoid having writeable and executable
    // segment in instrumented binary.
    if (opts::Instrument)
      Phnum += 2;

    NextAvailableAddress += Phnum * sizeof(ELF64LEPhdrTy);
    NextAvailableOffset += Phnum * sizeof(ELF64LEPhdrTy);

    // Align at cache line.
    NextAvailableAddress = alignTo(NextAvailableAddress, 64);
    NextAvailableOffset = alignTo(NextAvailableOffset, 64);
  }

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

Error RewriteInstance::run() {
  assert(BC && "failed to create a binary context");

  BC->outs() << "BOLT-INFO: Target architecture: "
             << Triple::getArchTypeName(
                    (llvm::Triple::ArchType)InputFile->getArch())
             << "\n";
  BC->outs() << "BOLT-INFO: BOLT version: " << BoltRevision << "\n";

  if (Error E = discoverStorage())
    return E;
  if (Error E = readSpecialSections())
    return E;
  adjustCommandLineOptions();
  discoverFileObjects();

  if (opts::Instrument && !BC->IsStaticExecutable)
    if (Error E = discoverRtFiniAddress())
      return E;

  preprocessProfileData();

  selectFunctionsToProcess();

  readDebugInfo();

  disassembleFunctions();

  processMetadataPreCFG();

  buildFunctionsCFG();

  processProfileData();

  // Save input binary metadata if BAT section needs to be emitted
  if (opts::EnableBAT)
    BAT->saveMetadata(*BC);

  postProcessFunctions();

  processMetadataPostCFG();

  if (opts::DiffOnly)
    return Error::success();

  if (opts::BinaryAnalysisMode) {
    runBinaryAnalyses();
    return Error::success();
  }

  preregisterSections();

  runOptimizationPasses();

  finalizeMetadataPreEmit();

  emitAndLink();

  updateMetadata();

  if (opts::Instrument && !BC->IsStaticExecutable)
    updateRtFiniReloc();

  if (opts::OutputFilename == "/dev/null") {
    BC->outs() << "BOLT-INFO: skipping writing final binary to disk\n";
    return Error::success();
  } else if (BC->IsLinuxKernel) {
    BC->errs() << "BOLT-WARNING: Linux kernel support is experimental\n";
  }

  // Rewrite allocatable contents and copy non-allocatable parts with mods.
  rewriteFile();
  return Error::success();
}

void RewriteInstance::discoverFileObjects() {
  NamedRegionTimer T("discoverFileObjects", "discover file objects",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);

  // For local symbols we want to keep track of associated FILE symbol name for
  // disambiguation by combined name.
  for (const ELFSymbolRef &Symbol : InputFile->symbols()) {
    Expected<StringRef> NameOrError = Symbol.getName();
    if (NameOrError && NameOrError->starts_with("__asan_init")) {
      BC->errs()
          << "BOLT-ERROR: input file was compiled or linked with sanitizer "
             "support. Cannot optimize.\n";
      exit(1);
    }
    if (NameOrError && NameOrError->starts_with("__llvm_coverage_mapping")) {
      BC->errs()
          << "BOLT-ERROR: input file was compiled or linked with coverage "
             "support. Cannot optimize.\n";
      exit(1);
    }

    if (cantFail(Symbol.getFlags()) & SymbolRef::SF_Undefined)
      continue;

    if (cantFail(Symbol.getType()) == SymbolRef::ST_File)
      FileSymbols.emplace_back(Symbol);
  }

  // Sort symbols in the file by value. Ignore symbols from non-allocatable
  // sections. We memoize getAddress(), as it has rather high overhead.
  struct SymbolInfo {
    uint64_t Address;
    SymbolRef Symbol;
  };
  std::vector<SymbolInfo> SortedSymbols;
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
  auto checkSymbolInSection = [this](const SymbolInfo &S) {
    // Sometimes, we encounter symbols with addresses outside their section. If
    // such symbols happen to fall into another section, they can interfere with
    // disassembly. Notably, this occurs with AArch64 marker symbols ($d and $t)
    // that belong to .eh_frame, but end up pointing into .text.
    // As a workaround, we ignore all symbols that lie outside their sections.
    auto Section = cantFail(S.Symbol.getSection());

    // Accept all absolute symbols.
    if (Section == InputFile->section_end())
      return true;

    uint64_t SecStart = Section->getAddress();
    uint64_t SecEnd = SecStart + Section->getSize();
    uint64_t SymEnd = S.Address + ELFSymbolRef(S.Symbol).getSize();
    if (S.Address >= SecStart && SymEnd <= SecEnd)
      return true;

    auto SymType = cantFail(S.Symbol.getType());
    // Skip warnings for common benign cases.
    if (opts::Verbosity < 1 && SymType == SymbolRef::ST_Other)
      return false; // E.g. ELF::STT_TLS.

    auto SymName = S.Symbol.getName();
    auto SecName = cantFail(S.Symbol.getSection())->getName();
    BC->errs() << "BOLT-WARNING: ignoring symbol "
               << (SymName ? *SymName : "[unnamed]") << " at 0x"
               << Twine::utohexstr(S.Address) << ", which lies outside "
               << (SecName ? *SecName : "[unnamed]") << "\n";

    return false;
  };
  for (const SymbolRef &Symbol : InputFile->symbols())
    if (isSymbolInMemory(Symbol)) {
      SymbolInfo SymInfo{cantFail(Symbol.getAddress()), Symbol};
      if (checkSymbolInSection(SymInfo))
        SortedSymbols.push_back(SymInfo);
    }

  auto CompareSymbols = [this](const SymbolInfo &A, const SymbolInfo &B) {
    if (A.Address != B.Address)
      return A.Address < B.Address;

    const bool AMarker = BC->isMarker(A.Symbol);
    const bool BMarker = BC->isMarker(B.Symbol);
    if (AMarker || BMarker) {
      return AMarker && !BMarker;
    }

    const auto AType = cantFail(A.Symbol.getType());
    const auto BType = cantFail(B.Symbol.getType());
    if (AType == SymbolRef::ST_Function && BType != SymbolRef::ST_Function)
      return true;
    if (BType == SymbolRef::ST_Debug && AType != SymbolRef::ST_Debug)
      return true;

    return false;
  };
  llvm::stable_sort(SortedSymbols, CompareSymbols);

  auto LastSymbol = SortedSymbols.end();
  if (!SortedSymbols.empty())
    --LastSymbol;

  // For aarch64, the ABI defines mapping symbols so we identify data in the
  // code section (see IHI0056B). $d identifies data contents.
  // Compilers usually merge multiple data objects in a single $d-$x interval,
  // but we need every data object to be marked with $d. Because of that we
  // keep track of marker symbols with all locations of data objects.

  DenseMap<uint64_t, MarkerSymType> MarkerSymbols;
  auto addExtraDataMarkerPerSymbol = [&]() {
    bool IsData = false;
    uint64_t LastAddr = 0;
    for (const auto &SymInfo : SortedSymbols) {
      if (LastAddr == SymInfo.Address) // don't repeat markers
        continue;

      MarkerSymType MarkerType = BC->getMarkerType(SymInfo.Symbol);

      // Treat ST_Function as code.
      Expected<object::SymbolRef::Type> TypeOrError = SymInfo.Symbol.getType();
      consumeError(TypeOrError.takeError());
      if (TypeOrError && *TypeOrError == SymbolRef::ST_Function) {
        if (IsData) {
          Expected<StringRef> NameOrError = SymInfo.Symbol.getName();
          consumeError(NameOrError.takeError());
          BC->errs() << "BOLT-WARNING: function symbol " << *NameOrError
                     << " lacks code marker\n";
        }
        MarkerType = MarkerSymType::CODE;
      }

      if (MarkerType != MarkerSymType::NONE) {
        MarkerSymbols[SymInfo.Address] = MarkerType;
        LastAddr = SymInfo.Address;
        IsData = MarkerType == MarkerSymType::DATA;
        continue;
      }

      if (IsData) {
        MarkerSymbols[SymInfo.Address] = MarkerSymType::DATA;
        LastAddr = SymInfo.Address;
      }
    }
  };

  if (BC->isAArch64() || BC->isRISCV()) {
    addExtraDataMarkerPerSymbol();
    LastSymbol = std::stable_partition(
        SortedSymbols.begin(), SortedSymbols.end(),
        [this](const SymbolInfo &S) { return !BC->isMarker(S.Symbol); });
    if (!SortedSymbols.empty())
      --LastSymbol;
  }

  BinaryFunction *PreviousFunction = nullptr;
  unsigned AnonymousId = 0;

  const auto SortedSymbolsEnd =
      LastSymbol == SortedSymbols.end() ? LastSymbol : std::next(LastSymbol);
  for (auto Iter = SortedSymbols.begin(); Iter != SortedSymbolsEnd; ++Iter) {
    const SymbolRef &Symbol = Iter->Symbol;
    const uint64_t SymbolAddress = Iter->Address;
    const auto SymbolFlags = cantFail(Symbol.getFlags());
    const SymbolRef::Type SymbolType = cantFail(Symbol.getType());

    if (SymbolType == SymbolRef::ST_File)
      continue;

    StringRef SymName = cantFail(Symbol.getName(), "cannot get symbol name");
    if (SymbolAddress == 0) {
      if (opts::Verbosity >= 1 && SymbolType == SymbolRef::ST_Function)
        BC->errs() << "BOLT-WARNING: function with 0 address seen\n";
      continue;
    }

    // Ignore input hot markers unless in heatmap mode
    if ((SymName == "__hot_start" || SymName == "__hot_end") &&
        !opts::HeatmapMode)
      continue;

    FileSymRefs.emplace(SymbolAddress, Symbol);

    // Skip section symbols that will be registered by disassemblePLT().
    if (SymbolType == SymbolRef::ST_Debug) {
      ErrorOr<BinarySection &> BSection =
          BC->getSectionForAddress(SymbolAddress);
      if (BSection && getPLTSectionInfo(BSection->getName()))
        continue;
    }

    /// It is possible we are seeing a globalized local. LLVM might treat it as
    /// a local if it has a "private global" prefix, e.g. ".L". Thus we have to
    /// change the prefix to enforce global scope of the symbol.
    std::string Name =
        SymName.starts_with(BC->AsmInfo->getPrivateGlobalPrefix())
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
    } else if (SymbolFlags & SymbolRef::SF_Global) {
      if (const BinaryData *BD = BC->getBinaryDataByName(Name)) {
        if (BD->getSize() == ELFSymbolRef(Symbol).getSize() &&
            BD->getAddress() == SymbolAddress) {
          if (opts::Verbosity > 1)
            BC->errs() << "BOLT-WARNING: ignoring duplicate global symbol "
                       << Name << "\n";
          // Ignore duplicate entry - possibly a bug in the linker
          continue;
        }
        BC->errs() << "BOLT-ERROR: bad input binary, global symbol \"" << Name
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
      auto SFI = llvm::upper_bound(FileSymbols, ELFSymbolRef(Symbol));
      if (SymbolType == SymbolRef::ST_Function && SFI != FileSymbols.begin()) {
        StringRef FileSymbolName = cantFail(SFI[-1].getName());
        if (!FileSymbolName.empty())
          AlternativeName = NR.uniquify(Name + "/" + FileSymbolName.str());
      }

      UniqueName = NR.uniquify(Name);
    }

    uint64_t SymbolSize = ELFSymbolRef(Symbol).getSize();
    uint64_t SymbolAlignment = Symbol.getAlignment();

    auto registerName = [&](uint64_t FinalSize) {
      // Register names even if it's not a function, e.g. for an entry point.
      BC->registerNameAtAddress(UniqueName, SymbolAddress, FinalSize,
                                SymbolAlignment, SymbolFlags);
      if (!AlternativeName.empty())
        BC->registerNameAtAddress(AlternativeName, SymbolAddress, FinalSize,
                                  SymbolAlignment, SymbolFlags);
    };

    section_iterator Section =
        cantFail(Symbol.getSection(), "cannot get symbol section");
    if (Section == InputFile->section_end()) {
      // Could be an absolute symbol. Used on RISC-V for __global_pointer$ so we
      // need to record it to handle relocations against it. For other instances
      // of absolute symbols, we record for pretty printing.
      LLVM_DEBUG(if (opts::Verbosity > 1) {
        dbgs() << "BOLT-INFO: absolute sym " << UniqueName << "\n";
      });
      registerName(SymbolSize);
      continue;
    }

    if (SymName == getBOLTReservedStart() || SymName == getBOLTReservedEnd()) {
      registerName(SymbolSize);
      continue;
    }

    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: considering symbol " << UniqueName
                      << " for function\n");

    if (SymbolAddress == Section->getAddress() + Section->getSize()) {
      assert(SymbolSize == 0 &&
             "unexpect non-zero sized symbol at end of section");
      LLVM_DEBUG(
          dbgs()
          << "BOLT-DEBUG: rejecting as symbol points to end of its section\n");
      registerName(SymbolSize);
      continue;
    }

    if (!Section->isText() || Section->isVirtual()) {
      assert(SymbolType != SymbolRef::ST_Function &&
             "unexpected function inside non-code section");
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: rejecting as symbol is not in code or "
                           "is in nobits section\n");
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
      if (PreviousFunction->containsAddress(SymbolAddress)) {
        if (PreviousFunction->isSymbolValidInScope(Symbol, SymbolSize)) {
          LLVM_DEBUG(dbgs()
                     << "BOLT-DEBUG: symbol is a function local symbol\n");
        } else if (SymbolAddress == PreviousFunction->getAddress() &&
                   !SymbolSize) {
          LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring symbol as a marker\n");
        } else if (opts::Verbosity > 1) {
          BC->errs() << "BOLT-WARNING: symbol " << UniqueName
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

    if (PreviousFunction && PreviousFunction->containsAddress(SymbolAddress) &&
        PreviousFunction->getAddress() != SymbolAddress) {
      if (PreviousFunction->isSymbolValidInScope(Symbol, SymbolSize)) {
        if (opts::Verbosity >= 1)
          BC->outs()
              << "BOLT-INFO: skipping possibly another entry for function "
              << *PreviousFunction << " : " << UniqueName << '\n';
        registerName(SymbolSize);
      } else {
        BC->outs() << "BOLT-INFO: using " << UniqueName
                   << " as another entry to "
                   << "function " << *PreviousFunction << '\n';

        registerName(0);

        PreviousFunction->addEntryPointAtOffset(SymbolAddress -
                                                PreviousFunction->getAddress());

        // Remove the symbol from FileSymRefs so that we can skip it from
        // in the future.
        auto SI = llvm::find_if(
            llvm::make_range(FileSymRefs.equal_range(SymbolAddress)),
            [&](auto SymIt) { return SymIt.second == Symbol; });
        assert(SI != FileSymRefs.end() && "symbol expected to be present");
        assert(SI->second == Symbol && "wrong symbol found");
        FileSymRefs.erase(SI);
      }
      continue;
    }

    // Checkout for conflicts with function data from FDEs.
    bool IsSimple = true;
    auto FDEI = CFIRdWrt->getFDEs().lower_bound(SymbolAddress);
    if (FDEI != CFIRdWrt->getFDEs().end()) {
      const dwarf::FDE &FDE = *FDEI->second;
      if (FDEI->first != SymbolAddress) {
        // There's no matching starting address in FDE. Make sure the previous
        // FDE does not contain this address.
        if (FDEI != CFIRdWrt->getFDEs().begin()) {
          --FDEI;
          const dwarf::FDE &PrevFDE = *FDEI->second;
          uint64_t PrevStart = PrevFDE.getInitialLocation();
          uint64_t PrevLength = PrevFDE.getAddressRange();
          if (SymbolAddress > PrevStart &&
              SymbolAddress < PrevStart + PrevLength) {
            BC->errs() << "BOLT-ERROR: function " << UniqueName
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
          BC->errs() << "BOLT-WARNING: sizes differ for function " << UniqueName
                     << ". FDE : " << FDE.getAddressRange()
                     << "; symbol table : " << SymbolSize
                     << ". Using max size.\n";
        }
        SymbolSize = std::max(SymbolSize, FDE.getAddressRange());
        if (BC->getBinaryDataAtAddress(SymbolAddress)) {
          BC->setBinaryDataSize(SymbolAddress, SymbolSize);
        } else {
          LLVM_DEBUG(dbgs() << "BOLT-DEBUG: No BD @ 0x"
                            << Twine::utohexstr(SymbolAddress) << "\n");
        }
      }
    }

    BinaryFunction *BF = nullptr;
    // Since function may not have yet obtained its real size, do a search
    // using the list of registered functions instead of calling
    // getBinaryFunctionAtAddress().
    auto BFI = BC->getBinaryFunctions().find(SymbolAddress);
    if (BFI != BC->getBinaryFunctions().end()) {
      BF = &BFI->second;
      // Duplicate the function name. Make sure everything matches before we add
      // an alternative name.
      if (SymbolSize != BF->getSize()) {
        if (opts::Verbosity >= 1) {
          if (SymbolSize && BF->getSize())
            BC->errs() << "BOLT-WARNING: size mismatch for duplicate entries "
                       << *BF << " and " << UniqueName << '\n';
          BC->outs() << "BOLT-INFO: adjusting size of function " << *BF
                     << " old " << BF->getSize() << " new " << SymbolSize
                     << "\n";
        }
        BF->setSize(std::max(SymbolSize, BF->getSize()));
        BC->setBinaryDataSize(SymbolAddress, BF->getSize());
      }
      BF->addAlternativeName(UniqueName);
    } else {
      ErrorOr<BinarySection &> Section =
          BC->getSectionForAddress(SymbolAddress);
      // Skip symbols from invalid sections
      if (!Section) {
        BC->errs() << "BOLT-WARNING: " << UniqueName << " (0x"
                   << Twine::utohexstr(SymbolAddress)
                   << ") does not have any section\n";
        continue;
      }

      // Skip symbols from zero-sized sections.
      if (!Section->getSize())
        continue;

      BF = BC->createBinaryFunction(UniqueName, *Section, SymbolAddress,
                                    SymbolSize);
      if (!IsSimple)
        BF->setSimple(false);
    }

    // Check if it's a cold function fragment.
    if (FunctionFragmentTemplate.match(SymName)) {
      static bool PrintedWarning = false;
      if (!PrintedWarning) {
        PrintedWarning = true;
        BC->errs() << "BOLT-WARNING: split function detected on input : "
                   << SymName;
        if (BC->HasRelocations)
          BC->errs() << ". The support is limited in relocation mode\n";
        else
          BC->errs() << '\n';
      }
      BC->HasSplitFunctions = true;
      BF->IsFragment = true;
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
      BC->errs() << "BOLT-WARNING: FDE [0x" << Twine::utohexstr(Address)
                 << ", 0x" << Twine::utohexstr(Address + FDE->getAddressRange())
                 << ") conflicts with function " << *BF << '\n';
      continue;
    }

    if (opts::Verbosity >= 1)
      BC->errs() << "BOLT-WARNING: FDE [0x" << Twine::utohexstr(Address)
                 << ", 0x" << Twine::utohexstr(Address + FDE->getAddressRange())
                 << ") has no corresponding symbol table entry\n";

    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
    assert(Section && "cannot get section for address from FDE");
    std::string FunctionName =
        "__BOLT_FDE_FUNCat" + Twine::utohexstr(Address).str();
    BC->createBinaryFunction(FunctionName, *Section, Address,
                             FDE->getAddressRange());
  }

  BC->setHasSymbolsWithFileName(FileSymbols.size());

  // Now that all the functions were created - adjust their boundaries.
  adjustFunctionBoundaries(MarkerSymbols);

  // Annotate functions with code/data markers in AArch64
  for (auto &[Address, Type] : MarkerSymbols) {
    auto *BF = BC->getBinaryFunctionContainingAddress(Address,
                                                      /*CheckPastEnd*/ false,
                                                      /*UseMaxSize*/ true);

    if (!BF) {
      // Stray marker
      continue;
    }
    const auto EntryOffset = Address - BF->getAddress();
    if (Type == MarkerSymType::CODE) {
      BF->markCodeAtOffset(EntryOffset);
      continue;
    }
    if (Type == MarkerSymType::DATA) {
      BF->markDataAtOffset(EntryOffset);
      BC->AddressToConstantIslandMap[Address] = BF;
      continue;
    }
    llvm_unreachable("Unknown marker");
  }

  if (BC->isAArch64()) {
    // Check for dynamic relocations that might be contained in
    // constant islands.
    for (const BinarySection &Section : BC->allocatableSections()) {
      const uint64_t SectionAddress = Section.getAddress();
      for (const Relocation &Rel : Section.dynamicRelocations()) {
        const uint64_t RelAddress = SectionAddress + Rel.Offset;
        BinaryFunction *BF =
            BC->getBinaryFunctionContainingAddress(RelAddress,
                                                   /*CheckPastEnd*/ false,
                                                   /*UseMaxSize*/ true);
        if (BF) {
          assert(Rel.isRelative() && "Expected relative relocation for island");
          BC->logBOLTErrorsAndQuitOnFatal(
              BF->markIslandDynamicRelocationAtAddress(RelAddress));
        }
      }
    }

    // The linker may omit data markers for absolute long veneers. Introduce
    // those markers artificially to assist the disassembler.
    for (BinaryFunction &BF :
         llvm::make_second_range(BC->getBinaryFunctions())) {
      if (BF.getOneName().starts_with("__AArch64AbsLongThunk_") &&
          BF.getSize() == 16 && !BF.getSizeOfDataInCodeAt(8)) {
        BC->errs() << "BOLT-WARNING: missing data marker detected in veneer "
                   << BF << '\n';
        BF.markDataAtOffset(8);
        BC->AddressToConstantIslandMap[BF.getAddress() + 8] = &BF;
      }
    }
  }

  if (!BC->IsLinuxKernel) {
    // Read all relocations now that we have binary functions mapped.
    processRelocations();
  }

  registerFragments();
  FileSymbols.clear();
  FileSymRefs.clear();

  discoverBOLTReserved();
}

void RewriteInstance::discoverBOLTReserved() {
  BinaryData *StartBD = BC->getBinaryDataByName(getBOLTReservedStart());
  BinaryData *EndBD = BC->getBinaryDataByName(getBOLTReservedEnd());
  if (!StartBD != !EndBD) {
    BC->errs() << "BOLT-ERROR: one of the symbols is missing from the binary: "
               << getBOLTReservedStart() << ", " << getBOLTReservedEnd()
               << '\n';
    exit(1);
  }

  if (!StartBD)
    return;

  if (StartBD->getAddress() >= EndBD->getAddress()) {
    BC->errs() << "BOLT-ERROR: invalid reserved space boundaries\n";
    exit(1);
  }
  BC->BOLTReserved = AddressRange(StartBD->getAddress(), EndBD->getAddress());
  BC->outs() << "BOLT-INFO: using reserved space for allocating new sections\n";

  PHDRTableOffset = 0;
  PHDRTableAddress = 0;
  NewTextSegmentAddress = 0;
  NewTextSegmentOffset = 0;
  NextAvailableAddress = BC->BOLTReserved.start();
}

Error RewriteInstance::discoverRtFiniAddress() {
  // Use DT_FINI if it's available.
  if (BC->FiniAddress) {
    BC->FiniFunctionAddress = BC->FiniAddress;
    return Error::success();
  }

  if (!BC->FiniArrayAddress || !BC->FiniArraySize) {
    return createStringError(
        std::errc::not_supported,
        "Instrumentation needs either DT_FINI or DT_FINI_ARRAY");
  }

  if (*BC->FiniArraySize < BC->AsmInfo->getCodePointerSize()) {
    return createStringError(std::errc::not_supported,
                             "Need at least 1 DT_FINI_ARRAY slot");
  }

  ErrorOr<BinarySection &> FiniArraySection =
      BC->getSectionForAddress(*BC->FiniArrayAddress);
  if (auto EC = FiniArraySection.getError())
    return errorCodeToError(EC);

  if (const Relocation *Reloc = FiniArraySection->getDynamicRelocationAt(0)) {
    BC->FiniFunctionAddress = Reloc->Addend;
    return Error::success();
  }

  if (const Relocation *Reloc = FiniArraySection->getRelocationAt(0)) {
    BC->FiniFunctionAddress = Reloc->Value;
    return Error::success();
  }

  return createStringError(std::errc::not_supported,
                           "No relocation for first DT_FINI_ARRAY slot");
}

void RewriteInstance::updateRtFiniReloc() {
  // Updating DT_FINI is handled by patchELFDynamic.
  if (BC->FiniAddress)
    return;

  const RuntimeLibrary *RT = BC->getRuntimeLibrary();
  if (!RT || !RT->getRuntimeFiniAddress())
    return;

  assert(BC->FiniArrayAddress && BC->FiniArraySize &&
         "inconsistent .fini_array state");

  ErrorOr<BinarySection &> FiniArraySection =
      BC->getSectionForAddress(*BC->FiniArrayAddress);
  assert(FiniArraySection && ".fini_array removed");

  if (std::optional<Relocation> Reloc =
          FiniArraySection->takeDynamicRelocationAt(0)) {
    assert(Reloc->Addend == BC->FiniFunctionAddress &&
           "inconsistent .fini_array dynamic relocation");
    Reloc->Addend = RT->getRuntimeFiniAddress();
    FiniArraySection->addDynamicRelocation(*Reloc);
  }

  // Update the static relocation by adding a pending relocation which will get
  // patched when flushPendingRelocations is called in rewriteFile. Note that
  // flushPendingRelocations will calculate the value to patch as
  // "Symbol + Addend". Since we don't have a symbol, just set the addend to the
  // desired value.
  FiniArraySection->addPendingRelocation(Relocation{
      /*Offset*/ 0, /*Symbol*/ nullptr, /*Type*/ Relocation::getAbs64(),
      /*Addend*/ RT->getRuntimeFiniAddress(), /*Value*/ 0});
}

void RewriteInstance::registerFragments() {
  if (!BC->HasSplitFunctions ||
      opts::HeatmapMode == opts::HeatmapModeKind::HM_Exclusive)
    return;

  // Process fragments with ambiguous parents separately as they are typically a
  // vanishing minority of cases and require expensive symbol table lookups.
  std::vector<std::pair<StringRef, BinaryFunction *>> AmbiguousFragments;
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isFragment())
      continue;
    for (StringRef Name : Function.getNames()) {
      StringRef BaseName = NR.restore(Name);
      const bool IsGlobal = BaseName == Name;
      SmallVector<StringRef> Matches;
      if (!FunctionFragmentTemplate.match(BaseName, &Matches))
        continue;
      StringRef ParentName = Matches[1];
      const BinaryData *BD = BC->getBinaryDataByName(ParentName);
      const uint64_t NumPossibleLocalParents =
          NR.getUniquifiedNameCount(ParentName);
      // The most common case: single local parent fragment.
      if (!BD && NumPossibleLocalParents == 1) {
        BD = BC->getBinaryDataByName(NR.getUniqueName(ParentName, 1));
      } else if (BD && (!NumPossibleLocalParents || IsGlobal)) {
        // Global parent and either no local candidates (second most common), or
        // the fragment is global as well (uncommon).
      } else {
        // Any other case: need to disambiguate using FILE symbols.
        AmbiguousFragments.emplace_back(ParentName, &Function);
        continue;
      }
      if (BD) {
        BinaryFunction *BF = BC->getFunctionForSymbol(BD->getSymbol());
        if (BF) {
          BC->registerFragment(Function, *BF);
          continue;
        }
      }
      BC->errs() << "BOLT-ERROR: parent function not found for " << Function
                 << '\n';
      exit(1);
    }
  }

  if (AmbiguousFragments.empty())
    return;

  if (!BC->hasSymbolsWithFileName()) {
    BC->errs() << "BOLT-ERROR: input file has split functions but does not "
                  "have FILE symbols. If the binary was stripped, preserve "
                  "FILE symbols with --keep-file-symbols strip option\n";
    exit(1);
  }

  // The first global symbol is identified by the symbol table sh_info value.
  // Used as local symbol search stopping point.
  auto *ELF64LEFile = cast<ELF64LEObjectFile>(InputFile);
  const ELFFile<ELF64LE> &Obj = ELF64LEFile->getELFFile();
  auto *SymTab = llvm::find_if(cantFail(Obj.sections()), [](const auto &Sec) {
    return Sec.sh_type == ELF::SHT_SYMTAB;
  });
  assert(SymTab);
  // Symtab sh_info contains the value one greater than the symbol table index
  // of the last local symbol.
  ELFSymbolRef LocalSymEnd = ELF64LEFile->toSymbolRef(SymTab, SymTab->sh_info);

  for (auto &Fragment : AmbiguousFragments) {
    const StringRef &ParentName = Fragment.first;
    BinaryFunction *BF = Fragment.second;
    const uint64_t Address = BF->getAddress();

    // Get fragment's own symbol
    const auto SymIt = llvm::find_if(
        llvm::make_range(FileSymRefs.equal_range(Address)), [&](auto SI) {
          StringRef Name = cantFail(SI.second.getName());
          return Name.contains(ParentName);
        });
    if (SymIt == FileSymRefs.end()) {
      BC->errs()
          << "BOLT-ERROR: symbol lookup failed for function at address 0x"
          << Twine::utohexstr(Address) << '\n';
      exit(1);
    }

    // Find containing FILE symbol
    ELFSymbolRef Symbol = SymIt->second;
    auto FSI = llvm::upper_bound(FileSymbols, Symbol);
    if (FSI == FileSymbols.begin()) {
      BC->errs() << "BOLT-ERROR: owning FILE symbol not found for symbol "
                 << cantFail(Symbol.getName()) << '\n';
      exit(1);
    }

    ELFSymbolRef StopSymbol = LocalSymEnd;
    if (FSI != FileSymbols.end())
      StopSymbol = *FSI;

    uint64_t ParentAddress{0};

    // Check if containing FILE symbol is BOLT emitted synthetic symbol marking
    // local fragments of global parents.
    if (cantFail(FSI[-1].getName()) == getBOLTFileSymbolName())
      goto registerParent;

    // BOLT split fragment symbols are emitted just before the main function
    // symbol.
    for (ELFSymbolRef NextSymbol = Symbol; NextSymbol < StopSymbol;
         NextSymbol.moveNext()) {
      StringRef Name = cantFail(NextSymbol.getName());
      if (Name == ParentName) {
        ParentAddress = cantFail(NextSymbol.getValue());
        goto registerParent;
      }
      if (Name.starts_with(ParentName))
        // With multi-way splitting, there are multiple fragments with different
        // suffixes. Parent follows the last fragment.
        continue;
      break;
    }

    // Iterate over local file symbols and check symbol names to match parent.
    for (ELFSymbolRef Symbol(FSI[-1]); Symbol < StopSymbol; Symbol.moveNext()) {
      if (cantFail(Symbol.getName()) == ParentName) {
        ParentAddress = cantFail(Symbol.getAddress());
        break;
      }
    }

registerParent:
    // No local parent is found, use global parent function.
    if (!ParentAddress)
      if (BinaryData *ParentBD = BC->getBinaryDataByName(ParentName))
        ParentAddress = ParentBD->getAddress();

    if (BinaryFunction *ParentBF =
            BC->getBinaryFunctionAtAddress(ParentAddress)) {
      BC->registerFragment(*BF, *ParentBF);
      continue;
    }
    BC->errs() << "BOLT-ERROR: parent function not found for " << *BF << '\n';
    exit(1);
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
    // Handle IFUNC trampoline with symbol
    setPLTSymbol(BF, BF->getOneName());
    return;
  }

  const Relocation *Rel = BC->getDynamicRelocationAt(TargetAddress);
  if (!Rel)
    return;

  MCSymbol *Symbol = Rel->Symbol;
  if (!Symbol) {
    if (BC->isRISCV() || !Rel->Addend || !Rel->isIRelative())
      return;

    // IFUNC trampoline without symbol
    BinaryFunction *TargetBF = BC->getBinaryFunctionAtAddress(Rel->Addend);
    if (!TargetBF) {
      BC->errs()
          << "BOLT-WARNING: Expected BF to be presented as IFUNC resolver at "
          << Twine::utohexstr(Rel->Addend) << ", skipping\n";
      return;
    }

    Symbol = TargetBF->getSymbol();
  }

  ErrorOr<BinarySection &> Section = BC->getSectionForAddress(EntryAddress);
  assert(Section && "cannot get section for address");
  if (!BF)
    BF = BC->createBinaryFunction(Symbol->getName().str() + "@PLT", *Section,
                                  EntryAddress, 0, EntrySize,
                                  Section->getAlignment());
  else
    BF->addAlternativeName(Symbol->getName().str() + "@PLT");
  setPLTSymbol(BF, Symbol->getName());
}

void RewriteInstance::disassemblePLTInstruction(const BinarySection &Section,
                                                uint64_t InstrOffset,
                                                MCInst &Instruction,
                                                uint64_t &InstrSize) {
  const uint64_t SectionAddress = Section.getAddress();
  const uint64_t SectionSize = Section.getSize();
  StringRef PLTContents = Section.getContents();
  ArrayRef<uint8_t> PLTData(
      reinterpret_cast<const uint8_t *>(PLTContents.data()), SectionSize);

  const uint64_t InstrAddr = SectionAddress + InstrOffset;
  if (!BC->DisAsm->getInstruction(Instruction, InstrSize,
                                  PLTData.slice(InstrOffset), InstrAddr,
                                  nulls())) {
    BC->errs()
        << "BOLT-ERROR: unable to disassemble instruction in PLT section "
        << Section.getName() << formatv(" at offset {0:x}\n", InstrOffset);
    exit(1);
  }
}

void RewriteInstance::disassemblePLTSectionAArch64(BinarySection &Section) {
  const uint64_t SectionAddress = Section.getAddress();
  const uint64_t SectionSize = Section.getSize();

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
      disassemblePLTInstruction(Section, InstrOffset, Instruction, InstrSize);
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
      disassemblePLTInstruction(Section, InstrOffset, Instruction, InstrSize);
      if (!BC->MIB->isNoop(Instruction))
        break;

      InstrOffset += InstrSize;
    }
  }
}

void RewriteInstance::disassemblePLTSectionRISCV(BinarySection &Section) {
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
      BC->errs()
          << "BOLT-ERROR: unable to disassemble instruction in PLT section "
          << Section.getName() << " at offset 0x"
          << Twine::utohexstr(InstrOffset) << '\n';
      exit(1);
    }
  };

  // Skip the first special entry since no relocation points to it.
  uint64_t InstrOffset = 32;

  while (InstrOffset < SectionSize) {
    InstructionListType Instructions;
    MCInst Instruction;
    const uint64_t EntryOffset = InstrOffset;
    const uint64_t EntrySize = 16;
    uint64_t InstrSize;

    while (InstrOffset < EntryOffset + EntrySize) {
      disassembleInstruction(InstrOffset, Instruction, InstrSize);
      Instructions.emplace_back(Instruction);
      InstrOffset += InstrSize;
    }

    const uint64_t EntryAddress = SectionAddress + EntryOffset;
    const uint64_t TargetAddress = BC->MIB->analyzePLTEntry(
        Instruction, Instructions.begin(), Instructions.end(), EntryAddress);

    createPLTBinaryFunction(TargetAddress, EntryAddress, EntrySize);
  }
}

void RewriteInstance::disassemblePLTSectionX86(BinarySection &Section,
                                               uint64_t EntrySize) {
  const uint64_t SectionAddress = Section.getAddress();
  const uint64_t SectionSize = Section.getSize();

  for (uint64_t EntryOffset = 0; EntryOffset + EntrySize <= SectionSize;
       EntryOffset += EntrySize) {
    MCInst Instruction;
    uint64_t InstrSize, InstrOffset = EntryOffset;
    while (InstrOffset < EntryOffset + EntrySize) {
      disassemblePLTInstruction(Section, InstrOffset, Instruction, InstrSize);
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
      BC->errs() << "BOLT-ERROR: error evaluating PLT instruction at offset 0x"
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
    if (BC->isRISCV())
      return disassemblePLTSectionRISCV(Section);
    if (BC->isX86())
      return disassemblePLTSectionX86(Section, EntrySize);
    llvm_unreachable("Unmplemented PLT");
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

void RewriteInstance::adjustFunctionBoundaries(
    DenseMap<uint64_t, MarkerSymType> &MarkerSyms) {
  for (auto BFI = BC->getBinaryFunctions().begin(),
            BFE = BC->getBinaryFunctions().end();
       BFI != BFE; ++BFI) {
    BinaryFunction &Function = BFI->second;
    const BinaryFunction *NextFunction = nullptr;
    if (std::next(BFI) != BFE)
      NextFunction = &std::next(BFI)->second;

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

      // Skip basic block labels. This happens on RISC-V with linker relaxation
      // enabled because every branch needs a relocation and corresponding
      // symbol. We don't want to add such symbols as entry points.
      const auto PrivateLabelPrefix = BC->AsmInfo->getPrivateLabelPrefix();
      if (!PrivateLabelPrefix.empty() &&
          cantFail(Symbol.getName()).starts_with(PrivateLabelPrefix)) {
        ++NextSymRefI;
        continue;
      }

      auto It = MarkerSyms.find(NextSymRefI->first);
      if (It == MarkerSyms.end() || It->second != MarkerSymType::DATA) {
        // This is potentially another entry point into the function.
        uint64_t EntryOffset = NextSymRefI->first - Function.getAddress();
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: adding entry point to function "
                          << Function << " at offset 0x"
                          << Twine::utohexstr(EntryOffset) << '\n');
        Function.addEntryPointAtOffset(EntryOffset);
      }

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
      BC->errs() << "BOLT-ERROR: symbol seen in the middle of the function "
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
        BC->outs() << "BOLT-INFO: setting size of function " << Function
                   << " to " << Function.getMaxSize() << " (was 0)\n";
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

    uint32_t RelType;
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
  }

  // Set IsRelro section attribute based on PT_GNU_RELRO segment.
  markGnuRelroSections();

  if (HasDebugInfo && !opts::UpdateDebugSections && !opts::AggregateOnly) {
    BC->errs() << "BOLT-WARNING: debug info will be stripped from the binary. "
                  "Use -update-debug-sections to keep it.\n";
  }

  HasTextRelocations = (bool)BC->getUniqueSectionByName(
      ".rela" + std::string(BC->getMainCodeSectionName()));
  HasSymbolTable = (bool)BC->getUniqueSectionByName(".symtab");
  EHFrameSection = BC->getUniqueSectionByName(".eh_frame");

  if (ErrorOr<BinarySection &> BATSec =
          BC->getUniqueSectionByName(BoltAddressTranslation::SECTION_NAME)) {
    BC->HasBATSection = true;
    // Do not read BAT when plotting a heatmap
    if (opts::HeatmapMode != opts::HeatmapModeKind::HM_Exclusive) {
      if (std::error_code EC = BAT->parse(BC->outs(), BATSec->getContents())) {
        BC->errs() << "BOLT-ERROR: failed to parse BOLT address translation "
                      "table.\n";
        exit(1);
      }
    }
  }

  if (opts::PrintSections) {
    BC->outs() << "BOLT-INFO: Sections from original binary:\n";
    BC->printSections(BC->outs());
  }

  if (opts::RelocationMode == cl::BOU_TRUE && !HasTextRelocations) {
    BC->errs()
        << "BOLT-ERROR: relocations against code are missing from the input "
           "file. Cannot proceed in relocations mode (-relocs).\n";
    exit(1);
  }

  BC->HasRelocations =
      HasTextRelocations && (opts::RelocationMode != cl::BOU_FALSE);

  if (BC->IsLinuxKernel && BC->HasRelocations) {
    BC->outs() << "BOLT-INFO: disabling relocation mode for Linux kernel\n";
    BC->HasRelocations = false;
  }

  BC->IsStripped = !HasSymbolTable;

  if (BC->IsStripped && !opts::AllowStripped) {
    BC->errs()
        << "BOLT-ERROR: stripped binaries are not supported. If you know "
           "what you're doing, use --allow-stripped to proceed";
    exit(1);
  }

  // Force non-relocation mode for heatmap generation
  if (opts::HeatmapMode == opts::HeatmapModeKind::HM_Exclusive)
    BC->HasRelocations = false;

  if (BC->HasRelocations)
    BC->outs() << "BOLT-INFO: enabling " << (opts::StrictMode ? "strict " : "")
               << "relocation mode\n";

  // Read EH frame for function boundaries info.
  Expected<const DWARFDebugFrame *> EHFrameOrError = BC->DwCtx->getEHFrame();
  if (!EHFrameOrError)
    report_error("expected valid eh_frame section", EHFrameOrError.takeError());
  CFIRdWrt.reset(new CFIReaderWriter(*BC, *EHFrameOrError.get()));

  processSectionMetadata();

  // Read .dynamic/PT_DYNAMIC.
  return readELFDynamic();
}

void RewriteInstance::adjustCommandLineOptions() {
  if (BC->isAArch64() && !BC->HasRelocations)
    BC->errs() << "BOLT-WARNING: non-relocation mode for AArch64 is not fully "
                  "supported\n";

  if (RuntimeLibrary *RtLibrary = BC->getRuntimeLibrary())
    RtLibrary->adjustCommandLineOptions(*BC);

  if (BC->isX86() && BC->MAB->allowAutoPadding()) {
    if (!BC->HasRelocations) {
      BC->errs()
          << "BOLT-ERROR: cannot apply mitigations for Intel JCC erratum in "
             "non-relocation mode\n";
      exit(1);
    }
    BC->outs()
        << "BOLT-WARNING: using mitigation for Intel JCC erratum, layout "
           "may take several minutes\n";
  }

  if (opts::SplitEH && !BC->HasRelocations) {
    BC->errs() << "BOLT-WARNING: disabling -split-eh in non-relocation mode\n";
    opts::SplitEH = false;
  }

  if (BC->isAArch64() && !opts::CompactCodeModel &&
      opts::SplitStrategy == opts::SplitFunctionsStrategy::CDSplit) {
    BC->errs() << "BOLT-ERROR: CDSplit is not supported with LongJmp. Try with "
                  "'--compact-code-model'\n";
    exit(1);
  }

  if (opts::StrictMode && !BC->HasRelocations) {
    BC->errs()
        << "BOLT-WARNING: disabling strict mode (-strict) in non-relocation "
           "mode\n";
    opts::StrictMode = false;
  }

  if (BC->HasRelocations && opts::AggregateOnly &&
      !opts::StrictMode.getNumOccurrences()) {
    BC->outs() << "BOLT-INFO: enabling strict relocation mode for aggregation "
                  "purposes\n";
    opts::StrictMode = true;
  }

  if (!BC->HasRelocations &&
      opts::ReorderFunctions != ReorderFunctions::RT_NONE) {
    BC->errs() << "BOLT-ERROR: function reordering only works when "
               << "relocations are enabled\n";
    exit(1);
  }

  if (!BC->HasRelocations &&
      opts::ICF == IdenticalCodeFolding::ICFLevel::Safe) {
    BC->errs() << "BOLT-ERROR: binary built without relocations. Safe ICF is "
                  "not supported\n";
    exit(1);
  }

  if (opts::Instrument ||
      (opts::ReorderFunctions != ReorderFunctions::RT_NONE &&
       !opts::HotText.getNumOccurrences())) {
    opts::HotText = true;
  } else if (opts::HotText && !BC->HasRelocations) {
    BC->errs() << "BOLT-WARNING: hot text is disabled in non-relocation mode\n";
    opts::HotText = false;
  }

  if (opts::Instrument && opts::UseGnuStack) {
    BC->errs() << "BOLT-ERROR: cannot avoid having writeable and executable "
                  "segment in instrumented binary if program headers will be "
                  "updated in place\n";
    exit(1);
  }

  if (opts::HotText && opts::HotTextMoveSections.getNumOccurrences() == 0) {
    opts::HotTextMoveSections.addValue(".stub");
    opts::HotTextMoveSections.addValue(".mover");
    opts::HotTextMoveSections.addValue(".never_hugify");
  }

  if (opts::UseOldText && !BC->OldTextSectionAddress) {
    BC->errs()
        << "BOLT-WARNING: cannot use old .text as the section was not found"
           "\n";
    opts::UseOldText = false;
  }
  if (opts::UseOldText && !BC->HasRelocations) {
    BC->errs() << "BOLT-WARNING: cannot use old .text in non-relocation mode\n";
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
    BC->errs() << "BOLT-WARNING: cannot combine -lite with -use-old-text. "
                  "Disabling -use-old-text.\n";
    opts::UseOldText = false;
  }

  if (opts::Lite && opts::StrictMode) {
    BC->errs()
        << "BOLT-ERROR: -strict and -lite cannot be used at the same time\n";
    exit(1);
  }

  if (opts::Lite)
    BC->outs() << "BOLT-INFO: enabling lite mode\n";

  if (BC->IsLinuxKernel) {
    if (!opts::KeepNops.getNumOccurrences())
      opts::KeepNops = true;

    // Linux kernel may resume execution after a trap or x86 HLT instruction.
    if (!opts::TerminalHLT.getNumOccurrences())
      opts::TerminalHLT = false;
    if (!opts::TerminalTrap.getNumOccurrences())
      opts::TerminalTrap = false;
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
  return getRelocationAddend(cast<ELF64LEObjectFile>(Obj), Rel);
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
  return getRelocationSymbol(cast<ELF64LEObjectFile>(Obj), Rel);
}
} // anonymous namespace

bool RewriteInstance::analyzeRelocation(
    const RelocationRef &Rel, uint32_t &RType, std::string &SymbolName,
    bool &IsSectionRelocation, uint64_t &SymbolAddress, int64_t &Addend,
    uint64_t &ExtractedValue) const {
  if (!Relocation::isSupported(RType))
    return false;

  auto IsWeakReference = [](const SymbolRef &Symbol) {
    Expected<uint32_t> SymFlagsOrErr = Symbol.getFlags();
    if (!SymFlagsOrErr)
      return false;
    return (*SymFlagsOrErr & SymbolRef::SF_Undefined) &&
           (*SymFlagsOrErr & SymbolRef::SF_Weak);
  };

  const bool IsAArch64 = BC->isAArch64();

  const size_t RelSize = Relocation::getSizeForType(RType);

  ErrorOr<uint64_t> Value =
      BC->getUnsignedValueAtAddress(Rel.getOffset(), RelSize);
  assert(Value && "failed to extract relocated value");

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
    if (!SymbolAddress && !IsWeakReference(Symbol) &&
        (IsAArch64 || BC->isRISCV())) {
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

  // GOT relocation can cause the underlying instruction to be modified by the
  // linker, resulting in the extracted value being different from the actual
  // symbol. It's also possible to have a GOT entry for a symbol defined in the
  // binary. In the latter case, the instruction can be using the GOT version
  // causing the extracted value mismatch. Similar cases can happen for TLS.
  // Pass the relocation information as is to the disassembler and let it decide
  // how to use it for the operand symbolization.
  if (Relocation::isGOT(RType) || Relocation::isTLS(RType)) {
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

    if (IsAArch64 || BC->isRISCV())
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
  // Read .relr.dyn section containing compressed R_*_RELATIVE relocations.
  if (DynamicRelrSize > 0) {
    ErrorOr<BinarySection &> DynamicRelrSectionOrErr =
        BC->getSectionForAddress(*DynamicRelrAddress);
    if (!DynamicRelrSectionOrErr)
      report_error("unable to find section corresponding to DT_RELR",
                   DynamicRelrSectionOrErr.getError());
    if (DynamicRelrSectionOrErr->getSize() != DynamicRelrSize)
      report_error("section size mismatch for DT_RELRSZ",
                   errc::executable_format_error);
    readDynamicRelrRelocations(*DynamicRelrSectionOrErr);
  }

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
  // The static executable might have .rela.dyn secion and not have PT_DYNAMIC
  if (!DynamicRelocationsSize && BC->IsStaticExecutable) {
    ErrorOr<BinarySection &> DynamicRelSectionOrErr =
        BC->getUniqueSectionByName(getRelaDynSectionName());
    if (DynamicRelSectionOrErr) {
      DynamicRelocationsAddress = DynamicRelSectionOrErr->getAddress();
      DynamicRelocationsSize = DynamicRelSectionOrErr->getSize();
      const SectionRef &SectionRef = DynamicRelSectionOrErr->getSectionRef();
      DynamicRelativeRelocationsCount = std::distance(
          SectionRef.relocation_begin(), SectionRef.relocation_end());
    }
  }

  if (DynamicRelocationsSize > 0) {
    ErrorOr<BinarySection &> DynamicRelSectionOrErr =
        BC->getSectionForAddress(*DynamicRelocationsAddress);
    if (!DynamicRelSectionOrErr)
      report_error("unable to find section corresponding to DT_RELA",
                   DynamicRelSectionOrErr.getError());
    auto DynamicRelSectionSize = DynamicRelSectionOrErr->getSize();
    // On RISC-V DT_RELASZ seems to include both .rela.dyn and .rela.plt
    if (DynamicRelocationsSize == DynamicRelSectionSize + PLTRelocationsSize)
      DynamicRelocationsSize = DynamicRelSectionSize;
    if (DynamicRelSectionSize != DynamicRelocationsSize)
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
    section_iterator SecIter = cantFail(Section.getRelocatedSection());
    if (SecIter == InputFile->section_end())
      continue;
    if (BinarySection(*BC, Section).isAllocatable())
      continue;

    readRelocations(Section);
  }

  if (NumFailedRelocations)
    BC->errs() << "BOLT-WARNING: Failed to analyze " << NumFailedRelocations
               << " relocations\n";
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
    const uint32_t RType = Relocation::getType(Rel);
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

    const uint64_t ReferencedAddress = SymbolAddress + Addend;
    BinaryFunction *Func =
        BC->getBinaryFunctionContainingAddress(ReferencedAddress);

    if (Relocation::isRelative(RType) && SymbolAddress == 0) {
      if (Func) {
        if (!Func->isInConstantIsland(ReferencedAddress)) {
          if (const uint64_t ReferenceOffset =
                  ReferencedAddress - Func->getAddress()) {
            Func->addEntryPointAtOffset(ReferenceOffset);
          }
        } else {
          BC->errs() << "BOLT-ERROR: referenced address at 0x"
                     << Twine::utohexstr(ReferencedAddress)
                     << " is in constant island of function " << *Func << "\n";
          exit(1);
        }
      }
    } else if (Relocation::isRelative(RType) && SymbolAddress != 0) {
      BC->errs() << "BOLT-ERROR: symbol address non zero for RELATIVE "
                    "relocation type\n";
      exit(1);
    }

    BC->addDynamicRelocation(Rel.getOffset(), Symbol, RType, Addend);
  }
}

void RewriteInstance::readDynamicRelrRelocations(BinarySection &Section) {
  assert(Section.isAllocatable() && "allocatable expected");

  LLVM_DEBUG({
    StringRef SectionName = Section.getName();
    dbgs() << "BOLT-DEBUG: reading relocations in section " << SectionName
           << ":\n";
  });

  const uint32_t RType = Relocation::getRelative();
  const uint8_t PSize = BC->AsmInfo->getCodePointerSize();
  const uint64_t MaxDelta = ((CHAR_BIT * DynamicRelrEntrySize) - 1) * PSize;

  auto ExtractAddendValue = [&](uint64_t Address) -> uint64_t {
    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
    assert(Section && "cannot get section for data address from RELR");
    DataExtractor DE = DataExtractor(Section->getContents(),
                                     BC->AsmInfo->isLittleEndian(), PSize);
    uint64_t Offset = Address - Section->getAddress();
    return DE.getUnsigned(&Offset, PSize);
  };

  auto AddRelocation = [&](uint64_t Address) {
    uint64_t Addend = ExtractAddendValue(Address);
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: R_*_RELATIVE relocation at 0x"
                      << Twine::utohexstr(Address) << " to 0x"
                      << Twine::utohexstr(Addend) << '\n';);
    BC->addDynamicRelocation(Address, nullptr, RType, Addend);
  };

  DataExtractor DE = DataExtractor(Section.getContents(),
                                   BC->AsmInfo->isLittleEndian(), PSize);
  uint64_t Offset = 0, Address = 0;
  uint64_t RelrCount = DynamicRelrSize / DynamicRelrEntrySize;
  while (RelrCount--) {
    assert(DE.isValidOffset(Offset));
    uint64_t Entry = DE.getUnsigned(&Offset, DynamicRelrEntrySize);
    if ((Entry & 1) == 0) {
      AddRelocation(Entry);
      Address = Entry + PSize;
    } else {
      const uint64_t StartAddress = Address;
      while (Entry >>= 1) {
        if (Entry & 1)
          AddRelocation(Address);

        Address += PSize;
      }

      Address = StartAddress + MaxDelta;
    }
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
  const bool IsX86 = BC->isX86();
  const bool IsFromCode = RelocatedSection.isText();
  const bool IsWritable = BinarySection(*BC, RelocatedSection).isWritable();

  SmallString<16> TypeName;
  Rel.getTypeName(TypeName);
  uint32_t RType = Relocation::getType(Rel);
  if (Relocation::skipRelocationType(RType))
    return;

  // Adjust the relocation type as the linker might have skewed it.
  if (IsX86 && (RType & ELF::R_X86_64_converted_reloc_bit)) {
    if (opts::Verbosity >= 1)
      dbgs() << "BOLT-WARNING: ignoring R_X86_64_converted_reloc_bit\n";
    RType &= ~ELF::R_X86_64_converted_reloc_bit;
  }

  if (Relocation::isTLS(RType)) {
    // No special handling required for TLS relocations on X86.
    if (IsX86)
      return;

    // The non-got related TLS relocations on AArch64 and RISC-V also could be
    // skipped.
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
  if (!analyzeRelocation(Rel, RType, SymbolName, IsSectionRelocation,
                         SymbolAddress, Addend, ExtractedValue)) {
    LLVM_DEBUG({
      dbgs() << "BOLT-WARNING: failed to analyze relocation @ offset = "
             << formatv("{0:x}; type name = {1}\n", Rel.getOffset(), TypeName);
    });
    ++NumFailedRelocations;
    return;
  }

  if (!IsFromCode && !IsWritable && (IsX86 || IsAArch64) &&
      Relocation::isPCRelative(RType)) {
    BinaryData *BD = BC->getBinaryDataContainingAddress(Rel.getOffset());
    if (BD && (BD->nameStartsWith("_ZTV") ||   // vtable
               BD->nameStartsWith("_ZTCN"))) { // construction vtable
      BinaryFunction *BF = BC->getBinaryFunctionContainingAddress(
          SymbolAddress, /*CheckPastEnd*/ false, /*UseMaxSize*/ true);
      if (BF) {
        if (BF->getAddress() != SymbolAddress) {
          BC->errs()
              << "BOLT-ERROR: the virtual function table entry at offset 0x"
              << Twine::utohexstr(Rel.getOffset())
              << " points to the middle of a function @ 0x"
              << Twine::utohexstr(BF->getAddress()) << "\n";
          exit(1);
        }
        BC->addRelocation(Rel.getOffset(), BF->getSymbol(), RType, Addend,
                          ExtractedValue);
        return;
      }
    }
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
        BC->outs() << formatv(
            "BOLT-INFO: {0} has relocations in padding area\n", *ContainingBF);
      ContainingBF->setSize(ContainingBF->getMaxSize());
      ContainingBF->setSimple(false);
      return;
    }
  }

  MCSymbol *ReferencedSymbol = nullptr;
  if (!IsSectionRelocation) {
    if (BinaryData *BD = BC->getBinaryDataByName(SymbolName)) {
      ReferencedSymbol = BD->getSymbol();
    } else if (BC->isGOTSymbol(SymbolName)) {
      if (BinaryData *BD = BC->getGOTSymbol())
        ReferencedSymbol = BD->getSymbol();
    } else if (BinaryData *BD = BC->getBinaryDataAtAddress(SymbolAddress)) {
      ReferencedSymbol = BD->getSymbol();
    }
  }

  ErrorOr<BinarySection &> ReferencedSection{std::errc::bad_address};
  symbol_iterator SymbolIter = Rel.getSymbol();
  if (SymbolIter != InputFile->symbol_end()) {
    SymbolRef Symbol = *SymbolIter;
    section_iterator Section =
        cantFail(Symbol.getSection(), "cannot get symbol section");
    if (Section != InputFile->section_end()) {
      Expected<StringRef> SectionName = Section->getName();
      if (SectionName && !SectionName->empty())
        ReferencedSection = BC->getUniqueSectionByName(*SectionName);
    } else if (BC->isRISCV() && ReferencedSymbol && ContainingBF &&
               (cantFail(Symbol.getFlags()) & SymbolRef::SF_Absolute)) {
      // This might be a relocation for an ABS symbols like __global_pointer$ on
      // RISC-V
      ContainingBF->addRelocation(Rel.getOffset(), ReferencedSymbol,
                                  Relocation::getType(Rel), 0,
                                  cantFail(Symbol.getValue()));
      return;
    }
  }

  if (!ReferencedSection)
    ReferencedSection = BC->getSectionForAddress(SymbolAddress);

  const bool IsToCode = ReferencedSection && ReferencedSection->isText();

  // Special handling of PC-relative relocations.
  if (IsX86 && Relocation::isPCRelative(RType)) {
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
  if ((BC->isAArch64() || BC->isRISCV()) && Relocation::isGOT(RType))
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
        BC->errs()
            << "BOLT-WARNING: non-standard function reference (e.g. bitmask)"
            << formatv(" detected against function {0} from ", *BF);
        if (IsFromCode)
          BC->errs() << formatv("function {0}\n", *ContainingBF);
        else
          BC->errs() << formatv("data section at {0:x}\n", Rel.getOffset());
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
        BC->errs()
            << "BOLT-WARNING: detected possible compiler de-virtualization "
               "bug: -1 addend used with non-pc-relative relocation against "
            << formatv("function {0} in function {1}\n", *RogueBF,
                       *ContainingBF);
        return;
      }
    }
  }

  if (ForceRelocation && !ReferencedBF) {
    // Create the relocation symbol if it's not defined in the binary.
    if (SymbolAddress == 0)
      ReferencedSymbol = BC->registerNameAtAddress(SymbolName, 0, 0, 0);

    LLVM_DEBUG(
        dbgs() << "BOLT-DEBUG: forcing relocation against symbol "
               << (ReferencedSymbol ? ReferencedSymbol->getName() : "<none>")
               << " with addend " << Addend << '\n');
  } else if (ReferencedBF) {
    ReferencedSymbol = ReferencedBF->getSymbol();
    uint64_t RefFunctionOffset = 0;

    // Adjust the point of reference to a code location inside a function.
    if (ReferencedBF->containsAddress(Address, /*UseMaxSize = */ true)) {
      RefFunctionOffset = Address - ReferencedBF->getAddress();
      if (Relocation::isInstructionReference(RType)) {
        // Instruction labels are created while disassembling so we just leave
        // the symbol empty for now. Since the extracted value is typically
        // unrelated to the referenced symbol (e.g., %pcrel_lo in RISC-V
        // references an instruction but the patched value references the low
        // bits of a data address), we set the extracted value to the symbol
        // address in order to be able to correctly reconstruct the reference
        // later.
        ReferencedSymbol = nullptr;
        ExtractedValue = Address;
      } else if (RefFunctionOffset) {
        if (ContainingBF && ContainingBF != ReferencedBF &&
            !ReferencedBF->isInConstantIsland(Address)) {
          ReferencedSymbol =
              ReferencedBF->addEntryPointAtOffset(RefFunctionOffset);
        } else {
          ReferencedSymbol =
              ReferencedBF->getOrCreateLocalLabel(Address,
                                                  /*CreatePastEnd =*/true);

          // If ContainingBF != nullptr, it equals ReferencedBF (see
          // if-condition above) so we're handling a relocation from a function
          // to itself. RISC-V uses such relocations for branches, for example.
          // These should not be registered as externally references offsets.
          if (!ContainingBF)
            ReferencedBF->registerReferencedOffset(RefFunctionOffset);
        }
        if (opts::Verbosity > 1 &&
            BinarySection(*BC, RelocatedSection).isWritable())
          BC->errs()
              << "BOLT-WARNING: writable reference into the middle of the "
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
      // but AArch64 and RISCV has inferred and incomplete object locations
      // coming from GOT/TLS or any other non-trivial relocation (that requires
      // creation of sections and whose symbol address is not really what should
      // be encoded in the instruction). So we essentially disabled this check
      // for AArch64 and live with bogus names for objects.
      assert((IsAArch64 || BC->isRISCV() || IsSectionRelocation ||
              BD->nameStartsWith(SymbolName) ||
              BD->nameStartsWith("PG" + SymbolName) ||
              (BD->nameStartsWith("ANONYMOUS") &&
               (BD->getSectionName().starts_with(".plt") ||
                BD->getSectionName().ends_with(".plt")))) &&
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
          (IsAArch64 || BC->isRISCV() || IsSectionRelocation ||
           BC->getSectionNameForAddress(SymbolAddress)->starts_with(".plt")) &&
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
                  .starts_with(BC->AsmInfo->getPrivateGlobalPrefix()))
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
      (opts::ForceToDataRelocations && checkMaxDataRelocations()) ||
      // RISC-V has ADD/SUB data-to-data relocations
      BC->isRISCV())
    ForceRelocation = true;

  if (IsFromCode)
    ContainingBF->addRelocation(Rel.getOffset(), ReferencedSymbol, RType,
                                Addend, ExtractedValue);
  else if (IsToCode || ForceRelocation)
    BC->addRelocation(Rel.getOffset(), ReferencedSymbol, RType, Addend,
                      ExtractedValue);
  else
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring relocation from data to data\n");
}

static BinaryFunction *getInitFunctionIfStaticBinary(BinaryContext &BC) {
  // Workaround for https://github.com/llvm/llvm-project/issues/100096
  // ("[BOLT] GOT array pointer incorrectly rewritten"). In aarch64
  // static glibc binaries, the .init section's _init function pointer can
  // alias with a data pointer for the end of an array. GOT rewriting
  // currently can't detect this and updates the data pointer to the
  // moved _init, causing a runtime crash. Skipping _init on the other
  // hand should be harmless.
  if (!BC.IsStaticExecutable)
    return nullptr;
  const BinaryData *BD = BC.getBinaryDataByName("_init");
  if (!BD || BD->getSectionName() != ".init")
    return nullptr;
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: skip _init in for GOT workaround.\n");
  return BC.getBinaryFunctionAtAddress(BD->getAddress());
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
    BC->errs()
        << "BOLT-ERROR: cannot select functions to process and skip at the "
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
    BC->outs() << "BOLT-INFO: limiting processing to functions with at least "
               << LiteThresholdExecCount << " invocations\n";
  }
  LiteThresholdExecCount = std::max(
      LiteThresholdExecCount, static_cast<uint64_t>(opts::LiteThresholdCount));

  StringSet<> ReorderFunctionsUserSet;
  StringSet<> ReorderFunctionsLTOCommonSet;
  if (opts::ReorderFunctions == ReorderFunctions::RT_USER) {
    std::vector<std::string> FunctionNames;
    BC->logBOLTErrorsAndQuitOnFatal(
        ReorderFunctions::readFunctionOrderFile(FunctionNames));
    for (const std::string &Function : FunctionNames) {
      ReorderFunctionsUserSet.insert(Function);
      if (std::optional<StringRef> LTOCommonName = getLTOCommonName(Function))
        ReorderFunctionsLTOCommonSet.insert(*LTOCommonName);
    }
  }

  uint64_t NumFunctionsToProcess = 0;
  auto mustSkip = [&](const BinaryFunction &Function) {
    if (opts::MaxFunctions.getNumOccurrences() &&
        NumFunctionsToProcess >= opts::MaxFunctions)
      return true;
    for (std::string &Name : opts::SkipFunctionNames)
      if (Function.hasNameRegex(Name))
        return true;

    return false;
  };

  auto shouldProcess = [&](const BinaryFunction &Function) {
    if (mustSkip(Function))
      return false;

    // If the list is not empty, only process functions from the list.
    if (!opts::ForceFunctionNames.empty() || !ForceFunctionsNR.empty()) {
      // Regex check (-funcs and -funcs-file options).
      for (std::string &Name : opts::ForceFunctionNames)
        if (Function.hasNameRegex(Name))
          return true;

      // Non-regex check (-funcs-no-regex and -funcs-file-no-regex).
      for (const StringRef Name : Function.getNames())
        if (ForceFunctionsNR.count(Name.str()))
          return true;

      return false;
    }

    if (opts::Lite) {
      // Forcibly include functions specified in the -function-order file.
      if (opts::ReorderFunctions == ReorderFunctions::RT_USER) {
        for (const StringRef Name : Function.getNames())
          if (ReorderFunctionsUserSet.contains(Name))
            return true;
        for (const StringRef Name : Function.getNames())
          if (std::optional<StringRef> LTOCommonName = getLTOCommonName(Name))
            if (ReorderFunctionsLTOCommonSet.contains(*LTOCommonName))
              return true;
      }

      if (ProfileReader && !ProfileReader->mayHaveProfileData(Function))
        return false;

      if (Function.getKnownExecutionCount() < LiteThresholdExecCount)
        return false;
    }

    return true;
  };

  if (BinaryFunction *Init = getInitFunctionIfStaticBinary(*BC))
    Init->setIgnored();

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    // Pseudo functions are explicitly marked by us not to be processed.
    if (Function.isPseudo()) {
      Function.IsIgnored = true;
      Function.HasExternalRefRelocations = true;
      continue;
    }

    // Decide what to do with fragments after parent functions are processed.
    if (Function.isFragment())
      continue;

    if (!shouldProcess(Function)) {
      if (opts::Verbosity >= 1) {
        BC->outs() << "BOLT-INFO: skipping processing " << Function
                   << " per user request\n";
      }
      Function.setIgnored();
    } else {
      ++NumFunctionsToProcess;
      if (opts::MaxFunctions.getNumOccurrences() &&
          NumFunctionsToProcess == opts::MaxFunctions)
        BC->outs() << "BOLT-INFO: processing ending on " << Function << '\n';
    }
  }

  if (!BC->HasSplitFunctions)
    return;

  // Fragment overrides:
  // - If the fragment must be skipped, then the parent must be skipped as well.
  // Otherwise, fragment should follow the parent function:
  // - if the parent is skipped, skip fragment,
  // - if the parent is processed, process the fragment(s) as well.
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isFragment())
      continue;
    if (mustSkip(Function)) {
      for (BinaryFunction *Parent : Function.ParentFragments) {
        if (opts::Verbosity >= 1) {
          BC->outs() << "BOLT-INFO: skipping processing " << *Parent
                     << " together with fragment function\n";
        }
        Parent->setIgnored();
        --NumFunctionsToProcess;
      }
      Function.setIgnored();
      continue;
    }

    bool IgnoredParent =
        llvm::any_of(Function.ParentFragments, [&](BinaryFunction *Parent) {
          return Parent->isIgnored();
        });
    if (IgnoredParent) {
      if (opts::Verbosity >= 1) {
        BC->outs() << "BOLT-INFO: skipping processing " << Function
                   << " together with parent function\n";
      }
      Function.setIgnored();
    } else {
      ++NumFunctionsToProcess;
      if (opts::Verbosity >= 1) {
        BC->outs() << "BOLT-INFO: processing " << Function
                   << " as a sibling of non-ignored function\n";
      }
      if (opts::MaxFunctions && NumFunctionsToProcess == opts::MaxFunctions)
        BC->outs() << "BOLT-INFO: processing ending on " << Function << '\n';
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

  BC->outs() << "BOLT-INFO: pre-processing profile using "
             << ProfileReader->getReaderName() << '\n';

  if (BAT->enabledFor(InputFile)) {
    BC->outs() << "BOLT-INFO: profile collection done on a binary already "
                  "processed by BOLT\n";
    ProfileReader->setBAT(&*BAT);
  }

  if (Error E = ProfileReader->preprocessProfile(*BC))
    report_error("cannot pre-process profile", std::move(E));

  if (!BC->hasSymbolsWithFileName() && ProfileReader->hasLocalsWithFileName() &&
      !opts::AllowStripped) {
    BC->errs()
        << "BOLT-ERROR: input binary does not have local file symbols "
           "but profile data includes function names with embedded file "
           "names. It appears that the input binary was stripped while a "
           "profiled binary was not. If you know what you are doing and "
           "wish to proceed, use -allow-stripped option.\n";
    exit(1);
  }
}

void RewriteInstance::initializeMetadataManager() {
  if (BC->IsLinuxKernel)
    MetadataManager.registerRewriter(createLinuxKernelRewriter(*BC));

  MetadataManager.registerRewriter(createBuildIDRewriter(*BC));

  MetadataManager.registerRewriter(createPseudoProbeRewriter(*BC));

  MetadataManager.registerRewriter(createSDTRewriter(*BC));

  MetadataManager.registerRewriter(createGNUPropertyRewriter(*BC));
}

void RewriteInstance::processSectionMetadata() {
  NamedRegionTimer T("processmetadata-section", "process section metadata",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  initializeMetadataManager();

  MetadataManager.runSectionInitializers();
}

void RewriteInstance::processMetadataPreCFG() {
  NamedRegionTimer T("processmetadata-precfg", "process metadata pre-CFG",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  MetadataManager.runInitializersPreCFG();

  processProfileDataPreCFG();
}

void RewriteInstance::processMetadataPostCFG() {
  NamedRegionTimer T("processmetadata-postcfg", "process metadata post-CFG",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  MetadataManager.runInitializersPostCFG();
}

void RewriteInstance::processProfileDataPreCFG() {
  if (!ProfileReader)
    return;

  NamedRegionTimer T("processprofile-precfg", "process profile data pre-CFG",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);

  if (Error E = ProfileReader->readProfilePreCFG(*BC))
    report_error("cannot read profile pre-CFG", std::move(E));
}

void RewriteInstance::processProfileData() {
  if (!ProfileReader)
    return;

  NamedRegionTimer T("processprofile", "process profile data", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);

  if (Error E = ProfileReader->readProfile(*BC))
    report_error("cannot read profile", std::move(E));

  if (opts::PrintProfile || opts::PrintAll) {
    for (auto &BFI : BC->getBinaryFunctions()) {
      BinaryFunction &Function = BFI.second;
      if (Function.empty())
        continue;

      Function.print(BC->outs(), "after attaching profile");
    }
  }

  if (!opts::SaveProfile.empty() && !BAT->enabledFor(InputFile)) {
    YAMLProfileWriter PW(opts::SaveProfile);
    PW.writeProfile(*this);
  }
  if (opts::AggregateOnly &&
      opts::ProfileFormat == opts::ProfileFormatKind::PF_YAML &&
      !BAT->enabledFor(InputFile)) {
    YAMLProfileWriter PW(opts::OutputFilename);
    PW.writeProfile(*this);
  }

  // Release memory used by profile reader.
  ProfileReader.reset();

  if (opts::AggregateOnly) {
    PrintProgramStats PPS(&*BAT);
    BC->logBOLTErrorsAndQuitOnFatal(PPS.runOnFunctions(*BC));
    TimerGroup::printAll(outs());
    exit(0);
  }
}

void RewriteInstance::disassembleFunctions() {
  NamedRegionTimer T("disassembleFunctions", "disassemble functions",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    ErrorOr<ArrayRef<uint8_t>> FunctionData = Function.getData();
    if (!FunctionData) {
      BC->errs() << "BOLT-ERROR: corresponding section is non-executable or "
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

    bool DisasmFailed{false};
    handleAllErrors(Function.disassemble(), [&](const BOLTError &E) {
      DisasmFailed = true;
      if (E.isFatal()) {
        E.log(BC->errs());
        exit(1);
      }
      if (opts::processAllFunctions()) {
        BC->errs() << BC->generateBugReportMessage(
            "function cannot be properly disassembled. "
            "Unable to continue in relocation mode.",
            Function);
        exit(1);
      }
      if (opts::Verbosity >= 1)
        BC->outs() << "BOLT-INFO: could not disassemble function " << Function
                   << ". Will ignore.\n";
      // Forcefully ignore the function.
      Function.scanExternalRefs();
      Function.setIgnored();
    });

    if (DisasmFailed)
      continue;

    if (opts::PrintAll || opts::PrintDisasm)
      Function.print(BC->outs(), "after disassembly");
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
        BC->errs() << BC->generateBugReportMessage("unable to fill CFI.",
                                                   Function);
        exit(1);
      } else {
        BC->errs() << "BOLT-WARNING: unable to fill CFI for function "
                   << Function << ". Skipping.\n";
        Function.setSimple(false);
        continue;
      }
    }

    // Parse LSDA.
    if (Function.getLSDAAddress() != 0 &&
        !BC->getFragmentsToSkip().count(&Function)) {
      ErrorOr<BinarySection &> LSDASection =
          BC->getSectionForAddress(Function.getLSDAAddress());
      check_error(LSDASection.getError(), "failed to get LSDA section");
      ArrayRef<uint8_t> LSDAData = ArrayRef<uint8_t>(
          LSDASection->getData(), LSDASection->getContents().size());
      BC->logBOLTErrorsAndQuitOnFatal(
          Function.parseLSDA(LSDAData, LSDASection->getAddress()));
    }
  }
}

void RewriteInstance::buildFunctionsCFG() {
  NamedRegionTimer T("buildCFG", "buildCFG", "buildfuncs",
                     "Build Binary Functions", opts::TimeBuild);

  // Create annotation indices to allow lock-free execution
  BC->MIB->getOrCreateAnnotationIndex("JTIndexReg");
  BC->MIB->getOrCreateAnnotationIndex("NOP");

  ParallelUtilities::WorkFuncWithAllocTy WorkFun =
      [&](BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocId) {
        bool HadErrors{false};
        handleAllErrors(BF.buildCFG(AllocId), [&](const BOLTError &E) {
          if (!E.getMessage().empty())
            E.log(BC->errs());
          if (E.isFatal())
            exit(1);
          HadErrors = true;
        });

        if (HadErrors)
          return;

        if (opts::PrintAll) {
          auto L = BC->scopeLock();
          BF.print(BC->outs(), "while building cfg");
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

    // Set function as non-simple if it has dynamic relocations
    // in constant island, we don't want this function to be optimized
    // e.g. function splitting is unsupported.
    if (Function.hasDynamicRelocationAtIsland())
      Function.setSimple(false);

    if (Function.empty())
      continue;

    Function.postProcessCFG();

    if (opts::PrintAll || opts::PrintCFG)
      Function.print(BC->outs(), "after building cfg");

    if (opts::shouldDumpDot(Function))
      Function.dumpGraphForPass("00_build-cfg");

    if (opts::PrintLoopInfo) {
      Function.calculateLoopInfo();
      Function.printLoopInfo(BC->outs());
    }

    BC->TotalScore += Function.getFunctionScore();
    BC->SumExecutionCount += Function.getKnownExecutionCount();
  }

  if (opts::PrintGlobals) {
    BC->outs() << "BOLT-INFO: Global symbols:\n";
    BC->printGlobalSymbols(BC->outs());
  }
}

void RewriteInstance::runOptimizationPasses() {
  NamedRegionTimer T("runOptimizationPasses", "run optimization passes",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  BC->logBOLTErrorsAndQuitOnFatal(BinaryFunctionPassManager::runAllPasses(*BC));
}

void RewriteInstance::runBinaryAnalyses() {
  NamedRegionTimer T("runBinaryAnalyses", "run binary analysis passes",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  BinaryFunctionPassManager Manager(*BC);
  // FIXME: add a pass that warns about which functions do not have CFG,
  // and therefore, analysis is most likely to be less accurate.
  using GSK = opts::GadgetScannerKind;
  using PAuthScanner = PAuthGadgetScanner::Analysis;

  // If no command line option was given, act as if "all" was specified.
  bool RunAll = !opts::GadgetScannersToRun.getBits() ||
                opts::GadgetScannersToRun.isSet(GSK::GS_ALL);

  if (RunAll || opts::GadgetScannersToRun.isSet(GSK::GS_PAUTH)) {
    Manager.registerPass(
        std::make_unique<PAuthScanner>(/*OnlyPacRetChecks=*/false));
  } else if (RunAll || opts::GadgetScannersToRun.isSet(GSK::GS_PACRET)) {
    Manager.registerPass(
        std::make_unique<PAuthScanner>(/*OnlyPacRetChecks=*/true));
  }

  BC->logBOLTErrorsAndQuitOnFatal(Manager.runPasses());
}

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

  SmallString<0> ObjectBuffer;
  raw_svector_ostream OS(ObjectBuffer);

  // Implicitly MCObjectStreamer takes ownership of MCAsmBackend (MAB)
  // and MCCodeEmitter (MCE). ~MCObjectStreamer() will delete these
  // two instances.
  std::unique_ptr<MCStreamer> Streamer = BC->createStreamer(OS);

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
    BC->errs() << "BOLT-ERROR: Emission failed.\n";
    exit(1);
  }

  if (opts::KeepTmp) {
    SmallString<128> OutObjectPath;
    sys::fs::getPotentiallyUniqueTempFileName("output", "o", OutObjectPath);
    std::error_code EC;
    raw_fd_ostream FOS(OutObjectPath, EC);
    check_error(EC, "cannot create output object file");
    FOS << ObjectBuffer;
    BC->outs()
        << "BOLT-INFO: intermediary output object file saved for debugging "
           "purposes: "
        << OutObjectPath << "\n";
  }

  ErrorOr<BinarySection &> TextSection =
      BC->getUniqueSectionByName(BC->getMainCodeSectionName());
  if (BC->HasRelocations && TextSection)
    BC->renameSection(*TextSection,
                      getOrgSecPrefix() + BC->getMainCodeSectionName());

  //////////////////////////////////////////////////////////////////////////////
  // Assign addresses to new sections.
  //////////////////////////////////////////////////////////////////////////////

  // Get output object as ObjectFile.
  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(ObjectBuffer, "in-memory object file", false);

  auto EFMM = std::make_unique<ExecutableFileMemoryManager>(*BC);
  EFMM->setNewSecPrefix(getNewSecPrefix());
  EFMM->setOrgSecPrefix(getOrgSecPrefix());

  Linker = std::make_unique<JITLinkLinker>(*BC, std::move(EFMM));
  Linker->loadObject(ObjectMemBuffer->getMemBufferRef(),
                     [this](auto MapSection) { mapFileSections(MapSection); });

  // Update output addresses based on the new section map and
  // layout. Only do this for the object created by ourselves.
  updateOutputValues(*Linker);

  if (opts::UpdateDebugSections) {
    DebugInfoRewriter->updateLineTableOffsets(
        static_cast<MCObjectStreamer &>(*Streamer).getAssembler());
  }

  if (RuntimeLibrary *RtLibrary = BC->getRuntimeLibrary()) {
    StartLinkingRuntimeLib = true;
    RtLibrary->link(*BC, ToolPath, *Linker, [this](auto MapSection) {
      // Map newly registered sections.
      this->mapAllocatableSections(MapSection);
    });
  }

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
    BC->outs() << "BOLT-INFO: cache metrics after emitting functions:\n";
    CacheMetrics::printAll(BC->outs(), BC->getSortedFunctions());
  }
}

void RewriteInstance::finalizeMetadataPreEmit() {
  NamedRegionTimer T("finalizemetadata-preemit", "finalize metadata pre-emit",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  MetadataManager.runFinalizersPreEmit();
}

void RewriteInstance::updateMetadata() {
  NamedRegionTimer T("updatemetadata-postemit", "update metadata post-emit",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  MetadataManager.runFinalizersAfterEmit();

  if (opts::UpdateDebugSections) {
    NamedRegionTimer T("updateDebugInfo", "update debug info", TimerGroupName,
                       TimerGroupDesc, opts::TimeRewrite);
    DebugInfoRewriter->updateDebugInfo();
  }

  if (opts::WriteBoltInfoSection)
    addBoltInfoSection();
}

void RewriteInstance::mapFileSections(BOLTLinker::SectionMapper MapSection) {
  BC->deregisterUnusedSections();

  // If no new .eh_frame was written, remove relocated original .eh_frame.
  BinarySection *RelocatedEHFrameSection =
      getSection(".relocated" + getEHFrameSectionName());
  if (RelocatedEHFrameSection && RelocatedEHFrameSection->hasValidSectionID()) {
    BinarySection *NewEHFrameSection =
        getSection(getNewSecPrefix() + getEHFrameSectionName());
    if (!NewEHFrameSection || !NewEHFrameSection->isFinalized()) {
      // JITLink will still have to process relocations for the section, hence
      // we need to assign it the address that wouldn't result in relocation
      // processing failure.
      MapSection(*RelocatedEHFrameSection, NextAvailableAddress);
      BC->deregisterSection(*RelocatedEHFrameSection);
    }
  }

  mapCodeSections(MapSection);

  // Map the rest of the sections.
  mapAllocatableSections(MapSection);

  if (!BC->BOLTReserved.empty()) {
    const uint64_t AllocatedSize =
        NextAvailableAddress - BC->BOLTReserved.start();
    if (BC->BOLTReserved.size() < AllocatedSize) {
      BC->errs() << "BOLT-ERROR: reserved space (" << BC->BOLTReserved.size()
                 << " byte" << (BC->BOLTReserved.size() == 1 ? "" : "s")
                 << ") is smaller than required for new allocations ("
                 << AllocatedSize << " bytes)\n";
      exit(1);
    }
  }
}

std::vector<BinarySection *> RewriteInstance::getCodeSections() {
  std::vector<BinarySection *> CodeSections;
  for (BinarySection &Section : BC->textSections())
    if (Section.hasValidSectionID())
      CodeSections.emplace_back(&Section);

  auto compareSections = [&](const BinarySection *A, const BinarySection *B) {
    // If both A and B have names starting with ".text.cold", then
    // - if opts::HotFunctionsAtEnd is true, we want order
    //   ".text.cold.T", ".text.cold.T-1", ... ".text.cold.1", ".text.cold"
    // - if opts::HotFunctionsAtEnd is false, we want order
    //   ".text.cold", ".text.cold.1", ... ".text.cold.T-1", ".text.cold.T"
    if (A->getName().starts_with(BC->getColdCodeSectionName()) &&
        B->getName().starts_with(BC->getColdCodeSectionName())) {
      if (A->getName().size() != B->getName().size())
        return (opts::HotFunctionsAtEnd)
                   ? (A->getName().size() > B->getName().size())
                   : (A->getName().size() < B->getName().size());
      return (opts::HotFunctionsAtEnd) ? (A->getName() > B->getName())
                                       : (A->getName() < B->getName());
    }

    // Place movers before anything else.
    if (A->getName() == BC->getHotTextMoverSectionName())
      return true;
    if (B->getName() == BC->getHotTextMoverSectionName())
      return false;

    // Depending on opts::HotFunctionsAtEnd, place main and warm sections in
    // order.
    if (opts::HotFunctionsAtEnd) {
      if (B->getName() == BC->getMainCodeSectionName())
        return true;
      if (A->getName() == BC->getMainCodeSectionName())
        return false;
      return (B->getName() == BC->getWarmCodeSectionName());
    } else {
      if (A->getName() == BC->getMainCodeSectionName())
        return true;
      if (B->getName() == BC->getMainCodeSectionName())
        return false;
      return (A->getName() == BC->getWarmCodeSectionName());
    }
  };

  // Determine the order of sections.
  llvm::stable_sort(CodeSections, compareSections);

  return CodeSections;
}

void RewriteInstance::mapCodeSections(BOLTLinker::SectionMapper MapSection) {
  if (!BC->HasRelocations) {
    mapCodeSectionsInPlace(MapSection);
    return;
  }

  // Map sections for functions with pre-assigned addresses.
  for (BinaryFunction *InjectedFunction : BC->getInjectedBinaryFunctions()) {
    const uint64_t OutputAddress = InjectedFunction->getOutputAddress();
    if (!OutputAddress)
      continue;

    ErrorOr<BinarySection &> FunctionSection =
        InjectedFunction->getCodeSection();
    assert(FunctionSection && "function should have section");
    FunctionSection->setOutputAddress(OutputAddress);
    MapSection(*FunctionSection, OutputAddress);
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
             for (const BinarySection *Section : CodeSections) dbgs()
             << Section->getName() << '\n';);

  uint64_t PaddingSize = 0; // size of padding required at the end

  // Allocate sections starting at a given Address.
  auto allocateAt = [&](uint64_t Address) {
    const char *LastNonColdSectionName = BC->HasWarmSection
                                             ? BC->getWarmCodeSectionName()
                                             : BC->getMainCodeSectionName();
    for (BinarySection *Section : CodeSections) {
      Address = alignTo(Address, Section->getAlignment());
      Section->setOutputAddress(Address);
      Address += Section->getOutputSize();

      // Hugify: Additional huge page from right side due to
      // weird ASLR mapping addresses (4KB aligned)
      if (opts::Hugify && !BC->HasFixedLoadAddress &&
          Section->getName() == LastNonColdSectionName)
        Address = alignTo(Address, Section->getAlignment());
    }

    // Make sure we allocate enough space for huge pages.
    ErrorOr<BinarySection &> TextSection =
        BC->getUniqueSectionByName(LastNonColdSectionName);
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

  // Try to allocate sections before the \p Address and return an address for
  // the allocation of the first section, or 0 if [0, Address) range is not
  // big enough to fit all sections.
  auto allocateBefore = [&](uint64_t Address) -> uint64_t {
    for (BinarySection *Section : llvm::reverse(CodeSections)) {
      if (Section->getOutputSize() > Address)
        return 0;
      Address -= Section->getOutputSize();
      Address = alignDown(Address, Section->getAlignment());
      Section->setOutputAddress(Address);
    }
    return Address;
  };

  // Check if we can fit code in the original .text
  bool AllocationDone = false;
  if (opts::UseOldText) {
    uint64_t StartAddress;
    uint64_t EndAddress;
    if (opts::HotFunctionsAtEnd) {
      EndAddress = BC->OldTextSectionAddress + BC->OldTextSectionSize;
      StartAddress = allocateBefore(EndAddress);
    } else {
      StartAddress = BC->OldTextSectionAddress;
      EndAddress = allocateAt(BC->OldTextSectionAddress);
    }

    const uint64_t CodeSize = EndAddress - StartAddress;
    if (CodeSize <= BC->OldTextSectionSize) {
      BC->outs() << "BOLT-INFO: using original .text for new code with 0x"
                 << Twine::utohexstr(opts::AlignText) << " alignment";
      if (StartAddress != BC->OldTextSectionAddress)
        BC->outs() << " at 0x" << Twine::utohexstr(StartAddress);
      BC->outs() << '\n';
      AllocationDone = true;
    } else {
      BC->errs() << "BOLT-WARNING: original .text too small to fit the new code"
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
    LLVM_DEBUG(dbgs() << "BOLT: mapping " << Section->getName() << " at 0x"
                      << Twine::utohexstr(Section->getAllocAddress())
                      << " to 0x"
                      << Twine::utohexstr(Section->getOutputAddress()) << '\n');
    MapSection(*Section, Section->getOutputAddress());
    Section->setOutputFileOffset(
        getFileOffsetForAddress(Section->getOutputAddress()));
  }

  // Check if we need to insert a padding section for hot text.
  if (PaddingSize && !opts::UseOldText)
    BC->outs() << "BOLT-INFO: padding code to 0x"
               << Twine::utohexstr(NextAvailableAddress)
               << " to accommodate hot text\n";
}

void RewriteInstance::mapCodeSectionsInPlace(
    BOLTLinker::SectionMapper MapSection) {
  // Processing in non-relocation mode.
  uint64_t NewTextSectionStartAddress = NextAvailableAddress;

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isEmitted())
      continue;

    ErrorOr<BinarySection &> FuncSection = Function.getCodeSection();
    assert(FuncSection && "cannot find section for function");
    FuncSection->setOutputAddress(Function.getAddress());
    LLVM_DEBUG(dbgs() << "BOLT: mapping 0x"
                      << Twine::utohexstr(FuncSection->getAllocAddress())
                      << " to 0x" << Twine::utohexstr(Function.getAddress())
                      << '\n');
    MapSection(*FuncSection, Function.getAddress());
    Function.setImageAddress(FuncSection->getAllocAddress());
    Function.setImageSize(FuncSection->getOutputSize());
    assert(Function.getImageSize() <= Function.getMaxSize() &&
           "Unexpected large function");

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
    FF.setAddress(NextAvailableAddress);
    FF.setImageAddress(ColdSection->getAllocAddress());
    FF.setImageSize(ColdSection->getOutputSize());
    FF.setFileOffset(getFileOffsetForAddress(NextAvailableAddress));
    ColdSection->setOutputAddress(FF.getAddress());

    LLVM_DEBUG(
        dbgs() << formatv(
            "BOLT: mapping cold fragment {0:x+} to {1:x+} with size {2:x+}\n",
            FF.getImageAddress(), FF.getAddress(), FF.getImageSize()));
    MapSection(*ColdSection, FF.getAddress());

    NextAvailableAddress += FF.getImageSize();
  }

  // Add the new text section aggregating all existing code sections.
  // This is pseudo-section that serves a purpose of creating a corresponding
  // entry in section header table.
  const uint64_t NewTextSectionSize =
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

void RewriteInstance::mapAllocatableSections(
    BOLTLinker::SectionMapper MapSection) {

  if (opts::UseOldText || opts::StrictMode) {
    auto tryRewriteSection = [&](BinarySection &OldSection,
                                 BinarySection &NewSection) {
      if (OldSection.getSize() < NewSection.getOutputSize())
        return;

      BC->outs() << "BOLT-INFO: rewriting " << OldSection.getName()
                 << " in-place\n";

      NewSection.setOutputAddress(OldSection.getAddress());
      NewSection.setOutputFileOffset(OldSection.getInputFileOffset());
      MapSection(NewSection, OldSection.getAddress());

      // Pad contents with zeros.
      NewSection.addPadding(OldSection.getSize() - NewSection.getOutputSize());

      // Prevent the original section name from appearing in the section header
      // table.
      OldSection.setAnonymous(true);
    };

    if (EHFrameSection) {
      BinarySection *NewEHFrameSection =
          getSection(getNewSecPrefix() + getEHFrameSectionName());
      assert(NewEHFrameSection && "New contents expected for .eh_frame");
      tryRewriteSection(*EHFrameSection, *NewEHFrameSection);
    }
    BinarySection *EHSection = getSection(".gcc_except_table");
    BinarySection *NewEHSection =
        getSection(getNewSecPrefix() + ".gcc_except_table");
    if (EHSection) {
      assert(NewEHSection && "New contents expected for .gcc_except_table");
      tryRewriteSection(*EHSection, *NewEHSection);
    }
  }

  // Allocate read-only sections first, then writable sections.
  enum : uint8_t { ST_READONLY, ST_READWRITE };
  for (uint8_t SType = ST_READONLY; SType <= ST_READWRITE; ++SType) {
    const uint64_t LastNextAvailableAddress = NextAvailableAddress;
    if (SType == ST_READWRITE) {
      // Align R+W segment to regular page size
      NextAvailableAddress = alignTo(NextAvailableAddress, BC->RegularPageSize);
      NewWritableSegmentAddress = NextAvailableAddress;
    }

    for (BinarySection &Section : BC->allocatableSections()) {
      if (Section.isLinkOnly())
        continue;

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
        MapSection(Section, Section.getAddress());
      } else {
        uint64_t Alignment = Section.getAlignment();
        if (opts::Instrument && StartLinkingRuntimeLib) {
          Alignment = BC->RegularPageSize;
          StartLinkingRuntimeLib = false;
        }
        NextAvailableAddress = alignTo(NextAvailableAddress, Alignment);

        LLVM_DEBUG({
          dbgs() << "BOLT-DEBUG: mapping section " << Section.getName()
                 << " (0x" << Twine::utohexstr(Section.getAllocAddress())
                 << ") to 0x" << Twine::utohexstr(NextAvailableAddress) << ":0x"
                 << Twine::utohexstr(NextAvailableAddress +
                                     Section.getOutputSize())
                 << '\n';
        });

        MapSection(Section, NextAvailableAddress);
        Section.setOutputAddress(NextAvailableAddress);
        Section.setOutputFileOffset(
            getFileOffsetForAddress(NextAvailableAddress));

        NextAvailableAddress += Section.getOutputSize();
      }
    }

    if (SType == ST_READONLY) {
      if (NewTextSegmentAddress)
        NewTextSegmentSize = NextAvailableAddress - NewTextSegmentAddress;
    } else if (SType == ST_READWRITE) {
      NewWritableSegmentSize = NextAvailableAddress - NewWritableSegmentAddress;
      // Restore NextAvailableAddress if no new writable sections
      if (!NewWritableSegmentSize)
        NextAvailableAddress = LastNextAvailableAddress;
    }
  }
}

void RewriteInstance::updateOutputValues(const BOLTLinker &Linker) {
  if (std::optional<AddressMap> Map = AddressMap::parse(*BC))
    BC->setIOAddressMap(std::move(*Map));

  for (BinaryFunction *Function : BC->getAllBinaryFunctions())
    Function->updateOutputValues(Linker);
}

void RewriteInstance::updateSegmentInfo() {
  // NOTE Currently .eh_frame_hdr appends to the last segment, recalculate
  // last segments size based on the NextAvailableAddress variable.
  if (!NewWritableSegmentSize) {
    if (NewTextSegmentAddress)
      NewTextSegmentSize = NextAvailableAddress - NewTextSegmentAddress;
  } else {
    NewWritableSegmentSize = NextAvailableAddress - NewWritableSegmentAddress;
  }

  if (NewTextSegmentSize) {
    SegmentInfo TextSegment = {NewTextSegmentAddress,
                               NewTextSegmentSize,
                               NewTextSegmentOffset,
                               NewTextSegmentSize,
                               BC->PageAlign,
                               true,
                               false};
    if (!opts::Instrument) {
      BC->NewSegments.push_back(TextSegment);
    } else {
      ErrorOr<BinarySection &> Sec =
          BC->getUniqueSectionByName(".bolt.instr.counters");
      assert(Sec && "expected one and only one `.bolt.instr.counters` section");
      const uint64_t Addr = Sec->getOutputAddress();
      const uint64_t Offset = Sec->getOutputFileOffset();
      const uint64_t Size = Sec->getOutputSize();
      assert(Addr > TextSegment.Address &&
             Addr + Size < TextSegment.Address + TextSegment.Size &&
             "`.bolt.instr.counters` section is expected to be included in the "
             "new text segment");

      // Set correct size for the previous header since we are breaking the
      // new text segment into three segments.
      uint64_t Delta = Addr - TextSegment.Address;
      TextSegment.Size = Delta;
      TextSegment.FileSize = Delta;
      BC->NewSegments.push_back(TextSegment);

      // Create RW segment that includes the `.bolt.instr.counters` section.
      SegmentInfo RWSegment = {Addr,  Size, Offset, Size, BC->RegularPageSize,
                               false, true};
      BC->NewSegments.push_back(RWSegment);

      // Create RX segment that includes all RX sections from runtime library.
      const uint64_t AddrRX = alignTo(Addr + Size, BC->RegularPageSize);
      const uint64_t OffsetRX = alignTo(Offset + Size, BC->RegularPageSize);
      const uint64_t SizeRX =
          NewTextSegmentSize - (AddrRX - TextSegment.Address);
      SegmentInfo RXSegment = {
          AddrRX, SizeRX, OffsetRX, SizeRX, BC->RegularPageSize, true, false};
      BC->NewSegments.push_back(RXSegment);
    }
  }

  if (NewWritableSegmentSize) {
    SegmentInfo DataSegmentInfo = {
        NewWritableSegmentAddress,
        NewWritableSegmentSize,
        getFileOffsetForAddress(NewWritableSegmentAddress),
        NewWritableSegmentSize,
        BC->RegularPageSize,
        false,
        true};
    BC->NewSegments.push_back(DataSegmentInfo);
  }
}

void RewriteInstance::patchELFPHDRTable() {
  auto ELF64LEFile = cast<ELF64LEObjectFile>(InputFile);
  const ELFFile<ELF64LE> &Obj = ELF64LEFile->getELFFile();
  raw_fd_ostream &OS = Out->os();

  Phnum = Obj.getHeader().e_phnum;

  if (BC->NewSegments.empty()) {
    BC->outs() << "BOLT-INFO: not adding new segments\n";
    return;
  }

  if (opts::UseGnuStack) {
    assert(!PHDRTableAddress && "unexpected address for program header table");
    if (BC->NewSegments.size() > 1) {
      BC->errs() << "BOLT-ERROR: unable to add writable segment\n";
      exit(1);
    }
  } else {
    Phnum += BC->NewSegments.size();
  }

  if (!PHDRTableOffset)
    PHDRTableOffset = Obj.getHeader().e_phoff;

  const uint64_t SavedPos = OS.tell();
  OS.seek(PHDRTableOffset);

  auto createPhdr = [](const SegmentInfo &SI) {
    ELF64LEPhdrTy Phdr;
    Phdr.p_type = ELF::PT_LOAD;
    Phdr.p_offset = SI.FileOffset;
    Phdr.p_vaddr = SI.Address;
    Phdr.p_paddr = SI.Address;
    Phdr.p_filesz = SI.FileSize;
    Phdr.p_memsz = SI.Size;
    Phdr.p_flags = ELF::PF_R;
    if (SI.IsExecutable)
      Phdr.p_flags |= ELF::PF_X;
    if (SI.IsWritable)
      Phdr.p_flags |= ELF::PF_W;
    Phdr.p_align = SI.Alignment;

    return Phdr;
  };

  auto writeNewSegmentPhdrs = [&]() {
    for (const SegmentInfo &SI : BC->NewSegments) {
      ELF64LEPhdrTy Phdr = createPhdr(SI);
      OS.write(reinterpret_cast<const char *>(&Phdr), sizeof(Phdr));
    }
  };

  bool ModdedGnuStack = false;
  bool AddedSegment = false;

  // Copy existing program headers with modifications.
  for (const ELF64LE::Phdr &Phdr : cantFail(Obj.program_headers())) {
    ELF64LE::Phdr NewPhdr = Phdr;
    switch (Phdr.p_type) {
    case ELF::PT_PHDR:
      if (PHDRTableAddress) {
        NewPhdr.p_offset = PHDRTableOffset;
        NewPhdr.p_vaddr = PHDRTableAddress;
        NewPhdr.p_paddr = PHDRTableAddress;
        NewPhdr.p_filesz = sizeof(NewPhdr) * Phnum;
        NewPhdr.p_memsz = sizeof(NewPhdr) * Phnum;
      }
      break;
    case ELF::PT_GNU_EH_FRAME: {
      ErrorOr<BinarySection &> EHFrameHdrSec = BC->getUniqueSectionByName(
          getNewSecPrefix() + getEHFrameHdrSectionName());
      if (EHFrameHdrSec && EHFrameHdrSec->isAllocatable() &&
          EHFrameHdrSec->isFinalized()) {
        NewPhdr.p_offset = EHFrameHdrSec->getOutputFileOffset();
        NewPhdr.p_vaddr = EHFrameHdrSec->getOutputAddress();
        NewPhdr.p_paddr = EHFrameHdrSec->getOutputAddress();
        NewPhdr.p_filesz = EHFrameHdrSec->getOutputSize();
        NewPhdr.p_memsz = EHFrameHdrSec->getOutputSize();
      }
      break;
    }
    case ELF::PT_GNU_STACK:
      if (opts::UseGnuStack) {
        // Overwrite the header with the new segment header.
        assert(BC->NewSegments.size() == 1 &&
               "Expected exactly one new segment");
        NewPhdr = createPhdr(BC->NewSegments.front());
        ModdedGnuStack = true;
      }
      break;
    case ELF::PT_DYNAMIC:
      if (!opts::UseGnuStack) {
        // Insert new headers before DYNAMIC.
        writeNewSegmentPhdrs();
        AddedSegment = true;
      }
      break;
    }
    OS.write(reinterpret_cast<const char *>(&NewPhdr), sizeof(NewPhdr));
  }

  if (!opts::UseGnuStack && !AddedSegment) {
    // Append new headers to the end of the table.
    writeNewSegmentPhdrs();
  }

  if (opts::UseGnuStack && !ModdedGnuStack) {
    BC->errs()
        << "BOLT-ERROR: could not find PT_GNU_STACK program header to modify\n";
    exit(1);
  }

  OS.seek(SavedPos);
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
  auto ELF64LEFile = cast<ELF64LEObjectFile>(InputFile);
  const ELFFile<ELF64LE> &Obj = ELF64LEFile->getELFFile();
  raw_fd_ostream &OS = Out->os();

  uint64_t NextAvailableOffset = std::max(
      getFileOffsetForAddress(NextAvailableAddress), FirstNonAllocatableOffset);
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
      Size += BSec->write(OS);
    }

    BSec->setOutputFileOffset(NextAvailableOffset);
    BSec->flushPendingRelocations(OS, [this](const MCSymbol *S) {
      return getNewValueForSymbol(S->getName());
    });

    // Section contents are no longer needed, but we need to update the size so
    // that it will be reflected in the section header table.
    BSec->updateContents(nullptr, Size);

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

    NextAvailableOffset += Section.write(OS);
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

  const std::string BoltInfo =
      BinarySection::encodeELFNote("BOLT", DescStr, BinarySection::NT_BOLT_BAT);
  BC->registerOrUpdateNoteSection(BoltAddressTranslation::SECTION_NAME,
                                  copyByteArray(BoltInfo), BoltInfo.size(),
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true, ELF::SHT_NOTE);
  BC->outs() << "BOLT-INFO: BAT section size (bytes): " << BoltInfo.size()
             << '\n';
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
  auto addSection = [&](const ELFShdrTy &Section, BinarySection &BinSec) {
    ELFShdrTy NewSection = Section;
    NewSection.sh_name = SHStrTab.getOffset(BinSec.getOutputName());
    OutputSections.emplace_back(&BinSec, std::move(NewSection));
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

    // Exclude anonymous sections.
    if (BinSec->isAnonymous())
      continue;

    addSection(Section, *BinSec);
  }

  for (BinarySection &Section : BC->allocatableSections()) {
    if (!Section.isFinalized())
      continue;

    if (Section.hasSectionRef() || Section.isAnonymous()) {
      if (opts::Verbosity)
        BC->outs() << "BOLT-INFO: not writing section header for section "
                   << Section.getOutputName() << '\n';
      continue;
    }

    if (opts::Verbosity >= 1)
      BC->outs() << "BOLT-INFO: writing section header for "
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
    addSection(NewSection, Section);
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

    // Ignore NOBITS sections as they don't take any space in the file.
    if (Section.sh_type == ELF::SHT_NOBITS)
      continue;

    // Note that address continuity is not guaranteed as sections could be
    // placed in different loadable segments.
    if (PrevSection &&
        PrevSection->sh_offset + PrevSection->sh_size > Section.sh_offset) {
      if (opts::Verbosity > 1)
        BC->outs() << "BOLT-INFO: adjusting size for section "
                   << PrevBinSec->getOutputName() << '\n';
      PrevSection->sh_size = Section.sh_offset - PrevSection->sh_offset;
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

    addSection(NewSection, *BinSec);

    LastFileOffset = BinSec->getOutputFileOffset();
  }

  // Create entries for new non-allocatable sections.
  for (BinarySection &Section : BC->nonAllocatableSections()) {
    if (Section.getOutputFileOffset() <= LastFileOffset)
      continue;

    if (opts::Verbosity >= 1)
      BC->outs() << "BOLT-INFO: writing section header for "
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

    addSection(NewSection, Section);
  }

  // Assign indices to sections.
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
  llvm::copy(llvm::make_second_range(OutputSections), SectionsOnly.begin());

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

  // Mapping from old section indices to new ones
  std::vector<uint32_t> NewSectionIndex;
  std::vector<ELFShdrTy> OutputSections =
      getOutputSections(File, NewSectionIndex);
  LLVM_DEBUG(
    dbgs() << "BOLT-DEBUG: old to new section index mapping:\n";
    for (uint64_t I = 0; I < NewSectionIndex.size(); ++I)
      dbgs() << "  " << I << " -> " << NewSectionIndex[I] << '\n';
  );

  // Align starting address for section header table. There's no architecutal
  // need to align this, it is just for pleasant human readability.
  uint64_t SHTOffset = OS.tell();
  SHTOffset = appendPadding(OS, SHTOffset, 16);

  // Write all section header entries while patching section references.
  for (ELFShdrTy &Section : OutputSections) {
    Section.sh_link = NewSectionIndex[Section.sh_link];
    if (Section.sh_type == ELF::SHT_REL || Section.sh_type == ELF::SHT_RELA)
      Section.sh_info = NewSectionIndex[Section.sh_info];
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
  if (PHDRTableOffset) {
    NewEhdr.e_phoff = PHDRTableOffset;
    NewEhdr.e_phnum = Phnum;
  }
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

  bool EmittedColdFileSymbol = false;

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

  // Get the extra symbol name of a split fragment; used in addExtraSymbols.
  auto getSplitSymbolName = [&](const FunctionFragment &FF,
                                const ELFSymTy &FunctionSymbol) {
    SmallString<256> SymbolName;
    if (BC->HasWarmSection)
      SymbolName =
          formatv("{0}.{1}", cantFail(FunctionSymbol.getName(StringSection)),
                  FF.getFragmentNum() == FragmentNum::warm() ? "warm" : "cold");
    else
      SymbolName = formatv("{0}.cold.{1}",
                           cantFail(FunctionSymbol.getName(StringSection)),
                           FF.getFragmentNum().get() - 1);
    return SymbolName;
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
      // Prepend synthetic FILE symbol to prevent local cold fragments from
      // colliding with existing symbols with the same name.
      if (!EmittedColdFileSymbol &&
          FunctionSymbol.getBinding() == ELF::STB_GLOBAL) {
        ELFSymTy FileSymbol;
        FileSymbol.st_shndx = ELF::SHN_ABS;
        FileSymbol.st_name = AddToStrTab(getBOLTFileSymbolName());
        FileSymbol.st_value = 0;
        FileSymbol.st_size = 0;
        FileSymbol.st_other = 0;
        FileSymbol.setBindingAndType(ELF::STB_LOCAL, ELF::STT_FILE);
        Symbols.emplace_back(FileSymbol);
        EmittedColdFileSymbol = true;
      }
      for (const FunctionFragment &FF :
           Function.getLayout().getSplitFragments()) {
        if (FF.getAddress()) {
          ELFSymTy NewColdSym = FunctionSymbol;
          const SmallString<256> SymbolName =
              getSplitSymbolName(FF, FunctionSymbol);
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

    // Handle special symbols based on their name.
    Expected<StringRef> SymbolName = Symbol.getName(StringSection);
    assert(SymbolName && "cannot get symbol name");

    auto updateSymbolValue = [&](const StringRef Name,
                                 std::optional<uint64_t> Value = std::nullopt) {
      NewSymbol.st_value = Value ? *Value : getNewValueForSymbol(Name);
      NewSymbol.st_shndx = ELF::SHN_ABS;
      BC->outs() << "BOLT-INFO: setting " << Name << " to 0x"
                 << Twine::utohexstr(NewSymbol.st_value) << '\n';
    };

    if (*SymbolName == "__hot_start" || *SymbolName == "__hot_end") {
      if (opts::HotText) {
        updateSymbolValue(*SymbolName);
        ++NumHotTextSymsUpdated;
      }
      goto registerSymbol;
    }

    if (*SymbolName == "__hot_data_start" || *SymbolName == "__hot_data_end") {
      if (opts::HotData) {
        updateSymbolValue(*SymbolName);
        ++NumHotDataSymsUpdated;
      }
      goto registerSymbol;
    }

    if (*SymbolName == "_end") {
      if (NextAvailableAddress > Symbol.st_value)
        updateSymbolValue(*SymbolName, NextAvailableAddress);
      goto registerSymbol;
    }

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
              "Cannot locate fragment containing secondary entrypoint");
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

  registerSymbol:
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
    if (Function->isAnonymous())
      continue;
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

  auto AddSymbol = [&](const StringRef &Name, uint64_t Address) {
    if (!Address)
      return;

    ELFSymTy Symbol;
    Symbol.st_value = Address;
    Symbol.st_shndx = ELF::SHN_ABS;
    Symbol.st_name = AddToStrTab(Name);
    Symbol.st_size = 0;
    Symbol.st_other = 0;
    Symbol.setBindingAndType(ELF::STB_WEAK, ELF::STT_NOTYPE);

    BC->outs() << "BOLT-INFO: setting " << Name << " to 0x"
               << Twine::utohexstr(Symbol.st_value) << '\n';

    Symbols.emplace_back(Symbol);
  };

  // Add runtime library start and fini address symbols
  if (RuntimeLibrary *RtLibrary = BC->getRuntimeLibrary()) {
    AddSymbol("__bolt_runtime_start", RtLibrary->getRuntimeStartAddress());
    AddSymbol("__bolt_runtime_fini", RtLibrary->getRuntimeFiniAddress());
  }

  assert((!NumHotTextSymsUpdated || NumHotTextSymsUpdated == 2) &&
         "either none or both __hot_start/__hot_end symbols were expected");
  assert((!NumHotDataSymsUpdated || NumHotDataSymsUpdated == 2) &&
         "either none or both __hot_data_start/__hot_data_end symbols were "
         "expected");

  auto AddEmittedSymbol = [&](const StringRef &Name) {
    AddSymbol(Name, getNewValueForSymbol(Name));
  };

  if (opts::HotText && !NumHotTextSymsUpdated) {
    AddEmittedSymbol("__hot_start");
    AddEmittedSymbol("__hot_end");
  }

  if (opts::HotData && !NumHotDataSymsUpdated) {
    AddEmittedSymbol("__hot_data_start");
    AddEmittedSymbol("__hot_data_end");
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
    BC->errs() << "BOLT-WARNING: no symbol table found\n";
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
void RewriteInstance::patchELFAllocatableRelrSection(
    ELFObjectFile<ELFT> *File) {
  if (!DynamicRelrAddress)
    return;

  raw_fd_ostream &OS = Out->os();
  const uint8_t PSize = BC->AsmInfo->getCodePointerSize();
  const uint64_t MaxDelta = ((CHAR_BIT * DynamicRelrEntrySize) - 1) * PSize;

  auto FixAddend = [&](const BinarySection &Section, const Relocation &Rel,
                       uint64_t FileOffset) {
    // Fix relocation symbol value in place if no static relocation found
    // on the same address. We won't check the BF relocations here since it
    // is rare case and no optimization is required.
    if (Section.getRelocationAt(Rel.Offset))
      return;

    // No fixup needed if symbol address was not changed
    const uint64_t Addend = getNewFunctionOrDataAddress(Rel.Addend);
    if (!Addend)
      return;

    OS.pwrite(reinterpret_cast<const char *>(&Addend), PSize, FileOffset);
  };

  // Fill new relative relocation offsets set
  std::set<uint64_t> RelOffsets;
  for (const BinarySection &Section : BC->allocatableSections()) {
    const uint64_t SectionInputAddress = Section.getAddress();
    uint64_t SectionAddress = Section.getOutputAddress();
    if (!SectionAddress)
      SectionAddress = SectionInputAddress;

    for (const Relocation &Rel : Section.dynamicRelocations()) {
      if (!Rel.isRelative())
        continue;

      uint64_t RelOffset =
          getNewFunctionOrDataAddress(SectionInputAddress + Rel.Offset);

      RelOffset = RelOffset == 0 ? SectionAddress + Rel.Offset : RelOffset;
      assert((RelOffset & 1) == 0 && "Wrong relocation offset");
      RelOffsets.emplace(RelOffset);
      FixAddend(Section, Rel, RelOffset);
    }
  }

  ErrorOr<BinarySection &> Section =
      BC->getSectionForAddress(*DynamicRelrAddress);
  assert(Section && "cannot get .relr.dyn section");
  assert(Section->isRelr() && "Expected section to be SHT_RELR type");
  uint64_t RelrDynOffset = Section->getInputFileOffset();
  const uint64_t RelrDynEndOffset = RelrDynOffset + Section->getSize();

  auto WriteRelr = [&](uint64_t Value) {
    if (RelrDynOffset + DynamicRelrEntrySize > RelrDynEndOffset) {
      BC->errs() << "BOLT-ERROR: Offset overflow for relr.dyn section\n";
      exit(1);
    }

    OS.pwrite(reinterpret_cast<const char *>(&Value), DynamicRelrEntrySize,
              RelrDynOffset);
    RelrDynOffset += DynamicRelrEntrySize;
  };

  for (auto RelIt = RelOffsets.begin(); RelIt != RelOffsets.end();) {
    WriteRelr(*RelIt);
    uint64_t Base = *RelIt++ + PSize;
    while (1) {
      uint64_t Bitmap = 0;
      for (; RelIt != RelOffsets.end(); ++RelIt) {
        const uint64_t Delta = *RelIt - Base;
        if (Delta >= MaxDelta || Delta % PSize)
          break;

        Bitmap |= (1ULL << (Delta / PSize));
      }

      if (!Bitmap)
        break;

      WriteRelr((Bitmap << 1) | 1);
      Base += MaxDelta;
    }
  }

  // Fill the rest of the section with empty bitmap value
  while (RelrDynOffset != RelrDynEndOffset)
    WriteRelr(1);
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
    assert(Section && "cannot get relocation section");
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
      const uint64_t SectionInputAddress = Section.getAddress();
      uint64_t SectionAddress = Section.getOutputAddress();
      if (!SectionAddress)
        SectionAddress = SectionInputAddress;

      for (const Relocation &Rel : Section.dynamicRelocations()) {
        const bool IsRelative = Rel.isRelative();
        if (PatchRelative != IsRelative)
          continue;

        if (IsRelative)
          ++DynamicRelativeRelocationsCount;

        Elf_Rela NewRelA;
        MCSymbol *Symbol = Rel.Symbol;
        uint32_t SymbolIdx = 0;
        uint64_t Addend = Rel.Addend;
        uint64_t RelOffset =
            getNewFunctionOrDataAddress(SectionInputAddress + Rel.Offset);

        RelOffset = RelOffset == 0 ? SectionAddress + Rel.Offset : RelOffset;
        if (Rel.Symbol) {
          SymbolIdx = getOutputDynamicSymbolIndex(Symbol);
        } else {
          // Usually this case is used for R_*_(I)RELATIVE relocations
          const uint64_t Address = getNewFunctionOrDataAddress(Addend);
          if (Address)
            Addend = Address;
        }

        NewRelA.setSymbolAndType(SymbolIdx, Rel.Type, EF.isMips64EL());
        NewRelA.r_offset = RelOffset;
        NewRelA.r_addend = Addend;

        const bool IsJmpRel = IsJmpRelocation.contains(Rel.Type);
        uint64_t &Offset = IsJmpRel ? RelPltOffset : RelDynOffset;
        const uint64_t &EndOffset =
            IsJmpRel ? RelPltEndOffset : RelDynEndOffset;
        if (!Offset || !EndOffset) {
          BC->errs() << "BOLT-ERROR: Invalid offsets for dynamic relocation\n";
          exit(1);
        }

        if (Offset + sizeof(NewRelA) > EndOffset) {
          BC->errs() << "BOLT-ERROR: Offset overflow for dynamic relocation\n";
          exit(1);
        }

        writeRela(&NewRelA, Offset);
      }
    }
  };

  // Place R_*_RELATIVE relocations in RELA section if RELR is not presented.
  // The dynamic linker expects all R_*_RELATIVE relocations in RELA
  // to be emitted first.
  if (!DynamicRelrAddress)
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
      BC->errs() << "BOLT-INFO: no .got section found\n";
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
    BC->errs()
        << "BOLT-ERROR: output binary requires immediate relocation "
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
    BC->outs() << "BOLT-INFO: static input executable detected\n";
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
      BC->FiniAddress = Dyn.getPtr();
      break;
    case ELF::DT_FINI_ARRAY:
      BC->FiniArrayAddress = Dyn.getPtr();
      break;
    case ELF::DT_FINI_ARRAYSZ:
      BC->FiniArraySize = Dyn.getPtr();
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
    case ELF::DT_RELR:
      DynamicRelrAddress = Dyn.getPtr();
      break;
    case ELF::DT_RELRSZ:
      DynamicRelrSize = Dyn.getVal();
      break;
    case ELF::DT_RELRENT:
      DynamicRelrEntrySize = Dyn.getVal();
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

  if (!DynamicRelrAddress || !DynamicRelrSize) {
    DynamicRelrAddress.reset();
    DynamicRelrSize = 0;
  } else if (!DynamicRelrEntrySize) {
    BC->errs() << "BOLT-ERROR: expected DT_RELRENT to be presented "
               << "in DYNAMIC section\n";
    exit(1);
  } else if (DynamicRelrSize % DynamicRelrEntrySize) {
    BC->errs() << "BOLT-ERROR: expected RELR table size to be divisible "
               << "by RELR entry size\n";
    exit(1);
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

  if (const BinaryFunction *BF =
          BC->getBinaryFunctionContainingAddress(OldAddress)) {
    if (BF->isEmitted()) {
      // If OldAddress is the another entry point of
      // the function, then BOLT could get the new address.
      if (BF->isMultiEntry()) {
        for (const BinaryBasicBlock &BB : *BF)
          if (BB.isEntryPoint() &&
              (BF->getAddress() + BB.getOffset()) == OldAddress)
            return BB.getOutputStartAddress();
      }
      BC->errs() << "BOLT-ERROR: unable to get new address corresponding to "
                    "input address 0x"
                 << Twine::utohexstr(OldAddress) << " in function " << *BF
                 << ". Consider adding this function to --skip-funcs=...\n";
      exit(1);
    }
  }

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

  auto Streamer = BC->createStreamer(OS);
  // Make sure output stream has enough reserved space, otherwise
  // pwrite() will fail.
  uint64_t Offset = std::max(getFileOffsetForAddress(NextAvailableAddress),
                             FirstNonAllocatableOffset);
  Offset = OS.seek(Offset);
  assert((Offset != (uint64_t)-1) && "Error resizing output file");

  // Overwrite functions with fixed output address. This is mostly used by
  // non-relocation mode, with one exception: injected functions are covered
  // here in both modes.
  uint64_t CountOverwrittenFunctions = 0;
  uint64_t OverwrittenScore = 0;
  for (BinaryFunction *Function : BC->getAllBinaryFunctions()) {
    if (Function->getImageAddress() == 0 || Function->getImageSize() == 0)
      continue;

    assert(Function->getImageSize() <= Function->getMaxSize() &&
           "Unexpected large function");

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
    ++CountOverwrittenFunctions;

    // Overwrite function in the output file.
    if (opts::Verbosity >= 2)
      BC->outs() << "BOLT: rewriting function \"" << *Function << "\"\n";

    OS.pwrite(reinterpret_cast<char *>(Function->getImageAddress()),
              Function->getImageSize(), Function->getFileOffset());

    // Write nops at the end of the function.
    if (Function->getMaxSize() != std::numeric_limits<uint64_t>::max()) {
      uint64_t Pos = OS.tell();
      OS.seek(Function->getFileOffset() + Function->getImageSize());
      BC->MAB->writeNopData(
          OS, Function->getMaxSize() - Function->getImageSize(), &*BC->STI);

      OS.seek(Pos);
    }

    if (!Function->isSplit())
      continue;

    // Write cold part
    if (opts::Verbosity >= 2) {
      BC->outs() << formatv("BOLT: rewriting function \"{0}\" (split parts)\n",
                            *Function);
    }

    for (const FunctionFragment &FF :
         Function->getLayout().getSplitFragments()) {
      OS.pwrite(reinterpret_cast<char *>(FF.getImageAddress()),
                FF.getImageSize(), FF.getFileOffset());
    }
  }

  // Print function statistics for non-relocation mode.
  if (!BC->HasRelocations) {
    BC->outs() << "BOLT: " << CountOverwrittenFunctions << " out of "
               << BC->getBinaryFunctions().size()
               << " functions were overwritten.\n";
    if (BC->TotalScore != 0) {
      double Coverage = OverwrittenScore / (double)BC->TotalScore * 100.0;
      BC->outs() << format("BOLT-INFO: rewritten functions cover %.2lf",
                           Coverage)
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
      StringRef TrapInstr = BC->MIB->getTrapFillValue();
      unsigned NInstr = BF.getMaxSize() / TrapInstr.size();
      for (unsigned I = 0; I < NInstr; ++I)
        OS.write(TrapInstr.data(), TrapInstr.size());
    }
    OS.seek(SavedPos);
  }

  // Write all allocatable sections - reloc-mode text is written here as well
  for (BinarySection &Section : BC->allocatableSections()) {
    if (!Section.isFinalized() || !Section.getOutputData()) {
      LLVM_DEBUG(if (opts::Verbosity > 1) {
        dbgs() << "BOLT-INFO: new section is finalized or !getOutputData, skip "
               << Section.getName() << '\n';
      });
      continue;
    }
    if (Section.isLinkOnly()) {
      LLVM_DEBUG(if (opts::Verbosity > 1) {
        dbgs() << "BOLT-INFO: new section is link only, skip "
               << Section.getName() << '\n';
      });
      continue;
    }

    if (opts::Verbosity >= 1)
      BC->outs() << "BOLT: writing new section " << Section.getName()
                 << "\n data at 0x"
                 << Twine::utohexstr(Section.getAllocAddress()) << "\n of size "
                 << Section.getOutputSize() << "\n at offset "
                 << Section.getOutputFileOffset() << " with content size "
                 << Section.getOutputContents().size() << '\n';
    OS.seek(Section.getOutputFileOffset());
    Section.write(OS);
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
  if (!BC->IsLinuxKernel) {
    updateSegmentInfo();
    patchELFPHDRTable();
  }

  // Finalize memory image of section string table.
  finalizeSectionStringTable();

  // Update symbol tables.
  patchELFSymTabs();

  if (opts::EnableBAT)
    encodeBATSection();

  // Copy non-allocatable sections once allocatable part is finished.
  rewriteNoteSections();

  if (BC->HasRelocations) {
    patchELFAllocatableRelaSections();
    patchELFAllocatableRelrSection();
    patchELFGOT();
  }

  // Patch dynamic section/segment.
  patchELFDynamic();

  // Update ELF book-keeping info.
  patchELFSectionHeaderTable();

  if (opts::PrintSections) {
    BC->outs() << "BOLT-INFO: Sections after processing:\n";
    BC->printSections(BC->outs());
  }

  Out->keep();
  EC = sys::fs::setPermissions(
      opts::OutputFilename,
      static_cast<sys::fs::perms>(sys::fs::perms::all_all &
                                  ~sys::fs::getUmask()));
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

  LLVM_DEBUG(dbgs() << "BOLT: writing a new " << getEHFrameHdrSectionName()
                    << '\n');

  // Try to overwrite the original .eh_frame_hdr if the size permits.
  uint64_t EHFrameHdrOutputAddress = 0;
  uint64_t EHFrameHdrFileOffset = 0;
  std::vector<char> NewEHFrameHdr;
  BinarySection *OldEHFrameHdrSection = getSection(getEHFrameHdrSectionName());
  if (OldEHFrameHdrSection) {
    NewEHFrameHdr = CFIRdWrt->generateEHFrameHeader(
        RelocatedEHFrame, NewEHFrame, OldEHFrameHdrSection->getAddress());
    if (NewEHFrameHdr.size() <= OldEHFrameHdrSection->getSize()) {
      BC->outs() << "BOLT-INFO: rewriting " << getEHFrameHdrSectionName()
                 << " in-place\n";
      EHFrameHdrOutputAddress = OldEHFrameHdrSection->getAddress();
      EHFrameHdrFileOffset = OldEHFrameHdrSection->getInputFileOffset();
    } else {
      OldEHFrameHdrSection->setOutputName(getOrgSecPrefix() +
                                          getEHFrameHdrSectionName());
      OldEHFrameHdrSection = nullptr;
    }
  }

  // If there was not enough space, allocate more memory for .eh_frame_hdr.
  if (!OldEHFrameHdrSection) {
    NextAvailableAddress =
        appendPadding(Out->os(), NextAvailableAddress, EHFrameHdrAlign);

    EHFrameHdrOutputAddress = NextAvailableAddress;
    EHFrameHdrFileOffset = getFileOffsetForAddress(NextAvailableAddress);

    NewEHFrameHdr = CFIRdWrt->generateEHFrameHeader(
        RelocatedEHFrame, NewEHFrame, EHFrameHdrOutputAddress);

    NextAvailableAddress += NewEHFrameHdr.size();
    if (!BC->BOLTReserved.empty() &&
        (NextAvailableAddress > BC->BOLTReserved.end())) {
      BC->errs() << "BOLT-ERROR: unable to fit " << getEHFrameHdrSectionName()
                 << " into reserved space\n";
      exit(1);
    }

    // Create a new entry in the section header table.
    const unsigned Flags = BinarySection::getFlags(/*IsReadOnly=*/true,
                                                   /*IsText=*/false,
                                                   /*IsAllocatable=*/true);
    BinarySection &EHFrameHdrSec = BC->registerOrUpdateSection(
        getNewSecPrefix() + getEHFrameHdrSectionName(), ELF::SHT_PROGBITS,
        Flags, nullptr, NewEHFrameHdr.size(), /*Alignment=*/1);
    EHFrameHdrSec.setOutputFileOffset(EHFrameHdrFileOffset);
    EHFrameHdrSec.setOutputAddress(EHFrameHdrOutputAddress);
    EHFrameHdrSec.setOutputName(getEHFrameHdrSectionName());
  }

  Out->os().seek(EHFrameHdrFileOffset);
  Out->os().write(NewEHFrameHdr.data(), NewEHFrameHdr.size());

  // Pad the contents if overwriting in-place.
  if (OldEHFrameHdrSection)
    Out->os().write_zeros(OldEHFrameHdrSection->getSize() -
                          NewEHFrameHdr.size());

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
  auto Value = Linker->lookupSymbolInfo(Name);
  if (Value)
    return Value->Address;

  // Return the original value if we haven't emitted the symbol.
  BinaryData *BD = BC->getBinaryDataByName(Name);
  if (!BD)
    return 0;

  return BD->getAddress();
}

uint64_t RewriteInstance::getFileOffsetForAddress(uint64_t Address) const {
  // Check if it's possibly part of the new segment.
  if (NewTextSegmentAddress && Address >= NewTextSegmentAddress)
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
  if (llvm::is_contained(SectionsToOverwrite, SectionName))
    return true;
  if (llvm::is_contained(DebugSectionsToOverwrite, SectionName))
    return true;

  ErrorOr<BinarySection &> Section = BC->getUniqueSectionByName(SectionName);
  return Section && Section->isAllocatable() && Section->isFinalized();
}

bool RewriteInstance::isDebugSection(StringRef SectionName) {
  if (SectionName.starts_with(".debug_") ||
      SectionName.starts_with(".zdebug_") || SectionName == ".gdb_index" ||
      SectionName == ".stab" || SectionName == ".stabstr")
    return true;

  return false;
}
