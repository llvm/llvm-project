//===- Driver.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The driver drives the entire linking process. It is responsible for
// parsing command line options and doing whatever it is instructed to do.
//
// One notable thing in the LLD's driver when compared to other linkers is
// that the LLD's driver is agnostic on the host operating system.
// Other linkers usually have implicit default values (such as a dynamic
// linker path or library paths) for each host OS.
//
// I don't think implicit default values are useful because they are
// usually explicitly specified by the compiler driver. They can even
// be harmful when you are doing cross-linking. Therefore, in LLD, we
// simply trust the compiler driver to pass all required options and
// don't try to make effort on our side.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Config.h"
#include "Error.h"
#include "Filesystem.h"
#include "ICF.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "LinkerScript.h"
#include "Memory.h"
#include "OutputSections.h"
#include "ScriptParser.h"
#include "Strings.h"
#include "SymbolTable.h"
#include "Target.h"
#include "Threads.h"
#include "Writer.h"
#include "lld/Config/Version.h"
#include "lld/Driver/Driver.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/Decompressor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TarWriter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <utility>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::sys;

using namespace lld;
using namespace lld::elf;

Configuration *elf::Config;
LinkerDriver *elf::Driver;

BumpPtrAllocator elf::BAlloc;
StringSaver elf::Saver{BAlloc};
std::vector<SpecificAllocBase *> elf::SpecificAllocBase::Instances;

static void setConfigs();

bool elf::link(ArrayRef<const char *> Args, bool CanExitEarly,
               raw_ostream &Error) {
  ErrorCount = 0;
  ErrorOS = &Error;
  Argv0 = Args[0];
  InputSections.clear();
  Tar = nullptr;

  Config = make<Configuration>();
  Driver = make<LinkerDriver>();
  Script = make<LinkerScript>();

  Driver->main(Args, CanExitEarly);
  freeArena();
  return !ErrorCount;
}

// Parses a linker -m option.
static std::tuple<ELFKind, uint16_t, uint8_t> parseEmulation(StringRef Emul) {
  uint8_t OSABI = 0;
  StringRef S = Emul;
  if (S.endswith("_fbsd")) {
    S = S.drop_back(5);
    OSABI = ELFOSABI_FREEBSD;
  }

  std::pair<ELFKind, uint16_t> Ret =
      StringSwitch<std::pair<ELFKind, uint16_t>>(S)
          .Cases("aarch64elf", "aarch64linux", {ELF64LEKind, EM_AARCH64})
          .Case("armelf_linux_eabi", {ELF32LEKind, EM_ARM})
          .Case("elf32_x86_64", {ELF32LEKind, EM_X86_64})
          .Cases("elf32btsmip", "elf32btsmipn32", {ELF32BEKind, EM_MIPS})
          .Cases("elf32ltsmip", "elf32ltsmipn32", {ELF32LEKind, EM_MIPS})
          .Case("elf32ppc", {ELF32BEKind, EM_PPC})
          .Case("elf64btsmip", {ELF64BEKind, EM_MIPS})
          .Case("elf64ltsmip", {ELF64LEKind, EM_MIPS})
          .Case("elf64ppc", {ELF64BEKind, EM_PPC64})
          .Cases("elf_amd64", "elf_x86_64", {ELF64LEKind, EM_X86_64})
          .Case("elf_i386", {ELF32LEKind, EM_386})
          .Case("elf_iamcu", {ELF32LEKind, EM_IAMCU})
          .Default({ELFNoneKind, EM_NONE});

  if (Ret.first == ELFNoneKind) {
    if (S == "i386pe" || S == "i386pep" || S == "thumb2pe")
      error("Windows targets are not supported on the ELF frontend: " + Emul);
    else
      error("unknown emulation: " + Emul);
  }
  return std::make_tuple(Ret.first, Ret.second, OSABI);
}

// Returns slices of MB by parsing MB as an archive file.
// Each slice consists of a member file in the archive.
std::vector<MemoryBufferRef>
LinkerDriver::getArchiveMembers(MemoryBufferRef MB) {
  std::unique_ptr<Archive> File =
      check(Archive::create(MB),
            MB.getBufferIdentifier() + ": failed to parse archive");

  std::vector<MemoryBufferRef> V;
  Error Err = Error::success();
  for (const ErrorOr<Archive::Child> &COrErr : File->children(Err)) {
    Archive::Child C =
        check(COrErr, MB.getBufferIdentifier() +
                          ": could not get the child of the archive");
    MemoryBufferRef MBRef =
        check(C.getMemoryBufferRef(),
              MB.getBufferIdentifier() +
                  ": could not get the buffer for a child of the archive");
    V.push_back(MBRef);
  }
  if (Err)
    fatal(MB.getBufferIdentifier() + ": Archive::children failed: " +
          toString(std::move(Err)));

  // Take ownership of memory buffers created for members of thin archives.
  for (std::unique_ptr<MemoryBuffer> &MB : File->takeThinBuffers())
    make<std::unique_ptr<MemoryBuffer>>(std::move(MB));

  return V;
}

// Opens and parses a file. Path has to be resolved already.
// Newly created memory buffers are owned by this driver.
void LinkerDriver::addFile(StringRef Path, bool WithLOption) {
  using namespace sys::fs;

  Optional<MemoryBufferRef> Buffer = readFile(Path);
  if (!Buffer.hasValue())
    return;
  MemoryBufferRef MBRef = *Buffer;

  if (InBinary) {
    Files.push_back(make<BinaryFile>(MBRef));
    return;
  }

  switch (identify_magic(MBRef.getBuffer())) {
  case file_magic::unknown:
    readLinkerScript(MBRef);
    return;
  case file_magic::archive:
    if (InWholeArchive) {
      for (MemoryBufferRef MB : getArchiveMembers(MBRef))
        Files.push_back(createObjectFile(MB, Path));
      return;
    }
    Files.push_back(make<ArchiveFile>(MBRef));
    return;
  case file_magic::elf_shared_object:
    if (Config->Relocatable) {
      error("attempted static link of dynamic object " + Path);
      return;
    }
    Files.push_back(createSharedFile(MBRef));

    // DSOs usually have DT_SONAME tags in their ELF headers, and the
    // sonames are used to identify DSOs. But if they are missing,
    // they are identified by filenames. We don't know whether the new
    // file has a DT_SONAME or not because we haven't parsed it yet.
    // Here, we set the default soname for the file because we might
    // need it later.
    //
    // If a file was specified by -lfoo, the directory part is not
    // significant, as a user did not specify it. This behavior is
    // compatible with GNU.
    Files.back()->DefaultSoName =
        WithLOption ? sys::path::filename(Path) : Path;
    return;
  default:
    if (InLib)
      Files.push_back(make<LazyObjectFile>(MBRef));
    else
      Files.push_back(createObjectFile(MBRef));
  }
}

// Add a given library by searching it from input search paths.
void LinkerDriver::addLibrary(StringRef Name) {
  if (Optional<std::string> Path = searchLibrary(Name))
    addFile(*Path, /*WithLOption=*/true);
  else
    error("unable to find library -l" + Name);
}

// This function is called on startup. We need this for LTO since
// LTO calls LLVM functions to compile bitcode files to native code.
// Technically this can be delayed until we read bitcode files, but
// we don't bother to do lazily because the initialization is fast.
static void initLLVM(opt::InputArgList &Args) {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // Parse and evaluate -mllvm options.
  std::vector<const char *> V;
  V.push_back("lld (LLVM option parsing)");
  for (auto *Arg : Args.filtered(OPT_mllvm))
    V.push_back(Arg->getValue());
  cl::ParseCommandLineOptions(V.size(), V.data());
}

// Some command line options or some combinations of them are not allowed.
// This function checks for such errors.
static void checkOptions(opt::InputArgList &Args) {
  // The MIPS ABI as of 2016 does not support the GNU-style symbol lookup
  // table which is a relatively new feature.
  if (Config->EMachine == EM_MIPS && Config->GnuHash)
    error("the .gnu.hash section is not compatible with the MIPS target.");

  if (Config->Pie && Config->Shared)
    error("-shared and -pie may not be used together");

  if (Config->Relocatable) {
    if (Config->Shared)
      error("-r and -shared may not be used together");
    if (Config->GcSections)
      error("-r and --gc-sections may not be used together");
    if (Config->ICF)
      error("-r and --icf may not be used together");
    if (Config->Pie)
      error("-r and -pie may not be used together");
  }
}

static StringRef getString(opt::InputArgList &Args, unsigned Key,
                           StringRef Default = "") {
  if (auto *Arg = Args.getLastArg(Key))
    return Arg->getValue();
  return Default;
}

static int getInteger(opt::InputArgList &Args, unsigned Key, int Default) {
  int V = Default;
  if (auto *Arg = Args.getLastArg(Key)) {
    StringRef S = Arg->getValue();
    if (S.getAsInteger(10, V))
      error(Arg->getSpelling() + ": number expected, but got " + S);
  }
  return V;
}

static const char *getReproduceOption(opt::InputArgList &Args) {
  if (auto *Arg = Args.getLastArg(OPT_reproduce))
    return Arg->getValue();
  return getenv("LLD_REPRODUCE");
}

static bool hasZOption(opt::InputArgList &Args, StringRef Key) {
  for (auto *Arg : Args.filtered(OPT_z))
    if (Key == Arg->getValue())
      return true;
  return false;
}

static uint64_t getZOptionValue(opt::InputArgList &Args, StringRef Key,
                                uint64_t Default) {
  for (auto *Arg : Args.filtered(OPT_z)) {
    StringRef Value = Arg->getValue();
    size_t Pos = Value.find("=");
    if (Pos != StringRef::npos && Key == Value.substr(0, Pos)) {
      Value = Value.substr(Pos + 1);
      uint64_t Result;
      if (Value.getAsInteger(0, Result))
        error("invalid " + Key + ": " + Value);
      return Result;
    }
  }
  return Default;
}

void LinkerDriver::main(ArrayRef<const char *> ArgsArr, bool CanExitEarly) {
  ELFOptTable Parser;
  opt::InputArgList Args = Parser.parse(ArgsArr.slice(1));

  // Interpret this flag early because error() depends on them.
  Config->ErrorLimit = getInteger(Args, OPT_error_limit, 20);

  // Handle -help
  if (Args.hasArg(OPT_help)) {
    printHelp(ArgsArr[0]);
    return;
  }

  // Handle -v or -version.
  //
  // A note about "compatible with GNU linkers" message: this is a hack for
  // scripts generated by GNU Libtool 2.4.6 (released in February 2014 and
  // still the newest version in March 2017) or earlier to recognize LLD as
  // a GNU compatible linker. As long as an output for the -v option
  // contains "GNU" or "with BFD", they recognize us as GNU-compatible.
  //
  // This is somewhat ugly hack, but in reality, we had no choice other
  // than doing this. Considering the very long release cycle of Libtool,
  // it is not easy to improve it to recognize LLD as a GNU compatible
  // linker in a timely manner. Even if we can make it, there are still a
  // lot of "configure" scripts out there that are generated by old version
  // of Libtool. We cannot convince every software developer to migrate to
  // the latest version and re-generate scripts. So we have this hack.
  if (Args.hasArg(OPT_v) || Args.hasArg(OPT_version))
    message(getLLDVersion() + " (compatible with GNU linkers)");

  // ld.bfd always exits after printing out the version string.
  // ld.gold proceeds if a given option is -v. Because gold's behavior
  // is more permissive than ld.bfd, we chose what gold does here.
  if (Args.hasArg(OPT_version))
    return;

  Config->ExitEarly = CanExitEarly && !Args.hasArg(OPT_full_shutdown);

  if (const char *Path = getReproduceOption(Args)) {
    // Note that --reproduce is a debug option so you can ignore it
    // if you are trying to understand the whole picture of the code.
    Expected<std::unique_ptr<TarWriter>> ErrOrWriter =
        TarWriter::create(Path, path::stem(Path));
    if (ErrOrWriter) {
      Tar = ErrOrWriter->get();
      Tar->append("response.txt", createResponseFile(Args));
      Tar->append("version.txt", getLLDVersion() + "\n");
      make<std::unique_ptr<TarWriter>>(std::move(*ErrOrWriter));
    } else {
      error(Twine("--reproduce: failed to open ") + Path + ": " +
            toString(ErrOrWriter.takeError()));
    }
  }

  readConfigs(Args);
  initLLVM(Args);
  createFiles(Args);
  inferMachineType();
  setConfigs();
  checkOptions(Args);
  if (ErrorCount)
    return;

  switch (Config->EKind) {
  case ELF32LEKind:
    link<ELF32LE>(Args);
    return;
  case ELF32BEKind:
    link<ELF32BE>(Args);
    return;
  case ELF64LEKind:
    link<ELF64LE>(Args);
    return;
  case ELF64BEKind:
    link<ELF64BE>(Args);
    return;
  default:
    llvm_unreachable("unknown Config->EKind");
  }
}

static bool getArg(opt::InputArgList &Args, unsigned K1, unsigned K2,
                   bool Default) {
  if (auto *Arg = Args.getLastArg(K1, K2))
    return Arg->getOption().getID() == K1;
  return Default;
}

static std::vector<StringRef> getArgs(opt::InputArgList &Args, int Id) {
  std::vector<StringRef> V;
  for (auto *Arg : Args.filtered(Id))
    V.push_back(Arg->getValue());
  return V;
}

static std::string getRPath(opt::InputArgList &Args) {
  std::vector<StringRef> V = getArgs(Args, OPT_rpath);
  return llvm::join(V.begin(), V.end(), ":");
}

// Determines what we should do if there are remaining unresolved
// symbols after the name resolution.
static UnresolvedPolicy getUnresolvedSymbolPolicy(opt::InputArgList &Args) {
  // -noinhibit-exec or -r imply some default values.
  if (Args.hasArg(OPT_noinhibit_exec))
    return UnresolvedPolicy::WarnAll;
  if (Args.hasArg(OPT_relocatable))
    return UnresolvedPolicy::IgnoreAll;

  UnresolvedPolicy ErrorOrWarn = getArg(Args, OPT_error_unresolved_symbols,
                                        OPT_warn_unresolved_symbols, true)
                                     ? UnresolvedPolicy::ReportError
                                     : UnresolvedPolicy::Warn;

  // Process the last of -unresolved-symbols, -no-undefined or -z defs.
  for (auto *Arg : llvm::reverse(Args)) {
    switch (Arg->getOption().getID()) {
    case OPT_unresolved_symbols: {
      StringRef S = Arg->getValue();
      if (S == "ignore-all" || S == "ignore-in-object-files")
        return UnresolvedPolicy::Ignore;
      if (S == "ignore-in-shared-libs" || S == "report-all")
        return ErrorOrWarn;
      error("unknown --unresolved-symbols value: " + S);
      continue;
    }
    case OPT_no_undefined:
      return ErrorOrWarn;
    case OPT_z:
      if (StringRef(Arg->getValue()) == "defs")
        return ErrorOrWarn;
      continue;
    }
  }

  // -shared implies -unresolved-symbols=ignore-all because missing
  // symbols are likely to be resolved at runtime using other DSOs.
  if (Config->Shared)
    return UnresolvedPolicy::Ignore;
  return ErrorOrWarn;
}

static Target2Policy getTarget2(opt::InputArgList &Args) {
  if (auto *Arg = Args.getLastArg(OPT_target2)) {
    StringRef S = Arg->getValue();
    if (S == "rel")
      return Target2Policy::Rel;
    if (S == "abs")
      return Target2Policy::Abs;
    if (S == "got-rel")
      return Target2Policy::GotRel;
    error("unknown --target2 option: " + S);
  }
  return Target2Policy::GotRel;
}

static bool isOutputFormatBinary(opt::InputArgList &Args) {
  if (auto *Arg = Args.getLastArg(OPT_oformat)) {
    StringRef S = Arg->getValue();
    if (S == "binary")
      return true;
    error("unknown --oformat value: " + S);
  }
  return false;
}

static DiscardPolicy getDiscard(opt::InputArgList &Args) {
  if (Args.hasArg(OPT_relocatable))
    return DiscardPolicy::None;

  auto *Arg =
      Args.getLastArg(OPT_discard_all, OPT_discard_locals, OPT_discard_none);
  if (!Arg)
    return DiscardPolicy::Default;
  if (Arg->getOption().getID() == OPT_discard_all)
    return DiscardPolicy::All;
  if (Arg->getOption().getID() == OPT_discard_locals)
    return DiscardPolicy::Locals;
  return DiscardPolicy::None;
}

static StringRef getDynamicLinker(opt::InputArgList &Args) {
  auto *Arg = Args.getLastArg(OPT_dynamic_linker, OPT_no_dynamic_linker);
  if (!Arg || Arg->getOption().getID() == OPT_no_dynamic_linker)
    return "";
  return Arg->getValue();
}

static StripPolicy getStrip(opt::InputArgList &Args) {
  if (Args.hasArg(OPT_relocatable))
    return StripPolicy::None;

  auto *Arg = Args.getLastArg(OPT_strip_all, OPT_strip_debug);
  if (!Arg)
    return StripPolicy::None;
  if (Arg->getOption().getID() == OPT_strip_all)
    return StripPolicy::All;
  return StripPolicy::Debug;
}

static uint64_t parseSectionAddress(StringRef S, opt::Arg *Arg) {
  uint64_t VA = 0;
  if (S.startswith("0x"))
    S = S.drop_front(2);
  if (S.getAsInteger(16, VA))
    error("invalid argument: " + toString(Arg));
  return VA;
}

static StringMap<uint64_t> getSectionStartMap(opt::InputArgList &Args) {
  StringMap<uint64_t> Ret;
  for (auto *Arg : Args.filtered(OPT_section_start)) {
    StringRef Name;
    StringRef Addr;
    std::tie(Name, Addr) = StringRef(Arg->getValue()).split('=');
    Ret[Name] = parseSectionAddress(Addr, Arg);
  }

  if (auto *Arg = Args.getLastArg(OPT_Ttext))
    Ret[".text"] = parseSectionAddress(Arg->getValue(), Arg);
  if (auto *Arg = Args.getLastArg(OPT_Tdata))
    Ret[".data"] = parseSectionAddress(Arg->getValue(), Arg);
  if (auto *Arg = Args.getLastArg(OPT_Tbss))
    Ret[".bss"] = parseSectionAddress(Arg->getValue(), Arg);
  return Ret;
}

static SortSectionPolicy getSortSection(opt::InputArgList &Args) {
  StringRef S = getString(Args, OPT_sort_section);
  if (S == "alignment")
    return SortSectionPolicy::Alignment;
  if (S == "name")
    return SortSectionPolicy::Name;
  if (!S.empty())
    error("unknown --sort-section rule: " + S);
  return SortSectionPolicy::Default;
}

static std::pair<bool, bool> getHashStyle(opt::InputArgList &Args) {
  StringRef S = getString(Args, OPT_hash_style, "sysv");
  if (S == "sysv")
    return {true, false};
  if (S == "gnu")
    return {false, true};
  if (S != "both")
    error("unknown -hash-style: " + S);
  return {true, true};
}

static std::vector<StringRef> getLines(MemoryBufferRef MB) {
  SmallVector<StringRef, 0> Arr;
  MB.getBuffer().split(Arr, '\n');

  std::vector<StringRef> Ret;
  for (StringRef S : Arr) {
    S = S.trim();
    if (!S.empty())
      Ret.push_back(S);
  }
  return Ret;
}

static bool getCompressDebugSections(opt::InputArgList &Args) {
  if (auto *Arg = Args.getLastArg(OPT_compress_debug_sections)) {
    StringRef S = Arg->getValue();
    if (S == "zlib")
      return zlib::isAvailable();
    if (S != "none")
      error("unknown --compress-debug-sections value: " + S);
  }
  return false;
}

// Initializes Config members by the command line options.
void LinkerDriver::readConfigs(opt::InputArgList &Args) {
  Config->AllowMultipleDefinition = Args.hasArg(OPT_allow_multiple_definition);
  Config->AuxiliaryList = getArgs(Args, OPT_auxiliary);
  Config->Bsymbolic = Args.hasArg(OPT_Bsymbolic);
  Config->BsymbolicFunctions = Args.hasArg(OPT_Bsymbolic_functions);
  Config->CompressDebugSections = getCompressDebugSections(Args);
  Config->DefineCommon = getArg(Args, OPT_define_common, OPT_no_define_common,
                                !Args.hasArg(OPT_relocatable));
  Config->Demangle = getArg(Args, OPT_demangle, OPT_no_demangle, true);
  Config->DisableVerify = Args.hasArg(OPT_disable_verify);
  Config->Discard = getDiscard(Args);
  Config->DynamicLinker = getDynamicLinker(Args);
  Config->EhFrameHdr = Args.hasArg(OPT_eh_frame_hdr);
  Config->EmitRelocs = Args.hasArg(OPT_emit_relocs);
  Config->EnableNewDtags = !Args.hasArg(OPT_disable_new_dtags);
  Config->Entry = getString(Args, OPT_entry);
  Config->ExportDynamic =
      getArg(Args, OPT_export_dynamic, OPT_no_export_dynamic, false);
  Config->FatalWarnings =
      getArg(Args, OPT_fatal_warnings, OPT_no_fatal_warnings, false);
  Config->Fini = getString(Args, OPT_fini, "_fini");
  Config->GcSections = getArg(Args, OPT_gc_sections, OPT_no_gc_sections, false);
  Config->GdbIndex = Args.hasArg(OPT_gdb_index);
  Config->ICF = Args.hasArg(OPT_icf);
  Config->Init = getString(Args, OPT_init, "_init");
  Config->LTOAAPipeline = getString(Args, OPT_lto_aa_pipeline);
  Config->LTONewPmPasses = getString(Args, OPT_lto_newpm_passes);
  Config->LTOO = getInteger(Args, OPT_lto_O, 2);
  Config->LTOPartitions = getInteger(Args, OPT_lto_partitions, 1);
  Config->MapFile = getString(Args, OPT_Map);
  Config->NoGnuUnique = Args.hasArg(OPT_no_gnu_unique);
  Config->NoUndefinedVersion = Args.hasArg(OPT_no_undefined_version);
  Config->Nostdlib = Args.hasArg(OPT_nostdlib);
  Config->OFormatBinary = isOutputFormatBinary(Args);
  Config->Omagic = Args.hasArg(OPT_omagic);
  Config->OptRemarksFilename = getString(Args, OPT_opt_remarks_filename);
  Config->OptRemarksWithHotness = Args.hasArg(OPT_opt_remarks_with_hotness);
  Config->Optimize = getInteger(Args, OPT_O, 1);
  Config->OutputFile = getString(Args, OPT_o);
  Config->Pie = getArg(Args, OPT_pie, OPT_nopie, false);
  Config->PrintGcSections = Args.hasArg(OPT_print_gc_sections);
  Config->RPath = getRPath(Args);
  Config->Relocatable = Args.hasArg(OPT_relocatable);
  Config->SaveTemps = Args.hasArg(OPT_save_temps);
  Config->SearchPaths = getArgs(Args, OPT_L);
  Config->SectionStartMap = getSectionStartMap(Args);
  Config->Shared = Args.hasArg(OPT_shared);
  Config->SingleRoRx = Args.hasArg(OPT_no_rosegment);
  Config->SoName = getString(Args, OPT_soname);
  Config->SortSection = getSortSection(Args);
  Config->Strip = getStrip(Args);
  Config->Sysroot = getString(Args, OPT_sysroot);
  Config->Target1Rel = getArg(Args, OPT_target1_rel, OPT_target1_abs, false);
  Config->Target2 = getTarget2(Args);
  Config->ThinLTOCacheDir = getString(Args, OPT_thinlto_cache_dir);
  Config->ThinLTOCachePolicy =
      check(parseCachePruningPolicy(getString(Args, OPT_thinlto_cache_policy)),
            "--thinlto-cache-policy: invalid cache policy");
  Config->ThinLTOJobs = getInteger(Args, OPT_thinlto_jobs, -1u);
  Config->Threads = getArg(Args, OPT_threads, OPT_no_threads, true);
  Config->Trace = Args.hasArg(OPT_trace);
  Config->Undefined = getArgs(Args, OPT_undefined);
  Config->UnresolvedSymbols = getUnresolvedSymbolPolicy(Args);
  Config->Verbose = Args.hasArg(OPT_verbose);
  Config->WarnCommon = Args.hasArg(OPT_warn_common);
  Config->ZCombreloc = !hasZOption(Args, "nocombreloc");
  Config->ZExecstack = hasZOption(Args, "execstack");
  Config->ZNocopyreloc = hasZOption(Args, "nocopyreloc");
  Config->ZNodelete = hasZOption(Args, "nodelete");
  Config->ZNodlopen = hasZOption(Args, "nodlopen");
  Config->ZNow = hasZOption(Args, "now");
  Config->ZOrigin = hasZOption(Args, "origin");
  Config->ZRelro = !hasZOption(Args, "norelro");
  Config->ZStackSize = getZOptionValue(Args, "stack-size", 0);
  Config->ZText = !hasZOption(Args, "notext");
  Config->ZWxneeded = hasZOption(Args, "wxneeded");

  if (Config->LTOO > 3)
    error("invalid optimization level for LTO: " + getString(Args, OPT_lto_O));
  if (Config->LTOPartitions == 0)
    error("--lto-partitions: number of threads must be > 0");
  if (Config->ThinLTOJobs == 0)
    error("--thinlto-jobs: number of threads must be > 0");

  if (auto *Arg = Args.getLastArg(OPT_m)) {
    // Parse ELF{32,64}{LE,BE} and CPU type.
    StringRef S = Arg->getValue();
    std::tie(Config->EKind, Config->EMachine, Config->OSABI) =
        parseEmulation(S);
    Config->MipsN32Abi = (S == "elf32btsmipn32" || S == "elf32ltsmipn32");
    Config->Emulation = S;
  }

  if (Args.hasArg(OPT_print_map))
    Config->MapFile = "-";

  // --omagic is an option to create old-fashioned executables in which
  // .text segments are writable. Today, the option is still in use to
  // create special-purpose programs such as boot loaders. It doesn't
  // make sense to create PT_GNU_RELRO for such executables.
  if (Config->Omagic)
    Config->ZRelro = false;

  std::tie(Config->SysvHash, Config->GnuHash) = getHashStyle(Args);

  // Parse --build-id or --build-id=<style>. We handle "tree" as a
  // synonym for "sha1" because all of our hash functions including
  // -build-id=sha1 are tree hashes for performance reasons.
  if (Args.hasArg(OPT_build_id))
    Config->BuildId = BuildIdKind::Fast;
  if (auto *Arg = Args.getLastArg(OPT_build_id_eq)) {
    StringRef S = Arg->getValue();
    if (S == "md5") {
      Config->BuildId = BuildIdKind::Md5;
    } else if (S == "sha1" || S == "tree") {
      Config->BuildId = BuildIdKind::Sha1;
    } else if (S == "uuid") {
      Config->BuildId = BuildIdKind::Uuid;
    } else if (S == "none") {
      Config->BuildId = BuildIdKind::None;
    } else if (S.startswith("0x")) {
      Config->BuildId = BuildIdKind::Hexstring;
      Config->BuildIdVector = parseHex(S.substr(2));
    } else {
      error("unknown --build-id style: " + S);
    }
  }

  if (!Config->Shared && !Config->AuxiliaryList.empty())
    error("-f may not be used without -shared");

  for (auto *Arg : Args.filtered(OPT_dynamic_list))
    if (Optional<MemoryBufferRef> Buffer = readFile(Arg->getValue()))
      readDynamicList(*Buffer);

  if (auto *Arg = Args.getLastArg(OPT_symbol_ordering_file))
    if (Optional<MemoryBufferRef> Buffer = readFile(Arg->getValue()))
      Config->SymbolOrderingFile = getLines(*Buffer);

  // If --retain-symbol-file is used, we'll keep only the symbols listed in
  // the file and discard all others.
  if (auto *Arg = Args.getLastArg(OPT_retain_symbols_file)) {
    Config->DefaultSymbolVersion = VER_NDX_LOCAL;
    if (Optional<MemoryBufferRef> Buffer = readFile(Arg->getValue()))
      for (StringRef S : getLines(*Buffer))
        Config->VersionScriptGlobals.push_back(
            {S, /*IsExternCpp*/ false, /*HasWildcard*/ false});
  }

  for (auto *Arg : Args.filtered(OPT_export_dynamic_symbol))
    Config->VersionScriptGlobals.push_back(
        {Arg->getValue(), /*IsExternCpp*/ false, /*HasWildcard*/ false});

  // Dynamic lists are a simplified linker script that doesn't need the
  // "global:" and implicitly ends with a "local:*". Set the variables needed to
  // simulate that.
  if (Args.hasArg(OPT_dynamic_list) || Args.hasArg(OPT_export_dynamic_symbol)) {
    Config->ExportDynamic = true;
    if (!Config->Shared)
      Config->DefaultSymbolVersion = VER_NDX_LOCAL;
  }

  if (getArg(Args, OPT_export_dynamic, OPT_no_export_dynamic, false))
    Config->DefaultSymbolVersion = VER_NDX_GLOBAL;

  if (auto *Arg = Args.getLastArg(OPT_version_script))
    if (Optional<MemoryBufferRef> Buffer = readFile(Arg->getValue()))
      readVersionScript(*Buffer);
}

// Some Config members do not directly correspond to any particular
// command line options, but computed based on other Config values.
// This function initialize such members. See Config.h for the details
// of these values.
static void setConfigs() {
  ELFKind Kind = Config->EKind;
  uint16_t Machine = Config->EMachine;

  // There is an ILP32 ABI for x86-64, although it's not very popular.
  // It is called the x32 ABI.
  bool IsX32 = (Kind == ELF32LEKind && Machine == EM_X86_64);

  Config->CopyRelocs = (Config->Relocatable || Config->EmitRelocs);
  Config->Is64 = (Kind == ELF64LEKind || Kind == ELF64BEKind);
  Config->IsLE = (Kind == ELF32LEKind || Kind == ELF64LEKind);
  Config->Endianness =
      Config->IsLE ? support::endianness::little : support::endianness::big;
  Config->IsMips64EL = (Kind == ELF64LEKind && Machine == EM_MIPS);
  Config->IsRela = Config->Is64 || IsX32 || Config->MipsN32Abi;
  Config->Pic = Config->Pie || Config->Shared;
  Config->Wordsize = Config->Is64 ? 8 : 4;
}

// Returns a value of "-format" option.
static bool getBinaryOption(StringRef S) {
  if (S == "binary")
    return true;
  if (S == "elf" || S == "default")
    return false;
  error("unknown -format value: " + S +
        " (supported formats: elf, default, binary)");
  return false;
}

void LinkerDriver::createFiles(opt::InputArgList &Args) {
  for (auto *Arg : Args) {
    switch (Arg->getOption().getID()) {
    case OPT_l:
      addLibrary(Arg->getValue());
      break;
    case OPT_INPUT:
      addFile(Arg->getValue(), /*WithLOption=*/false);
      break;
    case OPT_alias_script_T:
    case OPT_script:
      if (Optional<MemoryBufferRef> MB = readFile(Arg->getValue()))
        readLinkerScript(*MB);
      break;
    case OPT_as_needed:
      Config->AsNeeded = true;
      break;
    case OPT_format:
      InBinary = getBinaryOption(Arg->getValue());
      break;
    case OPT_no_as_needed:
      Config->AsNeeded = false;
      break;
    case OPT_Bstatic:
      Config->Static = true;
      break;
    case OPT_Bdynamic:
      Config->Static = false;
      break;
    case OPT_whole_archive:
      InWholeArchive = true;
      break;
    case OPT_no_whole_archive:
      InWholeArchive = false;
      break;
    case OPT_start_lib:
      InLib = true;
      break;
    case OPT_end_lib:
      InLib = false;
      break;
    }
  }

  if (Files.empty() && ErrorCount == 0)
    error("no input files");
}

// If -m <machine_type> was not given, infer it from object files.
void LinkerDriver::inferMachineType() {
  if (Config->EKind != ELFNoneKind)
    return;

  for (InputFile *F : Files) {
    if (F->EKind == ELFNoneKind)
      continue;
    Config->EKind = F->EKind;
    Config->EMachine = F->EMachine;
    Config->OSABI = F->OSABI;
    Config->MipsN32Abi = Config->EMachine == EM_MIPS && isMipsN32Abi(F);
    return;
  }
  error("target emulation unknown: -m or at least one .o file required");
}

// Parse -z max-page-size=<value>. The default value is defined by
// each target.
static uint64_t getMaxPageSize(opt::InputArgList &Args) {
  uint64_t Val =
      getZOptionValue(Args, "max-page-size", Target->DefaultMaxPageSize);
  if (!isPowerOf2_64(Val))
    error("max-page-size: value isn't a power of 2");
  return Val;
}

// Parses -image-base option.
static uint64_t getImageBase(opt::InputArgList &Args) {
  // Use default if no -image-base option is given.
  // Because we are using "Target" here, this function
  // has to be called after the variable is initialized.
  auto *Arg = Args.getLastArg(OPT_image_base);
  if (!Arg)
    return Config->Pic ? 0 : Target->DefaultImageBase;

  StringRef S = Arg->getValue();
  uint64_t V;
  if (S.getAsInteger(0, V)) {
    error("-image-base: number expected, but got " + S);
    return 0;
  }
  if ((V % Config->MaxPageSize) != 0)
    warn("-image-base: address isn't multiple of page size: " + S);
  return V;
}

// Do actual linking. Note that when this function is called,
// all linker scripts have already been parsed.
template <class ELFT> void LinkerDriver::link(opt::InputArgList &Args) {
  SymbolTable<ELFT> Symtab;
  elf::Symtab<ELFT>::X = &Symtab;
  Target = createTarget();

  Config->MaxPageSize = getMaxPageSize(Args);
  Config->ImageBase = getImageBase(Args);

  // Default output filename is "a.out" by the Unix tradition.
  if (Config->OutputFile.empty())
    Config->OutputFile = "a.out";

  // Fail early if the output file or map file is not writable. If a user has a
  // long link, e.g. due to a large LTO link, they do not wish to run it and
  // find that it failed because there was a mistake in their command-line.
  if (!isFileWritable(Config->OutputFile, "output file"))
    return;
  if (!isFileWritable(Config->MapFile, "map file"))
    return;

  // Use default entry point name if no name was given via the command
  // line nor linker scripts. For some reason, MIPS entry point name is
  // different from others.
  Config->WarnMissingEntry =
      (!Config->Entry.empty() || (!Config->Shared && !Config->Relocatable));
  if (Config->Entry.empty() && !Config->Relocatable)
    Config->Entry = (Config->EMachine == EM_MIPS) ? "__start" : "_start";

  // Handle --trace-symbol.
  for (auto *Arg : Args.filtered(OPT_trace_symbol))
    Symtab.trace(Arg->getValue());

  // Add all files to the symbol table. This will add almost all
  // symbols that we need to the symbol table.
  for (InputFile *F : Files)
    Symtab.addFile(F);

  // If an entry symbol is in a static archive, pull out that file now
  // to complete the symbol table. After this, no new names except a
  // few linker-synthesized ones will be added to the symbol table.
  if (Symtab.find(Config->Entry))
    Symtab.addUndefined(Config->Entry);

  // Return if there were name resolution errors.
  if (ErrorCount)
    return;

  Symtab.scanUndefinedFlags();
  Symtab.scanShlibUndefined();
  Symtab.scanVersionScript();

  Symtab.addCombinedLTOObject();
  if (ErrorCount)
    return;

  // Some symbols (such as __ehdr_start) are defined lazily only when there
  // are undefined symbols for them, so we add these to trigger that logic.
  for (StringRef Sym : Script->Opt.ReferencedSymbols)
    Symtab.addUndefined(Sym);

  for (auto *Arg : Args.filtered(OPT_wrap))
    Symtab.wrap(Arg->getValue());

  // Now that we have a complete list of input files.
  // Beyond this point, no new files are added.
  // Aggregate all input sections into one place.
  for (elf::ObjectFile<ELFT> *F : Symtab.getObjectFiles())
    for (InputSectionBase *S : F->getSections())
      if (S && S != &InputSection::Discarded)
        InputSections.push_back(S);
  for (BinaryFile *F : Symtab.getBinaryFiles())
    for (InputSectionBase *S : F->getSections())
      InputSections.push_back(cast<InputSection>(S));

  // Do size optimizations: garbage collection and identical code folding.
  if (Config->GcSections)
    markLive<ELFT>();
  if (Config->ICF)
    doIcf<ELFT>();

  // MergeInputSection::splitIntoPieces needs to be called before
  // any call of MergeInputSection::getOffset. Do that.
  parallelForEach(InputSections.begin(), InputSections.end(),
                  [](InputSectionBase *S) {
                    if (!S->Live)
                      return;
                    if (Decompressor::isCompressedELFSection(S->Flags, S->Name))
                      S->uncompress();
                    if (auto *MS = dyn_cast<MergeInputSection>(S))
                      MS->splitIntoPieces();
                  });

  // Write the result to the file.
  writeResult<ELFT>();
}
