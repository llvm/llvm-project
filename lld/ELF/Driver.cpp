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
// simply trust the compiler driver to pass all required options to us
// and don't try to make effort on our side.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Config.h"
#include "Error.h"
#include "ICF.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "LinkerScript.h"
#include "Memory.h"
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

bool elf::link(ArrayRef<const char *> Args, bool CanExitEarly,
               raw_ostream &Error) {
  ErrorCount = 0;
  ErrorOS = &Error;
  Argv0 = Args[0];
  Tar = nullptr;

  Config = make<Configuration>();
  Driver = make<LinkerDriver>();
  ScriptConfig = make<ScriptConfiguration>();

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
void LinkerDriver::addFile(StringRef Path) {
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
    addFile(*Path);
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

  // GNU linkers disagree here. Though both -version and -v are mentioned
  // in help to print the version information, GNU ld just normally exits,
  // while gold can continue linking. We are compatible with ld.bfd here.
  if (Args.hasArg(OPT_version) || Args.hasArg(OPT_v))
    outs() << getLLDVersion() << "\n";
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

// Determines what we should do if there are remaining unresolved
// symbols after the name resolution.
static UnresolvedPolicy getUnresolvedSymbolPolicy(opt::InputArgList &Args) {
  // -noinhibit-exec or -r imply some default values.
  if (Args.hasArg(OPT_noinhibit_exec))
    return UnresolvedPolicy::WarnAll;
  if (Config->Relocatable)
    return UnresolvedPolicy::IgnoreAll;

  UnresolvedPolicy ErrorOrWarn =
      getArg(Args, OPT_error_undef, OPT_warn_undef, true)
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

static Target2Policy getTarget2Option(opt::InputArgList &Args) {
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

static DiscardPolicy getDiscardOption(opt::InputArgList &Args) {
  if (Config->Relocatable)
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

static StripPolicy getStripOption(opt::InputArgList &Args) {
  if (auto *Arg = Args.getLastArg(OPT_strip_all, OPT_strip_debug)) {
    if (Arg->getOption().getID() == OPT_strip_all)
      return StripPolicy::All;
    return StripPolicy::Debug;
  }
  return StripPolicy::None;
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

static SortSectionPolicy getSortKind(opt::InputArgList &Args) {
  StringRef S = getString(Args, OPT_sort_section);
  if (S == "alignment")
    return SortSectionPolicy::Alignment;
  if (S == "name")
    return SortSectionPolicy::Name;
  if (!S.empty())
    error("unknown --sort-section rule: " + S);
  return SortSectionPolicy::Default;
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

// Initializes Config members by the command line options.
void LinkerDriver::readConfigs(opt::InputArgList &Args) {
  for (auto *Arg : Args.filtered(OPT_L))
    Config->SearchPaths.push_back(Arg->getValue());

  std::vector<StringRef> RPaths;
  for (auto *Arg : Args.filtered(OPT_rpath))
    RPaths.push_back(Arg->getValue());
  if (!RPaths.empty())
    Config->RPath = llvm::join(RPaths.begin(), RPaths.end(), ":");

  if (auto *Arg = Args.getLastArg(OPT_m)) {
    // Parse ELF{32,64}{LE,BE} and CPU type.
    StringRef S = Arg->getValue();
    std::tie(Config->EKind, Config->EMachine, Config->OSABI) =
        parseEmulation(S);
    Config->MipsN32Abi = (S == "elf32btsmipn32" || S == "elf32ltsmipn32");
    Config->Emulation = S;
  }

  Config->AllowMultipleDefinition = Args.hasArg(OPT_allow_multiple_definition);
  Config->Bsymbolic = Args.hasArg(OPT_Bsymbolic);
  Config->BsymbolicFunctions = Args.hasArg(OPT_Bsymbolic_functions);
  Config->Demangle = getArg(Args, OPT_demangle, OPT_no_demangle, true);
  Config->DisableVerify = Args.hasArg(OPT_disable_verify);
  Config->EmitRelocs = Args.hasArg(OPT_emit_relocs);
  Config->EhFrameHdr = Args.hasArg(OPT_eh_frame_hdr);
  Config->EnableNewDtags = !Args.hasArg(OPT_disable_new_dtags);
  Config->ExportDynamic =
      getArg(Args, OPT_export_dynamic, OPT_no_export_dynamic, false);
  Config->FatalWarnings =
      getArg(Args, OPT_fatal_warnings, OPT_no_fatal_warnings, false);
  Config->GcSections = getArg(Args, OPT_gc_sections, OPT_no_gc_sections, false);
  Config->GdbIndex = Args.hasArg(OPT_gdb_index);
  Config->ICF = Args.hasArg(OPT_icf);
  Config->NoGnuUnique = Args.hasArg(OPT_no_gnu_unique);
  Config->NoUndefinedVersion = Args.hasArg(OPT_no_undefined_version);
  Config->Nostdlib = Args.hasArg(OPT_nostdlib);
  Config->OMagic = Args.hasArg(OPT_omagic);
  Config->OptRemarksWithHotness = Args.hasArg(OPT_opt_remarks_with_hotness);
  Config->Pie = getArg(Args, OPT_pie, OPT_nopie, false);
  Config->PrintGcSections = Args.hasArg(OPT_print_gc_sections);
  Config->Relocatable = Args.hasArg(OPT_relocatable);
  Config->DefineCommon = getArg(Args, OPT_define_common, OPT_no_define_common,
                                !Config->Relocatable);
  Config->Discard = getDiscardOption(Args);
  Config->SaveTemps = Args.hasArg(OPT_save_temps);
  Config->SingleRoRx = Args.hasArg(OPT_no_rosegment);
  Config->Shared = Args.hasArg(OPT_shared);
  Config->Target1Rel = getArg(Args, OPT_target1_rel, OPT_target1_abs, false);
  Config->Threads = getArg(Args, OPT_threads, OPT_no_threads, true);
  Config->Trace = Args.hasArg(OPT_trace);
  Config->Verbose = Args.hasArg(OPT_verbose);
  Config->WarnCommon = Args.hasArg(OPT_warn_common);

  Config->DynamicLinker = getString(Args, OPT_dynamic_linker);
  Config->Entry = getString(Args, OPT_entry);
  Config->Fini = getString(Args, OPT_fini, "_fini");
  Config->Init = getString(Args, OPT_init, "_init");
  Config->LTOAAPipeline = getString(Args, OPT_lto_aa_pipeline);
  Config->LTONewPmPasses = getString(Args, OPT_lto_newpm_passes);
  Config->MapFile = getString(Args, OPT_Map);
  Config->OptRemarksFilename = getString(Args, OPT_opt_remarks_filename);
  Config->OutputFile = getString(Args, OPT_o);
  Config->SoName = getString(Args, OPT_soname);
  Config->Sysroot = getString(Args, OPT_sysroot);

  Config->Optimize = getInteger(Args, OPT_O, 1);
  Config->LTOO = getInteger(Args, OPT_lto_O, 2);
  if (Config->LTOO > 3)
    error("invalid optimization level for LTO: " + getString(Args, OPT_lto_O));
  Config->LTOPartitions = getInteger(Args, OPT_lto_partitions, 1);
  if (Config->LTOPartitions == 0)
    error("--lto-partitions: number of threads must be > 0");
  Config->ThinLTOJobs = getInteger(Args, OPT_thinlto_jobs, -1u);
  if (Config->ThinLTOJobs == 0)
    error("--thinlto-jobs: number of threads must be > 0");

  Config->ZCombreloc = !hasZOption(Args, "nocombreloc");
  Config->ZExecstack = hasZOption(Args, "execstack");
  Config->ZNodelete = hasZOption(Args, "nodelete");
  Config->ZNow = hasZOption(Args, "now");
  Config->ZOrigin = hasZOption(Args, "origin");
  Config->ZRelro = !hasZOption(Args, "norelro");
  Config->ZStackSize = getZOptionValue(Args, "stack-size", 0);
  Config->ZWxneeded = hasZOption(Args, "wxneeded");

  Config->OFormatBinary = isOutputFormatBinary(Args);
  Config->SectionStartMap = getSectionStartMap(Args);
  Config->SortSection = getSortKind(Args);
  Config->Target2 = getTarget2Option(Args);
  Config->UnresolvedSymbols = getUnresolvedSymbolPolicy(Args);

  if (Args.hasArg(OPT_print_map))
    Config->MapFile = "-";

  // --omagic is an option to create old-fashioned executables in which
  // .text segments are writable. Today, the option is still in use to
  // create special-purpose programs such as boot loaders. It doesn't
  // make sense to create PT_GNU_RELRO for such executables.
  if (Config->OMagic)
    Config->ZRelro = false;

  if (!Config->Relocatable)
    Config->Strip = getStripOption(Args);

  if (auto *Arg = Args.getLastArg(OPT_hash_style)) {
    StringRef S = Arg->getValue();
    if (S == "gnu") {
      Config->GnuHash = true;
      Config->SysvHash = false;
    } else if (S == "both") {
      Config->GnuHash = true;
    } else if (S != "sysv")
      error("unknown hash style: " + S);
  }

  // Parse --build-id or --build-id=<style>.
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

  for (auto *Arg : Args.filtered(OPT_auxiliary))
    Config->AuxiliaryList.push_back(Arg->getValue());
  if (!Config->Shared && !Config->AuxiliaryList.empty())
    error("-f may not be used without -shared");

  for (auto *Arg : Args.filtered(OPT_undefined))
    Config->Undefined.push_back(Arg->getValue());

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

  if (auto *Arg = Args.getLastArg(OPT_version_script))
    if (Optional<MemoryBufferRef> Buffer = readFile(Arg->getValue()))
      readVersionScript(*Buffer);
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
      addFile(Arg->getValue());
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
    return Config->pic() ? 0 : Target->DefaultImageBase;

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
  ScriptBase = Script<ELFT>::X = make<LinkerScript<ELFT>>();

  Config->Rela =
      ELFT::Is64Bits || Config->EMachine == EM_X86_64 || Config->MipsN32Abi;
  Config->Mips64EL =
      (Config->EMachine == EM_MIPS && Config->EKind == ELF64LEKind);
  Config->MaxPageSize = getMaxPageSize(Args);
  Config->ImageBase = getImageBase(Args);

  // Default output filename is "a.out" by the Unix tradition.
  if (Config->OutputFile.empty())
    Config->OutputFile = "a.out";

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

  for (auto *Arg : Args.filtered(OPT_wrap))
    Symtab.wrap(Arg->getValue());

  // Now that we have a complete list of input files.
  // Beyond this point, no new files are added.
  // Aggregate all input sections into one place.
  for (elf::ObjectFile<ELFT> *F : Symtab.getObjectFiles())
    for (InputSectionBase<ELFT> *S : F->getSections())
      if (S && S != &InputSection<ELFT>::Discarded)
        Symtab.Sections.push_back(S);
  for (BinaryFile *F : Symtab.getBinaryFiles())
    for (InputSectionData *S : F->getSections())
      Symtab.Sections.push_back(cast<InputSection<ELFT>>(S));

  // Do size optimizations: garbage collection and identical code folding.
  if (Config->GcSections)
    markLive<ELFT>();
  if (Config->ICF)
    doIcf<ELFT>();

  // MergeInputSection::splitIntoPieces needs to be called before
  // any call of MergeInputSection::getOffset. Do that.
  forEach(Symtab.Sections.begin(), Symtab.Sections.end(),
          [](InputSectionBase<ELFT> *S) {
            if (!S->Live)
              return;
            if (Decompressor::isCompressedELFSection(S->Flags, S->Name))
              S->uncompress();
            if (auto *MS = dyn_cast<MergeInputSection<ELFT>>(S))
              MS->splitIntoPieces();
          });

  // Write the result to the file.
  writeResult<ELFT>();
}
