//===- Driver.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Config.h"
#include "Error.h"
#include "ICF.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "LinkerScript.h"
#include "Strings.h"
#include "SymbolListFile.h"
#include "SymbolTable.h"
#include "Target.h"
#include "Writer.h"
#include "lld/Driver/Driver.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
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

bool elf::link(ArrayRef<const char *> Args, raw_ostream &Error) {
  HasError = false;
  ErrorOS = &Error;

  Configuration C;
  LinkerDriver D;
  ScriptConfiguration SC;
  Config = &C;
  Driver = &D;
  ScriptConfig = &SC;

  Driver->main(Args);
  InputFile::freePool();
  return !HasError;
}

// Parses a linker -m option.
static std::pair<ELFKind, uint16_t> parseEmulation(StringRef Emul) {
  StringRef S = Emul;
  if (S.endswith("_fbsd"))
    S = S.drop_back(5);

  std::pair<ELFKind, uint16_t> Ret =
      StringSwitch<std::pair<ELFKind, uint16_t>>(S)
          .Case("aarch64elf", {ELF64LEKind, EM_AARCH64})
          .Case("aarch64linux", {ELF64LEKind, EM_AARCH64})
          .Case("armelf_linux_eabi", {ELF32LEKind, EM_ARM})
          .Case("elf32_x86_64", {ELF32LEKind, EM_X86_64})
          .Case("elf32btsmip", {ELF32BEKind, EM_MIPS})
          .Case("elf32ltsmip", {ELF32LEKind, EM_MIPS})
          .Case("elf32ppc", {ELF32BEKind, EM_PPC})
          .Case("elf64btsmip", {ELF64BEKind, EM_MIPS})
          .Case("elf64ltsmip", {ELF64LEKind, EM_MIPS})
          .Case("elf64ppc", {ELF64BEKind, EM_PPC64})
          .Case("elf_amd64", {ELF64LEKind, EM_X86_64})
          .Case("elf_i386", {ELF32LEKind, EM_386})
          .Case("elf_iamcu", {ELF32LEKind, EM_IAMCU})
          .Case("elf_x86_64", {ELF64LEKind, EM_X86_64})
          .Default({ELFNoneKind, EM_NONE});

  if (Ret.first == ELFNoneKind) {
    if (S == "i386pe" || S == "i386pep" || S == "thumb2pe")
      error("Windows targets are not supported on the ELF frontend: " + Emul);
    else
      error("unknown emulation: " + Emul);
  }
  return Ret;
}

// Returns slices of MB by parsing MB as an archive file.
// Each slice consists of a member file in the archive.
std::vector<MemoryBufferRef>
LinkerDriver::getArchiveMembers(MemoryBufferRef MB) {
  std::unique_ptr<Archive> File =
      check(Archive::create(MB), "failed to parse archive");

  std::vector<MemoryBufferRef> V;
  Error Err;
  for (const ErrorOr<Archive::Child> &COrErr : File->children(Err)) {
    Archive::Child C = check(COrErr, "could not get the child of the archive " +
                                         File->getFileName());
    MemoryBufferRef MBRef =
        check(C.getMemoryBufferRef(),
              "could not get the buffer for a child of the archive " +
                  File->getFileName());
    V.push_back(MBRef);
  }
  if (Err)
    Error(Err);

  // Take ownership of memory buffers created for members of thin archives.
  for (std::unique_ptr<MemoryBuffer> &MB : File->takeThinBuffers())
    OwningMBs.push_back(std::move(MB));

  return V;
}

// Opens and parses a file. Path has to be resolved already.
// Newly created memory buffers are owned by this driver.
void LinkerDriver::addFile(StringRef Path, bool KnownScript) {
  using namespace sys::fs;
  if (Config->Verbose)
    outs() << Path << "\n";

  Optional<MemoryBufferRef> Buffer = readFile(Path);
  if (!Buffer.hasValue())
    return;
  MemoryBufferRef MBRef = *Buffer;

  if (Config->Binary && !KnownScript) {
    Files.push_back(new BinaryFile(MBRef));
    return;
  }

  switch (identify_magic(MBRef.getBuffer())) {
  case file_magic::unknown:
    readLinkerScript(MBRef);
    return;
  case file_magic::archive:
    if (WholeArchive) {
      for (MemoryBufferRef MB : getArchiveMembers(MBRef))
        Files.push_back(createObjectFile(MB, Path));
      return;
    }
    Files.push_back(new ArchiveFile(MBRef));
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
      Files.push_back(new LazyObjectFile(MBRef));
    else
      Files.push_back(createObjectFile(MBRef));
  }
}

Optional<MemoryBufferRef> LinkerDriver::readFile(StringRef Path) {
  auto MBOrErr = MemoryBuffer::getFile(Path);
  if (auto EC = MBOrErr.getError()) {
    error(EC, "cannot open " + Path);
    return None;
  }
  std::unique_ptr<MemoryBuffer> &MB = *MBOrErr;
  MemoryBufferRef MBRef = MB->getMemBufferRef();
  OwningMBs.push_back(std::move(MB)); // take MB ownership

  if (Cpio)
    Cpio->append(relativeToRoot(Path), MBRef.getBuffer());

  return MBRef;
}

// Add a given library by searching it from input search paths.
void LinkerDriver::addLibrary(StringRef Name) {
  std::string Path = searchLibrary(Name);
  if (Path.empty())
    error("unable to find library -l" + Name);
  else
    addFile(Path);
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

  // This is a flag to discard all but GlobalValue names.
  // We want to enable it by default because it saves memory.
  // Disable it only when a developer option (-save-temps) is given.
  Driver->Context.setDiscardValueNames(!Config->SaveTemps);
  Driver->Context.enableDebugTypeODRUniquing();

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

  if (Config->EMachine == EM_AMDGPU && !Config->Entry.empty())
    error("-e option is not valid for AMDGPU.");

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

static StringRef
getString(opt::InputArgList &Args, unsigned Key, StringRef Default = "") {
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

static Optional<StringRef>
getZOptionValue(opt::InputArgList &Args, StringRef Key) {
  for (auto *Arg : Args.filtered(OPT_z)) {
    StringRef Value = Arg->getValue();
    size_t Pos = Value.find("=");
    if (Pos != StringRef::npos && Key == Value.substr(0, Pos))
      return Value.substr(Pos + 1);
  }
  return None;
}

void LinkerDriver::main(ArrayRef<const char *> ArgsArr) {
  ELFOptTable Parser;
  opt::InputArgList Args = Parser.parse(ArgsArr.slice(1));
  if (Args.hasArg(OPT_help)) {
    printHelp(ArgsArr[0]);
    return;
  }
  if (Args.hasArg(OPT_version))
    outs() << getVersionString();

  if (const char *Path = getReproduceOption(Args)) {
    // Note that --reproduce is a debug option so you can ignore it
    // if you are trying to understand the whole picture of the code.
    ErrorOr<CpioFile *> F = CpioFile::create(Path);
    if (F) {
      Cpio.reset(*F);
      Cpio->append("response.txt", createResponseFile(Args));
      Cpio->append("version.txt", getVersionString());
    } else
      error(F.getError(),
            Twine("--reproduce: failed to open ") + Path + ".cpio");
  }

  readConfigs(Args);
  initLLVM(Args);
  createFiles(Args);
  checkOptions(Args);
  if (HasError)
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
    error("target emulation unknown: -m or at least one .o file required");
  }
}

static UnresolvedPolicy getUnresolvedSymbolOption(opt::InputArgList &Args) {
  if (Args.hasArg(OPT_noinhibit_exec))
    return UnresolvedPolicy::Warn;
  if (Args.hasArg(OPT_no_undefined) || hasZOption(Args, "defs"))
    return UnresolvedPolicy::NoUndef;
  if (Config->Relocatable)
    return UnresolvedPolicy::Ignore;

  if (auto *Arg = Args.getLastArg(OPT_unresolved_symbols)) {
    StringRef S = Arg->getValue();
    if (S == "ignore-all" || S == "ignore-in-object-files")
      return UnresolvedPolicy::Ignore;
    if (S == "ignore-in-shared-libs" || S == "report-all")
      return UnresolvedPolicy::ReportError;
    error("unknown --unresolved-symbols value: " + S);
  }
  return UnresolvedPolicy::ReportError;
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

static bool getArg(opt::InputArgList &Args, unsigned K1, unsigned K2,
                   bool Default) {
  if (auto *Arg = Args.getLastArg(K1, K2))
    return Arg->getOption().getID() == K1;
  return Default;
}

static DiscardPolicy getDiscardOption(opt::InputArgList &Args) {
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
    error("invalid argument: " + stringize(Arg));
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

// Initializes Config members by the command line options.
void LinkerDriver::readConfigs(opt::InputArgList &Args) {
  for (auto *Arg : Args.filtered(OPT_L))
    Config->SearchPaths.push_back(Arg->getValue());

  std::vector<StringRef> RPaths;
  for (auto *Arg : Args.filtered(OPT_rpath))
    RPaths.push_back(Arg->getValue());
  if (!RPaths.empty())
    Config->RPath = llvm::join(RPaths.begin(), RPaths.end(), ":");

  Config->SectionStartMap = getSectionStartMap(Args);

  if (auto *Arg = Args.getLastArg(OPT_m)) {
    // Parse ELF{32,64}{LE,BE} and CPU type.
    StringRef S = Arg->getValue();
    std::tie(Config->EKind, Config->EMachine) = parseEmulation(S);
    Config->Emulation = S;
  }

  Config->AllowMultipleDefinition = Args.hasArg(OPT_allow_multiple_definition);
  Config->Bsymbolic = Args.hasArg(OPT_Bsymbolic);
  Config->BsymbolicFunctions = Args.hasArg(OPT_Bsymbolic_functions);
  Config->Demangle = getArg(Args, OPT_demangle, OPT_no_demangle, true);
  Config->DisableVerify = Args.hasArg(OPT_disable_verify);
  Config->Discard = getDiscardOption(Args);
  Config->EhFrameHdr = Args.hasArg(OPT_eh_frame_hdr);
  Config->EnableNewDtags = !Args.hasArg(OPT_disable_new_dtags);
  Config->ExportDynamic = Args.hasArg(OPT_export_dynamic);
  Config->FatalWarnings = Args.hasArg(OPT_fatal_warnings);
  Config->GcSections = getArg(Args, OPT_gc_sections, OPT_no_gc_sections, false);
  Config->ICF = Args.hasArg(OPT_icf);
  Config->NoGnuUnique = Args.hasArg(OPT_no_gnu_unique);
  Config->NoUndefinedVersion = Args.hasArg(OPT_no_undefined_version);
  Config->Nostdlib = Args.hasArg(OPT_nostdlib);
  Config->Pie = Args.hasArg(OPT_pie);
  Config->PrintGcSections = Args.hasArg(OPT_print_gc_sections);
  Config->Relocatable = Args.hasArg(OPT_relocatable);
  Config->SaveTemps = Args.hasArg(OPT_save_temps);
  Config->Shared = Args.hasArg(OPT_shared);
  Config->Target1Rel = getArg(Args, OPT_target1_rel, OPT_target1_abs, false);
  Config->Threads = Args.hasArg(OPT_threads);
  Config->Trace = Args.hasArg(OPT_trace);
  Config->Verbose = Args.hasArg(OPT_verbose);
  Config->WarnCommon = Args.hasArg(OPT_warn_common);

  Config->DynamicLinker = getString(Args, OPT_dynamic_linker);
  Config->Entry = getString(Args, OPT_entry);
  Config->Fini = getString(Args, OPT_fini, "_fini");
  Config->Init = getString(Args, OPT_init, "_init");
  Config->LtoAAPipeline = getString(Args, OPT_lto_aa_pipeline);
  Config->LtoNewPmPasses = getString(Args, OPT_lto_newpm_passes);
  Config->OutputFile = getString(Args, OPT_o);
  Config->SoName = getString(Args, OPT_soname);
  Config->Sysroot = getString(Args, OPT_sysroot);

  Config->Optimize = getInteger(Args, OPT_O, 1);
  Config->LtoO = getInteger(Args, OPT_lto_O, 2);
  if (Config->LtoO > 3)
    error("invalid optimization level for LTO: " + getString(Args, OPT_lto_O));
  Config->LtoJobs = getInteger(Args, OPT_lto_jobs, 1);
  if (Config->LtoJobs == 0)
    error("number of threads must be > 0");

  Config->ZCombreloc = !hasZOption(Args, "nocombreloc");
  Config->ZExecStack = hasZOption(Args, "execstack");
  Config->ZNodelete = hasZOption(Args, "nodelete");
  Config->ZNow = hasZOption(Args, "now");
  Config->ZOrigin = hasZOption(Args, "origin");
  Config->ZRelro = !hasZOption(Args, "norelro");

  if (!Config->Relocatable)
    Config->Strip = getStripOption(Args);

  if (Optional<StringRef> Value = getZOptionValue(Args, "stack-size"))
    if (Value->getAsInteger(0, Config->ZStackSize))
      error("invalid stack size: " + *Value);

  // Config->Pic is true if we are generating position-independent code.
  Config->Pic = Config->Pie || Config->Shared;

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
    } else if (S == "sha1") {
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

  Config->OFormatBinary = isOutputFormatBinary(Args);

  for (auto *Arg : Args.filtered(OPT_auxiliary))
    Config->AuxiliaryList.push_back(Arg->getValue());
  if (!Config->Shared && !Config->AuxiliaryList.empty())
    error("-f may not be used without -shared");

  for (auto *Arg : Args.filtered(OPT_undefined))
    Config->Undefined.push_back(Arg->getValue());

  Config->SortSection = getSortKind(Args);

  Config->UnresolvedSymbols = getUnresolvedSymbolOption(Args);

  if (auto *Arg = Args.getLastArg(OPT_dynamic_list))
    if (Optional<MemoryBufferRef> Buffer = readFile(Arg->getValue()))
      parseDynamicList(*Buffer);

  for (auto *Arg : Args.filtered(OPT_export_dynamic_symbol))
    Config->DynamicList.push_back(Arg->getValue());

  if (auto *Arg = Args.getLastArg(OPT_version_script))
    if (Optional<MemoryBufferRef> Buffer = readFile(Arg->getValue()))
      readVersionScript(*Buffer);
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
      addFile(Arg->getValue(), true);
      break;
    case OPT_as_needed:
      Config->AsNeeded = true;
      break;
    case OPT_format: {
      StringRef Val = Arg->getValue();
      if (Val == "elf" || Val == "default")
        Config->Binary = false;
      else if (Val == "binary")
        Config->Binary = true;
      else
        error("unknown " + Arg->getSpelling() + " format: " + Arg->getValue() +
              " (supported formats: elf, default, binary)");
      break;
    }
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
      WholeArchive = true;
      break;
    case OPT_no_whole_archive:
      WholeArchive = false;
      break;
    case OPT_start_lib:
      InLib = true;
      break;
    case OPT_end_lib:
      InLib = false;
      break;
    }
  }

  if (Files.empty() && !HasError)
    error("no input files.");

  // If -m <machine_type> was not given, infer it from object files.
  if (Config->EKind == ELFNoneKind) {
    for (InputFile *F : Files) {
      if (F->EKind == ELFNoneKind)
        continue;
      Config->EKind = F->EKind;
      Config->EMachine = F->EMachine;
      break;
    }
  }
}

// Do actual linking. Note that when this function is called,
// all linker scripts have already been parsed.
template <class ELFT> void LinkerDriver::link(opt::InputArgList &Args) {
  SymbolTable<ELFT> Symtab;
  elf::Symtab<ELFT>::X = &Symtab;

  std::unique_ptr<TargetInfo> TI(createTarget());
  Target = TI.get();
  LinkerScript<ELFT> LS;
  ScriptBase = Script<ELFT>::X = &LS;

  Config->Rela = ELFT::Is64Bits || Config->EMachine == EM_X86_64;
  Config->Mips64EL =
      (Config->EMachine == EM_MIPS && Config->EKind == ELF64LEKind);

  // Default output filename is "a.out" by the Unix tradition.
  if (Config->OutputFile.empty())
    Config->OutputFile = "a.out";

  // Handle --trace-symbol.
  for (auto *Arg : Args.filtered(OPT_trace_symbol))
    Symtab.trace(Arg->getValue());

  // Initialize Config->ImageBase.
  if (auto *Arg = Args.getLastArg(OPT_image_base)) {
    StringRef S = Arg->getValue();
    if (S.getAsInteger(0, Config->ImageBase))
      error(Arg->getSpelling() + ": number expected, but got " + S);
    else if ((Config->ImageBase % Target->PageSize) != 0)
      warning(Arg->getSpelling() + ": address isn't multiple of page size");
  } else {
    Config->ImageBase = Config->Pic ? 0 : Target->DefaultImageBase;
  }

  // Add all files to the symbol table. After this, the symbol table
  // contains all known names except a few linker-synthesized symbols.
  for (InputFile *F : Files)
    Symtab.addFile(F);

  // Add the start symbol.
  // It initializes either Config->Entry or Config->EntryAddr.
  // Note that AMDGPU binaries have no entries.
  bool HasEntryAddr = false;
  if (!Config->Entry.empty()) {
    // It is either "-e <addr>" or "-e <symbol>".
    HasEntryAddr = !Config->Entry.getAsInteger(0, Config->EntryAddr);
  } else if (!Config->Shared && !Config->Relocatable &&
             Config->EMachine != EM_AMDGPU) {
    // -e was not specified. Use the default start symbol name
    // if it is resolvable.
    Config->Entry = (Config->EMachine == EM_MIPS) ? "__start" : "_start";
  }
  if (!HasEntryAddr) {
    if (Symtab.find(Config->Entry))
      Config->EntrySym = Symtab.addUndefined(Config->Entry);
    else
      warning("entry symbol " + Config->Entry + " not found, assuming 0");
  }

  if (HasError)
    return; // There were duplicate symbols or incompatible files

  Symtab.scanUndefinedFlags();
  Symtab.scanShlibUndefined();
  Symtab.scanDynamicList();
  Symtab.scanVersionScript();

  Symtab.addCombinedLtoObject();
  if (HasError)
    return;

  for (auto *Arg : Args.filtered(OPT_wrap))
    Symtab.wrap(Arg->getValue());

  // Do size optimizations: garbage collection and identical code folding.
  if (Config->GcSections)
    markLive<ELFT>();
  if (Config->ICF)
    doIcf<ELFT>();

  // MergeInputSection::splitIntoPieces needs to be called before
  // any call of MergeInputSection::getOffset. Do that.
  for (elf::ObjectFile<ELFT> *F : Symtab.getObjectFiles()) {
    for (InputSectionBase<ELFT> *S : F->getSections()) {
      if (!S || S == &InputSection<ELFT>::Discarded || !S->Live)
        continue;
      if (S->Compressed)
        S->uncompress();
      if (auto *MS = dyn_cast<MergeInputSection<ELFT>>(S))
        MS->splitIntoPieces();
    }
  }

  // Write the result to the file.
  writeResult<ELFT>();
}
