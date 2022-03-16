//===- Driver.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "ICF.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "LinkerScript.h"
#include "MarkLive.h"
#include "OutputSections.h"
#include "ScriptParser.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "Writer.h"
#include "lld/Common/Args.h"
#include "lld/Common/Driver.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Filesystem.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Strings.h"
#include "lld/Common/TargetOptionsCommandFlags.h"
#include "lld/Common/Version.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Remarks/HotnessThresholdParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TarWriter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <utility>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::sys;
using namespace llvm::support;
using namespace lld;
using namespace lld::elf;

std::unique_ptr<Configuration> elf::config;
std::unique_ptr<LinkerDriver> elf::driver;

static void setConfigs(opt::InputArgList &args);
static void readConfigs(opt::InputArgList &args);

void elf::errorOrWarn(const Twine &msg) {
  if (config->noinhibitExec)
    warn(msg);
  else
    error(msg);
}

bool elf::link(ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
               llvm::raw_ostream &stderrOS, bool exitEarly,
               bool disableOutput) {
  // This driver-specific context will be freed later by lldMain().
  auto *ctx = new CommonLinkerContext;

  ctx->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);
  ctx->e.cleanupCallback = []() {
    inputSections.clear();
    outputSections.clear();
    memoryBuffers.clear();
    binaryFiles.clear();
    bitcodeFiles.clear();
    lazyBitcodeFiles.clear();
    objectFiles.clear();
    sharedFiles.clear();
    symAux.clear();

    tar = nullptr;
    in.reset();

    partitions.clear();
    partitions.emplace_back();

    SharedFile::vernauxNum = 0;
  };
  ctx->e.logName = args::getFilenameWithoutExe(args[0]);
  ctx->e.errorLimitExceededMsg = "too many errors emitted, stopping now (use "
                                 "--error-limit=0 to see all errors)";

  config = std::make_unique<Configuration>();
  driver = std::make_unique<LinkerDriver>();
  script = std::make_unique<LinkerScript>();
  symtab = std::make_unique<SymbolTable>();

  partitions.clear();
  partitions.emplace_back();

  config->progName = args[0];

  driver->linkerMain(args);

  return errorCount() == 0;
}

// Parses a linker -m option.
static std::tuple<ELFKind, uint16_t, uint8_t> parseEmulation(StringRef emul) {
  uint8_t osabi = 0;
  StringRef s = emul;
  if (s.endswith("_fbsd")) {
    s = s.drop_back(5);
    osabi = ELFOSABI_FREEBSD;
  }

  std::pair<ELFKind, uint16_t> ret =
      StringSwitch<std::pair<ELFKind, uint16_t>>(s)
          .Cases("aarch64elf", "aarch64linux", {ELF64LEKind, EM_AARCH64})
          .Cases("aarch64elfb", "aarch64linuxb", {ELF64BEKind, EM_AARCH64})
          .Cases("armelf", "armelf_linux_eabi", {ELF32LEKind, EM_ARM})
          .Case("elf32_x86_64", {ELF32LEKind, EM_X86_64})
          .Cases("elf32btsmip", "elf32btsmipn32", {ELF32BEKind, EM_MIPS})
          .Cases("elf32ltsmip", "elf32ltsmipn32", {ELF32LEKind, EM_MIPS})
          .Case("elf32lriscv", {ELF32LEKind, EM_RISCV})
          .Cases("elf32ppc", "elf32ppclinux", {ELF32BEKind, EM_PPC})
          .Cases("elf32lppc", "elf32lppclinux", {ELF32LEKind, EM_PPC})
          .Case("elf64btsmip", {ELF64BEKind, EM_MIPS})
          .Case("elf64ltsmip", {ELF64LEKind, EM_MIPS})
          .Case("elf64lriscv", {ELF64LEKind, EM_RISCV})
          .Case("elf64ppc", {ELF64BEKind, EM_PPC64})
          .Case("elf64lppc", {ELF64LEKind, EM_PPC64})
          .Cases("elf_amd64", "elf_x86_64", {ELF64LEKind, EM_X86_64})
          .Case("elf_i386", {ELF32LEKind, EM_386})
          .Case("elf_iamcu", {ELF32LEKind, EM_IAMCU})
          .Case("elf64_sparc", {ELF64BEKind, EM_SPARCV9})
          .Case("msp430elf", {ELF32LEKind, EM_MSP430})
          .Default({ELFNoneKind, EM_NONE});

  if (ret.first == ELFNoneKind)
    error("unknown emulation: " + emul);
  if (ret.second == EM_MSP430)
    osabi = ELFOSABI_STANDALONE;
  return std::make_tuple(ret.first, ret.second, osabi);
}

// Returns slices of MB by parsing MB as an archive file.
// Each slice consists of a member file in the archive.
std::vector<std::pair<MemoryBufferRef, uint64_t>> static getArchiveMembers(
    MemoryBufferRef mb) {
  std::unique_ptr<Archive> file =
      CHECK(Archive::create(mb),
            mb.getBufferIdentifier() + ": failed to parse archive");

  std::vector<std::pair<MemoryBufferRef, uint64_t>> v;
  Error err = Error::success();
  bool addToTar = file->isThin() && tar;
  for (const Archive::Child &c : file->children(err)) {
    MemoryBufferRef mbref =
        CHECK(c.getMemoryBufferRef(),
              mb.getBufferIdentifier() +
                  ": could not get the buffer for a child of the archive");
    if (addToTar)
      tar->append(relativeToRoot(check(c.getFullName())), mbref.getBuffer());
    v.push_back(std::make_pair(mbref, c.getChildOffset()));
  }
  if (err)
    fatal(mb.getBufferIdentifier() + ": Archive::children failed: " +
          toString(std::move(err)));

  // Take ownership of memory buffers created for members of thin archives.
  std::vector<std::unique_ptr<MemoryBuffer>> mbs = file->takeThinBuffers();
  std::move(mbs.begin(), mbs.end(), std::back_inserter(memoryBuffers));

  return v;
}

// Opens a file and create a file object. Path has to be resolved already.
void LinkerDriver::addFile(StringRef path, bool withLOption) {
  using namespace sys::fs;

  Optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer.hasValue())
    return;
  MemoryBufferRef mbref = *buffer;

  if (config->formatBinary) {
    files.push_back(make<BinaryFile>(mbref));
    return;
  }

  switch (identify_magic(mbref.getBuffer())) {
  case file_magic::unknown:
    readLinkerScript(mbref);
    return;
  case file_magic::archive: {
    if (inWholeArchive) {
      for (const auto &p : getArchiveMembers(mbref))
        files.push_back(createObjectFile(p.first, path, p.second));
      return;
    }

    auto members = getArchiveMembers(mbref);
    archiveFiles.emplace_back(path, members.size());

    // Handle archives and --start-lib/--end-lib using the same code path. This
    // scans all the ELF relocatable object files and bitcode files in the
    // archive rather than just the index file, with the benefit that the
    // symbols are only loaded once. For many projects archives see high
    // utilization rates and it is a net performance win. --start-lib scans
    // symbols in the same order that llvm-ar adds them to the index, so in the
    // common case the semantics are identical. If the archive symbol table was
    // created in a different order, or is incomplete, this strategy has
    // different semantics. Such output differences are considered user error.
    //
    // All files within the archive get the same group ID to allow mutual
    // references for --warn-backrefs.
    bool saved = InputFile::isInGroup;
    InputFile::isInGroup = true;
    for (const std::pair<MemoryBufferRef, uint64_t> &p : members) {
      auto magic = identify_magic(p.first.getBuffer());
      if (magic == file_magic::bitcode || magic == file_magic::elf_relocatable)
        files.push_back(createLazyFile(p.first, path, p.second));
      else
        warn(path + ": archive member '" + p.first.getBufferIdentifier() +
             "' is neither ET_REL nor LLVM bitcode");
    }
    InputFile::isInGroup = saved;
    if (!saved)
      ++InputFile::nextGroupId;
    return;
  }
  case file_magic::elf_shared_object:
    if (config->isStatic || config->relocatable) {
      error("attempted static link of dynamic object " + path);
      return;
    }

    // Shared objects are identified by soname. soname is (if specified)
    // DT_SONAME and falls back to filename. If a file was specified by -lfoo,
    // the directory part is ignored. Note that path may be a temporary and
    // cannot be stored into SharedFile::soName.
    path = mbref.getBufferIdentifier();
    files.push_back(
        make<SharedFile>(mbref, withLOption ? path::filename(path) : path));
    return;
  case file_magic::bitcode:
  case file_magic::elf_relocatable:
    if (inLib)
      files.push_back(createLazyFile(mbref, "", 0));
    else
      files.push_back(createObjectFile(mbref));
    break;
  default:
    error(path + ": unknown file type");
  }
}

// Add a given library by searching it from input search paths.
void LinkerDriver::addLibrary(StringRef name) {
  if (Optional<std::string> path = searchLibrary(name))
    addFile(*path, /*withLOption=*/true);
  else
    error("unable to find library -l" + name, ErrorTag::LibNotFound, {name});
}

// This function is called on startup. We need this for LTO since
// LTO calls LLVM functions to compile bitcode files to native code.
// Technically this can be delayed until we read bitcode files, but
// we don't bother to do lazily because the initialization is fast.
static void initLLVM() {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();
}

// Some command line options or some combinations of them are not allowed.
// This function checks for such errors.
static void checkOptions() {
  // The MIPS ABI as of 2016 does not support the GNU-style symbol lookup
  // table which is a relatively new feature.
  if (config->emachine == EM_MIPS && config->gnuHash)
    error("the .gnu.hash section is not compatible with the MIPS target");

  if (config->fixCortexA53Errata843419 && config->emachine != EM_AARCH64)
    error("--fix-cortex-a53-843419 is only supported on AArch64 targets");

  if (config->fixCortexA8 && config->emachine != EM_ARM)
    error("--fix-cortex-a8 is only supported on ARM targets");

  if (config->tocOptimize && config->emachine != EM_PPC64)
    error("--toc-optimize is only supported on PowerPC64 targets");

  if (config->pcRelOptimize && config->emachine != EM_PPC64)
    error("--pcrel-optimize is only supported on PowerPC64 targets");

  if (config->pie && config->shared)
    error("-shared and -pie may not be used together");

  if (!config->shared && !config->filterList.empty())
    error("-F may not be used without -shared");

  if (!config->shared && !config->auxiliaryList.empty())
    error("-f may not be used without -shared");

  if (config->strip == StripPolicy::All && config->emitRelocs)
    error("--strip-all and --emit-relocs may not be used together");

  if (config->zText && config->zIfuncNoplt)
    error("-z text and -z ifunc-noplt may not be used together");

  if (config->relocatable) {
    if (config->shared)
      error("-r and -shared may not be used together");
    if (config->gdbIndex)
      error("-r and --gdb-index may not be used together");
    if (config->icf != ICFLevel::None)
      error("-r and --icf may not be used together");
    if (config->pie)
      error("-r and -pie may not be used together");
    if (config->exportDynamic)
      error("-r and --export-dynamic may not be used together");
  }

  if (config->executeOnly) {
    if (config->emachine != EM_AARCH64)
      error("--execute-only is only supported on AArch64 targets");

    if (config->singleRoRx && !script->hasSectionsCommand)
      error("--execute-only and --no-rosegment cannot be used together");
  }

  if (config->zRetpolineplt && config->zForceIbt)
    error("-z force-ibt may not be used with -z retpolineplt");

  if (config->emachine != EM_AARCH64) {
    if (config->zPacPlt)
      error("-z pac-plt only supported on AArch64");
    if (config->zForceBti)
      error("-z force-bti only supported on AArch64");
    if (config->zBtiReport != "none")
      error("-z bti-report only supported on AArch64");
  }

  if (config->emachine != EM_386 && config->emachine != EM_X86_64 &&
      config->zCetReport != "none")
    error("-z cet-report only supported on X86 and X86_64");
}

static const char *getReproduceOption(opt::InputArgList &args) {
  if (auto *arg = args.getLastArg(OPT_reproduce))
    return arg->getValue();
  return getenv("LLD_REPRODUCE");
}

static bool hasZOption(opt::InputArgList &args, StringRef key) {
  for (auto *arg : args.filtered(OPT_z))
    if (key == arg->getValue())
      return true;
  return false;
}

static bool getZFlag(opt::InputArgList &args, StringRef k1, StringRef k2,
                     bool Default) {
  for (auto *arg : args.filtered_reverse(OPT_z)) {
    if (k1 == arg->getValue())
      return true;
    if (k2 == arg->getValue())
      return false;
  }
  return Default;
}

static SeparateSegmentKind getZSeparate(opt::InputArgList &args) {
  for (auto *arg : args.filtered_reverse(OPT_z)) {
    StringRef v = arg->getValue();
    if (v == "noseparate-code")
      return SeparateSegmentKind::None;
    if (v == "separate-code")
      return SeparateSegmentKind::Code;
    if (v == "separate-loadable-segments")
      return SeparateSegmentKind::Loadable;
  }
  return SeparateSegmentKind::None;
}

static GnuStackKind getZGnuStack(opt::InputArgList &args) {
  for (auto *arg : args.filtered_reverse(OPT_z)) {
    if (StringRef("execstack") == arg->getValue())
      return GnuStackKind::Exec;
    if (StringRef("noexecstack") == arg->getValue())
      return GnuStackKind::NoExec;
    if (StringRef("nognustack") == arg->getValue())
      return GnuStackKind::None;
  }

  return GnuStackKind::NoExec;
}

static uint8_t getZStartStopVisibility(opt::InputArgList &args) {
  for (auto *arg : args.filtered_reverse(OPT_z)) {
    std::pair<StringRef, StringRef> kv = StringRef(arg->getValue()).split('=');
    if (kv.first == "start-stop-visibility") {
      if (kv.second == "default")
        return STV_DEFAULT;
      else if (kv.second == "internal")
        return STV_INTERNAL;
      else if (kv.second == "hidden")
        return STV_HIDDEN;
      else if (kv.second == "protected")
        return STV_PROTECTED;
      error("unknown -z start-stop-visibility= value: " + StringRef(kv.second));
    }
  }
  return STV_PROTECTED;
}

constexpr const char *knownZFlags[] = {
    "combreloc",
    "copyreloc",
    "defs",
    "execstack",
    "force-bti",
    "force-ibt",
    "global",
    "hazardplt",
    "ifunc-noplt",
    "initfirst",
    "interpose",
    "keep-text-section-prefix",
    "lazy",
    "muldefs",
    "nocombreloc",
    "nocopyreloc",
    "nodefaultlib",
    "nodelete",
    "nodlopen",
    "noexecstack",
    "nognustack",
    "nokeep-text-section-prefix",
    "nopack-relative-relocs",
    "norelro",
    "noseparate-code",
    "nostart-stop-gc",
    "notext",
    "now",
    "origin",
    "pac-plt",
    "pack-relative-relocs",
    "rel",
    "rela",
    "relro",
    "retpolineplt",
    "rodynamic",
    "separate-code",
    "separate-loadable-segments",
    "shstk",
    "start-stop-gc",
    "text",
    "undefs",
    "wxneeded",
};

static bool isKnownZFlag(StringRef s) {
  return llvm::is_contained(knownZFlags, s) ||
         s.startswith("common-page-size=") || s.startswith("bti-report=") ||
         s.startswith("cet-report=") ||
         s.startswith("dead-reloc-in-nonalloc=") ||
         s.startswith("max-page-size=") || s.startswith("stack-size=") ||
         s.startswith("start-stop-visibility=");
}

// Report a warning for an unknown -z option.
static void checkZOptions(opt::InputArgList &args) {
  for (auto *arg : args.filtered(OPT_z))
    if (!isKnownZFlag(arg->getValue()))
      warn("unknown -z value: " + StringRef(arg->getValue()));
}

void LinkerDriver::linkerMain(ArrayRef<const char *> argsArr) {
  ELFOptTable parser;
  opt::InputArgList args = parser.parse(argsArr.slice(1));

  // Interpret these flags early because error()/warn() depend on them.
  errorHandler().errorLimit = args::getInteger(args, OPT_error_limit, 20);
  errorHandler().fatalWarnings =
      args.hasFlag(OPT_fatal_warnings, OPT_no_fatal_warnings, false);
  checkZOptions(args);

  // Handle -help
  if (args.hasArg(OPT_help)) {
    printHelp();
    return;
  }

  // Handle -v or -version.
  //
  // A note about "compatible with GNU linkers" message: this is a hack for
  // scripts generated by GNU Libtool up to 2021-10 to recognize LLD as
  // a GNU compatible linker. See
  // <https://lists.gnu.org/archive/html/libtool/2017-01/msg00007.html>.
  //
  // This is somewhat ugly hack, but in reality, we had no choice other
  // than doing this. Considering the very long release cycle of Libtool,
  // it is not easy to improve it to recognize LLD as a GNU compatible
  // linker in a timely manner. Even if we can make it, there are still a
  // lot of "configure" scripts out there that are generated by old version
  // of Libtool. We cannot convince every software developer to migrate to
  // the latest version and re-generate scripts. So we have this hack.
  if (args.hasArg(OPT_v) || args.hasArg(OPT_version))
    message(getLLDVersion() + " (compatible with GNU linkers)");

  if (const char *path = getReproduceOption(args)) {
    // Note that --reproduce is a debug option so you can ignore it
    // if you are trying to understand the whole picture of the code.
    Expected<std::unique_ptr<TarWriter>> errOrWriter =
        TarWriter::create(path, path::stem(path));
    if (errOrWriter) {
      tar = std::move(*errOrWriter);
      tar->append("response.txt", createResponseFile(args));
      tar->append("version.txt", getLLDVersion() + "\n");
      StringRef ltoSampleProfile = args.getLastArgValue(OPT_lto_sample_profile);
      if (!ltoSampleProfile.empty())
        readFile(ltoSampleProfile);
    } else {
      error("--reproduce: " + toString(errOrWriter.takeError()));
    }
  }

  readConfigs(args);

  // The behavior of -v or --version is a bit strange, but this is
  // needed for compatibility with GNU linkers.
  if (args.hasArg(OPT_v) && !args.hasArg(OPT_INPUT))
    return;
  if (args.hasArg(OPT_version))
    return;

  // Initialize time trace profiler.
  if (config->timeTraceEnabled)
    timeTraceProfilerInitialize(config->timeTraceGranularity, config->progName);

  {
    llvm::TimeTraceScope timeScope("ExecuteLinker");

    initLLVM();
    createFiles(args);
    if (errorCount())
      return;

    inferMachineType();
    setConfigs(args);
    checkOptions();
    if (errorCount())
      return;

    // The Target instance handles target-specific stuff, such as applying
    // relocations or writing a PLT section. It also contains target-dependent
    // values such as a default image base address.
    target = getTarget();

    link(args);
  }

  if (config->timeTraceEnabled) {
    checkError(timeTraceProfilerWrite(
        args.getLastArgValue(OPT_time_trace_file_eq).str(),
        config->outputFile));
    timeTraceProfilerCleanup();
  }
}

static std::string getRpath(opt::InputArgList &args) {
  std::vector<StringRef> v = args::getStrings(args, OPT_rpath);
  return llvm::join(v.begin(), v.end(), ":");
}

// Determines what we should do if there are remaining unresolved
// symbols after the name resolution.
static void setUnresolvedSymbolPolicy(opt::InputArgList &args) {
  UnresolvedPolicy errorOrWarn = args.hasFlag(OPT_error_unresolved_symbols,
                                              OPT_warn_unresolved_symbols, true)
                                     ? UnresolvedPolicy::ReportError
                                     : UnresolvedPolicy::Warn;
  // -shared implies --unresolved-symbols=ignore-all because missing
  // symbols are likely to be resolved at runtime.
  bool diagRegular = !config->shared, diagShlib = !config->shared;

  for (const opt::Arg *arg : args) {
    switch (arg->getOption().getID()) {
    case OPT_unresolved_symbols: {
      StringRef s = arg->getValue();
      if (s == "ignore-all") {
        diagRegular = false;
        diagShlib = false;
      } else if (s == "ignore-in-object-files") {
        diagRegular = false;
        diagShlib = true;
      } else if (s == "ignore-in-shared-libs") {
        diagRegular = true;
        diagShlib = false;
      } else if (s == "report-all") {
        diagRegular = true;
        diagShlib = true;
      } else {
        error("unknown --unresolved-symbols value: " + s);
      }
      break;
    }
    case OPT_no_undefined:
      diagRegular = true;
      break;
    case OPT_z:
      if (StringRef(arg->getValue()) == "defs")
        diagRegular = true;
      else if (StringRef(arg->getValue()) == "undefs")
        diagRegular = false;
      break;
    case OPT_allow_shlib_undefined:
      diagShlib = false;
      break;
    case OPT_no_allow_shlib_undefined:
      diagShlib = true;
      break;
    }
  }

  config->unresolvedSymbols =
      diagRegular ? errorOrWarn : UnresolvedPolicy::Ignore;
  config->unresolvedSymbolsInShlib =
      diagShlib ? errorOrWarn : UnresolvedPolicy::Ignore;
}

static Target2Policy getTarget2(opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_target2, "got-rel");
  if (s == "rel")
    return Target2Policy::Rel;
  if (s == "abs")
    return Target2Policy::Abs;
  if (s == "got-rel")
    return Target2Policy::GotRel;
  error("unknown --target2 option: " + s);
  return Target2Policy::GotRel;
}

static bool isOutputFormatBinary(opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_oformat, "elf");
  if (s == "binary")
    return true;
  if (!s.startswith("elf"))
    error("unknown --oformat value: " + s);
  return false;
}

static DiscardPolicy getDiscard(opt::InputArgList &args) {
  auto *arg =
      args.getLastArg(OPT_discard_all, OPT_discard_locals, OPT_discard_none);
  if (!arg)
    return DiscardPolicy::Default;
  if (arg->getOption().getID() == OPT_discard_all)
    return DiscardPolicy::All;
  if (arg->getOption().getID() == OPT_discard_locals)
    return DiscardPolicy::Locals;
  return DiscardPolicy::None;
}

static StringRef getDynamicLinker(opt::InputArgList &args) {
  auto *arg = args.getLastArg(OPT_dynamic_linker, OPT_no_dynamic_linker);
  if (!arg)
    return "";
  if (arg->getOption().getID() == OPT_no_dynamic_linker) {
    // --no-dynamic-linker suppresses undefined weak symbols in .dynsym
    config->noDynamicLinker = true;
    return "";
  }
  return arg->getValue();
}

static ICFLevel getICF(opt::InputArgList &args) {
  auto *arg = args.getLastArg(OPT_icf_none, OPT_icf_safe, OPT_icf_all);
  if (!arg || arg->getOption().getID() == OPT_icf_none)
    return ICFLevel::None;
  if (arg->getOption().getID() == OPT_icf_safe)
    return ICFLevel::Safe;
  return ICFLevel::All;
}

static StripPolicy getStrip(opt::InputArgList &args) {
  if (args.hasArg(OPT_relocatable))
    return StripPolicy::None;

  auto *arg = args.getLastArg(OPT_strip_all, OPT_strip_debug);
  if (!arg)
    return StripPolicy::None;
  if (arg->getOption().getID() == OPT_strip_all)
    return StripPolicy::All;
  return StripPolicy::Debug;
}

static uint64_t parseSectionAddress(StringRef s, opt::InputArgList &args,
                                    const opt::Arg &arg) {
  uint64_t va = 0;
  if (s.startswith("0x"))
    s = s.drop_front(2);
  if (!to_integer(s, va, 16))
    error("invalid argument: " + arg.getAsString(args));
  return va;
}

static StringMap<uint64_t> getSectionStartMap(opt::InputArgList &args) {
  StringMap<uint64_t> ret;
  for (auto *arg : args.filtered(OPT_section_start)) {
    StringRef name;
    StringRef addr;
    std::tie(name, addr) = StringRef(arg->getValue()).split('=');
    ret[name] = parseSectionAddress(addr, args, *arg);
  }

  if (auto *arg = args.getLastArg(OPT_Ttext))
    ret[".text"] = parseSectionAddress(arg->getValue(), args, *arg);
  if (auto *arg = args.getLastArg(OPT_Tdata))
    ret[".data"] = parseSectionAddress(arg->getValue(), args, *arg);
  if (auto *arg = args.getLastArg(OPT_Tbss))
    ret[".bss"] = parseSectionAddress(arg->getValue(), args, *arg);
  return ret;
}

static SortSectionPolicy getSortSection(opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_sort_section);
  if (s == "alignment")
    return SortSectionPolicy::Alignment;
  if (s == "name")
    return SortSectionPolicy::Name;
  if (!s.empty())
    error("unknown --sort-section rule: " + s);
  return SortSectionPolicy::Default;
}

static OrphanHandlingPolicy getOrphanHandling(opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_orphan_handling, "place");
  if (s == "warn")
    return OrphanHandlingPolicy::Warn;
  if (s == "error")
    return OrphanHandlingPolicy::Error;
  if (s != "place")
    error("unknown --orphan-handling mode: " + s);
  return OrphanHandlingPolicy::Place;
}

// Parse --build-id or --build-id=<style>. We handle "tree" as a
// synonym for "sha1" because all our hash functions including
// --build-id=sha1 are actually tree hashes for performance reasons.
static std::pair<BuildIdKind, std::vector<uint8_t>>
getBuildId(opt::InputArgList &args) {
  auto *arg = args.getLastArg(OPT_build_id, OPT_build_id_eq);
  if (!arg)
    return {BuildIdKind::None, {}};

  if (arg->getOption().getID() == OPT_build_id)
    return {BuildIdKind::Fast, {}};

  StringRef s = arg->getValue();
  if (s == "fast")
    return {BuildIdKind::Fast, {}};
  if (s == "md5")
    return {BuildIdKind::Md5, {}};
  if (s == "sha1" || s == "tree")
    return {BuildIdKind::Sha1, {}};
  if (s == "uuid")
    return {BuildIdKind::Uuid, {}};
  if (s.startswith("0x"))
    return {BuildIdKind::Hexstring, parseHex(s.substr(2))};

  if (s != "none")
    error("unknown --build-id style: " + s);
  return {BuildIdKind::None, {}};
}

static std::pair<bool, bool> getPackDynRelocs(opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_pack_dyn_relocs, "none");
  if (s == "android")
    return {true, false};
  if (s == "relr")
    return {false, true};
  if (s == "android+relr")
    return {true, true};

  if (s != "none")
    error("unknown --pack-dyn-relocs format: " + s);
  return {false, false};
}

static void readCallGraph(MemoryBufferRef mb) {
  // Build a map from symbol name to section
  DenseMap<StringRef, Symbol *> map;
  for (ELFFileBase *file : objectFiles)
    for (Symbol *sym : file->getSymbols())
      map[sym->getName()] = sym;

  auto findSection = [&](StringRef name) -> InputSectionBase * {
    Symbol *sym = map.lookup(name);
    if (!sym) {
      if (config->warnSymbolOrdering)
        warn(mb.getBufferIdentifier() + ": no such symbol: " + name);
      return nullptr;
    }
    maybeWarnUnorderableSymbol(sym);

    if (Defined *dr = dyn_cast_or_null<Defined>(sym))
      return dyn_cast_or_null<InputSectionBase>(dr->section);
    return nullptr;
  };

  for (StringRef line : args::getLines(mb)) {
    SmallVector<StringRef, 3> fields;
    line.split(fields, ' ');
    uint64_t count;

    if (fields.size() != 3 || !to_integer(fields[2], count)) {
      error(mb.getBufferIdentifier() + ": parse error");
      return;
    }

    if (InputSectionBase *from = findSection(fields[0]))
      if (InputSectionBase *to = findSection(fields[1]))
        config->callGraphProfile[std::make_pair(from, to)] += count;
  }
}

// If SHT_LLVM_CALL_GRAPH_PROFILE and its relocation section exist, returns
// true and populates cgProfile and symbolIndices.
template <class ELFT>
static bool
processCallGraphRelocations(SmallVector<uint32_t, 32> &symbolIndices,
                            ArrayRef<typename ELFT::CGProfile> &cgProfile,
                            ObjFile<ELFT> *inputObj) {
  if (inputObj->cgProfileSectionIndex == SHN_UNDEF)
    return false;

  ArrayRef<Elf_Shdr_Impl<ELFT>> objSections =
      inputObj->template getELFShdrs<ELFT>();
  symbolIndices.clear();
  const ELFFile<ELFT> &obj = inputObj->getObj();
  cgProfile =
      check(obj.template getSectionContentsAsArray<typename ELFT::CGProfile>(
          objSections[inputObj->cgProfileSectionIndex]));

  for (size_t i = 0, e = objSections.size(); i < e; ++i) {
    const Elf_Shdr_Impl<ELFT> &sec = objSections[i];
    if (sec.sh_info == inputObj->cgProfileSectionIndex) {
      if (sec.sh_type == SHT_RELA) {
        ArrayRef<typename ELFT::Rela> relas =
            CHECK(obj.relas(sec), "could not retrieve cg profile rela section");
        for (const typename ELFT::Rela &rel : relas)
          symbolIndices.push_back(rel.getSymbol(config->isMips64EL));
        break;
      }
      if (sec.sh_type == SHT_REL) {
        ArrayRef<typename ELFT::Rel> rels =
            CHECK(obj.rels(sec), "could not retrieve cg profile rel section");
        for (const typename ELFT::Rel &rel : rels)
          symbolIndices.push_back(rel.getSymbol(config->isMips64EL));
        break;
      }
    }
  }
  if (symbolIndices.empty())
    warn("SHT_LLVM_CALL_GRAPH_PROFILE exists, but relocation section doesn't");
  return !symbolIndices.empty();
}

template <class ELFT> static void readCallGraphsFromObjectFiles() {
  SmallVector<uint32_t, 32> symbolIndices;
  ArrayRef<typename ELFT::CGProfile> cgProfile;
  for (auto file : objectFiles) {
    auto *obj = cast<ObjFile<ELFT>>(file);
    if (!processCallGraphRelocations(symbolIndices, cgProfile, obj))
      continue;

    if (symbolIndices.size() != cgProfile.size() * 2)
      fatal("number of relocations doesn't match Weights");

    for (uint32_t i = 0, size = cgProfile.size(); i < size; ++i) {
      const Elf_CGProfile_Impl<ELFT> &cgpe = cgProfile[i];
      uint32_t fromIndex = symbolIndices[i * 2];
      uint32_t toIndex = symbolIndices[i * 2 + 1];
      auto *fromSym = dyn_cast<Defined>(&obj->getSymbol(fromIndex));
      auto *toSym = dyn_cast<Defined>(&obj->getSymbol(toIndex));
      if (!fromSym || !toSym)
        continue;

      auto *from = dyn_cast_or_null<InputSectionBase>(fromSym->section);
      auto *to = dyn_cast_or_null<InputSectionBase>(toSym->section);
      if (from && to)
        config->callGraphProfile[{from, to}] += cgpe.cgp_weight;
    }
  }
}

static bool getCompressDebugSections(opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_compress_debug_sections, "none");
  if (s == "none")
    return false;
  if (s != "zlib")
    error("unknown --compress-debug-sections value: " + s);
  if (!zlib::isAvailable())
    error("--compress-debug-sections: zlib is not available");
  return true;
}

static StringRef getAliasSpelling(opt::Arg *arg) {
  if (const opt::Arg *alias = arg->getAlias())
    return alias->getSpelling();
  return arg->getSpelling();
}

static std::pair<StringRef, StringRef> getOldNewOptions(opt::InputArgList &args,
                                                        unsigned id) {
  auto *arg = args.getLastArg(id);
  if (!arg)
    return {"", ""};

  StringRef s = arg->getValue();
  std::pair<StringRef, StringRef> ret = s.split(';');
  if (ret.second.empty())
    error(getAliasSpelling(arg) + " expects 'old;new' format, but got " + s);
  return ret;
}

// Parse the symbol ordering file and warn for any duplicate entries.
static std::vector<StringRef> getSymbolOrderingFile(MemoryBufferRef mb) {
  SetVector<StringRef> names;
  for (StringRef s : args::getLines(mb))
    if (!names.insert(s) && config->warnSymbolOrdering)
      warn(mb.getBufferIdentifier() + ": duplicate ordered symbol: " + s);

  return names.takeVector();
}

static bool getIsRela(opt::InputArgList &args) {
  // If -z rel or -z rela is specified, use the last option.
  for (auto *arg : args.filtered_reverse(OPT_z)) {
    StringRef s(arg->getValue());
    if (s == "rel")
      return false;
    if (s == "rela")
      return true;
  }

  // Otherwise use the psABI defined relocation entry format.
  uint16_t m = config->emachine;
  return m == EM_AARCH64 || m == EM_AMDGPU || m == EM_HEXAGON || m == EM_PPC ||
         m == EM_PPC64 || m == EM_RISCV || m == EM_X86_64;
}

static void parseClangOption(StringRef opt, const Twine &msg) {
  std::string err;
  raw_string_ostream os(err);

  const char *argv[] = {config->progName.data(), opt.data()};
  if (cl::ParseCommandLineOptions(2, argv, "", &os))
    return;
  os.flush();
  error(msg + ": " + StringRef(err).trim());
}

// Checks the parameter of the bti-report and cet-report options.
static bool isValidReportString(StringRef arg) {
  return arg == "none" || arg == "warning" || arg == "error";
}

// Initializes Config members by the command line options.
static void readConfigs(opt::InputArgList &args) {
  errorHandler().verbose = args.hasArg(OPT_verbose);
  errorHandler().vsDiagnostics =
      args.hasArg(OPT_visual_studio_diagnostics_format, false);

  config->allowMultipleDefinition =
      args.hasFlag(OPT_allow_multiple_definition,
                   OPT_no_allow_multiple_definition, false) ||
      hasZOption(args, "muldefs");
  config->auxiliaryList = args::getStrings(args, OPT_auxiliary);
  if (opt::Arg *arg =
          args.getLastArg(OPT_Bno_symbolic, OPT_Bsymbolic_non_weak_functions,
                          OPT_Bsymbolic_functions, OPT_Bsymbolic)) {
    if (arg->getOption().matches(OPT_Bsymbolic_non_weak_functions))
      config->bsymbolic = BsymbolicKind::NonWeakFunctions;
    else if (arg->getOption().matches(OPT_Bsymbolic_functions))
      config->bsymbolic = BsymbolicKind::Functions;
    else if (arg->getOption().matches(OPT_Bsymbolic))
      config->bsymbolic = BsymbolicKind::All;
  }
  config->checkSections =
      args.hasFlag(OPT_check_sections, OPT_no_check_sections, true);
  config->chroot = args.getLastArgValue(OPT_chroot);
  config->compressDebugSections = getCompressDebugSections(args);
  config->cref = args.hasArg(OPT_cref);
  config->optimizeBBJumps =
      args.hasFlag(OPT_optimize_bb_jumps, OPT_no_optimize_bb_jumps, false);
  config->demangle = args.hasFlag(OPT_demangle, OPT_no_demangle, true);
  config->dependencyFile = args.getLastArgValue(OPT_dependency_file);
  config->dependentLibraries = args.hasFlag(OPT_dependent_libraries, OPT_no_dependent_libraries, true);
  config->disableVerify = args.hasArg(OPT_disable_verify);
  config->discard = getDiscard(args);
  config->dwoDir = args.getLastArgValue(OPT_plugin_opt_dwo_dir_eq);
  config->dynamicLinker = getDynamicLinker(args);
  config->ehFrameHdr =
      args.hasFlag(OPT_eh_frame_hdr, OPT_no_eh_frame_hdr, false);
  config->emitLLVM = args.hasArg(OPT_plugin_opt_emit_llvm, false);
  config->emitRelocs = args.hasArg(OPT_emit_relocs);
  config->callGraphProfileSort = args.hasFlag(
      OPT_call_graph_profile_sort, OPT_no_call_graph_profile_sort, true);
  config->enableNewDtags =
      args.hasFlag(OPT_enable_new_dtags, OPT_disable_new_dtags, true);
  config->entry = args.getLastArgValue(OPT_entry);

  errorHandler().errorHandlingScript =
      args.getLastArgValue(OPT_error_handling_script);

  config->executeOnly =
      args.hasFlag(OPT_execute_only, OPT_no_execute_only, false);
  config->exportDynamic =
      args.hasFlag(OPT_export_dynamic, OPT_no_export_dynamic, false) ||
      args.hasArg(OPT_shared);
  config->filterList = args::getStrings(args, OPT_filter);
  config->fini = args.getLastArgValue(OPT_fini, "_fini");
  config->fixCortexA53Errata843419 = args.hasArg(OPT_fix_cortex_a53_843419) &&
                                     !args.hasArg(OPT_relocatable);
  config->fixCortexA8 =
      args.hasArg(OPT_fix_cortex_a8) && !args.hasArg(OPT_relocatable);
  config->fortranCommon =
      args.hasFlag(OPT_fortran_common, OPT_no_fortran_common, true);
  config->gcSections = args.hasFlag(OPT_gc_sections, OPT_no_gc_sections, false);
  config->gnuUnique = args.hasFlag(OPT_gnu_unique, OPT_no_gnu_unique, true);
  config->gdbIndex = args.hasFlag(OPT_gdb_index, OPT_no_gdb_index, false);
  config->icf = getICF(args);
  config->ignoreDataAddressEquality =
      args.hasArg(OPT_ignore_data_address_equality);
  config->ignoreFunctionAddressEquality =
      args.hasArg(OPT_ignore_function_address_equality);
  config->init = args.getLastArgValue(OPT_init, "_init");
  config->ltoAAPipeline = args.getLastArgValue(OPT_lto_aa_pipeline);
  config->ltoCSProfileGenerate = args.hasArg(OPT_lto_cs_profile_generate);
  config->ltoCSProfileFile = args.getLastArgValue(OPT_lto_cs_profile_file);
  config->ltoPGOWarnMismatch = args.hasFlag(OPT_lto_pgo_warn_mismatch,
                                            OPT_no_lto_pgo_warn_mismatch, true);
  config->ltoDebugPassManager = args.hasArg(OPT_lto_debug_pass_manager);
  config->ltoEmitAsm = args.hasArg(OPT_lto_emit_asm);
  config->ltoNewPassManager =
      args.hasFlag(OPT_no_lto_legacy_pass_manager, OPT_lto_legacy_pass_manager,
                   LLVM_ENABLE_NEW_PASS_MANAGER);
  config->ltoNewPmPasses = args.getLastArgValue(OPT_lto_newpm_passes);
  config->ltoWholeProgramVisibility =
      args.hasFlag(OPT_lto_whole_program_visibility,
                   OPT_no_lto_whole_program_visibility, false);
  config->ltoo = args::getInteger(args, OPT_lto_O, 2);
  config->ltoObjPath = args.getLastArgValue(OPT_lto_obj_path_eq);
  config->ltoPartitions = args::getInteger(args, OPT_lto_partitions, 1);
  config->ltoSampleProfile = args.getLastArgValue(OPT_lto_sample_profile);
  config->ltoBasicBlockSections =
      args.getLastArgValue(OPT_lto_basic_block_sections);
  config->ltoUniqueBasicBlockSectionNames =
      args.hasFlag(OPT_lto_unique_basic_block_section_names,
                   OPT_no_lto_unique_basic_block_section_names, false);
  config->mapFile = args.getLastArgValue(OPT_Map);
  config->mipsGotSize = args::getInteger(args, OPT_mips_got_size, 0xfff0);
  config->mergeArmExidx =
      args.hasFlag(OPT_merge_exidx_entries, OPT_no_merge_exidx_entries, true);
  config->mmapOutputFile =
      args.hasFlag(OPT_mmap_output_file, OPT_no_mmap_output_file, true);
  config->nmagic = args.hasFlag(OPT_nmagic, OPT_no_nmagic, false);
  config->noinhibitExec = args.hasArg(OPT_noinhibit_exec);
  config->nostdlib = args.hasArg(OPT_nostdlib);
  config->oFormatBinary = isOutputFormatBinary(args);
  config->omagic = args.hasFlag(OPT_omagic, OPT_no_omagic, false);
  config->optRemarksFilename = args.getLastArgValue(OPT_opt_remarks_filename);

  // Parse remarks hotness threshold. Valid value is either integer or 'auto'.
  if (auto *arg = args.getLastArg(OPT_opt_remarks_hotness_threshold)) {
    auto resultOrErr = remarks::parseHotnessThresholdOption(arg->getValue());
    if (!resultOrErr)
      error(arg->getSpelling() + ": invalid argument '" + arg->getValue() +
            "', only integer or 'auto' is supported");
    else
      config->optRemarksHotnessThreshold = *resultOrErr;
  }

  config->optRemarksPasses = args.getLastArgValue(OPT_opt_remarks_passes);
  config->optRemarksWithHotness = args.hasArg(OPT_opt_remarks_with_hotness);
  config->optRemarksFormat = args.getLastArgValue(OPT_opt_remarks_format);
  config->optimize = args::getInteger(args, OPT_O, 1);
  config->orphanHandling = getOrphanHandling(args);
  config->outputFile = args.getLastArgValue(OPT_o);
  config->pie = args.hasFlag(OPT_pie, OPT_no_pie, false);
  config->printIcfSections =
      args.hasFlag(OPT_print_icf_sections, OPT_no_print_icf_sections, false);
  config->printGcSections =
      args.hasFlag(OPT_print_gc_sections, OPT_no_print_gc_sections, false);
  config->printArchiveStats = args.getLastArgValue(OPT_print_archive_stats);
  config->printSymbolOrder =
      args.getLastArgValue(OPT_print_symbol_order);
  config->relax = args.hasFlag(OPT_relax, OPT_no_relax, true);
  config->rpath = getRpath(args);
  config->relocatable = args.hasArg(OPT_relocatable);
  config->saveTemps = args.hasArg(OPT_save_temps);
  config->searchPaths = args::getStrings(args, OPT_library_path);
  config->sectionStartMap = getSectionStartMap(args);
  config->shared = args.hasArg(OPT_shared);
  config->singleRoRx = !args.hasFlag(OPT_rosegment, OPT_no_rosegment, true);
  config->soName = args.getLastArgValue(OPT_soname);
  config->sortSection = getSortSection(args);
  config->splitStackAdjustSize = args::getInteger(args, OPT_split_stack_adjust_size, 16384);
  config->strip = getStrip(args);
  config->sysroot = args.getLastArgValue(OPT_sysroot);
  config->target1Rel = args.hasFlag(OPT_target1_rel, OPT_target1_abs, false);
  config->target2 = getTarget2(args);
  config->thinLTOCacheDir = args.getLastArgValue(OPT_thinlto_cache_dir);
  config->thinLTOCachePolicy = CHECK(
      parseCachePruningPolicy(args.getLastArgValue(OPT_thinlto_cache_policy)),
      "--thinlto-cache-policy: invalid cache policy");
  config->thinLTOEmitImportsFiles = args.hasArg(OPT_thinlto_emit_imports_files);
  config->thinLTOIndexOnly = args.hasArg(OPT_thinlto_index_only) ||
                             args.hasArg(OPT_thinlto_index_only_eq);
  config->thinLTOIndexOnlyArg = args.getLastArgValue(OPT_thinlto_index_only_eq);
  config->thinLTOObjectSuffixReplace =
      getOldNewOptions(args, OPT_thinlto_object_suffix_replace_eq);
  config->thinLTOPrefixReplace =
      getOldNewOptions(args, OPT_thinlto_prefix_replace_eq);
  config->thinLTOModulesToCompile =
      args::getStrings(args, OPT_thinlto_single_module_eq);
  config->timeTraceEnabled = args.hasArg(OPT_time_trace);
  config->timeTraceGranularity =
      args::getInteger(args, OPT_time_trace_granularity, 500);
  config->trace = args.hasArg(OPT_trace);
  config->undefined = args::getStrings(args, OPT_undefined);
  config->undefinedVersion =
      args.hasFlag(OPT_undefined_version, OPT_no_undefined_version, true);
  config->unique = args.hasArg(OPT_unique);
  config->useAndroidRelrTags = args.hasFlag(
      OPT_use_android_relr_tags, OPT_no_use_android_relr_tags, false);
  config->warnBackrefs =
      args.hasFlag(OPT_warn_backrefs, OPT_no_warn_backrefs, false);
  config->warnCommon = args.hasFlag(OPT_warn_common, OPT_no_warn_common, false);
  config->warnSymbolOrdering =
      args.hasFlag(OPT_warn_symbol_ordering, OPT_no_warn_symbol_ordering, true);
  config->whyExtract = args.getLastArgValue(OPT_why_extract);
  config->zCombreloc = getZFlag(args, "combreloc", "nocombreloc", true);
  config->zCopyreloc = getZFlag(args, "copyreloc", "nocopyreloc", true);
  config->zForceBti = hasZOption(args, "force-bti");
  config->zForceIbt = hasZOption(args, "force-ibt");
  config->zGlobal = hasZOption(args, "global");
  config->zGnustack = getZGnuStack(args);
  config->zHazardplt = hasZOption(args, "hazardplt");
  config->zIfuncNoplt = hasZOption(args, "ifunc-noplt");
  config->zInitfirst = hasZOption(args, "initfirst");
  config->zInterpose = hasZOption(args, "interpose");
  config->zKeepTextSectionPrefix = getZFlag(
      args, "keep-text-section-prefix", "nokeep-text-section-prefix", false);
  config->zNodefaultlib = hasZOption(args, "nodefaultlib");
  config->zNodelete = hasZOption(args, "nodelete");
  config->zNodlopen = hasZOption(args, "nodlopen");
  config->zNow = getZFlag(args, "now", "lazy", false);
  config->zOrigin = hasZOption(args, "origin");
  config->zPacPlt = hasZOption(args, "pac-plt");
  config->zRelro = getZFlag(args, "relro", "norelro", true);
  config->zRetpolineplt = hasZOption(args, "retpolineplt");
  config->zRodynamic = hasZOption(args, "rodynamic");
  config->zSeparate = getZSeparate(args);
  config->zShstk = hasZOption(args, "shstk");
  config->zStackSize = args::getZOptionValue(args, OPT_z, "stack-size", 0);
  config->zStartStopGC =
      getZFlag(args, "start-stop-gc", "nostart-stop-gc", true);
  config->zStartStopVisibility = getZStartStopVisibility(args);
  config->zText = getZFlag(args, "text", "notext", true);
  config->zWxneeded = hasZOption(args, "wxneeded");
  setUnresolvedSymbolPolicy(args);
  config->power10Stubs = args.getLastArgValue(OPT_power10_stubs_eq) != "no";

  if (opt::Arg *arg = args.getLastArg(OPT_eb, OPT_el)) {
    if (arg->getOption().matches(OPT_eb))
      config->optEB = true;
    else
      config->optEL = true;
  }

  for (opt::Arg *arg : args.filtered(OPT_shuffle_sections)) {
    constexpr StringRef errPrefix = "--shuffle-sections=: ";
    std::pair<StringRef, StringRef> kv = StringRef(arg->getValue()).split('=');
    if (kv.first.empty() || kv.second.empty()) {
      error(errPrefix + "expected <section_glob>=<seed>, but got '" +
            arg->getValue() + "'");
      continue;
    }
    // Signed so that <section_glob>=-1 is allowed.
    int64_t v;
    if (!to_integer(kv.second, v))
      error(errPrefix + "expected an integer, but got '" + kv.second + "'");
    else if (Expected<GlobPattern> pat = GlobPattern::create(kv.first))
      config->shuffleSections.emplace_back(std::move(*pat), uint32_t(v));
    else
      error(errPrefix + toString(pat.takeError()));
  }

  auto reports = {std::make_pair("bti-report", &config->zBtiReport),
                  std::make_pair("cet-report", &config->zCetReport)};
  for (opt::Arg *arg : args.filtered(OPT_z)) {
    std::pair<StringRef, StringRef> option =
        StringRef(arg->getValue()).split('=');
    for (auto reportArg : reports) {
      if (option.first != reportArg.first)
        continue;
      if (!isValidReportString(option.second)) {
        error(Twine("-z ") + reportArg.first + "= parameter " + option.second +
              " is not recognized");
        continue;
      }
      *reportArg.second = option.second;
    }
  }

  for (opt::Arg *arg : args.filtered(OPT_z)) {
    std::pair<StringRef, StringRef> option =
        StringRef(arg->getValue()).split('=');
    if (option.first != "dead-reloc-in-nonalloc")
      continue;
    constexpr StringRef errPrefix = "-z dead-reloc-in-nonalloc=: ";
    std::pair<StringRef, StringRef> kv = option.second.split('=');
    if (kv.first.empty() || kv.second.empty()) {
      error(errPrefix + "expected <section_glob>=<value>");
      continue;
    }
    uint64_t v;
    if (!to_integer(kv.second, v))
      error(errPrefix + "expected a non-negative integer, but got '" +
            kv.second + "'");
    else if (Expected<GlobPattern> pat = GlobPattern::create(kv.first))
      config->deadRelocInNonAlloc.emplace_back(std::move(*pat), v);
    else
      error(errPrefix + toString(pat.takeError()));
  }

  cl::ResetAllOptionOccurrences();

  // Parse LTO options.
  if (auto *arg = args.getLastArg(OPT_plugin_opt_mcpu_eq))
    parseClangOption(saver().save("-mcpu=" + StringRef(arg->getValue())),
                     arg->getSpelling());

  for (opt::Arg *arg : args.filtered(OPT_plugin_opt_eq_minus))
    parseClangOption(std::string("-") + arg->getValue(), arg->getSpelling());

  // GCC collect2 passes -plugin-opt=path/to/lto-wrapper with an absolute or
  // relative path. Just ignore. If not ended with "lto-wrapper", consider it an
  // unsupported LLVMgold.so option and error.
  for (opt::Arg *arg : args.filtered(OPT_plugin_opt_eq))
    if (!StringRef(arg->getValue()).endswith("lto-wrapper"))
      error(arg->getSpelling() + ": unknown plugin option '" + arg->getValue() +
            "'");

  // Parse -mllvm options.
  for (auto *arg : args.filtered(OPT_mllvm))
    parseClangOption(arg->getValue(), arg->getSpelling());

  // --threads= takes a positive integer and provides the default value for
  // --thinlto-jobs=.
  if (auto *arg = args.getLastArg(OPT_threads)) {
    StringRef v(arg->getValue());
    unsigned threads = 0;
    if (!llvm::to_integer(v, threads, 0) || threads == 0)
      error(arg->getSpelling() + ": expected a positive integer, but got '" +
            arg->getValue() + "'");
    parallel::strategy = hardware_concurrency(threads);
    config->thinLTOJobs = v;
  }
  if (auto *arg = args.getLastArg(OPT_thinlto_jobs))
    config->thinLTOJobs = arg->getValue();

  if (config->ltoo > 3)
    error("invalid optimization level for LTO: " + Twine(config->ltoo));
  if (config->ltoPartitions == 0)
    error("--lto-partitions: number of threads must be > 0");
  if (!get_threadpool_strategy(config->thinLTOJobs))
    error("--thinlto-jobs: invalid job count: " + config->thinLTOJobs);

  if (config->splitStackAdjustSize < 0)
    error("--split-stack-adjust-size: size must be >= 0");

  // The text segment is traditionally the first segment, whose address equals
  // the base address. However, lld places the R PT_LOAD first. -Ttext-segment
  // is an old-fashioned option that does not play well with lld's layout.
  // Suggest --image-base as a likely alternative.
  if (args.hasArg(OPT_Ttext_segment))
    error("-Ttext-segment is not supported. Use --image-base if you "
          "intend to set the base address");

  // Parse ELF{32,64}{LE,BE} and CPU type.
  if (auto *arg = args.getLastArg(OPT_m)) {
    StringRef s = arg->getValue();
    std::tie(config->ekind, config->emachine, config->osabi) =
        parseEmulation(s);
    config->mipsN32Abi =
        (s.startswith("elf32btsmipn32") || s.startswith("elf32ltsmipn32"));
    config->emulation = s;
  }

  // Parse --hash-style={sysv,gnu,both}.
  if (auto *arg = args.getLastArg(OPT_hash_style)) {
    StringRef s = arg->getValue();
    if (s == "sysv")
      config->sysvHash = true;
    else if (s == "gnu")
      config->gnuHash = true;
    else if (s == "both")
      config->sysvHash = config->gnuHash = true;
    else
      error("unknown --hash-style: " + s);
  }

  if (args.hasArg(OPT_print_map))
    config->mapFile = "-";

  // Page alignment can be disabled by the -n (--nmagic) and -N (--omagic).
  // As PT_GNU_RELRO relies on Paging, do not create it when we have disabled
  // it.
  if (config->nmagic || config->omagic)
    config->zRelro = false;

  std::tie(config->buildId, config->buildIdVector) = getBuildId(args);

  if (getZFlag(args, "pack-relative-relocs", "nopack-relative-relocs", false)) {
    config->relrGlibc = true;
    config->relrPackDynRelocs = true;
  } else {
    std::tie(config->androidPackDynRelocs, config->relrPackDynRelocs) =
        getPackDynRelocs(args);
  }

  if (auto *arg = args.getLastArg(OPT_symbol_ordering_file)){
    if (args.hasArg(OPT_call_graph_ordering_file))
      error("--symbol-ordering-file and --call-graph-order-file "
            "may not be used together");
    if (Optional<MemoryBufferRef> buffer = readFile(arg->getValue())){
      config->symbolOrderingFile = getSymbolOrderingFile(*buffer);
      // Also need to disable CallGraphProfileSort to prevent
      // LLD order symbols with CGProfile
      config->callGraphProfileSort = false;
    }
  }

  assert(config->versionDefinitions.empty());
  config->versionDefinitions.push_back(
      {"local", (uint16_t)VER_NDX_LOCAL, {}, {}});
  config->versionDefinitions.push_back(
      {"global", (uint16_t)VER_NDX_GLOBAL, {}, {}});

  // If --retain-symbol-file is used, we'll keep only the symbols listed in
  // the file and discard all others.
  if (auto *arg = args.getLastArg(OPT_retain_symbols_file)) {
    config->versionDefinitions[VER_NDX_LOCAL].nonLocalPatterns.push_back(
        {"*", /*isExternCpp=*/false, /*hasWildcard=*/true});
    if (Optional<MemoryBufferRef> buffer = readFile(arg->getValue()))
      for (StringRef s : args::getLines(*buffer))
        config->versionDefinitions[VER_NDX_GLOBAL].nonLocalPatterns.push_back(
            {s, /*isExternCpp=*/false, /*hasWildcard=*/false});
  }

  for (opt::Arg *arg : args.filtered(OPT_warn_backrefs_exclude)) {
    StringRef pattern(arg->getValue());
    if (Expected<GlobPattern> pat = GlobPattern::create(pattern))
      config->warnBackrefsExclude.push_back(std::move(*pat));
    else
      error(arg->getSpelling() + ": " + toString(pat.takeError()));
  }

  // For -no-pie and -pie, --export-dynamic-symbol specifies defined symbols
  // which should be exported. For -shared, references to matched non-local
  // STV_DEFAULT symbols are not bound to definitions within the shared object,
  // even if other options express a symbolic intention: -Bsymbolic,
  // -Bsymbolic-functions (if STT_FUNC), --dynamic-list.
  for (auto *arg : args.filtered(OPT_export_dynamic_symbol))
    config->dynamicList.push_back(
        {arg->getValue(), /*isExternCpp=*/false,
         /*hasWildcard=*/hasWildcard(arg->getValue())});

  // --export-dynamic-symbol-list specifies a list of --export-dynamic-symbol
  // patterns. --dynamic-list is --export-dynamic-symbol-list plus -Bsymbolic
  // like semantics.
  config->symbolic =
      config->bsymbolic == BsymbolicKind::All || args.hasArg(OPT_dynamic_list);
  for (auto *arg :
       args.filtered(OPT_dynamic_list, OPT_export_dynamic_symbol_list))
    if (Optional<MemoryBufferRef> buffer = readFile(arg->getValue()))
      readDynamicList(*buffer);

  for (auto *arg : args.filtered(OPT_version_script))
    if (Optional<std::string> path = searchScript(arg->getValue())) {
      if (Optional<MemoryBufferRef> buffer = readFile(*path))
        readVersionScript(*buffer);
    } else {
      error(Twine("cannot find version script ") + arg->getValue());
    }
}

// Some Config members do not directly correspond to any particular
// command line options, but computed based on other Config values.
// This function initialize such members. See Config.h for the details
// of these values.
static void setConfigs(opt::InputArgList &args) {
  ELFKind k = config->ekind;
  uint16_t m = config->emachine;

  config->copyRelocs = (config->relocatable || config->emitRelocs);
  config->is64 = (k == ELF64LEKind || k == ELF64BEKind);
  config->isLE = (k == ELF32LEKind || k == ELF64LEKind);
  config->endianness = config->isLE ? endianness::little : endianness::big;
  config->isMips64EL = (k == ELF64LEKind && m == EM_MIPS);
  config->isPic = config->pie || config->shared;
  config->picThunk = args.hasArg(OPT_pic_veneer, config->isPic);
  config->wordsize = config->is64 ? 8 : 4;

  // ELF defines two different ways to store relocation addends as shown below:
  //
  //  Rel: Addends are stored to the location where relocations are applied. It
  //  cannot pack the full range of addend values for all relocation types, but
  //  this only affects relocation types that we don't support emitting as
  //  dynamic relocations (see getDynRel).
  //  Rela: Addends are stored as part of relocation entry.
  //
  // In other words, Rela makes it easy to read addends at the price of extra
  // 4 or 8 byte for each relocation entry.
  //
  // We pick the format for dynamic relocations according to the psABI for each
  // processor, but a contrary choice can be made if the dynamic loader
  // supports.
  config->isRela = getIsRela(args);

  // If the output uses REL relocations we must store the dynamic relocation
  // addends to the output sections. We also store addends for RELA relocations
  // if --apply-dynamic-relocs is used.
  // We default to not writing the addends when using RELA relocations since
  // any standard conforming tool can find it in r_addend.
  config->writeAddends = args.hasFlag(OPT_apply_dynamic_relocs,
                                      OPT_no_apply_dynamic_relocs, false) ||
                         !config->isRela;
  // Validation of dynamic relocation addends is on by default for assertions
  // builds (for supported targets) and disabled otherwise. Ideally we would
  // enable the debug checks for all targets, but currently not all targets
  // have support for reading Elf_Rel addends, so we only enable for a subset.
#ifndef NDEBUG
  bool checkDynamicRelocsDefault = m == EM_ARM || m == EM_386 || m == EM_MIPS ||
                                   m == EM_X86_64 || m == EM_RISCV;
#else
  bool checkDynamicRelocsDefault = false;
#endif
  config->checkDynamicRelocs =
      args.hasFlag(OPT_check_dynamic_relocations,
                   OPT_no_check_dynamic_relocations, checkDynamicRelocsDefault);
  config->tocOptimize =
      args.hasFlag(OPT_toc_optimize, OPT_no_toc_optimize, m == EM_PPC64);
  config->pcRelOptimize =
      args.hasFlag(OPT_pcrel_optimize, OPT_no_pcrel_optimize, m == EM_PPC64);
}

static bool isFormatBinary(StringRef s) {
  if (s == "binary")
    return true;
  if (s == "elf" || s == "default")
    return false;
  error("unknown --format value: " + s +
        " (supported formats: elf, default, binary)");
  return false;
}

void LinkerDriver::createFiles(opt::InputArgList &args) {
  llvm::TimeTraceScope timeScope("Load input files");
  // For --{push,pop}-state.
  std::vector<std::tuple<bool, bool, bool>> stack;

  // Iterate over argv to process input files and positional arguments.
  InputFile::isInGroup = false;
  bool hasInput = false;
  for (auto *arg : args) {
    switch (arg->getOption().getID()) {
    case OPT_library:
      addLibrary(arg->getValue());
      hasInput = true;
      break;
    case OPT_INPUT:
      addFile(arg->getValue(), /*withLOption=*/false);
      hasInput = true;
      break;
    case OPT_defsym: {
      StringRef from;
      StringRef to;
      std::tie(from, to) = StringRef(arg->getValue()).split('=');
      if (from.empty() || to.empty())
        error("--defsym: syntax error: " + StringRef(arg->getValue()));
      else
        readDefsym(from, MemoryBufferRef(to, "--defsym"));
      break;
    }
    case OPT_script:
      if (Optional<std::string> path = searchScript(arg->getValue())) {
        if (Optional<MemoryBufferRef> mb = readFile(*path))
          readLinkerScript(*mb);
        break;
      }
      error(Twine("cannot find linker script ") + arg->getValue());
      break;
    case OPT_as_needed:
      config->asNeeded = true;
      break;
    case OPT_format:
      config->formatBinary = isFormatBinary(arg->getValue());
      break;
    case OPT_no_as_needed:
      config->asNeeded = false;
      break;
    case OPT_Bstatic:
    case OPT_omagic:
    case OPT_nmagic:
      config->isStatic = true;
      break;
    case OPT_Bdynamic:
      config->isStatic = false;
      break;
    case OPT_whole_archive:
      inWholeArchive = true;
      break;
    case OPT_no_whole_archive:
      inWholeArchive = false;
      break;
    case OPT_just_symbols:
      if (Optional<MemoryBufferRef> mb = readFile(arg->getValue())) {
        files.push_back(createObjectFile(*mb));
        files.back()->justSymbols = true;
      }
      break;
    case OPT_start_group:
      if (InputFile::isInGroup)
        error("nested --start-group");
      InputFile::isInGroup = true;
      break;
    case OPT_end_group:
      if (!InputFile::isInGroup)
        error("stray --end-group");
      InputFile::isInGroup = false;
      ++InputFile::nextGroupId;
      break;
    case OPT_start_lib:
      if (inLib)
        error("nested --start-lib");
      if (InputFile::isInGroup)
        error("may not nest --start-lib in --start-group");
      inLib = true;
      InputFile::isInGroup = true;
      break;
    case OPT_end_lib:
      if (!inLib)
        error("stray --end-lib");
      inLib = false;
      InputFile::isInGroup = false;
      ++InputFile::nextGroupId;
      break;
    case OPT_push_state:
      stack.emplace_back(config->asNeeded, config->isStatic, inWholeArchive);
      break;
    case OPT_pop_state:
      if (stack.empty()) {
        error("unbalanced --push-state/--pop-state");
        break;
      }
      std::tie(config->asNeeded, config->isStatic, inWholeArchive) = stack.back();
      stack.pop_back();
      break;
    }
  }

  if (files.empty() && !hasInput && errorCount() == 0)
    error("no input files");
}

// If -m <machine_type> was not given, infer it from object files.
void LinkerDriver::inferMachineType() {
  if (config->ekind != ELFNoneKind)
    return;

  for (InputFile *f : files) {
    if (f->ekind == ELFNoneKind)
      continue;
    config->ekind = f->ekind;
    config->emachine = f->emachine;
    config->osabi = f->osabi;
    config->mipsN32Abi = config->emachine == EM_MIPS && isMipsN32Abi(f);
    return;
  }
  error("target emulation unknown: -m or at least one .o file required");
}

// Parse -z max-page-size=<value>. The default value is defined by
// each target.
static uint64_t getMaxPageSize(opt::InputArgList &args) {
  uint64_t val = args::getZOptionValue(args, OPT_z, "max-page-size",
                                       target->defaultMaxPageSize);
  if (!isPowerOf2_64(val))
    error("max-page-size: value isn't a power of 2");
  if (config->nmagic || config->omagic) {
    if (val != target->defaultMaxPageSize)
      warn("-z max-page-size set, but paging disabled by omagic or nmagic");
    return 1;
  }
  return val;
}

// Parse -z common-page-size=<value>. The default value is defined by
// each target.
static uint64_t getCommonPageSize(opt::InputArgList &args) {
  uint64_t val = args::getZOptionValue(args, OPT_z, "common-page-size",
                                       target->defaultCommonPageSize);
  if (!isPowerOf2_64(val))
    error("common-page-size: value isn't a power of 2");
  if (config->nmagic || config->omagic) {
    if (val != target->defaultCommonPageSize)
      warn("-z common-page-size set, but paging disabled by omagic or nmagic");
    return 1;
  }
  // commonPageSize can't be larger than maxPageSize.
  if (val > config->maxPageSize)
    val = config->maxPageSize;
  return val;
}

// Parses --image-base option.
static Optional<uint64_t> getImageBase(opt::InputArgList &args) {
  // Because we are using "Config->maxPageSize" here, this function has to be
  // called after the variable is initialized.
  auto *arg = args.getLastArg(OPT_image_base);
  if (!arg)
    return None;

  StringRef s = arg->getValue();
  uint64_t v;
  if (!to_integer(s, v)) {
    error("--image-base: number expected, but got " + s);
    return 0;
  }
  if ((v % config->maxPageSize) != 0)
    warn("--image-base: address isn't multiple of page size: " + s);
  return v;
}

// Parses `--exclude-libs=lib,lib,...`.
// The library names may be delimited by commas or colons.
static DenseSet<StringRef> getExcludeLibs(opt::InputArgList &args) {
  DenseSet<StringRef> ret;
  for (auto *arg : args.filtered(OPT_exclude_libs)) {
    StringRef s = arg->getValue();
    for (;;) {
      size_t pos = s.find_first_of(",:");
      if (pos == StringRef::npos)
        break;
      ret.insert(s.substr(0, pos));
      s = s.substr(pos + 1);
    }
    ret.insert(s);
  }
  return ret;
}

// Handles the --exclude-libs option. If a static library file is specified
// by the --exclude-libs option, all public symbols from the archive become
// private unless otherwise specified by version scripts or something.
// A special library name "ALL" means all archive files.
//
// This is not a popular option, but some programs such as bionic libc use it.
static void excludeLibs(opt::InputArgList &args) {
  DenseSet<StringRef> libs = getExcludeLibs(args);
  bool all = libs.count("ALL");

  auto visit = [&](InputFile *file) {
    if (file->archiveName.empty() ||
        !(all || libs.count(path::filename(file->archiveName))))
      return;
    ArrayRef<Symbol *> symbols = file->getSymbols();
    if (isa<ELFFileBase>(file))
      symbols = cast<ELFFileBase>(file)->getGlobalSymbols();
    for (Symbol *sym : symbols)
      if (!sym->isUndefined() && sym->file == file)
        sym->versionId = VER_NDX_LOCAL;
  };

  for (ELFFileBase *file : objectFiles)
    visit(file);

  for (BitcodeFile *file : bitcodeFiles)
    visit(file);
}

// Force Sym to be entered in the output.
static void handleUndefined(Symbol *sym, const char *option) {
  // Since a symbol may not be used inside the program, LTO may
  // eliminate it. Mark the symbol as "used" to prevent it.
  sym->isUsedInRegularObj = true;

  if (!sym->isLazy())
    return;
  sym->extract();
  if (!config->whyExtract.empty())
    driver->whyExtract.emplace_back(option, sym->file, *sym);
}

// As an extension to GNU linkers, lld supports a variant of `-u`
// which accepts wildcard patterns. All symbols that match a given
// pattern are handled as if they were given by `-u`.
static void handleUndefinedGlob(StringRef arg) {
  Expected<GlobPattern> pat = GlobPattern::create(arg);
  if (!pat) {
    error("--undefined-glob: " + toString(pat.takeError()));
    return;
  }

  // Calling sym->extract() in the loop is not safe because it may add new
  // symbols to the symbol table, invalidating the current iterator.
  SmallVector<Symbol *, 0> syms;
  for (Symbol *sym : symtab->symbols())
    if (!sym->isPlaceholder() && pat->match(sym->getName()))
      syms.push_back(sym);

  for (Symbol *sym : syms)
    handleUndefined(sym, "--undefined-glob");
}

static void handleLibcall(StringRef name) {
  Symbol *sym = symtab->find(name);
  if (!sym || !sym->isLazy())
    return;

  MemoryBufferRef mb;
  mb = cast<LazyObject>(sym)->file->mb;

  if (isBitcode(mb))
    sym->extract();
}

void LinkerDriver::writeArchiveStats() const {
  if (config->printArchiveStats.empty())
    return;

  std::error_code ec;
  raw_fd_ostream os(config->printArchiveStats, ec, sys::fs::OF_None);
  if (ec) {
    error("--print-archive-stats=: cannot open " + config->printArchiveStats +
          ": " + ec.message());
    return;
  }

  os << "members\textracted\tarchive\n";

  SmallVector<StringRef, 0> archives;
  DenseMap<CachedHashStringRef, unsigned> all, extracted;
  for (ELFFileBase *file : objectFiles)
    if (file->archiveName.size())
      ++extracted[CachedHashStringRef(file->archiveName)];
  for (BitcodeFile *file : bitcodeFiles)
    if (file->archiveName.size())
      ++extracted[CachedHashStringRef(file->archiveName)];
  for (std::pair<StringRef, unsigned> f : archiveFiles) {
    unsigned &v = extracted[CachedHashString(f.first)];
    os << f.second << '\t' << v << '\t' << f.first << '\n';
    // If the archive occurs multiple times, other instances have a count of 0.
    v = 0;
  }
}

void LinkerDriver::writeWhyExtract() const {
  if (config->whyExtract.empty())
    return;

  std::error_code ec;
  raw_fd_ostream os(config->whyExtract, ec, sys::fs::OF_None);
  if (ec) {
    error("cannot open --why-extract= file " + config->whyExtract + ": " +
          ec.message());
    return;
  }

  os << "reference\textracted\tsymbol\n";
  for (auto &entry : whyExtract) {
    os << std::get<0>(entry) << '\t' << toString(std::get<1>(entry)) << '\t'
       << toString(std::get<2>(entry)) << '\n';
  }
}

void LinkerDriver::reportBackrefs() const {
  for (auto &ref : backwardReferences) {
    const Symbol &sym = *ref.first;
    std::string to = toString(ref.second.second);
    // Some libraries have known problems and can cause noise. Filter them out
    // with --warn-backrefs-exclude=. The value may look like (for --start-lib)
    // *.o or (archive member) *.a(*.o).
    bool exclude = false;
    for (const llvm::GlobPattern &pat : config->warnBackrefsExclude)
      if (pat.match(to)) {
        exclude = true;
        break;
      }
    if (!exclude)
      warn("backward reference detected: " + sym.getName() + " in " +
           toString(ref.second.first) + " refers to " + to);
  }
}

// Handle --dependency-file=<path>. If that option is given, lld creates a
// file at a given path with the following contents:
//
//   <output-file>: <input-file> ...
//
//   <input-file>:
//
// where <output-file> is a pathname of an output file and <input-file>
// ... is a list of pathnames of all input files. `make` command can read a
// file in the above format and interpret it as a dependency info. We write
// phony targets for every <input-file> to avoid an error when that file is
// removed.
//
// This option is useful if you want to make your final executable to depend
// on all input files including system libraries. Here is why.
//
// When you write a Makefile, you usually write it so that the final
// executable depends on all user-generated object files. Normally, you
// don't make your executable to depend on system libraries (such as libc)
// because you don't know the exact paths of libraries, even though system
// libraries that are linked to your executable statically are technically a
// part of your program. By using --dependency-file option, you can make
// lld to dump dependency info so that you can maintain exact dependencies
// easily.
static void writeDependencyFile() {
  std::error_code ec;
  raw_fd_ostream os(config->dependencyFile, ec, sys::fs::OF_None);
  if (ec) {
    error("cannot open " + config->dependencyFile + ": " + ec.message());
    return;
  }

  // We use the same escape rules as Clang/GCC which are accepted by Make/Ninja:
  // * A space is escaped by a backslash which itself must be escaped.
  // * A hash sign is escaped by a single backslash.
  // * $ is escapes as $$.
  auto printFilename = [](raw_fd_ostream &os, StringRef filename) {
    llvm::SmallString<256> nativePath;
    llvm::sys::path::native(filename.str(), nativePath);
    llvm::sys::path::remove_dots(nativePath, /*remove_dot_dot=*/true);
    for (unsigned i = 0, e = nativePath.size(); i != e; ++i) {
      if (nativePath[i] == '#') {
        os << '\\';
      } else if (nativePath[i] == ' ') {
        os << '\\';
        unsigned j = i;
        while (j > 0 && nativePath[--j] == '\\')
          os << '\\';
      } else if (nativePath[i] == '$') {
        os << '$';
      }
      os << nativePath[i];
    }
  };

  os << config->outputFile << ":";
  for (StringRef path : config->dependencyFiles) {
    os << " \\\n ";
    printFilename(os, path);
  }
  os << "\n";

  for (StringRef path : config->dependencyFiles) {
    os << "\n";
    printFilename(os, path);
    os << ":\n";
  }
}

// Replaces common symbols with defined symbols reside in .bss sections.
// This function is called after all symbol names are resolved. As a
// result, the passes after the symbol resolution won't see any
// symbols of type CommonSymbol.
static void replaceCommonSymbols() {
  llvm::TimeTraceScope timeScope("Replace common symbols");
  for (ELFFileBase *file : objectFiles) {
    if (!file->hasCommonSyms)
      continue;
    for (Symbol *sym : file->getGlobalSymbols()) {
      auto *s = dyn_cast<CommonSymbol>(sym);
      if (!s)
        continue;

      auto *bss = make<BssSection>("COMMON", s->size, s->alignment);
      bss->file = s->file;
      inputSections.push_back(bss);
      s->replace(Defined{s->file, StringRef(), s->binding, s->stOther, s->type,
                         /*value=*/0, s->size, bss});
    }
  }
}

// If all references to a DSO happen to be weak, the DSO is not added to
// DT_NEEDED. If that happens, replace ShardSymbol with Undefined to avoid
// dangling references to an unneeded DSO. Use a weak binding to avoid
// --no-allow-shlib-undefined diagnostics. Similarly, demote lazy symbols.
static void demoteSharedAndLazySymbols() {
  llvm::TimeTraceScope timeScope("Demote shared and lazy symbols");
  for (Symbol *sym : symtab->symbols()) {
    auto *s = dyn_cast<SharedSymbol>(sym);
    if (!(s && !cast<SharedFile>(s->file)->isNeeded) && !sym->isLazy())
      continue;

    bool used = sym->used;
    uint8_t binding = sym->isLazy() ? sym->binding : uint8_t(STB_WEAK);
    sym->replace(
        Undefined{nullptr, sym->getName(), binding, sym->stOther, sym->type});
    sym->used = used;
    sym->versionId = VER_NDX_GLOBAL;
  }
}

// The section referred to by `s` is considered address-significant. Set the
// keepUnique flag on the section if appropriate.
static void markAddrsig(Symbol *s) {
  if (auto *d = dyn_cast_or_null<Defined>(s))
    if (d->section)
      // We don't need to keep text sections unique under --icf=all even if they
      // are address-significant.
      if (config->icf == ICFLevel::Safe || !(d->section->flags & SHF_EXECINSTR))
        d->section->keepUnique = true;
}

// Record sections that define symbols mentioned in --keep-unique <symbol>
// and symbols referred to by address-significance tables. These sections are
// ineligible for ICF.
template <class ELFT>
static void findKeepUniqueSections(opt::InputArgList &args) {
  for (auto *arg : args.filtered(OPT_keep_unique)) {
    StringRef name = arg->getValue();
    auto *d = dyn_cast_or_null<Defined>(symtab->find(name));
    if (!d || !d->section) {
      warn("could not find symbol " + name + " to keep unique");
      continue;
    }
    d->section->keepUnique = true;
  }

  // --icf=all --ignore-data-address-equality means that we can ignore
  // the dynsym and address-significance tables entirely.
  if (config->icf == ICFLevel::All && config->ignoreDataAddressEquality)
    return;

  // Symbols in the dynsym could be address-significant in other executables
  // or DSOs, so we conservatively mark them as address-significant.
  for (Symbol *sym : symtab->symbols())
    if (sym->includeInDynsym())
      markAddrsig(sym);

  // Visit the address-significance table in each object file and mark each
  // referenced symbol as address-significant.
  for (InputFile *f : objectFiles) {
    auto *obj = cast<ObjFile<ELFT>>(f);
    ArrayRef<Symbol *> syms = obj->getSymbols();
    if (obj->addrsigSec) {
      ArrayRef<uint8_t> contents =
          check(obj->getObj().getSectionContents(*obj->addrsigSec));
      const uint8_t *cur = contents.begin();
      while (cur != contents.end()) {
        unsigned size;
        const char *err;
        uint64_t symIndex = decodeULEB128(cur, &size, contents.end(), &err);
        if (err)
          fatal(toString(f) + ": could not decode addrsig section: " + err);
        markAddrsig(syms[symIndex]);
        cur += size;
      }
    } else {
      // If an object file does not have an address-significance table,
      // conservatively mark all of its symbols as address-significant.
      for (Symbol *s : syms)
        markAddrsig(s);
    }
  }
}

// This function reads a symbol partition specification section. These sections
// are used to control which partition a symbol is allocated to. See
// https://lld.llvm.org/Partitions.html for more details on partitions.
template <typename ELFT>
static void readSymbolPartitionSection(InputSectionBase *s) {
  // Read the relocation that refers to the partition's entry point symbol.
  Symbol *sym;
  const RelsOrRelas<ELFT> rels = s->template relsOrRelas<ELFT>();
  if (rels.areRelocsRel())
    sym = &s->getFile<ELFT>()->getRelocTargetSym(rels.rels[0]);
  else
    sym = &s->getFile<ELFT>()->getRelocTargetSym(rels.relas[0]);
  if (!isa<Defined>(sym) || !sym->includeInDynsym())
    return;

  StringRef partName = reinterpret_cast<const char *>(s->rawData.data());
  for (Partition &part : partitions) {
    if (part.name == partName) {
      sym->partition = part.getNumber();
      return;
    }
  }

  // Forbid partitions from being used on incompatible targets, and forbid them
  // from being used together with various linker features that assume a single
  // set of output sections.
  if (script->hasSectionsCommand)
    error(toString(s->file) +
          ": partitions cannot be used with the SECTIONS command");
  if (script->hasPhdrsCommands())
    error(toString(s->file) +
          ": partitions cannot be used with the PHDRS command");
  if (!config->sectionStartMap.empty())
    error(toString(s->file) + ": partitions cannot be used with "
                              "--section-start, -Ttext, -Tdata or -Tbss");
  if (config->emachine == EM_MIPS)
    error(toString(s->file) + ": partitions cannot be used on this target");

  // Impose a limit of no more than 254 partitions. This limit comes from the
  // sizes of the Partition fields in InputSectionBase and Symbol, as well as
  // the amount of space devoted to the partition number in RankFlags.
  if (partitions.size() == 254)
    fatal("may not have more than 254 partitions");

  partitions.emplace_back();
  Partition &newPart = partitions.back();
  newPart.name = partName;
  sym->partition = newPart.getNumber();
}

static Symbol *addUnusedUndefined(StringRef name,
                                  uint8_t binding = STB_GLOBAL) {
  return symtab->addSymbol(Undefined{nullptr, name, binding, STV_DEFAULT, 0});
}

static void markBuffersAsDontNeed(bool skipLinkedOutput) {
  // With --thinlto-index-only, all buffers are nearly unused from now on
  // (except symbol/section names used by infrequent passes). Mark input file
  // buffers as MADV_DONTNEED so that these pages can be reused by the expensive
  // thin link, saving memory.
  if (skipLinkedOutput) {
    for (MemoryBuffer &mb : llvm::make_pointee_range(memoryBuffers))
      mb.dontNeedIfMmap();
    return;
  }

  // Otherwise, just mark MemoryBuffers backing BitcodeFiles.
  DenseSet<const char *> bufs;
  for (BitcodeFile *file : bitcodeFiles)
    bufs.insert(file->mb.getBufferStart());
  for (BitcodeFile *file : lazyBitcodeFiles)
    bufs.insert(file->mb.getBufferStart());
  for (MemoryBuffer &mb : llvm::make_pointee_range(memoryBuffers))
    if (bufs.count(mb.getBufferStart()))
      mb.dontNeedIfMmap();
}

// This function is where all the optimizations of link-time
// optimization takes place. When LTO is in use, some input files are
// not in native object file format but in the LLVM bitcode format.
// This function compiles bitcode files into a few big native files
// using LLVM functions and replaces bitcode symbols with the results.
// Because all bitcode files that the program consists of are passed to
// the compiler at once, it can do a whole-program optimization.
template <class ELFT>
void LinkerDriver::compileBitcodeFiles(bool skipLinkedOutput) {
  llvm::TimeTraceScope timeScope("LTO");
  // Compile bitcode files and replace bitcode symbols.
  lto.reset(new BitcodeCompiler);
  for (BitcodeFile *file : bitcodeFiles)
    lto->add(*file);

  if (!bitcodeFiles.empty())
    markBuffersAsDontNeed(skipLinkedOutput);

  for (InputFile *file : lto->compile()) {
    auto *obj = cast<ObjFile<ELFT>>(file);
    obj->parse(/*ignoreComdats=*/true);

    // Parse '@' in symbol names for non-relocatable output.
    if (!config->relocatable)
      for (Symbol *sym : obj->getGlobalSymbols())
        if (sym->hasVersionSuffix)
          sym->parseSymbolVersion();
    objectFiles.push_back(obj);
  }
}

// The --wrap option is a feature to rename symbols so that you can write
// wrappers for existing functions. If you pass `--wrap=foo`, all
// occurrences of symbol `foo` are resolved to `__wrap_foo` (so, you are
// expected to write `__wrap_foo` function as a wrapper). The original
// symbol becomes accessible as `__real_foo`, so you can call that from your
// wrapper.
//
// This data structure is instantiated for each --wrap option.
struct WrappedSymbol {
  Symbol *sym;
  Symbol *real;
  Symbol *wrap;
};

// Handles --wrap option.
//
// This function instantiates wrapper symbols. At this point, they seem
// like they are not being used at all, so we explicitly set some flags so
// that LTO won't eliminate them.
static std::vector<WrappedSymbol> addWrappedSymbols(opt::InputArgList &args) {
  std::vector<WrappedSymbol> v;
  DenseSet<StringRef> seen;

  for (auto *arg : args.filtered(OPT_wrap)) {
    StringRef name = arg->getValue();
    if (!seen.insert(name).second)
      continue;

    Symbol *sym = symtab->find(name);
    // Avoid wrapping symbols that are lazy and unreferenced at this point, to
    // not create undefined references. The isUsedInRegularObj check handles the
    // case of a weak reference, which we still want to wrap even though it
    // doesn't cause lazy symbols to be extracted.
    if (!sym || (sym->isLazy() && !sym->isUsedInRegularObj))
      continue;

    Symbol *real = addUnusedUndefined(saver().save("__real_" + name));
    Symbol *wrap =
        addUnusedUndefined(saver().save("__wrap_" + name), sym->binding);
    v.push_back({sym, real, wrap});

    // We want to tell LTO not to inline symbols to be overwritten
    // because LTO doesn't know the final symbol contents after renaming.
    real->scriptDefined = true;
    sym->scriptDefined = true;

    // Tell LTO not to eliminate these symbols.
    sym->isUsedInRegularObj = true;
    // If sym is referenced in any object file, bitcode file or shared object,
    // retain wrap which is the redirection target of sym. If the object file
    // defining sym has sym references, we cannot easily distinguish the case
    // from cases where sym is not referenced. Retain wrap because we choose to
    // wrap sym references regardless of whether sym is defined
    // (https://sourceware.org/bugzilla/show_bug.cgi?id=26358).
    if (sym->referenced || sym->isDefined())
      wrap->isUsedInRegularObj = true;
  }
  return v;
}

// Do renaming for --wrap and foo@v1 by updating pointers to symbols.
//
// When this function is executed, only InputFiles and symbol table
// contain pointers to symbol objects. We visit them to replace pointers,
// so that wrapped symbols are swapped as instructed by the command line.
static void redirectSymbols(ArrayRef<WrappedSymbol> wrapped) {
  llvm::TimeTraceScope timeScope("Redirect symbols");
  DenseMap<Symbol *, Symbol *> map;
  for (const WrappedSymbol &w : wrapped) {
    map[w.sym] = w.wrap;
    map[w.real] = w.sym;
  }
  for (Symbol *sym : symtab->symbols()) {
    // Enumerate symbols with a non-default version (foo@v1). hasVersionSuffix
    // filters out most symbols but is not sufficient.
    if (!sym->hasVersionSuffix)
      continue;
    const char *suffix1 = sym->getVersionSuffix();
    if (suffix1[0] != '@' || suffix1[1] == '@')
      continue;

    // Check the existing symbol foo. We have two special cases to handle:
    //
    // * There is a definition of foo@v1 and foo@@v1.
    // * There is a definition of foo@v1 and foo.
    Defined *sym2 = dyn_cast_or_null<Defined>(symtab->find(sym->getName()));
    if (!sym2)
      continue;
    const char *suffix2 = sym2->getVersionSuffix();
    if (suffix2[0] == '@' && suffix2[1] == '@' &&
        strcmp(suffix1 + 1, suffix2 + 2) == 0) {
      // foo@v1 and foo@@v1 should be merged, so redirect foo@v1 to foo@@v1.
      map.try_emplace(sym, sym2);
      // If both foo@v1 and foo@@v1 are defined and non-weak, report a duplicate
      // definition error.
      if (sym->isDefined())
        sym2->checkDuplicate(cast<Defined>(*sym));
      sym2->resolve(*sym);
      // Eliminate foo@v1 from the symbol table.
      sym->symbolKind = Symbol::PlaceholderKind;
      sym->isUsedInRegularObj = false;
    } else if (auto *sym1 = dyn_cast<Defined>(sym)) {
      if (sym2->versionId > VER_NDX_GLOBAL
              ? config->versionDefinitions[sym2->versionId].name == suffix1 + 1
              : sym1->section == sym2->section && sym1->value == sym2->value) {
        // Due to an assembler design flaw, if foo is defined, .symver foo,
        // foo@v1 defines both foo and foo@v1. Unless foo is bound to a
        // different version, GNU ld makes foo@v1 canonical and eliminates foo.
        // Emulate its behavior, otherwise we would have foo or foo@@v1 beside
        // foo@v1. foo@v1 and foo combining does not apply if they are not
        // defined in the same place.
        map.try_emplace(sym2, sym);
        sym2->symbolKind = Symbol::PlaceholderKind;
        sym2->isUsedInRegularObj = false;
      }
    }
  }

  if (map.empty())
    return;

  // Update pointers in input files.
  parallelForEach(objectFiles, [&](ELFFileBase *file) {
    for (Symbol *&sym : file->getMutableGlobalSymbols())
      if (Symbol *s = map.lookup(sym))
        sym = s;
  });

  // Update pointers in the symbol table.
  for (const WrappedSymbol &w : wrapped)
    symtab->wrap(w.sym, w.real, w.wrap);
}

static void checkAndReportMissingFeature(StringRef config, uint32_t features,
                                         uint32_t mask, const Twine &report) {
  if (!(features & mask)) {
    if (config == "error")
      error(report);
    else if (config == "warning")
      warn(report);
  }
}

// To enable CET (x86's hardware-assited control flow enforcement), each
// source file must be compiled with -fcf-protection. Object files compiled
// with the flag contain feature flags indicating that they are compatible
// with CET. We enable the feature only when all object files are compatible
// with CET.
//
// This is also the case with AARCH64's BTI and PAC which use the similar
// GNU_PROPERTY_AARCH64_FEATURE_1_AND mechanism.
static uint32_t getAndFeatures() {
  if (config->emachine != EM_386 && config->emachine != EM_X86_64 &&
      config->emachine != EM_AARCH64)
    return 0;

  uint32_t ret = -1;
  for (ELFFileBase *f : objectFiles) {
    uint32_t features = f->andFeatures;

    checkAndReportMissingFeature(
        config->zBtiReport, features, GNU_PROPERTY_AARCH64_FEATURE_1_BTI,
        toString(f) + ": -z bti-report: file does not have "
                      "GNU_PROPERTY_AARCH64_FEATURE_1_BTI property");

    checkAndReportMissingFeature(
        config->zCetReport, features, GNU_PROPERTY_X86_FEATURE_1_IBT,
        toString(f) + ": -z cet-report: file does not have "
                      "GNU_PROPERTY_X86_FEATURE_1_IBT property");

    checkAndReportMissingFeature(
        config->zCetReport, features, GNU_PROPERTY_X86_FEATURE_1_SHSTK,
        toString(f) + ": -z cet-report: file does not have "
                      "GNU_PROPERTY_X86_FEATURE_1_SHSTK property");

    if (config->zForceBti && !(features & GNU_PROPERTY_AARCH64_FEATURE_1_BTI)) {
      features |= GNU_PROPERTY_AARCH64_FEATURE_1_BTI;
      if (config->zBtiReport == "none")
        warn(toString(f) + ": -z force-bti: file does not have "
                           "GNU_PROPERTY_AARCH64_FEATURE_1_BTI property");
    } else if (config->zForceIbt &&
               !(features & GNU_PROPERTY_X86_FEATURE_1_IBT)) {
      if (config->zCetReport == "none")
        warn(toString(f) + ": -z force-ibt: file does not have "
                           "GNU_PROPERTY_X86_FEATURE_1_IBT property");
      features |= GNU_PROPERTY_X86_FEATURE_1_IBT;
    }
    if (config->zPacPlt && !(features & GNU_PROPERTY_AARCH64_FEATURE_1_PAC)) {
      warn(toString(f) + ": -z pac-plt: file does not have "
                         "GNU_PROPERTY_AARCH64_FEATURE_1_PAC property");
      features |= GNU_PROPERTY_AARCH64_FEATURE_1_PAC;
    }
    ret &= features;
  }

  // Force enable Shadow Stack.
  if (config->zShstk)
    ret |= GNU_PROPERTY_X86_FEATURE_1_SHSTK;

  return ret;
}

static void initializeLocalSymbols(ELFFileBase *file) {
  switch (config->ekind) {
  case ELF32LEKind:
    cast<ObjFile<ELF32LE>>(file)->initializeLocalSymbols();
    break;
  case ELF32BEKind:
    cast<ObjFile<ELF32BE>>(file)->initializeLocalSymbols();
    break;
  case ELF64LEKind:
    cast<ObjFile<ELF64LE>>(file)->initializeLocalSymbols();
    break;
  case ELF64BEKind:
    cast<ObjFile<ELF64BE>>(file)->initializeLocalSymbols();
    break;
  default:
    llvm_unreachable("");
  }
}

static void postParseObjectFile(ELFFileBase *file) {
  switch (config->ekind) {
  case ELF32LEKind:
    cast<ObjFile<ELF32LE>>(file)->postParse();
    break;
  case ELF32BEKind:
    cast<ObjFile<ELF32BE>>(file)->postParse();
    break;
  case ELF64LEKind:
    cast<ObjFile<ELF64LE>>(file)->postParse();
    break;
  case ELF64BEKind:
    cast<ObjFile<ELF64BE>>(file)->postParse();
    break;
  default:
    llvm_unreachable("");
  }
}

// Do actual linking. Note that when this function is called,
// all linker scripts have already been parsed.
void LinkerDriver::link(opt::InputArgList &args) {
  llvm::TimeTraceScope timeScope("Link", StringRef("LinkerDriver::Link"));
  // If a --hash-style option was not given, set to a default value,
  // which varies depending on the target.
  if (!args.hasArg(OPT_hash_style)) {
    if (config->emachine == EM_MIPS)
      config->sysvHash = true;
    else
      config->sysvHash = config->gnuHash = true;
  }

  // Default output filename is "a.out" by the Unix tradition.
  if (config->outputFile.empty())
    config->outputFile = "a.out";

  // Fail early if the output file or map file is not writable. If a user has a
  // long link, e.g. due to a large LTO link, they do not wish to run it and
  // find that it failed because there was a mistake in their command-line.
  {
    llvm::TimeTraceScope timeScope("Create output files");
    if (auto e = tryCreateFile(config->outputFile))
      error("cannot open output file " + config->outputFile + ": " +
            e.message());
    if (auto e = tryCreateFile(config->mapFile))
      error("cannot open map file " + config->mapFile + ": " + e.message());
    if (auto e = tryCreateFile(config->whyExtract))
      error("cannot open --why-extract= file " + config->whyExtract + ": " +
            e.message());
  }
  if (errorCount())
    return;

  // Use default entry point name if no name was given via the command
  // line nor linker scripts. For some reason, MIPS entry point name is
  // different from others.
  config->warnMissingEntry =
      (!config->entry.empty() || (!config->shared && !config->relocatable));
  if (config->entry.empty() && !config->relocatable)
    config->entry = (config->emachine == EM_MIPS) ? "__start" : "_start";

  // Handle --trace-symbol.
  for (auto *arg : args.filtered(OPT_trace_symbol))
    symtab->insert(arg->getValue())->traced = true;

  // Handle -u/--undefined before input files. If both a.a and b.so define foo,
  // -u foo a.a b.so will extract a.a.
  for (StringRef name : config->undefined)
    addUnusedUndefined(name)->referenced = true;

  // Add all files to the symbol table. This will add almost all
  // symbols that we need to the symbol table. This process might
  // add files to the link, via autolinking, these files are always
  // appended to the Files vector.
  {
    llvm::TimeTraceScope timeScope("Parse input files");
    for (size_t i = 0; i < files.size(); ++i) {
      llvm::TimeTraceScope timeScope("Parse input files", files[i]->getName());
      parseFile(files[i]);
    }
  }

  // Now that we have every file, we can decide if we will need a
  // dynamic symbol table.
  // We need one if we were asked to export dynamic symbols or if we are
  // producing a shared library.
  // We also need one if any shared libraries are used and for pie executables
  // (probably because the dynamic linker needs it).
  config->hasDynSymTab =
      !sharedFiles.empty() || config->isPic || config->exportDynamic;

  // Some symbols (such as __ehdr_start) are defined lazily only when there
  // are undefined symbols for them, so we add these to trigger that logic.
  for (StringRef name : script->referencedSymbols)
    addUnusedUndefined(name)->isUsedInRegularObj = true;

  // Prevent LTO from removing any definition referenced by -u.
  for (StringRef name : config->undefined)
    if (Defined *sym = dyn_cast_or_null<Defined>(symtab->find(name)))
      sym->isUsedInRegularObj = true;

  // If an entry symbol is in a static archive, pull out that file now.
  if (Symbol *sym = symtab->find(config->entry))
    handleUndefined(sym, "--entry");

  // Handle the `--undefined-glob <pattern>` options.
  for (StringRef pat : args::getStrings(args, OPT_undefined_glob))
    handleUndefinedGlob(pat);

  // Mark -init and -fini symbols so that the LTO doesn't eliminate them.
  if (Symbol *sym = dyn_cast_or_null<Defined>(symtab->find(config->init)))
    sym->isUsedInRegularObj = true;
  if (Symbol *sym = dyn_cast_or_null<Defined>(symtab->find(config->fini)))
    sym->isUsedInRegularObj = true;

  // If any of our inputs are bitcode files, the LTO code generator may create
  // references to certain library functions that might not be explicit in the
  // bitcode file's symbol table. If any of those library functions are defined
  // in a bitcode file in an archive member, we need to arrange to use LTO to
  // compile those archive members by adding them to the link beforehand.
  //
  // However, adding all libcall symbols to the link can have undesired
  // consequences. For example, the libgcc implementation of
  // __sync_val_compare_and_swap_8 on 32-bit ARM pulls in an .init_array entry
  // that aborts the program if the Linux kernel does not support 64-bit
  // atomics, which would prevent the program from running even if it does not
  // use 64-bit atomics.
  //
  // Therefore, we only add libcall symbols to the link before LTO if we have
  // to, i.e. if the symbol's definition is in bitcode. Any other required
  // libcall symbols will be added to the link after LTO when we add the LTO
  // object file to the link.
  if (!bitcodeFiles.empty())
    for (auto *s : lto::LTO::getRuntimeLibcallSymbols())
      handleLibcall(s);

  // Archive members defining __wrap symbols may be extracted.
  std::vector<WrappedSymbol> wrapped = addWrappedSymbols(args);

  // No more lazy bitcode can be extracted at this point. Do post parse work
  // like checking duplicate symbols.
  parallelForEach(objectFiles, initializeLocalSymbols);
  parallelForEach(objectFiles, postParseObjectFile);
  parallelForEach(bitcodeFiles, [](BitcodeFile *file) { file->postParse(); });

  // Return if there were name resolution errors.
  if (errorCount())
    return;

  // We want to declare linker script's symbols early,
  // so that we can version them.
  // They also might be exported if referenced by DSOs.
  script->declareSymbols();

  // Handle --exclude-libs. This is before scanVersionScript() due to a
  // workaround for Android ndk: for a defined versioned symbol in an archive
  // without a version node in the version script, Android does not expect a
  // 'has undefined version' error in -shared --exclude-libs=ALL mode (PR36295).
  // GNU ld errors in this case.
  if (args.hasArg(OPT_exclude_libs))
    excludeLibs(args);

  // Create elfHeader early. We need a dummy section in
  // addReservedSymbols to mark the created symbols as not absolute.
  Out::elfHeader = make<OutputSection>("", 0, SHF_ALLOC);

  // We need to create some reserved symbols such as _end. Create them.
  if (!config->relocatable)
    addReservedSymbols();

  // Apply version scripts.
  //
  // For a relocatable output, version scripts don't make sense, and
  // parsing a symbol version string (e.g. dropping "@ver1" from a symbol
  // name "foo@ver1") rather do harm, so we don't call this if -r is given.
  if (!config->relocatable) {
    llvm::TimeTraceScope timeScope("Process symbol versions");
    symtab->scanVersionScript();
  }

  // Skip the normal linked output if some LTO options are specified.
  //
  // For --thinlto-index-only, index file creation is performed in
  // compileBitcodeFiles, so we are done afterwards. --plugin-opt=emit-llvm and
  // --plugin-opt=emit-asm create output files in bitcode or assembly code,
  // respectively. When only certain thinLTO modules are specified for
  // compilation, the intermediate object file are the expected output.
  const bool skipLinkedOutput = config->thinLTOIndexOnly || config->emitLLVM ||
                                config->ltoEmitAsm ||
                                !config->thinLTOModulesToCompile.empty();

  // Do link-time optimization if given files are LLVM bitcode files.
  // This compiles bitcode files into real object files.
  //
  // With this the symbol table should be complete. After this, no new names
  // except a few linker-synthesized ones will be added to the symbol table.
  const size_t numObjsBeforeLTO = objectFiles.size();
  invokeELFT(compileBitcodeFiles, skipLinkedOutput);

  // Symbol resolution finished. Report backward reference problems,
  // --print-archive-stats=, and --why-extract=.
  reportBackrefs();
  writeArchiveStats();
  writeWhyExtract();
  if (errorCount())
    return;

  // Bail out if normal linked output is skipped due to LTO.
  if (skipLinkedOutput)
    return;

  // compileBitcodeFiles may have produced lto.tmp object files. After this, no
  // more file will be added.
  auto newObjectFiles = makeArrayRef(objectFiles).slice(numObjsBeforeLTO);
  parallelForEach(newObjectFiles, initializeLocalSymbols);
  parallelForEach(newObjectFiles, postParseObjectFile);

  // Handle --exclude-libs again because lto.tmp may reference additional
  // libcalls symbols defined in an excluded archive. This may override
  // versionId set by scanVersionScript().
  if (args.hasArg(OPT_exclude_libs))
    excludeLibs(args);

  // Apply symbol renames for --wrap and combine foo@v1 and foo@@v1.
  redirectSymbols(wrapped);

  // Replace common symbols with regular symbols.
  replaceCommonSymbols();

  {
    llvm::TimeTraceScope timeScope("Aggregate sections");
    // Now that we have a complete list of input files.
    // Beyond this point, no new files are added.
    // Aggregate all input sections into one place.
    for (InputFile *f : objectFiles)
      for (InputSectionBase *s : f->getSections())
        if (s && s != &InputSection::discarded)
          inputSections.push_back(s);
    for (BinaryFile *f : binaryFiles)
      for (InputSectionBase *s : f->getSections())
        inputSections.push_back(cast<InputSection>(s));
  }

  {
    llvm::TimeTraceScope timeScope("Strip sections");
    llvm::erase_if(inputSections, [](InputSectionBase *s) {
      if (s->type == SHT_LLVM_SYMPART) {
        invokeELFT(readSymbolPartitionSection, s);
        return true;
      }

      // We do not want to emit debug sections if --strip-all
      // or --strip-debug are given.
      if (config->strip == StripPolicy::None)
        return false;

      if (isDebugSection(*s))
        return true;
      if (auto *isec = dyn_cast<InputSection>(s))
        if (InputSectionBase *rel = isec->getRelocatedSection())
          if (isDebugSection(*rel))
            return true;

      return false;
    });
  }

  // Since we now have a complete set of input files, we can create
  // a .d file to record build dependencies.
  if (!config->dependencyFile.empty())
    writeDependencyFile();

  // Now that the number of partitions is fixed, save a pointer to the main
  // partition.
  mainPart = &partitions[0];

  // Read .note.gnu.property sections from input object files which
  // contain a hint to tweak linker's and loader's behaviors.
  config->andFeatures = getAndFeatures();

  // The Target instance handles target-specific stuff, such as applying
  // relocations or writing a PLT section. It also contains target-dependent
  // values such as a default image base address.
  target = getTarget();

  config->eflags = target->calcEFlags();
  // maxPageSize (sometimes called abi page size) is the maximum page size that
  // the output can be run on. For example if the OS can use 4k or 64k page
  // sizes then maxPageSize must be 64k for the output to be useable on both.
  // All important alignment decisions must use this value.
  config->maxPageSize = getMaxPageSize(args);
  // commonPageSize is the most common page size that the output will be run on.
  // For example if an OS can use 4k or 64k page sizes and 4k is more common
  // than 64k then commonPageSize is set to 4k. commonPageSize can be used for
  // optimizations such as DATA_SEGMENT_ALIGN in linker scripts. LLD's use of it
  // is limited to writing trap instructions on the last executable segment.
  config->commonPageSize = getCommonPageSize(args);

  config->imageBase = getImageBase(args);

  if (config->emachine == EM_ARM) {
    // FIXME: These warnings can be removed when lld only uses these features
    // when the input objects have been compiled with an architecture that
    // supports them.
    if (config->armHasBlx == false)
      warn("lld uses blx instruction, no object with architecture supporting "
           "feature detected");
  }

  // This adds a .comment section containing a version string.
  if (!config->relocatable)
    inputSections.push_back(createCommentSection());

  // Split SHF_MERGE and .eh_frame sections into pieces in preparation for garbage collection.
  invokeELFT(splitSections);

  // Garbage collection and removal of shared symbols from unused shared objects.
  invokeELFT(markLive);
  demoteSharedAndLazySymbols();

  // Make copies of any input sections that need to be copied into each
  // partition.
  copySectionsIntoPartitions();

  // Create synthesized sections such as .got and .plt. This is called before
  // processSectionCommands() so that they can be placed by SECTIONS commands.
  invokeELFT(createSyntheticSections);

  // Some input sections that are used for exception handling need to be moved
  // into synthetic sections. Do that now so that they aren't assigned to
  // output sections in the usual way.
  if (!config->relocatable)
    combineEhSections();

  {
    llvm::TimeTraceScope timeScope("Assign sections");

    // Create output sections described by SECTIONS commands.
    script->processSectionCommands();

    // Linker scripts control how input sections are assigned to output
    // sections. Input sections that were not handled by scripts are called
    // "orphans", and they are assigned to output sections by the default rule.
    // Process that.
    script->addOrphanSections();
  }

  {
    llvm::TimeTraceScope timeScope("Merge/finalize input sections");

    // Migrate InputSectionDescription::sectionBases to sections. This includes
    // merging MergeInputSections into a single MergeSyntheticSection. From this
    // point onwards InputSectionDescription::sections should be used instead of
    // sectionBases.
    for (SectionCommand *cmd : script->sectionCommands)
      if (auto *osd = dyn_cast<OutputDesc>(cmd))
        osd->osec.finalizeInputSections();
    llvm::erase_if(inputSections, [](InputSectionBase *s) {
      return isa<MergeInputSection>(s);
    });
  }

  // Two input sections with different output sections should not be folded.
  // ICF runs after processSectionCommands() so that we know the output sections.
  if (config->icf != ICFLevel::None) {
    invokeELFT(findKeepUniqueSections, args);
    invokeELFT(doIcf);
  }

  // Read the callgraph now that we know what was gced or icfed
  if (config->callGraphProfileSort) {
    if (auto *arg = args.getLastArg(OPT_call_graph_ordering_file))
      if (Optional<MemoryBufferRef> buffer = readFile(arg->getValue()))
        readCallGraph(*buffer);
    invokeELFT(readCallGraphsFromObjectFiles);
  }

  // Write the result to the file.
  invokeELFT(writeResult);
}
