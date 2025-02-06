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
// usually explicitly specified by the compiler ctx.driver. They can even
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
#include "LTO.h"
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
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Filesystem.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Strings.h"
#include "lld/Common/TargetOptionsCommandFlags.h"
#include "lld/Common/Version.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Remarks/HotnessThresholdParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SaveAndRestore.h"
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
#include <tuple>
#include <utility>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::sys;
using namespace llvm::support;
using namespace lld;
using namespace lld::elf;

static void setConfigs(Ctx &ctx, opt::InputArgList &args);
static void readConfigs(Ctx &ctx, opt::InputArgList &args);

ELFSyncStream elf::Log(Ctx &ctx) { return {ctx, DiagLevel::Log}; }
ELFSyncStream elf::Msg(Ctx &ctx) { return {ctx, DiagLevel::Msg}; }
ELFSyncStream elf::Warn(Ctx &ctx) { return {ctx, DiagLevel::Warn}; }
ELFSyncStream elf::Err(Ctx &ctx) {
  return {ctx, ctx.arg.noinhibitExec ? DiagLevel::Warn : DiagLevel::Err};
}
ELFSyncStream elf::ErrAlways(Ctx &ctx) { return {ctx, DiagLevel::Err}; }
ELFSyncStream elf::Fatal(Ctx &ctx) { return {ctx, DiagLevel::Fatal}; }
uint64_t elf::errCount(Ctx &ctx) { return ctx.e.errorCount; }

ELFSyncStream elf::InternalErr(Ctx &ctx, const uint8_t *buf) {
  ELFSyncStream s(ctx, DiagLevel::Err);
  s << "internal linker error: ";
  return s;
}

Ctx::Ctx() : driver(*this) {}

llvm::raw_fd_ostream Ctx::openAuxiliaryFile(llvm::StringRef filename,
                                            std::error_code &ec) {
  using namespace llvm::sys::fs;
  OpenFlags flags =
      auxiliaryFiles.insert(filename).second ? OF_None : OF_Append;
  if (e.disableOutput && filename == "-") {
#ifdef _WIN32
    filename = "NUL";
#else
    filename = "/dev/null";
#endif
  }
  return {filename, ec, flags};
}

namespace lld {
namespace elf {
bool link(ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) {
  // This driver-specific context will be freed later by unsafeLldMain().
  auto *context = new Ctx;
  Ctx &ctx = *context;

  context->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);
  context->e.logName = args::getFilenameWithoutExe(args[0]);
  context->e.errorLimitExceededMsg =
      "too many errors emitted, stopping now (use "
      "--error-limit=0 to see all errors)";

  LinkerScript script(ctx);
  ctx.script = &script;
  ctx.symAux.emplace_back();
  ctx.symtab = std::make_unique<SymbolTable>(ctx);

  ctx.partitions.clear();
  ctx.partitions.emplace_back(ctx);

  ctx.arg.progName = args[0];

  ctx.driver.linkerMain(args);

  return errCount(ctx) == 0;
}
} // namespace elf
} // namespace lld

// Parses a linker -m option.
static std::tuple<ELFKind, uint16_t, uint8_t> parseEmulation(Ctx &ctx,
                                                             StringRef emul) {
  uint8_t osabi = 0;
  StringRef s = emul;
  if (s.ends_with("_fbsd")) {
    s = s.drop_back(5);
    osabi = ELFOSABI_FREEBSD;
  }

  std::pair<ELFKind, uint16_t> ret =
      StringSwitch<std::pair<ELFKind, uint16_t>>(s)
          .Cases("aarch64elf", "aarch64linux", {ELF64LEKind, EM_AARCH64})
          .Cases("aarch64elfb", "aarch64linuxb", {ELF64BEKind, EM_AARCH64})
          .Cases("armelf", "armelf_linux_eabi", {ELF32LEKind, EM_ARM})
          .Cases("armelfb", "armelfb_linux_eabi", {ELF32BEKind, EM_ARM})
          .Case("elf32_x86_64", {ELF32LEKind, EM_X86_64})
          .Cases("elf32btsmip", "elf32btsmipn32", {ELF32BEKind, EM_MIPS})
          .Cases("elf32ltsmip", "elf32ltsmipn32", {ELF32LEKind, EM_MIPS})
          .Case("elf32lriscv", {ELF32LEKind, EM_RISCV})
          .Cases("elf32ppc", "elf32ppclinux", {ELF32BEKind, EM_PPC})
          .Cases("elf32lppc", "elf32lppclinux", {ELF32LEKind, EM_PPC})
          .Case("elf32loongarch", {ELF32LEKind, EM_LOONGARCH})
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
          .Case("elf64_amdgpu", {ELF64LEKind, EM_AMDGPU})
          .Case("elf64loongarch", {ELF64LEKind, EM_LOONGARCH})
          .Case("elf64_s390", {ELF64BEKind, EM_S390})
          .Case("hexagonelf", {ELF32LEKind, EM_HEXAGON})
          .Default({ELFNoneKind, EM_NONE});

  if (ret.first == ELFNoneKind)
    ErrAlways(ctx) << "unknown emulation: " << emul;
  if (ret.second == EM_MSP430)
    osabi = ELFOSABI_STANDALONE;
  else if (ret.second == EM_AMDGPU)
    osabi = ELFOSABI_AMDGPU_HSA;
  return std::make_tuple(ret.first, ret.second, osabi);
}

// Returns slices of MB by parsing MB as an archive file.
// Each slice consists of a member file in the archive.
std::vector<std::pair<MemoryBufferRef, uint64_t>> static getArchiveMembers(
    Ctx &ctx, MemoryBufferRef mb) {
  std::unique_ptr<Archive> file =
      CHECK(Archive::create(mb),
            mb.getBufferIdentifier() + ": failed to parse archive");

  std::vector<std::pair<MemoryBufferRef, uint64_t>> v;
  Error err = Error::success();
  bool addToTar = file->isThin() && ctx.tar;
  for (const Archive::Child &c : file->children(err)) {
    MemoryBufferRef mbref =
        CHECK(c.getMemoryBufferRef(),
              mb.getBufferIdentifier() +
                  ": could not get the buffer for a child of the archive");
    if (addToTar)
      ctx.tar->append(relativeToRoot(check(c.getFullName())),
                      mbref.getBuffer());
    v.push_back(std::make_pair(mbref, c.getChildOffset()));
  }
  if (err)
    Fatal(ctx) << mb.getBufferIdentifier()
               << ": Archive::children failed: " << std::move(err);

  // Take ownership of memory buffers created for members of thin archives.
  std::vector<std::unique_ptr<MemoryBuffer>> mbs = file->takeThinBuffers();
  std::move(mbs.begin(), mbs.end(), std::back_inserter(ctx.memoryBuffers));

  return v;
}

static bool isBitcode(MemoryBufferRef mb) {
  return identify_magic(mb.getBuffer()) == llvm::file_magic::bitcode;
}

bool LinkerDriver::tryAddFatLTOFile(MemoryBufferRef mb, StringRef archiveName,
                                    uint64_t offsetInArchive, bool lazy) {
  if (!ctx.arg.fatLTOObjects)
    return false;
  Expected<MemoryBufferRef> fatLTOData =
      IRObjectFile::findBitcodeInMemBuffer(mb);
  if (errorToBool(fatLTOData.takeError()))
    return false;
  files.push_back(std::make_unique<BitcodeFile>(ctx, *fatLTOData, archiveName,
                                                offsetInArchive, lazy));
  return true;
}

// Opens a file and create a file object. Path has to be resolved already.
void LinkerDriver::addFile(StringRef path, bool withLOption) {
  using namespace sys::fs;

  std::optional<MemoryBufferRef> buffer = readFile(ctx, path);
  if (!buffer)
    return;
  MemoryBufferRef mbref = *buffer;

  if (ctx.arg.formatBinary) {
    files.push_back(std::make_unique<BinaryFile>(ctx, mbref));
    return;
  }

  switch (identify_magic(mbref.getBuffer())) {
  case file_magic::unknown:
    readLinkerScript(ctx, mbref);
    return;
  case file_magic::archive: {
    auto members = getArchiveMembers(ctx, mbref);
    if (inWholeArchive) {
      for (const std::pair<MemoryBufferRef, uint64_t> &p : members) {
        if (isBitcode(p.first))
          files.push_back(std::make_unique<BitcodeFile>(ctx, p.first, path,
                                                        p.second, false));
        else if (!tryAddFatLTOFile(p.first, path, p.second, false))
          files.push_back(createObjFile(ctx, p.first, path));
      }
      return;
    }

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
    SaveAndRestore saved(isInGroup, true);
    for (const std::pair<MemoryBufferRef, uint64_t> &p : members) {
      auto magic = identify_magic(p.first.getBuffer());
      if (magic == file_magic::elf_relocatable) {
        if (!tryAddFatLTOFile(p.first, path, p.second, true))
          files.push_back(createObjFile(ctx, p.first, path, true));
      } else if (magic == file_magic::bitcode)
        files.push_back(
            std::make_unique<BitcodeFile>(ctx, p.first, path, p.second, true));
      else
        Warn(ctx) << path << ": archive member '"
                  << p.first.getBufferIdentifier()
                  << "' is neither ET_REL nor LLVM bitcode";
    }
    if (!saved.get())
      ++nextGroupId;
    return;
  }
  case file_magic::elf_shared_object: {
    if (ctx.arg.isStatic) {
      ErrAlways(ctx) << "attempted static link of dynamic object " << path;
      return;
    }

    // Shared objects are identified by soname. soname is (if specified)
    // DT_SONAME and falls back to filename. If a file was specified by -lfoo,
    // the directory part is ignored. Note that path may be a temporary and
    // cannot be stored into SharedFile::soName.
    path = mbref.getBufferIdentifier();
    auto f = std::make_unique<SharedFile>(
        ctx, mbref, withLOption ? path::filename(path) : path);
    f->init();
    files.push_back(std::move(f));
    return;
  }
  case file_magic::bitcode:
    files.push_back(std::make_unique<BitcodeFile>(ctx, mbref, "", 0, inLib));
    break;
  case file_magic::elf_relocatable:
    if (!tryAddFatLTOFile(mbref, "", 0, inLib))
      files.push_back(createObjFile(ctx, mbref, "", inLib));
    break;
  default:
    ErrAlways(ctx) << path << ": unknown file type";
  }
}

// Add a given library by searching it from input search paths.
void LinkerDriver::addLibrary(StringRef name) {
  if (std::optional<std::string> path = searchLibrary(ctx, name))
    addFile(ctx.saver.save(*path), /*withLOption=*/true);
  else
    ctx.e.error("unable to find library -l" + name, ErrorTag::LibNotFound,
                {name});
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
static void checkOptions(Ctx &ctx) {
  // The MIPS ABI as of 2016 does not support the GNU-style symbol lookup
  // table which is a relatively new feature.
  if (ctx.arg.emachine == EM_MIPS && ctx.arg.gnuHash)
    ErrAlways(ctx)
        << "the .gnu.hash section is not compatible with the MIPS target";

  if (ctx.arg.emachine == EM_ARM) {
    if (!ctx.arg.cmseImplib) {
      if (!ctx.arg.cmseInputLib.empty())
        ErrAlways(ctx) << "--in-implib may not be used without --cmse-implib";
      if (!ctx.arg.cmseOutputLib.empty())
        ErrAlways(ctx) << "--out-implib may not be used without --cmse-implib";
    }
  } else {
    if (ctx.arg.cmseImplib)
      ErrAlways(ctx) << "--cmse-implib is only supported on ARM targets";
    if (!ctx.arg.cmseInputLib.empty())
      ErrAlways(ctx) << "--in-implib is only supported on ARM targets";
    if (!ctx.arg.cmseOutputLib.empty())
      ErrAlways(ctx) << "--out-implib is only supported on ARM targets";
  }

  if (ctx.arg.fixCortexA53Errata843419 && ctx.arg.emachine != EM_AARCH64)
    ErrAlways(ctx)
        << "--fix-cortex-a53-843419 is only supported on AArch64 targets";

  if (ctx.arg.fixCortexA8 && ctx.arg.emachine != EM_ARM)
    ErrAlways(ctx) << "--fix-cortex-a8 is only supported on ARM targets";

  if (ctx.arg.armBe8 && ctx.arg.emachine != EM_ARM)
    ErrAlways(ctx) << "--be8 is only supported on ARM targets";

  if (ctx.arg.fixCortexA8 && !ctx.arg.isLE)
    ErrAlways(ctx) << "--fix-cortex-a8 is not supported on big endian targets";

  if (ctx.arg.tocOptimize && ctx.arg.emachine != EM_PPC64)
    ErrAlways(ctx) << "--toc-optimize is only supported on PowerPC64 targets";

  if (ctx.arg.pcRelOptimize && ctx.arg.emachine != EM_PPC64)
    ErrAlways(ctx) << "--pcrel-optimize is only supported on PowerPC64 targets";

  if (ctx.arg.relaxGP && ctx.arg.emachine != EM_RISCV)
    ErrAlways(ctx) << "--relax-gp is only supported on RISC-V targets";

  if (ctx.arg.pie && ctx.arg.shared)
    ErrAlways(ctx) << "-shared and -pie may not be used together";

  if (!ctx.arg.shared && !ctx.arg.filterList.empty())
    ErrAlways(ctx) << "-F may not be used without -shared";

  if (!ctx.arg.shared && !ctx.arg.auxiliaryList.empty())
    ErrAlways(ctx) << "-f may not be used without -shared";

  if (ctx.arg.strip == StripPolicy::All && ctx.arg.emitRelocs)
    ErrAlways(ctx) << "--strip-all and --emit-relocs may not be used together";

  if (ctx.arg.zText && ctx.arg.zIfuncNoplt)
    ErrAlways(ctx) << "-z text and -z ifunc-noplt may not be used together";

  if (ctx.arg.relocatable) {
    if (ctx.arg.shared)
      ErrAlways(ctx) << "-r and -shared may not be used together";
    if (ctx.arg.gdbIndex)
      ErrAlways(ctx) << "-r and --gdb-index may not be used together";
    if (ctx.arg.icf != ICFLevel::None)
      ErrAlways(ctx) << "-r and --icf may not be used together";
    if (ctx.arg.pie)
      ErrAlways(ctx) << "-r and -pie may not be used together";
    if (ctx.arg.exportDynamic)
      ErrAlways(ctx) << "-r and --export-dynamic may not be used together";
    if (ctx.arg.debugNames)
      ErrAlways(ctx) << "-r and --debug-names may not be used together";
    if (!ctx.arg.zSectionHeader)
      ErrAlways(ctx) << "-r and -z nosectionheader may not be used together";
  }

  if (ctx.arg.executeOnly) {
    if (ctx.arg.emachine != EM_AARCH64)
      ErrAlways(ctx) << "--execute-only is only supported on AArch64 targets";

    if (ctx.arg.singleRoRx && !ctx.script->hasSectionsCommand)
      ErrAlways(ctx)
          << "--execute-only and --no-rosegment cannot be used together";
  }

  if (ctx.arg.zRetpolineplt && ctx.arg.zForceIbt)
    ErrAlways(ctx) << "-z force-ibt may not be used with -z retpolineplt";

  if (ctx.arg.emachine != EM_AARCH64) {
    if (ctx.arg.zPacPlt)
      ErrAlways(ctx) << "-z pac-plt only supported on AArch64";
    if (ctx.arg.zForceBti)
      ErrAlways(ctx) << "-z force-bti only supported on AArch64";
    if (ctx.arg.zBtiReport != "none")
      ErrAlways(ctx) << "-z bti-report only supported on AArch64";
    if (ctx.arg.zPauthReport != "none")
      ErrAlways(ctx) << "-z pauth-report only supported on AArch64";
    if (ctx.arg.zGcsReport != "none")
      ErrAlways(ctx) << "-z gcs-report only supported on AArch64";
    if (ctx.arg.zGcs != GcsPolicy::Implicit)
      ErrAlways(ctx) << "-z gcs only supported on AArch64";
  }

  if (ctx.arg.emachine != EM_386 && ctx.arg.emachine != EM_X86_64 &&
      ctx.arg.zCetReport != "none")
    ErrAlways(ctx) << "-z cet-report only supported on X86 and X86_64";
}

static const char *getReproduceOption(opt::InputArgList &args) {
  if (auto *arg = args.getLastArg(OPT_reproduce))
    return arg->getValue();
  return getenv("LLD_REPRODUCE");
}

static bool hasZOption(opt::InputArgList &args, StringRef key) {
  bool ret = false;
  for (auto *arg : args.filtered(OPT_z))
    if (key == arg->getValue()) {
      ret = true;
      arg->claim();
    }
  return ret;
}

static bool getZFlag(opt::InputArgList &args, StringRef k1, StringRef k2,
                     bool defaultValue) {
  for (auto *arg : args.filtered(OPT_z)) {
    StringRef v = arg->getValue();
    if (k1 == v)
      defaultValue = true;
    else if (k2 == v)
      defaultValue = false;
    else
      continue;
    arg->claim();
  }
  return defaultValue;
}

static SeparateSegmentKind getZSeparate(opt::InputArgList &args) {
  auto ret = SeparateSegmentKind::None;
  for (auto *arg : args.filtered(OPT_z)) {
    StringRef v = arg->getValue();
    if (v == "noseparate-code")
      ret = SeparateSegmentKind::None;
    else if (v == "separate-code")
      ret = SeparateSegmentKind::Code;
    else if (v == "separate-loadable-segments")
      ret = SeparateSegmentKind::Loadable;
    else
      continue;
    arg->claim();
  }
  return ret;
}

static GnuStackKind getZGnuStack(opt::InputArgList &args) {
  auto ret = GnuStackKind::NoExec;
  for (auto *arg : args.filtered(OPT_z)) {
    StringRef v = arg->getValue();
    if (v == "execstack")
      ret = GnuStackKind::Exec;
    else if (v == "noexecstack")
      ret = GnuStackKind::NoExec;
    else if (v == "nognustack")
      ret = GnuStackKind::None;
    else
      continue;
    arg->claim();
  }
  return ret;
}

static uint8_t getZStartStopVisibility(Ctx &ctx, opt::InputArgList &args) {
  uint8_t ret = STV_PROTECTED;
  for (auto *arg : args.filtered(OPT_z)) {
    std::pair<StringRef, StringRef> kv = StringRef(arg->getValue()).split('=');
    if (kv.first == "start-stop-visibility") {
      arg->claim();
      if (kv.second == "default")
        ret = STV_DEFAULT;
      else if (kv.second == "internal")
        ret = STV_INTERNAL;
      else if (kv.second == "hidden")
        ret = STV_HIDDEN;
      else if (kv.second == "protected")
        ret = STV_PROTECTED;
      else
        ErrAlways(ctx) << "unknown -z start-stop-visibility= value: "
                       << StringRef(kv.second);
    }
  }
  return ret;
}

static GcsPolicy getZGcs(Ctx &ctx, opt::InputArgList &args) {
  GcsPolicy ret = GcsPolicy::Implicit;
  for (auto *arg : args.filtered(OPT_z)) {
    std::pair<StringRef, StringRef> kv = StringRef(arg->getValue()).split('=');
    if (kv.first == "gcs") {
      arg->claim();
      if (kv.second == "implicit")
        ret = GcsPolicy::Implicit;
      else if (kv.second == "never")
        ret = GcsPolicy::Never;
      else if (kv.second == "always")
        ret = GcsPolicy::Always;
      else
        ErrAlways(ctx) << "unknown -z gcs= value: " << kv.second;
    }
  }
  return ret;
}

// Report a warning for an unknown -z option.
static void checkZOptions(Ctx &ctx, opt::InputArgList &args) {
  // This function is called before getTarget(), when certain options are not
  // initialized yet. Claim them here.
  args::getZOptionValue(args, OPT_z, "max-page-size", 0);
  args::getZOptionValue(args, OPT_z, "common-page-size", 0);
  getZFlag(args, "rel", "rela", false);
  for (auto *arg : args.filtered(OPT_z))
    if (!arg->isClaimed())
      Warn(ctx) << "unknown -z value: " << StringRef(arg->getValue());
}

constexpr const char *saveTempsValues[] = {
    "resolution", "preopt",     "promote", "internalize",  "import",
    "opt",        "precodegen", "prelink", "combinedindex"};

LinkerDriver::LinkerDriver(Ctx &ctx) : ctx(ctx) {}

void LinkerDriver::linkerMain(ArrayRef<const char *> argsArr) {
  ELFOptTable parser;
  opt::InputArgList args = parser.parse(ctx, argsArr.slice(1));

  // Interpret these flags early because Err/Warn depend on them.
  ctx.e.errorLimit = args::getInteger(args, OPT_error_limit, 20);
  ctx.e.fatalWarnings =
      args.hasFlag(OPT_fatal_warnings, OPT_no_fatal_warnings, false) &&
      !args.hasArg(OPT_no_warnings);
  ctx.e.suppressWarnings = args.hasArg(OPT_no_warnings);

  // Handle -help
  if (args.hasArg(OPT_help)) {
    printHelp(ctx);
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
    Msg(ctx) << getLLDVersion() << " (compatible with GNU linkers)";

  if (const char *path = getReproduceOption(args)) {
    // Note that --reproduce is a debug option so you can ignore it
    // if you are trying to understand the whole picture of the code.
    Expected<std::unique_ptr<TarWriter>> errOrWriter =
        TarWriter::create(path, path::stem(path));
    if (errOrWriter) {
      ctx.tar = std::move(*errOrWriter);
      ctx.tar->append("response.txt", createResponseFile(args));
      ctx.tar->append("version.txt", getLLDVersion() + "\n");
      StringRef ltoSampleProfile = args.getLastArgValue(OPT_lto_sample_profile);
      if (!ltoSampleProfile.empty())
        readFile(ctx, ltoSampleProfile);
    } else {
      ErrAlways(ctx) << "--reproduce: " << errOrWriter.takeError();
    }
  }

  readConfigs(ctx, args);
  checkZOptions(ctx, args);

  // The behavior of -v or --version is a bit strange, but this is
  // needed for compatibility with GNU linkers.
  if (args.hasArg(OPT_v) && !args.hasArg(OPT_INPUT))
    return;
  if (args.hasArg(OPT_version))
    return;

  // Initialize time trace profiler.
  if (ctx.arg.timeTraceEnabled)
    timeTraceProfilerInitialize(ctx.arg.timeTraceGranularity, ctx.arg.progName);

  {
    llvm::TimeTraceScope timeScope("ExecuteLinker");

    initLLVM();
    createFiles(args);
    if (errCount(ctx))
      return;

    inferMachineType();
    setConfigs(ctx, args);
    checkOptions(ctx);
    if (errCount(ctx))
      return;

    invokeELFT(link, args);
  }

  if (ctx.arg.timeTraceEnabled) {
    checkError(ctx.e, timeTraceProfilerWrite(
                          args.getLastArgValue(OPT_time_trace_eq).str(),
                          ctx.arg.outputFile));
    timeTraceProfilerCleanup();
  }
}

static std::string getRpath(opt::InputArgList &args) {
  SmallVector<StringRef, 0> v = args::getStrings(args, OPT_rpath);
  return llvm::join(v.begin(), v.end(), ":");
}

// Determines what we should do if there are remaining unresolved
// symbols after the name resolution.
static void setUnresolvedSymbolPolicy(Ctx &ctx, opt::InputArgList &args) {
  UnresolvedPolicy errorOrWarn = args.hasFlag(OPT_error_unresolved_symbols,
                                              OPT_warn_unresolved_symbols, true)
                                     ? UnresolvedPolicy::ReportError
                                     : UnresolvedPolicy::Warn;
  // -shared implies --unresolved-symbols=ignore-all because missing
  // symbols are likely to be resolved at runtime.
  bool diagRegular = !ctx.arg.shared, diagShlib = !ctx.arg.shared;

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
        ErrAlways(ctx) << "unknown --unresolved-symbols value: " << s;
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
      else
        break;
      arg->claim();
      break;
    case OPT_allow_shlib_undefined:
      diagShlib = false;
      break;
    case OPT_no_allow_shlib_undefined:
      diagShlib = true;
      break;
    }
  }

  ctx.arg.unresolvedSymbols =
      diagRegular ? errorOrWarn : UnresolvedPolicy::Ignore;
  ctx.arg.unresolvedSymbolsInShlib =
      diagShlib ? errorOrWarn : UnresolvedPolicy::Ignore;
}

static Target2Policy getTarget2(Ctx &ctx, opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_target2, "got-rel");
  if (s == "rel")
    return Target2Policy::Rel;
  if (s == "abs")
    return Target2Policy::Abs;
  if (s == "got-rel")
    return Target2Policy::GotRel;
  ErrAlways(ctx) << "unknown --target2 option: " << s;
  return Target2Policy::GotRel;
}

static bool isOutputFormatBinary(Ctx &ctx, opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_oformat, "elf");
  if (s == "binary")
    return true;
  if (!s.starts_with("elf"))
    ErrAlways(ctx) << "unknown --oformat value: " << s;
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

static StringRef getDynamicLinker(Ctx &ctx, opt::InputArgList &args) {
  auto *arg = args.getLastArg(OPT_dynamic_linker, OPT_no_dynamic_linker);
  if (!arg)
    return "";
  if (arg->getOption().getID() == OPT_no_dynamic_linker) {
    // --no-dynamic-linker suppresses undefined weak symbols in .dynsym
    ctx.arg.noDynamicLinker = true;
    return "";
  }
  return arg->getValue();
}

static int getMemtagMode(Ctx &ctx, opt::InputArgList &args) {
  StringRef memtagModeArg = args.getLastArgValue(OPT_android_memtag_mode);
  if (memtagModeArg.empty()) {
    if (ctx.arg.androidMemtagStack)
      Warn(ctx) << "--android-memtag-mode is unspecified, leaving "
                   "--android-memtag-stack a no-op";
    else if (ctx.arg.androidMemtagHeap)
      Warn(ctx) << "--android-memtag-mode is unspecified, leaving "
                   "--android-memtag-heap a no-op";
    return ELF::NT_MEMTAG_LEVEL_NONE;
  }

  if (memtagModeArg == "sync")
    return ELF::NT_MEMTAG_LEVEL_SYNC;
  if (memtagModeArg == "async")
    return ELF::NT_MEMTAG_LEVEL_ASYNC;
  if (memtagModeArg == "none")
    return ELF::NT_MEMTAG_LEVEL_NONE;

  ErrAlways(ctx) << "unknown --android-memtag-mode value: \"" << memtagModeArg
                 << "\", should be one of {async, sync, none}";
  return ELF::NT_MEMTAG_LEVEL_NONE;
}

static ICFLevel getICF(opt::InputArgList &args) {
  auto *arg = args.getLastArg(OPT_icf_none, OPT_icf_safe, OPT_icf_all);
  if (!arg || arg->getOption().getID() == OPT_icf_none)
    return ICFLevel::None;
  if (arg->getOption().getID() == OPT_icf_safe)
    return ICFLevel::Safe;
  return ICFLevel::All;
}

static StripPolicy getStrip(Ctx &ctx, opt::InputArgList &args) {
  if (args.hasArg(OPT_relocatable))
    return StripPolicy::None;
  if (!ctx.arg.zSectionHeader)
    return StripPolicy::All;

  auto *arg = args.getLastArg(OPT_strip_all, OPT_strip_debug);
  if (!arg)
    return StripPolicy::None;
  if (arg->getOption().getID() == OPT_strip_all)
    return StripPolicy::All;
  return StripPolicy::Debug;
}

static uint64_t parseSectionAddress(Ctx &ctx, StringRef s,
                                    opt::InputArgList &args,
                                    const opt::Arg &arg) {
  uint64_t va = 0;
  s.consume_front("0x");
  if (!to_integer(s, va, 16))
    ErrAlways(ctx) << "invalid argument: " << arg.getAsString(args);
  return va;
}

static StringMap<uint64_t> getSectionStartMap(Ctx &ctx,
                                              opt::InputArgList &args) {
  StringMap<uint64_t> ret;
  for (auto *arg : args.filtered(OPT_section_start)) {
    StringRef name;
    StringRef addr;
    std::tie(name, addr) = StringRef(arg->getValue()).split('=');
    ret[name] = parseSectionAddress(ctx, addr, args, *arg);
  }

  if (auto *arg = args.getLastArg(OPT_Ttext))
    ret[".text"] = parseSectionAddress(ctx, arg->getValue(), args, *arg);
  if (auto *arg = args.getLastArg(OPT_Tdata))
    ret[".data"] = parseSectionAddress(ctx, arg->getValue(), args, *arg);
  if (auto *arg = args.getLastArg(OPT_Tbss))
    ret[".bss"] = parseSectionAddress(ctx, arg->getValue(), args, *arg);
  return ret;
}

static SortSectionPolicy getSortSection(Ctx &ctx, opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_sort_section);
  if (s == "alignment")
    return SortSectionPolicy::Alignment;
  if (s == "name")
    return SortSectionPolicy::Name;
  if (!s.empty())
    ErrAlways(ctx) << "unknown --sort-section rule: " << s;
  return SortSectionPolicy::Default;
}

static OrphanHandlingPolicy getOrphanHandling(Ctx &ctx,
                                              opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_orphan_handling, "place");
  if (s == "warn")
    return OrphanHandlingPolicy::Warn;
  if (s == "error")
    return OrphanHandlingPolicy::Error;
  if (s != "place")
    ErrAlways(ctx) << "unknown --orphan-handling mode: " << s;
  return OrphanHandlingPolicy::Place;
}

// Parse --build-id or --build-id=<style>. We handle "tree" as a
// synonym for "sha1" because all our hash functions including
// --build-id=sha1 are actually tree hashes for performance reasons.
static std::pair<BuildIdKind, SmallVector<uint8_t, 0>>
getBuildId(Ctx &ctx, opt::InputArgList &args) {
  auto *arg = args.getLastArg(OPT_build_id);
  if (!arg)
    return {BuildIdKind::None, {}};

  StringRef s = arg->getValue();
  if (s == "fast")
    return {BuildIdKind::Fast, {}};
  if (s == "md5")
    return {BuildIdKind::Md5, {}};
  if (s == "sha1" || s == "tree")
    return {BuildIdKind::Sha1, {}};
  if (s == "uuid")
    return {BuildIdKind::Uuid, {}};
  if (s.starts_with("0x"))
    return {BuildIdKind::Hexstring, parseHex(s.substr(2))};

  if (s != "none")
    ErrAlways(ctx) << "unknown --build-id style: " << s;
  return {BuildIdKind::None, {}};
}

static std::pair<bool, bool> getPackDynRelocs(Ctx &ctx,
                                              opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_pack_dyn_relocs, "none");
  if (s == "android")
    return {true, false};
  if (s == "relr")
    return {false, true};
  if (s == "android+relr")
    return {true, true};

  if (s != "none")
    ErrAlways(ctx) << "unknown --pack-dyn-relocs format: " << s;
  return {false, false};
}

static void readCallGraph(Ctx &ctx, MemoryBufferRef mb) {
  // Build a map from symbol name to section
  DenseMap<StringRef, Symbol *> map;
  for (ELFFileBase *file : ctx.objectFiles)
    for (Symbol *sym : file->getSymbols())
      map[sym->getName()] = sym;

  auto findSection = [&](StringRef name) -> InputSectionBase * {
    Symbol *sym = map.lookup(name);
    if (!sym) {
      if (ctx.arg.warnSymbolOrdering)
        Warn(ctx) << mb.getBufferIdentifier() << ": no such symbol: " << name;
      return nullptr;
    }
    maybeWarnUnorderableSymbol(ctx, sym);

    if (Defined *dr = dyn_cast_or_null<Defined>(sym))
      return dyn_cast_or_null<InputSectionBase>(dr->section);
    return nullptr;
  };

  for (StringRef line : args::getLines(mb)) {
    SmallVector<StringRef, 3> fields;
    line.split(fields, ' ');
    uint64_t count;

    if (fields.size() != 3 || !to_integer(fields[2], count)) {
      ErrAlways(ctx) << mb.getBufferIdentifier() << ": parse error";
      return;
    }

    if (InputSectionBase *from = findSection(fields[0]))
      if (InputSectionBase *to = findSection(fields[1]))
        ctx.arg.callGraphProfile[std::make_pair(from, to)] += count;
  }
}

// If SHT_LLVM_CALL_GRAPH_PROFILE and its relocation section exist, returns
// true and populates cgProfile and symbolIndices.
template <class ELFT>
static bool
processCallGraphRelocations(Ctx &ctx, SmallVector<uint32_t, 32> &symbolIndices,
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
      if (sec.sh_type == SHT_CREL) {
        auto crels =
            CHECK(obj.crels(sec), "could not retrieve cg profile rela section");
        for (const auto &rel : crels.first)
          symbolIndices.push_back(rel.getSymbol(false));
        for (const auto &rel : crels.second)
          symbolIndices.push_back(rel.getSymbol(false));
        break;
      }
      if (sec.sh_type == SHT_RELA) {
        ArrayRef<typename ELFT::Rela> relas =
            CHECK(obj.relas(sec), "could not retrieve cg profile rela section");
        for (const typename ELFT::Rela &rel : relas)
          symbolIndices.push_back(rel.getSymbol(ctx.arg.isMips64EL));
        break;
      }
      if (sec.sh_type == SHT_REL) {
        ArrayRef<typename ELFT::Rel> rels =
            CHECK(obj.rels(sec), "could not retrieve cg profile rel section");
        for (const typename ELFT::Rel &rel : rels)
          symbolIndices.push_back(rel.getSymbol(ctx.arg.isMips64EL));
        break;
      }
    }
  }
  if (symbolIndices.empty())
    Warn(ctx)
        << "SHT_LLVM_CALL_GRAPH_PROFILE exists, but relocation section doesn't";
  return !symbolIndices.empty();
}

template <class ELFT> static void readCallGraphsFromObjectFiles(Ctx &ctx) {
  SmallVector<uint32_t, 32> symbolIndices;
  ArrayRef<typename ELFT::CGProfile> cgProfile;
  for (auto file : ctx.objectFiles) {
    auto *obj = cast<ObjFile<ELFT>>(file);
    if (!processCallGraphRelocations(ctx, symbolIndices, cgProfile, obj))
      continue;

    if (symbolIndices.size() != cgProfile.size() * 2)
      Fatal(ctx) << "number of relocations doesn't match Weights";

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
        ctx.arg.callGraphProfile[{from, to}] += cgpe.cgp_weight;
    }
  }
}

template <class ELFT>
static void ltoValidateAllVtablesHaveTypeInfos(Ctx &ctx,
                                               opt::InputArgList &args) {
  DenseSet<StringRef> typeInfoSymbols;
  SmallSetVector<StringRef, 0> vtableSymbols;
  auto processVtableAndTypeInfoSymbols = [&](StringRef name) {
    if (name.consume_front("_ZTI"))
      typeInfoSymbols.insert(name);
    else if (name.consume_front("_ZTV"))
      vtableSymbols.insert(name);
  };

  // Examine all native symbol tables.
  for (ELFFileBase *f : ctx.objectFiles) {
    using Elf_Sym = typename ELFT::Sym;
    for (const Elf_Sym &s : f->template getGlobalELFSyms<ELFT>()) {
      if (s.st_shndx != SHN_UNDEF) {
        StringRef name = check(s.getName(f->getStringTable()));
        processVtableAndTypeInfoSymbols(name);
      }
    }
  }

  for (SharedFile *f : ctx.sharedFiles) {
    using Elf_Sym = typename ELFT::Sym;
    for (const Elf_Sym &s : f->template getELFSyms<ELFT>()) {
      if (s.st_shndx != SHN_UNDEF) {
        StringRef name = check(s.getName(f->getStringTable()));
        processVtableAndTypeInfoSymbols(name);
      }
    }
  }

  SmallSetVector<StringRef, 0> vtableSymbolsWithNoRTTI;
  for (StringRef s : vtableSymbols)
    if (!typeInfoSymbols.count(s))
      vtableSymbolsWithNoRTTI.insert(s);

  // Remove known safe symbols.
  for (auto *arg : args.filtered(OPT_lto_known_safe_vtables)) {
    StringRef knownSafeName = arg->getValue();
    if (!knownSafeName.consume_front("_ZTV"))
      ErrAlways(ctx)
          << "--lto-known-safe-vtables=: expected symbol to start with _ZTV, "
             "but got "
          << knownSafeName;
    Expected<GlobPattern> pat = GlobPattern::create(knownSafeName);
    if (!pat)
      ErrAlways(ctx) << "--lto-known-safe-vtables=: " << pat.takeError();
    vtableSymbolsWithNoRTTI.remove_if(
        [&](StringRef s) { return pat->match(s); });
  }

  ctx.ltoAllVtablesHaveTypeInfos = vtableSymbolsWithNoRTTI.empty();
  // Check for unmatched RTTI symbols
  for (StringRef s : vtableSymbolsWithNoRTTI) {
    Msg(ctx) << "--lto-validate-all-vtables-have-type-infos: RTTI missing for "
                "vtable "
                "_ZTV"
             << s << ", --lto-whole-program-visibility disabled";
  }
}

static CGProfileSortKind getCGProfileSortKind(Ctx &ctx,
                                              opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_call_graph_profile_sort, "cdsort");
  if (s == "hfsort")
    return CGProfileSortKind::Hfsort;
  if (s == "cdsort")
    return CGProfileSortKind::Cdsort;
  if (s != "none")
    ErrAlways(ctx) << "unknown --call-graph-profile-sort= value: " << s;
  return CGProfileSortKind::None;
}

static void parseBPOrdererOptions(Ctx &ctx, opt::InputArgList &args) {
  if (auto *arg = args.getLastArg(OPT_bp_compression_sort)) {
    StringRef s = arg->getValue();
    if (s == "function") {
      ctx.arg.bpFunctionOrderForCompression = true;
    } else if (s == "data") {
      ctx.arg.bpDataOrderForCompression = true;
    } else if (s == "both") {
      ctx.arg.bpFunctionOrderForCompression = true;
      ctx.arg.bpDataOrderForCompression = true;
    } else if (s != "none") {
      ErrAlways(ctx) << arg->getSpelling()
                     << ": expected [none|function|data|both]";
    }
    if (s != "none" && args.hasArg(OPT_call_graph_ordering_file))
      ErrAlways(ctx) << "--bp-compression-sort is incompatible with "
                        "--call-graph-ordering-file";
  }
  if (auto *arg = args.getLastArg(OPT_bp_startup_sort)) {
    StringRef s = arg->getValue();
    if (s == "function") {
      ctx.arg.bpStartupFunctionSort = true;
    } else if (s != "none") {
      ErrAlways(ctx) << arg->getSpelling() << ": expected [none|function]";
    }
    if (s != "none" && args.hasArg(OPT_call_graph_ordering_file))
      ErrAlways(ctx) << "--bp-startup-sort=function is incompatible with "
                        "--call-graph-ordering-file";
  }

  ctx.arg.bpCompressionSortStartupFunctions =
      args.hasFlag(OPT_bp_compression_sort_startup_functions,
                   OPT_no_bp_compression_sort_startup_functions, false);
  ctx.arg.bpVerboseSectionOrderer = args.hasArg(OPT_verbose_bp_section_orderer);

  ctx.arg.irpgoProfilePath = args.getLastArgValue(OPT_irpgo_profile);
  if (ctx.arg.irpgoProfilePath.empty()) {
    if (ctx.arg.bpStartupFunctionSort)
      ErrAlways(ctx) << "--bp-startup-sort=function must be used with "
                        "--irpgo-profile";
    if (ctx.arg.bpCompressionSortStartupFunctions)
      ErrAlways(ctx)
          << "--bp-compression-sort-startup-functions must be used with "
             "--irpgo-profile";
  }
}

static DebugCompressionType getCompressionType(Ctx &ctx, StringRef s,
                                               StringRef option) {
  DebugCompressionType type = StringSwitch<DebugCompressionType>(s)
                                  .Case("zlib", DebugCompressionType::Zlib)
                                  .Case("zstd", DebugCompressionType::Zstd)
                                  .Default(DebugCompressionType::None);
  if (type == DebugCompressionType::None) {
    if (s != "none")
      ErrAlways(ctx) << "unknown " << option << " value: " << s;
  } else if (const char *reason = compression::getReasonIfUnsupported(
                 compression::formatFor(type))) {
    ErrAlways(ctx) << option << ": " << reason;
  }
  return type;
}

static StringRef getAliasSpelling(opt::Arg *arg) {
  if (const opt::Arg *alias = arg->getAlias())
    return alias->getSpelling();
  return arg->getSpelling();
}

static std::pair<StringRef, StringRef>
getOldNewOptions(Ctx &ctx, opt::InputArgList &args, unsigned id) {
  auto *arg = args.getLastArg(id);
  if (!arg)
    return {"", ""};

  StringRef s = arg->getValue();
  std::pair<StringRef, StringRef> ret = s.split(';');
  if (ret.second.empty())
    ErrAlways(ctx) << getAliasSpelling(arg)
                   << " expects 'old;new' format, but got " << s;
  return ret;
}

// Parse options of the form "old;new[;extra]".
static std::tuple<StringRef, StringRef, StringRef>
getOldNewOptionsExtra(Ctx &ctx, opt::InputArgList &args, unsigned id) {
  auto [oldDir, second] = getOldNewOptions(ctx, args, id);
  auto [newDir, extraDir] = second.split(';');
  return {oldDir, newDir, extraDir};
}

// Parse the symbol ordering file and warn for any duplicate entries.
static SmallVector<StringRef, 0> getSymbolOrderingFile(Ctx &ctx,
                                                       MemoryBufferRef mb) {
  SetVector<StringRef, SmallVector<StringRef, 0>> names;
  for (StringRef s : args::getLines(mb))
    if (!names.insert(s) && ctx.arg.warnSymbolOrdering)
      Warn(ctx) << mb.getBufferIdentifier()
                << ": duplicate ordered symbol: " << s;

  return names.takeVector();
}

static bool getIsRela(Ctx &ctx, opt::InputArgList &args) {
  // The psABI specifies the default relocation entry format.
  bool rela = is_contained({EM_AARCH64, EM_AMDGPU, EM_HEXAGON, EM_LOONGARCH,
                            EM_PPC, EM_PPC64, EM_RISCV, EM_S390, EM_X86_64},
                           ctx.arg.emachine);
  // If -z rel or -z rela is specified, use the last option.
  for (auto *arg : args.filtered(OPT_z)) {
    StringRef s(arg->getValue());
    if (s == "rel")
      rela = false;
    else if (s == "rela")
      rela = true;
    else
      continue;
    arg->claim();
  }
  return rela;
}

static void parseClangOption(Ctx &ctx, StringRef opt, const Twine &msg) {
  std::string err;
  raw_string_ostream os(err);

  const char *argv[] = {ctx.arg.progName.data(), opt.data()};
  if (cl::ParseCommandLineOptions(2, argv, "", &os))
    return;
  ErrAlways(ctx) << msg << ": " << StringRef(err).trim();
}

// Checks the parameter of the bti-report and cet-report options.
static bool isValidReportString(StringRef arg) {
  return arg == "none" || arg == "warning" || arg == "error";
}

// Process a remap pattern 'from-glob=to-file'.
static bool remapInputs(Ctx &ctx, StringRef line, const Twine &location) {
  SmallVector<StringRef, 0> fields;
  line.split(fields, '=');
  if (fields.size() != 2 || fields[1].empty()) {
    ErrAlways(ctx) << location << ": parse error, not 'from-glob=to-file'";
    return true;
  }
  if (!hasWildcard(fields[0]))
    ctx.arg.remapInputs[fields[0]] = fields[1];
  else if (Expected<GlobPattern> pat = GlobPattern::create(fields[0]))
    ctx.arg.remapInputsWildcards.emplace_back(std::move(*pat), fields[1]);
  else {
    ErrAlways(ctx) << location << ": " << pat.takeError() << ": " << fields[0];
    return true;
  }
  return false;
}

// Initializes Config members by the command line options.
static void readConfigs(Ctx &ctx, opt::InputArgList &args) {
  ctx.e.verbose = args.hasArg(OPT_verbose);
  ctx.e.vsDiagnostics =
      args.hasArg(OPT_visual_studio_diagnostics_format, false);

  ctx.arg.allowMultipleDefinition =
      hasZOption(args, "muldefs") ||
      args.hasFlag(OPT_allow_multiple_definition,
                   OPT_no_allow_multiple_definition, false);
  ctx.arg.androidMemtagHeap =
      args.hasFlag(OPT_android_memtag_heap, OPT_no_android_memtag_heap, false);
  ctx.arg.androidMemtagStack = args.hasFlag(OPT_android_memtag_stack,
                                            OPT_no_android_memtag_stack, false);
  ctx.arg.fatLTOObjects =
      args.hasFlag(OPT_fat_lto_objects, OPT_no_fat_lto_objects, false);
  ctx.arg.androidMemtagMode = getMemtagMode(ctx, args);
  ctx.arg.auxiliaryList = args::getStrings(args, OPT_auxiliary);
  ctx.arg.armBe8 = args.hasArg(OPT_be8);
  if (opt::Arg *arg = args.getLastArg(
          OPT_Bno_symbolic, OPT_Bsymbolic_non_weak_functions,
          OPT_Bsymbolic_functions, OPT_Bsymbolic_non_weak, OPT_Bsymbolic)) {
    if (arg->getOption().matches(OPT_Bsymbolic_non_weak_functions))
      ctx.arg.bsymbolic = BsymbolicKind::NonWeakFunctions;
    else if (arg->getOption().matches(OPT_Bsymbolic_functions))
      ctx.arg.bsymbolic = BsymbolicKind::Functions;
    else if (arg->getOption().matches(OPT_Bsymbolic_non_weak))
      ctx.arg.bsymbolic = BsymbolicKind::NonWeak;
    else if (arg->getOption().matches(OPT_Bsymbolic))
      ctx.arg.bsymbolic = BsymbolicKind::All;
  }
  ctx.arg.callGraphProfileSort = getCGProfileSortKind(ctx, args);
  parseBPOrdererOptions(ctx, args);
  ctx.arg.checkSections =
      args.hasFlag(OPT_check_sections, OPT_no_check_sections, true);
  ctx.arg.chroot = args.getLastArgValue(OPT_chroot);
  if (auto *arg = args.getLastArg(OPT_compress_debug_sections)) {
    ctx.arg.compressDebugSections =
        getCompressionType(ctx, arg->getValue(), "--compress-debug-sections");
  }
  ctx.arg.cref = args.hasArg(OPT_cref);
  ctx.arg.optimizeBBJumps =
      args.hasFlag(OPT_optimize_bb_jumps, OPT_no_optimize_bb_jumps, false);
  ctx.arg.debugNames = args.hasFlag(OPT_debug_names, OPT_no_debug_names, false);
  ctx.arg.demangle = args.hasFlag(OPT_demangle, OPT_no_demangle, true);
  ctx.arg.dependencyFile = args.getLastArgValue(OPT_dependency_file);
  ctx.arg.dependentLibraries =
      args.hasFlag(OPT_dependent_libraries, OPT_no_dependent_libraries, true);
  ctx.arg.disableVerify = args.hasArg(OPT_disable_verify);
  ctx.arg.discard = getDiscard(args);
  ctx.arg.dwoDir = args.getLastArgValue(OPT_plugin_opt_dwo_dir_eq);
  ctx.arg.dynamicLinker = getDynamicLinker(ctx, args);
  ctx.arg.ehFrameHdr =
      args.hasFlag(OPT_eh_frame_hdr, OPT_no_eh_frame_hdr, false);
  ctx.arg.emitLLVM = args.hasArg(OPT_lto_emit_llvm);
  ctx.arg.emitRelocs = args.hasArg(OPT_emit_relocs);
  ctx.arg.enableNewDtags =
      args.hasFlag(OPT_enable_new_dtags, OPT_disable_new_dtags, true);
  ctx.arg.enableNonContiguousRegions =
      args.hasArg(OPT_enable_non_contiguous_regions);
  ctx.arg.entry = args.getLastArgValue(OPT_entry);

  ctx.e.errorHandlingScript = args.getLastArgValue(OPT_error_handling_script);

  ctx.arg.executeOnly =
      args.hasFlag(OPT_execute_only, OPT_no_execute_only, false);
  ctx.arg.exportDynamic =
      args.hasFlag(OPT_export_dynamic, OPT_no_export_dynamic, false) ||
      args.hasArg(OPT_shared);
  ctx.arg.filterList = args::getStrings(args, OPT_filter);
  ctx.arg.fini = args.getLastArgValue(OPT_fini, "_fini");
  ctx.arg.fixCortexA53Errata843419 =
      args.hasArg(OPT_fix_cortex_a53_843419) && !args.hasArg(OPT_relocatable);
  ctx.arg.cmseImplib = args.hasArg(OPT_cmse_implib);
  ctx.arg.cmseInputLib = args.getLastArgValue(OPT_in_implib);
  ctx.arg.cmseOutputLib = args.getLastArgValue(OPT_out_implib);
  ctx.arg.fixCortexA8 =
      args.hasArg(OPT_fix_cortex_a8) && !args.hasArg(OPT_relocatable);
  ctx.arg.fortranCommon =
      args.hasFlag(OPT_fortran_common, OPT_no_fortran_common, false);
  ctx.arg.gcSections = args.hasFlag(OPT_gc_sections, OPT_no_gc_sections, false);
  ctx.arg.gnuUnique = args.hasFlag(OPT_gnu_unique, OPT_no_gnu_unique, true);
  ctx.arg.gdbIndex = args.hasFlag(OPT_gdb_index, OPT_no_gdb_index, false);
  ctx.arg.icf = getICF(args);
  ctx.arg.ignoreDataAddressEquality =
      args.hasArg(OPT_ignore_data_address_equality);
  ctx.arg.ignoreFunctionAddressEquality =
      args.hasArg(OPT_ignore_function_address_equality);
  ctx.arg.init = args.getLastArgValue(OPT_init, "_init");
  ctx.arg.ltoAAPipeline = args.getLastArgValue(OPT_lto_aa_pipeline);
  ctx.arg.ltoCSProfileGenerate = args.hasArg(OPT_lto_cs_profile_generate);
  ctx.arg.ltoCSProfileFile = args.getLastArgValue(OPT_lto_cs_profile_file);
  ctx.arg.ltoPGOWarnMismatch = args.hasFlag(OPT_lto_pgo_warn_mismatch,
                                            OPT_no_lto_pgo_warn_mismatch, true);
  ctx.arg.ltoDebugPassManager = args.hasArg(OPT_lto_debug_pass_manager);
  ctx.arg.ltoEmitAsm = args.hasArg(OPT_lto_emit_asm);
  ctx.arg.ltoNewPmPasses = args.getLastArgValue(OPT_lto_newpm_passes);
  ctx.arg.ltoWholeProgramVisibility =
      args.hasFlag(OPT_lto_whole_program_visibility,
                   OPT_no_lto_whole_program_visibility, false);
  ctx.arg.ltoValidateAllVtablesHaveTypeInfos =
      args.hasFlag(OPT_lto_validate_all_vtables_have_type_infos,
                   OPT_no_lto_validate_all_vtables_have_type_infos, false);
  ctx.arg.ltoo = args::getInteger(args, OPT_lto_O, 2);
  if (ctx.arg.ltoo > 3)
    ErrAlways(ctx) << "invalid optimization level for LTO: " << ctx.arg.ltoo;
  unsigned ltoCgo =
      args::getInteger(args, OPT_lto_CGO, args::getCGOptLevel(ctx.arg.ltoo));
  if (auto level = CodeGenOpt::getLevel(ltoCgo))
    ctx.arg.ltoCgo = *level;
  else
    ErrAlways(ctx) << "invalid codegen optimization level for LTO: " << ltoCgo;
  ctx.arg.ltoObjPath = args.getLastArgValue(OPT_lto_obj_path_eq);
  ctx.arg.ltoPartitions = args::getInteger(args, OPT_lto_partitions, 1);
  ctx.arg.ltoSampleProfile = args.getLastArgValue(OPT_lto_sample_profile);
  ctx.arg.ltoBBAddrMap =
      args.hasFlag(OPT_lto_basic_block_address_map,
                   OPT_no_lto_basic_block_address_map, false);
  ctx.arg.ltoBasicBlockSections =
      args.getLastArgValue(OPT_lto_basic_block_sections);
  ctx.arg.ltoUniqueBasicBlockSectionNames =
      args.hasFlag(OPT_lto_unique_basic_block_section_names,
                   OPT_no_lto_unique_basic_block_section_names, false);
  ctx.arg.mapFile = args.getLastArgValue(OPT_Map);
  ctx.arg.mipsGotSize = args::getInteger(args, OPT_mips_got_size, 0xfff0);
  ctx.arg.mergeArmExidx =
      args.hasFlag(OPT_merge_exidx_entries, OPT_no_merge_exidx_entries, true);
  ctx.arg.mmapOutputFile =
      args.hasFlag(OPT_mmap_output_file, OPT_no_mmap_output_file, true);
  ctx.arg.nmagic = args.hasFlag(OPT_nmagic, OPT_no_nmagic, false);
  ctx.arg.noinhibitExec = args.hasArg(OPT_noinhibit_exec);
  ctx.arg.nostdlib = args.hasArg(OPT_nostdlib);
  ctx.arg.oFormatBinary = isOutputFormatBinary(ctx, args);
  ctx.arg.omagic = args.hasFlag(OPT_omagic, OPT_no_omagic, false);
  ctx.arg.optRemarksFilename = args.getLastArgValue(OPT_opt_remarks_filename);
  ctx.arg.optStatsFilename = args.getLastArgValue(OPT_plugin_opt_stats_file);

  // Parse remarks hotness threshold. Valid value is either integer or 'auto'.
  if (auto *arg = args.getLastArg(OPT_opt_remarks_hotness_threshold)) {
    auto resultOrErr = remarks::parseHotnessThresholdOption(arg->getValue());
    if (!resultOrErr)
      ErrAlways(ctx) << arg->getSpelling() << ": invalid argument '"
                     << arg->getValue()
                     << "', only integer or 'auto' is supported";
    else
      ctx.arg.optRemarksHotnessThreshold = *resultOrErr;
  }

  ctx.arg.optRemarksPasses = args.getLastArgValue(OPT_opt_remarks_passes);
  ctx.arg.optRemarksWithHotness = args.hasArg(OPT_opt_remarks_with_hotness);
  ctx.arg.optRemarksFormat = args.getLastArgValue(OPT_opt_remarks_format);
  ctx.arg.optimize = args::getInteger(args, OPT_O, 1);
  ctx.arg.orphanHandling = getOrphanHandling(ctx, args);
  ctx.arg.outputFile = args.getLastArgValue(OPT_o);
  ctx.arg.packageMetadata = args.getLastArgValue(OPT_package_metadata);
  ctx.arg.pie = args.hasFlag(OPT_pie, OPT_no_pie, false);
  ctx.arg.printIcfSections =
      args.hasFlag(OPT_print_icf_sections, OPT_no_print_icf_sections, false);
  ctx.arg.printGcSections =
      args.hasFlag(OPT_print_gc_sections, OPT_no_print_gc_sections, false);
  ctx.arg.printMemoryUsage = args.hasArg(OPT_print_memory_usage);
  ctx.arg.printArchiveStats = args.getLastArgValue(OPT_print_archive_stats);
  ctx.arg.printSymbolOrder = args.getLastArgValue(OPT_print_symbol_order);
  ctx.arg.rejectMismatch = !args.hasArg(OPT_no_warn_mismatch);
  ctx.arg.relax = args.hasFlag(OPT_relax, OPT_no_relax, true);
  ctx.arg.relaxGP = args.hasFlag(OPT_relax_gp, OPT_no_relax_gp, false);
  ctx.arg.rpath = getRpath(args);
  ctx.arg.relocatable = args.hasArg(OPT_relocatable);
  ctx.arg.resolveGroups =
      !args.hasArg(OPT_relocatable) || args.hasArg(OPT_force_group_allocation);

  if (args.hasArg(OPT_save_temps)) {
    // --save-temps implies saving all temps.
    for (const char *s : saveTempsValues)
      ctx.arg.saveTempsArgs.insert(s);
  } else {
    for (auto *arg : args.filtered(OPT_save_temps_eq)) {
      StringRef s = arg->getValue();
      if (llvm::is_contained(saveTempsValues, s))
        ctx.arg.saveTempsArgs.insert(s);
      else
        ErrAlways(ctx) << "unknown --save-temps value: " << s;
    }
  }

  ctx.arg.searchPaths = args::getStrings(args, OPT_library_path);
  ctx.arg.sectionStartMap = getSectionStartMap(ctx, args);
  ctx.arg.shared = args.hasArg(OPT_shared);
  if (args.hasArg(OPT_randomize_section_padding))
    ctx.arg.randomizeSectionPadding =
        args::getInteger(args, OPT_randomize_section_padding, 0);
  ctx.arg.singleRoRx = !args.hasFlag(OPT_rosegment, OPT_no_rosegment, true);
  ctx.arg.soName = args.getLastArgValue(OPT_soname);
  ctx.arg.sortSection = getSortSection(ctx, args);
  ctx.arg.splitStackAdjustSize =
      args::getInteger(args, OPT_split_stack_adjust_size, 16384);
  ctx.arg.zSectionHeader =
      getZFlag(args, "sectionheader", "nosectionheader", true);
  ctx.arg.strip = getStrip(ctx, args); // needs zSectionHeader
  ctx.arg.sysroot = args.getLastArgValue(OPT_sysroot);
  ctx.arg.target1Rel = args.hasFlag(OPT_target1_rel, OPT_target1_abs, false);
  ctx.arg.target2 = getTarget2(ctx, args);
  ctx.arg.thinLTOCacheDir = args.getLastArgValue(OPT_thinlto_cache_dir);
  ctx.arg.thinLTOCachePolicy = CHECK(
      parseCachePruningPolicy(args.getLastArgValue(OPT_thinlto_cache_policy)),
      "--thinlto-cache-policy: invalid cache policy");
  ctx.arg.thinLTOEmitImportsFiles = args.hasArg(OPT_thinlto_emit_imports_files);
  ctx.arg.thinLTOEmitIndexFiles = args.hasArg(OPT_thinlto_emit_index_files) ||
                                  args.hasArg(OPT_thinlto_index_only) ||
                                  args.hasArg(OPT_thinlto_index_only_eq);
  ctx.arg.thinLTOIndexOnly = args.hasArg(OPT_thinlto_index_only) ||
                             args.hasArg(OPT_thinlto_index_only_eq);
  ctx.arg.thinLTOIndexOnlyArg = args.getLastArgValue(OPT_thinlto_index_only_eq);
  ctx.arg.thinLTOObjectSuffixReplace =
      getOldNewOptions(ctx, args, OPT_thinlto_object_suffix_replace_eq);
  std::tie(ctx.arg.thinLTOPrefixReplaceOld, ctx.arg.thinLTOPrefixReplaceNew,
           ctx.arg.thinLTOPrefixReplaceNativeObject) =
      getOldNewOptionsExtra(ctx, args, OPT_thinlto_prefix_replace_eq);
  if (ctx.arg.thinLTOEmitIndexFiles && !ctx.arg.thinLTOIndexOnly) {
    if (args.hasArg(OPT_thinlto_object_suffix_replace_eq))
      ErrAlways(ctx) << "--thinlto-object-suffix-replace is not supported with "
                        "--thinlto-emit-index-files";
    else if (args.hasArg(OPT_thinlto_prefix_replace_eq))
      ErrAlways(ctx) << "--thinlto-prefix-replace is not supported with "
                        "--thinlto-emit-index-files";
  }
  if (!ctx.arg.thinLTOPrefixReplaceNativeObject.empty() &&
      ctx.arg.thinLTOIndexOnlyArg.empty()) {
    ErrAlways(ctx)
        << "--thinlto-prefix-replace=old_dir;new_dir;obj_dir must be used with "
           "--thinlto-index-only=";
  }
  ctx.arg.thinLTOModulesToCompile =
      args::getStrings(args, OPT_thinlto_single_module_eq);
  ctx.arg.timeTraceEnabled =
      args.hasArg(OPT_time_trace_eq) && !ctx.e.disableOutput;
  ctx.arg.timeTraceGranularity =
      args::getInteger(args, OPT_time_trace_granularity, 500);
  ctx.arg.trace = args.hasArg(OPT_trace);
  ctx.arg.undefined = args::getStrings(args, OPT_undefined);
  ctx.arg.undefinedVersion =
      args.hasFlag(OPT_undefined_version, OPT_no_undefined_version, false);
  ctx.arg.unique = args.hasArg(OPT_unique);
  ctx.arg.useAndroidRelrTags = args.hasFlag(
      OPT_use_android_relr_tags, OPT_no_use_android_relr_tags, false);
  ctx.arg.warnBackrefs =
      args.hasFlag(OPT_warn_backrefs, OPT_no_warn_backrefs, false);
  ctx.arg.warnCommon = args.hasFlag(OPT_warn_common, OPT_no_warn_common, false);
  ctx.arg.warnSymbolOrdering =
      args.hasFlag(OPT_warn_symbol_ordering, OPT_no_warn_symbol_ordering, true);
  ctx.arg.whyExtract = args.getLastArgValue(OPT_why_extract);
  ctx.arg.zCombreloc = getZFlag(args, "combreloc", "nocombreloc", true);
  ctx.arg.zCopyreloc = getZFlag(args, "copyreloc", "nocopyreloc", true);
  ctx.arg.zForceBti = hasZOption(args, "force-bti");
  ctx.arg.zForceIbt = hasZOption(args, "force-ibt");
  ctx.arg.zGcs = getZGcs(ctx, args);
  ctx.arg.zGlobal = hasZOption(args, "global");
  ctx.arg.zGnustack = getZGnuStack(args);
  ctx.arg.zHazardplt = hasZOption(args, "hazardplt");
  ctx.arg.zIfuncNoplt = hasZOption(args, "ifunc-noplt");
  ctx.arg.zInitfirst = hasZOption(args, "initfirst");
  ctx.arg.zInterpose = hasZOption(args, "interpose");
  ctx.arg.zKeepTextSectionPrefix = getZFlag(
      args, "keep-text-section-prefix", "nokeep-text-section-prefix", false);
  ctx.arg.zLrodataAfterBss =
      getZFlag(args, "lrodata-after-bss", "nolrodata-after-bss", false);
  ctx.arg.zNoBtCfi = hasZOption(args, "nobtcfi");
  ctx.arg.zNodefaultlib = hasZOption(args, "nodefaultlib");
  ctx.arg.zNodelete = hasZOption(args, "nodelete");
  ctx.arg.zNodlopen = hasZOption(args, "nodlopen");
  ctx.arg.zNow = getZFlag(args, "now", "lazy", false);
  ctx.arg.zOrigin = hasZOption(args, "origin");
  ctx.arg.zPacPlt = hasZOption(args, "pac-plt");
  ctx.arg.zRelro = getZFlag(args, "relro", "norelro", true);
  ctx.arg.zRetpolineplt = hasZOption(args, "retpolineplt");
  ctx.arg.zRodynamic = hasZOption(args, "rodynamic");
  ctx.arg.zSeparate = getZSeparate(args);
  ctx.arg.zShstk = hasZOption(args, "shstk");
  ctx.arg.zStackSize = args::getZOptionValue(args, OPT_z, "stack-size", 0);
  ctx.arg.zStartStopGC =
      getZFlag(args, "start-stop-gc", "nostart-stop-gc", true);
  ctx.arg.zStartStopVisibility = getZStartStopVisibility(ctx, args);
  ctx.arg.zText = getZFlag(args, "text", "notext", true);
  ctx.arg.zWxneeded = hasZOption(args, "wxneeded");
  setUnresolvedSymbolPolicy(ctx, args);
  ctx.arg.power10Stubs = args.getLastArgValue(OPT_power10_stubs_eq) != "no";

  if (opt::Arg *arg = args.getLastArg(OPT_eb, OPT_el)) {
    if (arg->getOption().matches(OPT_eb))
      ctx.arg.optEB = true;
    else
      ctx.arg.optEL = true;
  }

  for (opt::Arg *arg : args.filtered(OPT_remap_inputs)) {
    StringRef value(arg->getValue());
    remapInputs(ctx, value, arg->getSpelling());
  }
  for (opt::Arg *arg : args.filtered(OPT_remap_inputs_file)) {
    StringRef filename(arg->getValue());
    std::optional<MemoryBufferRef> buffer = readFile(ctx, filename);
    if (!buffer)
      continue;
    // Parse 'from-glob=to-file' lines, ignoring #-led comments.
    for (auto [lineno, line] : llvm::enumerate(args::getLines(*buffer)))
      if (remapInputs(ctx, line, filename + ":" + Twine(lineno + 1)))
        break;
  }

  for (opt::Arg *arg : args.filtered(OPT_shuffle_sections)) {
    constexpr StringRef errPrefix = "--shuffle-sections=: ";
    std::pair<StringRef, StringRef> kv = StringRef(arg->getValue()).split('=');
    if (kv.first.empty() || kv.second.empty()) {
      ErrAlways(ctx) << errPrefix << "expected <section_glob>=<seed>, but got '"
                     << arg->getValue() << "'";
      continue;
    }
    // Signed so that <section_glob>=-1 is allowed.
    int64_t v;
    if (!to_integer(kv.second, v))
      ErrAlways(ctx) << errPrefix << "expected an integer, but got '"
                     << kv.second << "'";
    else if (Expected<GlobPattern> pat = GlobPattern::create(kv.first))
      ctx.arg.shuffleSections.emplace_back(std::move(*pat), uint32_t(v));
    else
      ErrAlways(ctx) << errPrefix << pat.takeError() << ": " << kv.first;
  }

  auto reports = {std::make_pair("bti-report", &ctx.arg.zBtiReport),
                  std::make_pair("cet-report", &ctx.arg.zCetReport),
                  std::make_pair("gcs-report", &ctx.arg.zGcsReport),
                  std::make_pair("pauth-report", &ctx.arg.zPauthReport)};
  for (opt::Arg *arg : args.filtered(OPT_z)) {
    std::pair<StringRef, StringRef> option =
        StringRef(arg->getValue()).split('=');
    for (auto reportArg : reports) {
      if (option.first != reportArg.first)
        continue;
      arg->claim();
      if (!isValidReportString(option.second)) {
        ErrAlways(ctx) << "-z " << reportArg.first << "= parameter "
                       << option.second << " is not recognized";
        continue;
      }
      *reportArg.second = option.second;
    }
  }

  for (opt::Arg *arg : args.filtered(OPT_compress_sections)) {
    SmallVector<StringRef, 0> fields;
    StringRef(arg->getValue()).split(fields, '=');
    if (fields.size() != 2 || fields[1].empty()) {
      ErrAlways(ctx) << arg->getSpelling()
                     << ": parse error, not 'section-glob=[none|zlib|zstd]'";
      continue;
    }
    auto [typeStr, levelStr] = fields[1].split(':');
    auto type = getCompressionType(ctx, typeStr, arg->getSpelling());
    unsigned level = 0;
    if (fields[1].size() != typeStr.size() &&
        !llvm::to_integer(levelStr, level)) {
      ErrAlways(ctx)
          << arg->getSpelling()
          << ": expected a non-negative integer compression level, but got '"
          << levelStr << "'";
    }
    if (Expected<GlobPattern> pat = GlobPattern::create(fields[0])) {
      ctx.arg.compressSections.emplace_back(std::move(*pat), type, level);
    } else {
      ErrAlways(ctx) << arg->getSpelling() << ": " << pat.takeError();
      continue;
    }
  }

  for (opt::Arg *arg : args.filtered(OPT_z)) {
    std::pair<StringRef, StringRef> option =
        StringRef(arg->getValue()).split('=');
    if (option.first != "dead-reloc-in-nonalloc")
      continue;
    arg->claim();
    constexpr StringRef errPrefix = "-z dead-reloc-in-nonalloc=: ";
    std::pair<StringRef, StringRef> kv = option.second.split('=');
    if (kv.first.empty() || kv.second.empty()) {
      ErrAlways(ctx) << errPrefix << "expected <section_glob>=<value>";
      continue;
    }
    uint64_t v;
    if (!to_integer(kv.second, v))
      ErrAlways(ctx) << errPrefix
                     << "expected a non-negative integer, but got '"
                     << kv.second << "'";
    else if (Expected<GlobPattern> pat = GlobPattern::create(kv.first))
      ctx.arg.deadRelocInNonAlloc.emplace_back(std::move(*pat), v);
    else
      ErrAlways(ctx) << errPrefix << pat.takeError() << ": " << kv.first;
  }

  cl::ResetAllOptionOccurrences();

  // Parse LTO options.
  if (auto *arg = args.getLastArg(OPT_plugin_opt_mcpu_eq))
    parseClangOption(ctx, ctx.saver.save("-mcpu=" + StringRef(arg->getValue())),
                     arg->getSpelling());

  for (opt::Arg *arg : args.filtered(OPT_plugin_opt_eq_minus))
    parseClangOption(ctx, std::string("-") + arg->getValue(),
                     arg->getSpelling());

  // GCC collect2 passes -plugin-opt=path/to/lto-wrapper with an absolute or
  // relative path. Just ignore. If not ended with "lto-wrapper" (or
  // "lto-wrapper.exe" for GCC cross-compiled for Windows), consider it an
  // unsupported LLVMgold.so option and error.
  for (opt::Arg *arg : args.filtered(OPT_plugin_opt_eq)) {
    StringRef v(arg->getValue());
    if (!v.ends_with("lto-wrapper") && !v.ends_with("lto-wrapper.exe"))
      ErrAlways(ctx) << arg->getSpelling() << ": unknown plugin option '"
                     << arg->getValue() << "'";
  }

  ctx.arg.passPlugins = args::getStrings(args, OPT_load_pass_plugins);

  // Parse -mllvm options.
  for (const auto *arg : args.filtered(OPT_mllvm)) {
    parseClangOption(ctx, arg->getValue(), arg->getSpelling());
    ctx.arg.mllvmOpts.emplace_back(arg->getValue());
  }

  ctx.arg.ltoKind = LtoKind::Default;
  if (auto *arg = args.getLastArg(OPT_lto)) {
    StringRef s = arg->getValue();
    if (s == "thin")
      ctx.arg.ltoKind = LtoKind::UnifiedThin;
    else if (s == "full")
      ctx.arg.ltoKind = LtoKind::UnifiedRegular;
    else if (s == "default")
      ctx.arg.ltoKind = LtoKind::Default;
    else
      ErrAlways(ctx) << "unknown LTO mode: " << s;
  }

  // --threads= takes a positive integer and provides the default value for
  // --thinlto-jobs=. If unspecified, cap the number of threads since
  // overhead outweighs optimization for used parallel algorithms for the
  // non-LTO parts.
  if (auto *arg = args.getLastArg(OPT_threads)) {
    StringRef v(arg->getValue());
    unsigned threads = 0;
    if (!llvm::to_integer(v, threads, 0) || threads == 0)
      ErrAlways(ctx) << arg->getSpelling()
                     << ": expected a positive integer, but got '"
                     << arg->getValue() << "'";
    parallel::strategy = hardware_concurrency(threads);
    ctx.arg.thinLTOJobs = v;
  } else if (parallel::strategy.compute_thread_count() > 16) {
    Log(ctx) << "set maximum concurrency to 16, specify --threads= to change";
    parallel::strategy = hardware_concurrency(16);
  }
  if (auto *arg = args.getLastArg(OPT_thinlto_jobs_eq))
    ctx.arg.thinLTOJobs = arg->getValue();
  ctx.arg.threadCount = parallel::strategy.compute_thread_count();

  if (ctx.arg.ltoPartitions == 0)
    ErrAlways(ctx) << "--lto-partitions: number of threads must be > 0";
  if (!get_threadpool_strategy(ctx.arg.thinLTOJobs))
    ErrAlways(ctx) << "--thinlto-jobs: invalid job count: "
                   << ctx.arg.thinLTOJobs;

  if (ctx.arg.splitStackAdjustSize < 0)
    ErrAlways(ctx) << "--split-stack-adjust-size: size must be >= 0";

  // The text segment is traditionally the first segment, whose address equals
  // the base address. However, lld places the R PT_LOAD first. -Ttext-segment
  // is an old-fashioned option that does not play well with lld's layout.
  // Suggest --image-base as a likely alternative.
  if (args.hasArg(OPT_Ttext_segment))
    ErrAlways(ctx)
        << "-Ttext-segment is not supported. Use --image-base if you "
           "intend to set the base address";

  // Parse ELF{32,64}{LE,BE} and CPU type.
  if (auto *arg = args.getLastArg(OPT_m)) {
    StringRef s = arg->getValue();
    std::tie(ctx.arg.ekind, ctx.arg.emachine, ctx.arg.osabi) =
        parseEmulation(ctx, s);
    ctx.arg.mipsN32Abi =
        (s.starts_with("elf32btsmipn32") || s.starts_with("elf32ltsmipn32"));
    ctx.arg.emulation = s;
  }

  // Parse --hash-style={sysv,gnu,both}.
  if (auto *arg = args.getLastArg(OPT_hash_style)) {
    StringRef s = arg->getValue();
    if (s == "sysv")
      ctx.arg.sysvHash = true;
    else if (s == "gnu")
      ctx.arg.gnuHash = true;
    else if (s == "both")
      ctx.arg.sysvHash = ctx.arg.gnuHash = true;
    else
      ErrAlways(ctx) << "unknown --hash-style: " << s;
  }

  if (args.hasArg(OPT_print_map))
    ctx.arg.mapFile = "-";

  // Page alignment can be disabled by the -n (--nmagic) and -N (--omagic).
  // As PT_GNU_RELRO relies on Paging, do not create it when we have disabled
  // it. Also disable RELRO for -r.
  if (ctx.arg.nmagic || ctx.arg.omagic || ctx.arg.relocatable)
    ctx.arg.zRelro = false;

  std::tie(ctx.arg.buildId, ctx.arg.buildIdVector) = getBuildId(ctx, args);

  if (getZFlag(args, "pack-relative-relocs", "nopack-relative-relocs", false)) {
    ctx.arg.relrGlibc = true;
    ctx.arg.relrPackDynRelocs = true;
  } else {
    std::tie(ctx.arg.androidPackDynRelocs, ctx.arg.relrPackDynRelocs) =
        getPackDynRelocs(ctx, args);
  }

  if (auto *arg = args.getLastArg(OPT_symbol_ordering_file)){
    if (args.hasArg(OPT_call_graph_ordering_file))
      ErrAlways(ctx) << "--symbol-ordering-file and --call-graph-order-file "
                        "may not be used together";
    if (auto buffer = readFile(ctx, arg->getValue()))
      ctx.arg.symbolOrderingFile = getSymbolOrderingFile(ctx, *buffer);
  }

  assert(ctx.arg.versionDefinitions.empty());
  ctx.arg.versionDefinitions.push_back(
      {"local", (uint16_t)VER_NDX_LOCAL, {}, {}});
  ctx.arg.versionDefinitions.push_back(
      {"global", (uint16_t)VER_NDX_GLOBAL, {}, {}});

  // If --retain-symbol-file is used, we'll keep only the symbols listed in
  // the file and discard all others.
  if (auto *arg = args.getLastArg(OPT_retain_symbols_file)) {
    ctx.arg.versionDefinitions[VER_NDX_LOCAL].nonLocalPatterns.push_back(
        {"*", /*isExternCpp=*/false, /*hasWildcard=*/true});
    if (std::optional<MemoryBufferRef> buffer = readFile(ctx, arg->getValue()))
      for (StringRef s : args::getLines(*buffer))
        ctx.arg.versionDefinitions[VER_NDX_GLOBAL].nonLocalPatterns.push_back(
            {s, /*isExternCpp=*/false, /*hasWildcard=*/false});
  }

  for (opt::Arg *arg : args.filtered(OPT_warn_backrefs_exclude)) {
    StringRef pattern(arg->getValue());
    if (Expected<GlobPattern> pat = GlobPattern::create(pattern))
      ctx.arg.warnBackrefsExclude.push_back(std::move(*pat));
    else
      ErrAlways(ctx) << arg->getSpelling() << ": " << pat.takeError() << ": "
                     << pattern;
  }

  // For -no-pie and -pie, --export-dynamic-symbol specifies defined symbols
  // which should be exported. For -shared, references to matched non-local
  // STV_DEFAULT symbols are not bound to definitions within the shared object,
  // even if other options express a symbolic intention: -Bsymbolic,
  // -Bsymbolic-functions (if STT_FUNC), --dynamic-list.
  for (auto *arg : args.filtered(OPT_export_dynamic_symbol))
    ctx.arg.dynamicList.push_back(
        {arg->getValue(), /*isExternCpp=*/false,
         /*hasWildcard=*/hasWildcard(arg->getValue())});

  // --export-dynamic-symbol-list specifies a list of --export-dynamic-symbol
  // patterns. --dynamic-list is --export-dynamic-symbol-list plus -Bsymbolic
  // like semantics.
  ctx.arg.symbolic =
      ctx.arg.bsymbolic == BsymbolicKind::All || args.hasArg(OPT_dynamic_list);
  for (auto *arg :
       args.filtered(OPT_dynamic_list, OPT_export_dynamic_symbol_list))
    if (std::optional<MemoryBufferRef> buffer = readFile(ctx, arg->getValue()))
      readDynamicList(ctx, *buffer);

  for (auto *arg : args.filtered(OPT_version_script))
    if (std::optional<std::string> path = searchScript(ctx, arg->getValue())) {
      if (std::optional<MemoryBufferRef> buffer = readFile(ctx, *path))
        readVersionScript(ctx, *buffer);
    } else {
      ErrAlways(ctx) << "cannot find version script " << arg->getValue();
    }
}

// Some Config members do not directly correspond to any particular
// command line options, but computed based on other Config values.
// This function initialize such members. See Config.h for the details
// of these values.
static void setConfigs(Ctx &ctx, opt::InputArgList &args) {
  ELFKind k = ctx.arg.ekind;
  uint16_t m = ctx.arg.emachine;

  ctx.arg.copyRelocs = (ctx.arg.relocatable || ctx.arg.emitRelocs);
  ctx.arg.is64 = (k == ELF64LEKind || k == ELF64BEKind);
  ctx.arg.isLE = (k == ELF32LEKind || k == ELF64LEKind);
  ctx.arg.endianness = ctx.arg.isLE ? endianness::little : endianness::big;
  ctx.arg.isMips64EL = (k == ELF64LEKind && m == EM_MIPS);
  ctx.arg.isPic = ctx.arg.pie || ctx.arg.shared;
  ctx.arg.picThunk = args.hasArg(OPT_pic_veneer, ctx.arg.isPic);
  ctx.arg.wordsize = ctx.arg.is64 ? 8 : 4;

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
  ctx.arg.isRela = getIsRela(ctx, args);

  // If the output uses REL relocations we must store the dynamic relocation
  // addends to the output sections. We also store addends for RELA relocations
  // if --apply-dynamic-relocs is used.
  // We default to not writing the addends when using RELA relocations since
  // any standard conforming tool can find it in r_addend.
  ctx.arg.writeAddends = args.hasFlag(OPT_apply_dynamic_relocs,
                                      OPT_no_apply_dynamic_relocs, false) ||
                         !ctx.arg.isRela;
  // Validation of dynamic relocation addends is on by default for assertions
  // builds and disabled otherwise. This check is enabled when writeAddends is
  // true.
#ifndef NDEBUG
  bool checkDynamicRelocsDefault = true;
#else
  bool checkDynamicRelocsDefault = false;
#endif
  ctx.arg.checkDynamicRelocs =
      args.hasFlag(OPT_check_dynamic_relocations,
                   OPT_no_check_dynamic_relocations, checkDynamicRelocsDefault);
  ctx.arg.tocOptimize =
      args.hasFlag(OPT_toc_optimize, OPT_no_toc_optimize, m == EM_PPC64);
  ctx.arg.pcRelOptimize =
      args.hasFlag(OPT_pcrel_optimize, OPT_no_pcrel_optimize, m == EM_PPC64);

  if (!args.hasArg(OPT_hash_style)) {
    if (ctx.arg.emachine == EM_MIPS)
      ctx.arg.sysvHash = true;
    else
      ctx.arg.sysvHash = ctx.arg.gnuHash = true;
  }

  // Set default entry point and output file if not specified by command line or
  // linker scripts.
  ctx.arg.warnMissingEntry =
      (!ctx.arg.entry.empty() || (!ctx.arg.shared && !ctx.arg.relocatable));
  if (ctx.arg.entry.empty() && !ctx.arg.relocatable)
    ctx.arg.entry = ctx.arg.emachine == EM_MIPS ? "__start" : "_start";
  if (ctx.arg.outputFile.empty())
    ctx.arg.outputFile = "a.out";

  // Fail early if the output file or map file is not writable. If a user has a
  // long link, e.g. due to a large LTO link, they do not wish to run it and
  // find that it failed because there was a mistake in their command-line.
  {
    llvm::TimeTraceScope timeScope("Create output files");
    if (auto e = tryCreateFile(ctx.arg.outputFile))
      ErrAlways(ctx) << "cannot open output file " << ctx.arg.outputFile << ": "
                     << e.message();
    if (auto e = tryCreateFile(ctx.arg.mapFile))
      ErrAlways(ctx) << "cannot open map file " << ctx.arg.mapFile << ": "
                     << e.message();
    if (auto e = tryCreateFile(ctx.arg.whyExtract))
      ErrAlways(ctx) << "cannot open --why-extract= file " << ctx.arg.whyExtract
                     << ": " << e.message();
  }
}

static bool isFormatBinary(Ctx &ctx, StringRef s) {
  if (s == "binary")
    return true;
  if (s == "elf" || s == "default")
    return false;
  ErrAlways(ctx) << "unknown --format value: " << s
                 << " (supported formats: elf, default, binary)";
  return false;
}

void LinkerDriver::createFiles(opt::InputArgList &args) {
  llvm::TimeTraceScope timeScope("Load input files");
  // For --{push,pop}-state.
  std::vector<std::tuple<bool, bool, bool>> stack;

  // -r implies -Bstatic and has precedence over -Bdynamic.
  ctx.arg.isStatic = ctx.arg.relocatable;

  // Iterate over argv to process input files and positional arguments.
  std::optional<MemoryBufferRef> defaultScript;
  nextGroupId = 0;
  isInGroup = false;
  bool hasInput = false, hasScript = false;
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
      readDefsym(ctx, MemoryBufferRef(arg->getValue(), "--defsym"));
      break;
    }
    case OPT_script:
    case OPT_default_script:
      if (std::optional<std::string> path =
              searchScript(ctx, arg->getValue())) {
        if (std::optional<MemoryBufferRef> mb = readFile(ctx, *path)) {
          if (arg->getOption().matches(OPT_default_script)) {
            defaultScript = mb;
          } else {
            readLinkerScript(ctx, *mb);
            hasScript = true;
          }
        }
        break;
      }
      ErrAlways(ctx) << "cannot find linker script " << arg->getValue();
      break;
    case OPT_as_needed:
      ctx.arg.asNeeded = true;
      break;
    case OPT_format:
      ctx.arg.formatBinary = isFormatBinary(ctx, arg->getValue());
      break;
    case OPT_no_as_needed:
      ctx.arg.asNeeded = false;
      break;
    case OPT_Bstatic:
    case OPT_omagic:
    case OPT_nmagic:
      ctx.arg.isStatic = true;
      break;
    case OPT_Bdynamic:
      if (!ctx.arg.relocatable)
        ctx.arg.isStatic = false;
      break;
    case OPT_whole_archive:
      inWholeArchive = true;
      break;
    case OPT_no_whole_archive:
      inWholeArchive = false;
      break;
    case OPT_just_symbols:
      if (std::optional<MemoryBufferRef> mb = readFile(ctx, arg->getValue())) {
        files.push_back(createObjFile(ctx, *mb));
        files.back()->justSymbols = true;
      }
      break;
    case OPT_in_implib:
      if (armCmseImpLib)
        ErrAlways(ctx) << "multiple CMSE import libraries not supported";
      else if (std::optional<MemoryBufferRef> mb =
                   readFile(ctx, arg->getValue()))
        armCmseImpLib = createObjFile(ctx, *mb);
      break;
    case OPT_start_group:
      if (isInGroup)
        ErrAlways(ctx) << "nested --start-group";
      isInGroup = true;
      break;
    case OPT_end_group:
      if (!isInGroup)
        ErrAlways(ctx) << "stray --end-group";
      isInGroup = false;
      ++nextGroupId;
      break;
    case OPT_start_lib:
      if (inLib)
        ErrAlways(ctx) << "nested --start-lib";
      if (isInGroup)
        ErrAlways(ctx) << "may not nest --start-lib in --start-group";
      inLib = true;
      isInGroup = true;
      break;
    case OPT_end_lib:
      if (!inLib)
        ErrAlways(ctx) << "stray --end-lib";
      inLib = false;
      isInGroup = false;
      ++nextGroupId;
      break;
    case OPT_push_state:
      stack.emplace_back(ctx.arg.asNeeded, ctx.arg.isStatic, inWholeArchive);
      break;
    case OPT_pop_state:
      if (stack.empty()) {
        ErrAlways(ctx) << "unbalanced --push-state/--pop-state";
        break;
      }
      std::tie(ctx.arg.asNeeded, ctx.arg.isStatic, inWholeArchive) =
          stack.back();
      stack.pop_back();
      break;
    }
  }

  if (defaultScript && !hasScript)
    readLinkerScript(ctx, *defaultScript);
  if (files.empty() && !hasInput && errCount(ctx) == 0)
    ErrAlways(ctx) << "no input files";
}

// If -m <machine_type> was not given, infer it from object files.
void LinkerDriver::inferMachineType() {
  if (ctx.arg.ekind != ELFNoneKind)
    return;

  bool inferred = false;
  for (auto &f : files) {
    if (f->ekind == ELFNoneKind)
      continue;
    if (!inferred) {
      inferred = true;
      ctx.arg.ekind = f->ekind;
      ctx.arg.emachine = f->emachine;
      ctx.arg.mipsN32Abi = ctx.arg.emachine == EM_MIPS && isMipsN32Abi(ctx, *f);
    }
    ctx.arg.osabi = f->osabi;
    if (f->osabi != ELFOSABI_NONE)
      return;
  }
  if (!inferred)
    ErrAlways(ctx)
        << "target emulation unknown: -m or at least one .o file required";
}

// Parse -z max-page-size=<value>. The default value is defined by
// each target.
static uint64_t getMaxPageSize(Ctx &ctx, opt::InputArgList &args) {
  uint64_t val = args::getZOptionValue(args, OPT_z, "max-page-size",
                                       ctx.target->defaultMaxPageSize);
  if (!isPowerOf2_64(val)) {
    ErrAlways(ctx) << "max-page-size: value isn't a power of 2";
    return ctx.target->defaultMaxPageSize;
  }
  if (ctx.arg.nmagic || ctx.arg.omagic) {
    if (val != ctx.target->defaultMaxPageSize)
      Warn(ctx)
          << "-z max-page-size set, but paging disabled by omagic or nmagic";
    return 1;
  }
  return val;
}

// Parse -z common-page-size=<value>. The default value is defined by
// each target.
static uint64_t getCommonPageSize(Ctx &ctx, opt::InputArgList &args) {
  uint64_t val = args::getZOptionValue(args, OPT_z, "common-page-size",
                                       ctx.target->defaultCommonPageSize);
  if (!isPowerOf2_64(val)) {
    ErrAlways(ctx) << "common-page-size: value isn't a power of 2";
    return ctx.target->defaultCommonPageSize;
  }
  if (ctx.arg.nmagic || ctx.arg.omagic) {
    if (val != ctx.target->defaultCommonPageSize)
      Warn(ctx)
          << "-z common-page-size set, but paging disabled by omagic or nmagic";
    return 1;
  }
  // commonPageSize can't be larger than maxPageSize.
  if (val > ctx.arg.maxPageSize)
    val = ctx.arg.maxPageSize;
  return val;
}

// Parses --image-base option.
static std::optional<uint64_t> getImageBase(Ctx &ctx, opt::InputArgList &args) {
  // Because we are using `ctx.arg.maxPageSize` here, this function has to be
  // called after the variable is initialized.
  auto *arg = args.getLastArg(OPT_image_base);
  if (!arg)
    return std::nullopt;

  StringRef s = arg->getValue();
  uint64_t v;
  if (!to_integer(s, v)) {
    ErrAlways(ctx) << "--image-base: number expected, but got " << s;
    return 0;
  }
  if ((v % ctx.arg.maxPageSize) != 0)
    Warn(ctx) << "--image-base: address isn't multiple of page size: " << s;
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
static void excludeLibs(Ctx &ctx, opt::InputArgList &args) {
  DenseSet<StringRef> libs = getExcludeLibs(args);
  bool all = libs.count("ALL");

  auto visit = [&](InputFile *file) {
    if (file->archiveName.empty() ||
        !(all || libs.count(path::filename(file->archiveName))))
      return;
    ArrayRef<Symbol *> symbols = file->getSymbols();
    if (isa<ELFFileBase>(file))
      symbols = cast<ELFFileBase>(file)->getGlobalSymbols();
    for (Symbol *sym : symbols) {
      if (!sym->isUndefined() && sym->file == file) {
        sym->versionId = VER_NDX_LOCAL;
        sym->isExported = false;
      }
    }
  };

  for (ELFFileBase *file : ctx.objectFiles)
    visit(file);

  for (BitcodeFile *file : ctx.bitcodeFiles)
    visit(file);
}

// Force Sym to be entered in the output.
static void handleUndefined(Ctx &ctx, Symbol *sym, const char *option) {
  // Since a symbol may not be used inside the program, LTO may
  // eliminate it. Mark the symbol as "used" to prevent it.
  sym->isUsedInRegularObj = true;

  if (!sym->isLazy())
    return;
  sym->extract(ctx);
  if (!ctx.arg.whyExtract.empty())
    ctx.whyExtractRecords.emplace_back(option, sym->file, *sym);
}

// As an extension to GNU linkers, lld supports a variant of `-u`
// which accepts wildcard patterns. All symbols that match a given
// pattern are handled as if they were given by `-u`.
static void handleUndefinedGlob(Ctx &ctx, StringRef arg) {
  Expected<GlobPattern> pat = GlobPattern::create(arg);
  if (!pat) {
    ErrAlways(ctx) << "--undefined-glob: " << pat.takeError() << ": " << arg;
    return;
  }

  // Calling sym->extract() in the loop is not safe because it may add new
  // symbols to the symbol table, invalidating the current iterator.
  SmallVector<Symbol *, 0> syms;
  for (Symbol *sym : ctx.symtab->getSymbols())
    if (!sym->isPlaceholder() && pat->match(sym->getName()))
      syms.push_back(sym);

  for (Symbol *sym : syms)
    handleUndefined(ctx, sym, "--undefined-glob");
}

static void handleLibcall(Ctx &ctx, StringRef name) {
  Symbol *sym = ctx.symtab->find(name);
  if (sym && sym->isLazy() && isa<BitcodeFile>(sym->file)) {
    if (!ctx.arg.whyExtract.empty())
      ctx.whyExtractRecords.emplace_back("<libcall>", sym->file, *sym);
    sym->extract(ctx);
  }
}

static void writeArchiveStats(Ctx &ctx) {
  if (ctx.arg.printArchiveStats.empty())
    return;

  std::error_code ec;
  raw_fd_ostream os = ctx.openAuxiliaryFile(ctx.arg.printArchiveStats, ec);
  if (ec) {
    ErrAlways(ctx) << "--print-archive-stats=: cannot open "
                   << ctx.arg.printArchiveStats << ": " << ec.message();
    return;
  }

  os << "members\textracted\tarchive\n";

  SmallVector<StringRef, 0> archives;
  DenseMap<CachedHashStringRef, unsigned> all, extracted;
  for (ELFFileBase *file : ctx.objectFiles)
    if (file->archiveName.size())
      ++extracted[CachedHashStringRef(file->archiveName)];
  for (BitcodeFile *file : ctx.bitcodeFiles)
    if (file->archiveName.size())
      ++extracted[CachedHashStringRef(file->archiveName)];
  for (std::pair<StringRef, unsigned> f : ctx.driver.archiveFiles) {
    unsigned &v = extracted[CachedHashString(f.first)];
    os << f.second << '\t' << v << '\t' << f.first << '\n';
    // If the archive occurs multiple times, other instances have a count of 0.
    v = 0;
  }
}

static void writeWhyExtract(Ctx &ctx) {
  if (ctx.arg.whyExtract.empty())
    return;

  std::error_code ec;
  raw_fd_ostream os = ctx.openAuxiliaryFile(ctx.arg.whyExtract, ec);
  if (ec) {
    ErrAlways(ctx) << "cannot open --why-extract= file " << ctx.arg.whyExtract
                   << ": " << ec.message();
    return;
  }

  os << "reference\textracted\tsymbol\n";
  for (auto &entry : ctx.whyExtractRecords) {
    os << std::get<0>(entry) << '\t' << toStr(ctx, std::get<1>(entry)) << '\t'
       << toStr(ctx, std::get<2>(entry)) << '\n';
  }
}

static void reportBackrefs(Ctx &ctx) {
  for (auto &ref : ctx.backwardReferences) {
    const Symbol &sym = *ref.first;
    std::string to = toStr(ctx, ref.second.second);
    // Some libraries have known problems and can cause noise. Filter them out
    // with --warn-backrefs-exclude=. The value may look like (for --start-lib)
    // *.o or (archive member) *.a(*.o).
    bool exclude = false;
    for (const llvm::GlobPattern &pat : ctx.arg.warnBackrefsExclude)
      if (pat.match(to)) {
        exclude = true;
        break;
      }
    if (!exclude)
      Warn(ctx) << "backward reference detected: " << sym.getName() << " in "
                << ref.second.first << " refers to " << to;
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
static void writeDependencyFile(Ctx &ctx) {
  std::error_code ec;
  raw_fd_ostream os = ctx.openAuxiliaryFile(ctx.arg.dependencyFile, ec);
  if (ec) {
    ErrAlways(ctx) << "cannot open " << ctx.arg.dependencyFile << ": "
                   << ec.message();
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

  os << ctx.arg.outputFile << ":";
  for (StringRef path : ctx.arg.dependencyFiles) {
    os << " \\\n ";
    printFilename(os, path);
  }
  os << "\n";

  for (StringRef path : ctx.arg.dependencyFiles) {
    os << "\n";
    printFilename(os, path);
    os << ":\n";
  }
}

// Replaces common symbols with defined symbols reside in .bss sections.
// This function is called after all symbol names are resolved. As a
// result, the passes after the symbol resolution won't see any
// symbols of type CommonSymbol.
static void replaceCommonSymbols(Ctx &ctx) {
  llvm::TimeTraceScope timeScope("Replace common symbols");
  for (ELFFileBase *file : ctx.objectFiles) {
    if (!file->hasCommonSyms)
      continue;
    for (Symbol *sym : file->getGlobalSymbols()) {
      auto *s = dyn_cast<CommonSymbol>(sym);
      if (!s)
        continue;

      auto *bss = make<BssSection>(ctx, "COMMON", s->size, s->alignment);
      bss->file = s->file;
      ctx.inputSections.push_back(bss);
      Defined(ctx, s->file, StringRef(), s->binding, s->stOther, s->type,
              /*value=*/0, s->size, bss)
          .overwrite(*s);
    }
  }
}

// The section referred to by `s` is considered address-significant. Set the
// keepUnique flag on the section if appropriate.
static void markAddrsig(bool icfSafe, Symbol *s) {
  // We don't need to keep text sections unique under --icf=all even if they
  // are address-significant.
  if (auto *d = dyn_cast_or_null<Defined>(s))
    if (auto *sec = dyn_cast_or_null<InputSectionBase>(d->section))
      if (icfSafe || !(sec->flags & SHF_EXECINSTR))
        sec->keepUnique = true;
}

// Record sections that define symbols mentioned in --keep-unique <symbol>
// and symbols referred to by address-significance tables. These sections are
// ineligible for ICF.
template <class ELFT>
static void findKeepUniqueSections(Ctx &ctx, opt::InputArgList &args) {
  for (auto *arg : args.filtered(OPT_keep_unique)) {
    StringRef name = arg->getValue();
    auto *d = dyn_cast_or_null<Defined>(ctx.symtab->find(name));
    if (!d || !d->section) {
      Warn(ctx) << "could not find symbol " << name << " to keep unique";
      continue;
    }
    if (auto *sec = dyn_cast<InputSectionBase>(d->section))
      sec->keepUnique = true;
  }

  // --icf=all --ignore-data-address-equality means that we can ignore
  // the dynsym and address-significance tables entirely.
  if (ctx.arg.icf == ICFLevel::All && ctx.arg.ignoreDataAddressEquality)
    return;

  // Symbols in the dynsym could be address-significant in other executables
  // or DSOs, so we conservatively mark them as address-significant.
  bool icfSafe = ctx.arg.icf == ICFLevel::Safe;
  for (Symbol *sym : ctx.symtab->getSymbols())
    if (sym->isExported)
      markAddrsig(icfSafe, sym);

  // Visit the address-significance table in each object file and mark each
  // referenced symbol as address-significant.
  for (InputFile *f : ctx.objectFiles) {
    auto *obj = cast<ObjFile<ELFT>>(f);
    ArrayRef<Symbol *> syms = obj->getSymbols();
    if (obj->addrsigSec) {
      ArrayRef<uint8_t> contents =
          check(obj->getObj().getSectionContents(*obj->addrsigSec));
      const uint8_t *cur = contents.begin();
      while (cur != contents.end()) {
        unsigned size;
        const char *err = nullptr;
        uint64_t symIndex = decodeULEB128(cur, &size, contents.end(), &err);
        if (err) {
          Err(ctx) << f << ": could not decode addrsig section: " << err;
          break;
        }
        markAddrsig(icfSafe, syms[symIndex]);
        cur += size;
      }
    } else {
      // If an object file does not have an address-significance table,
      // conservatively mark all of its symbols as address-significant.
      for (Symbol *s : syms)
        markAddrsig(icfSafe, s);
    }
  }
}

// This function reads a symbol partition specification section. These sections
// are used to control which partition a symbol is allocated to. See
// https://lld.llvm.org/Partitions.html for more details on partitions.
template <typename ELFT>
static void readSymbolPartitionSection(Ctx &ctx, InputSectionBase *s) {
  // Read the relocation that refers to the partition's entry point symbol.
  Symbol *sym;
  const RelsOrRelas<ELFT> rels = s->template relsOrRelas<ELFT>();
  auto readEntry = [](InputFile *file, const auto &rels) -> Symbol * {
    for (const auto &rel : rels)
      return &file->getRelocTargetSym(rel);
    return nullptr;
  };
  if (rels.areRelocsCrel())
    sym = readEntry(s->file, rels.crels);
  else if (rels.areRelocsRel())
    sym = readEntry(s->file, rels.rels);
  else
    sym = readEntry(s->file, rels.relas);
  if (!isa_and_nonnull<Defined>(sym) || !sym->isExported)
    return;

  StringRef partName = reinterpret_cast<const char *>(s->content().data());
  for (Partition &part : ctx.partitions) {
    if (part.name == partName) {
      sym->partition = part.getNumber(ctx);
      return;
    }
  }

  // Forbid partitions from being used on incompatible targets, and forbid them
  // from being used together with various linker features that assume a single
  // set of output sections.
  if (ctx.script->hasSectionsCommand)
    ErrAlways(ctx) << s->file
                   << ": partitions cannot be used with the SECTIONS command";
  if (ctx.script->hasPhdrsCommands())
    ErrAlways(ctx) << s->file
                   << ": partitions cannot be used with the PHDRS command";
  if (!ctx.arg.sectionStartMap.empty())
    ErrAlways(ctx) << s->file
                   << ": partitions cannot be used with "
                      "--section-start, -Ttext, -Tdata or -Tbss";
  if (ctx.arg.emachine == EM_MIPS)
    ErrAlways(ctx) << s->file << ": partitions cannot be used on this target";

  // Impose a limit of no more than 254 partitions. This limit comes from the
  // sizes of the Partition fields in InputSectionBase and Symbol, as well as
  // the amount of space devoted to the partition number in RankFlags.
  if (ctx.partitions.size() == 254)
    Fatal(ctx) << "may not have more than 254 partitions";

  ctx.partitions.emplace_back(ctx);
  Partition &newPart = ctx.partitions.back();
  newPart.name = partName;
  sym->partition = newPart.getNumber(ctx);
}

static void markBuffersAsDontNeed(Ctx &ctx, bool skipLinkedOutput) {
  // With --thinlto-index-only, all buffers are nearly unused from now on
  // (except symbol/section names used by infrequent passes). Mark input file
  // buffers as MADV_DONTNEED so that these pages can be reused by the expensive
  // thin link, saving memory.
  if (skipLinkedOutput) {
    for (MemoryBuffer &mb : llvm::make_pointee_range(ctx.memoryBuffers))
      mb.dontNeedIfMmap();
    return;
  }

  // Otherwise, just mark MemoryBuffers backing BitcodeFiles.
  DenseSet<const char *> bufs;
  for (BitcodeFile *file : ctx.bitcodeFiles)
    bufs.insert(file->mb.getBufferStart());
  for (BitcodeFile *file : ctx.lazyBitcodeFiles)
    bufs.insert(file->mb.getBufferStart());
  for (MemoryBuffer &mb : llvm::make_pointee_range(ctx.memoryBuffers))
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
  lto.reset(new BitcodeCompiler(ctx));
  for (BitcodeFile *file : ctx.bitcodeFiles)
    lto->add(*file);

  if (!ctx.bitcodeFiles.empty())
    markBuffersAsDontNeed(ctx, skipLinkedOutput);

  ltoObjectFiles = lto->compile();
  for (auto &file : ltoObjectFiles) {
    auto *obj = cast<ObjFile<ELFT>>(file.get());
    obj->parse(/*ignoreComdats=*/true);

    // For defined symbols in non-relocatable output,
    // compute isExported and parse '@'.
    if (!ctx.arg.relocatable)
      for (Symbol *sym : obj->getGlobalSymbols()) {
        if (!sym->isDefined())
          continue;
        if (ctx.hasDynsym && ctx.arg.exportDynamic &&
            sym->computeBinding(ctx) != STB_LOCAL)
          sym->isExported = true;
        if (sym->hasVersionSuffix)
          sym->parseSymbolVersion(ctx);
      }
    ctx.objectFiles.push_back(obj);
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
static std::vector<WrappedSymbol> addWrappedSymbols(Ctx &ctx,
                                                    opt::InputArgList &args) {
  std::vector<WrappedSymbol> v;
  DenseSet<StringRef> seen;
  auto &ss = ctx.saver;
  for (auto *arg : args.filtered(OPT_wrap)) {
    StringRef name = arg->getValue();
    if (!seen.insert(name).second)
      continue;

    Symbol *sym = ctx.symtab->find(name);
    if (!sym)
      continue;

    Symbol *wrap =
        ctx.symtab->addUnusedUndefined(ss.save("__wrap_" + name), sym->binding);

    // If __real_ is referenced, pull in the symbol if it is lazy. Do this after
    // processing __wrap_ as that may have referenced __real_.
    StringRef realName = ctx.saver.save("__real_" + name);
    if (Symbol *real = ctx.symtab->find(realName)) {
      ctx.symtab->addUnusedUndefined(name, sym->binding);
      // Update sym's binding, which will replace real's later in
      // SymbolTable::wrap.
      sym->binding = real->binding;
    }

    Symbol *real = ctx.symtab->addUnusedUndefined(realName);
    v.push_back({sym, real, wrap});

    // We want to tell LTO not to inline symbols to be overwritten
    // because LTO doesn't know the final symbol contents after renaming.
    real->scriptDefined = true;
    sym->scriptDefined = true;

    // If a symbol is referenced in any object file, bitcode file or shared
    // object, mark its redirection target (foo for __real_foo and __wrap_foo
    // for foo) as referenced after redirection, which will be used to tell LTO
    // to not eliminate the redirection target. If the object file defining the
    // symbol also references it, we cannot easily distinguish the case from
    // cases where the symbol is not referenced. Retain the redirection target
    // in this case because we choose to wrap symbol references regardless of
    // whether the symbol is defined
    // (https://sourceware.org/bugzilla/show_bug.cgi?id=26358).
    if (real->referenced || real->isDefined())
      sym->referencedAfterWrap = true;
    if (sym->referenced || sym->isDefined())
      wrap->referencedAfterWrap = true;
  }
  return v;
}

static void combineVersionedSymbol(Ctx &ctx, Symbol &sym,
                                   DenseMap<Symbol *, Symbol *> &map) {
  const char *suffix1 = sym.getVersionSuffix();
  if (suffix1[0] != '@' || suffix1[1] == '@')
    return;

  // Check the existing symbol foo. We have two special cases to handle:
  //
  // * There is a definition of foo@v1 and foo@@v1.
  // * There is a definition of foo@v1 and foo.
  Defined *sym2 = dyn_cast_or_null<Defined>(ctx.symtab->find(sym.getName()));
  if (!sym2)
    return;
  const char *suffix2 = sym2->getVersionSuffix();
  if (suffix2[0] == '@' && suffix2[1] == '@' &&
      strcmp(suffix1 + 1, suffix2 + 2) == 0) {
    // foo@v1 and foo@@v1 should be merged, so redirect foo@v1 to foo@@v1.
    map.try_emplace(&sym, sym2);
    // If both foo@v1 and foo@@v1 are defined and non-weak, report a
    // duplicate definition error.
    if (sym.isDefined()) {
      sym2->checkDuplicate(ctx, cast<Defined>(sym));
      sym2->resolve(ctx, cast<Defined>(sym));
    } else if (sym.isUndefined()) {
      sym2->resolve(ctx, cast<Undefined>(sym));
    } else {
      sym2->resolve(ctx, cast<SharedSymbol>(sym));
    }
    // Eliminate foo@v1 from the symbol table.
    sym.symbolKind = Symbol::PlaceholderKind;
    sym.isUsedInRegularObj = false;
  } else if (auto *sym1 = dyn_cast<Defined>(&sym)) {
    if (sym2->versionId > VER_NDX_GLOBAL
            ? ctx.arg.versionDefinitions[sym2->versionId].name == suffix1 + 1
            : sym1->section == sym2->section && sym1->value == sym2->value) {
      // Due to an assembler design flaw, if foo is defined, .symver foo,
      // foo@v1 defines both foo and foo@v1. Unless foo is bound to a
      // different version, GNU ld makes foo@v1 canonical and eliminates
      // foo. Emulate its behavior, otherwise we would have foo or foo@@v1
      // beside foo@v1. foo@v1 and foo combining does not apply if they are
      // not defined in the same place.
      map.try_emplace(sym2, &sym);
      sym2->symbolKind = Symbol::PlaceholderKind;
      sym2->isUsedInRegularObj = false;
    }
  }
}

// Do renaming for --wrap and foo@v1 by updating pointers to symbols.
//
// When this function is executed, only InputFiles and symbol table
// contain pointers to symbol objects. We visit them to replace pointers,
// so that wrapped symbols are swapped as instructed by the command line.
static void redirectSymbols(Ctx &ctx, ArrayRef<WrappedSymbol> wrapped) {
  llvm::TimeTraceScope timeScope("Redirect symbols");
  DenseMap<Symbol *, Symbol *> map;
  for (const WrappedSymbol &w : wrapped) {
    map[w.sym] = w.wrap;
    map[w.real] = w.sym;
  }

  // If there are version definitions (versionDefinitions.size() > 2), enumerate
  // symbols with a non-default version (foo@v1) and check whether it should be
  // combined with foo or foo@@v1.
  if (ctx.arg.versionDefinitions.size() > 2)
    for (Symbol *sym : ctx.symtab->getSymbols())
      if (sym->hasVersionSuffix)
        combineVersionedSymbol(ctx, *sym, map);

  if (map.empty())
    return;

  // Update pointers in input files.
  parallelForEach(ctx.objectFiles, [&](ELFFileBase *file) {
    for (Symbol *&sym : file->getMutableGlobalSymbols())
      if (Symbol *s = map.lookup(sym))
        sym = s;
  });

  // Update pointers in the symbol table.
  for (const WrappedSymbol &w : wrapped)
    ctx.symtab->wrap(w.sym, w.real, w.wrap);
}

// To enable CET (x86's hardware-assisted control flow enforcement), each
// source file must be compiled with -fcf-protection. Object files compiled
// with the flag contain feature flags indicating that they are compatible
// with CET. We enable the feature only when all object files are compatible
// with CET.
//
// This is also the case with AARCH64's BTI and PAC which use the similar
// GNU_PROPERTY_AARCH64_FEATURE_1_AND mechanism.
//
// For AArch64 PAuth-enabled object files, the core info of all of them must
// match. Missing info for some object files with matching info for remaining
// ones can be allowed (see -z pauth-report).
static void readSecurityNotes(Ctx &ctx) {
  if (ctx.arg.emachine != EM_386 && ctx.arg.emachine != EM_X86_64 &&
      ctx.arg.emachine != EM_AARCH64)
    return;

  ctx.arg.andFeatures = -1;

  StringRef referenceFileName;
  if (ctx.arg.emachine == EM_AARCH64) {
    auto it = llvm::find_if(ctx.objectFiles, [](const ELFFileBase *f) {
      return !f->aarch64PauthAbiCoreInfo.empty();
    });
    if (it != ctx.objectFiles.end()) {
      ctx.aarch64PauthAbiCoreInfo = (*it)->aarch64PauthAbiCoreInfo;
      referenceFileName = (*it)->getName();
    }
  }
  bool hasValidPauthAbiCoreInfo = llvm::any_of(
      ctx.aarch64PauthAbiCoreInfo, [](uint8_t c) { return c != 0; });

  auto report = [&](StringRef config) -> ELFSyncStream {
    if (config == "error")
      return {ctx, DiagLevel::Err};
    else if (config == "warning")
      return {ctx, DiagLevel::Warn};
    return {ctx, DiagLevel::None};
  };
  auto reportUnless = [&](StringRef config, bool cond) -> ELFSyncStream {
    if (cond)
      return {ctx, DiagLevel::None};
    return report(config);
  };
  for (ELFFileBase *f : ctx.objectFiles) {
    uint32_t features = f->andFeatures;

    reportUnless(ctx.arg.zBtiReport,
                 features & GNU_PROPERTY_AARCH64_FEATURE_1_BTI)
        << f
        << ": -z bti-report: file does not have "
           "GNU_PROPERTY_AARCH64_FEATURE_1_BTI property";

    reportUnless(ctx.arg.zGcsReport,
                 features & GNU_PROPERTY_AARCH64_FEATURE_1_GCS)
        << f
        << ": -z gcs-report: file does not have "
           "GNU_PROPERTY_AARCH64_FEATURE_1_GCS property";

    reportUnless(ctx.arg.zCetReport, features & GNU_PROPERTY_X86_FEATURE_1_IBT)
        << f
        << ": -z cet-report: file does not have "
           "GNU_PROPERTY_X86_FEATURE_1_IBT property";

    reportUnless(ctx.arg.zCetReport,
                 features & GNU_PROPERTY_X86_FEATURE_1_SHSTK)
        << f
        << ": -z cet-report: file does not have "
           "GNU_PROPERTY_X86_FEATURE_1_SHSTK property";

    if (ctx.arg.zForceBti && !(features & GNU_PROPERTY_AARCH64_FEATURE_1_BTI)) {
      features |= GNU_PROPERTY_AARCH64_FEATURE_1_BTI;
      if (ctx.arg.zBtiReport == "none")
        Warn(ctx) << f
                  << ": -z force-bti: file does not have "
                     "GNU_PROPERTY_AARCH64_FEATURE_1_BTI property";
    } else if (ctx.arg.zForceIbt &&
               !(features & GNU_PROPERTY_X86_FEATURE_1_IBT)) {
      if (ctx.arg.zCetReport == "none")
        Warn(ctx) << f
                  << ": -z force-ibt: file does not have "
                     "GNU_PROPERTY_X86_FEATURE_1_IBT property";
      features |= GNU_PROPERTY_X86_FEATURE_1_IBT;
    }
    if (ctx.arg.zPacPlt && !(hasValidPauthAbiCoreInfo ||
                             (features & GNU_PROPERTY_AARCH64_FEATURE_1_PAC))) {
      Warn(ctx) << f
                << ": -z pac-plt: file does not have "
                   "GNU_PROPERTY_AARCH64_FEATURE_1_PAC property and no valid "
                   "PAuth core info present for this link job";
      features |= GNU_PROPERTY_AARCH64_FEATURE_1_PAC;
    }
    ctx.arg.andFeatures &= features;

    if (ctx.aarch64PauthAbiCoreInfo.empty())
      continue;

    if (f->aarch64PauthAbiCoreInfo.empty()) {
      report(ctx.arg.zPauthReport)
          << f
          << ": -z pauth-report: file does not have AArch64 "
             "PAuth core info while '"
          << referenceFileName << "' has one";
      continue;
    }

    if (ctx.aarch64PauthAbiCoreInfo != f->aarch64PauthAbiCoreInfo)
      Err(ctx) << "incompatible values of AArch64 PAuth core info found\n>>> "
               << referenceFileName << ": 0x"
               << toHex(ctx.aarch64PauthAbiCoreInfo, /*LowerCase=*/true)
               << "\n>>> " << f << ": 0x"
               << toHex(f->aarch64PauthAbiCoreInfo, /*LowerCase=*/true);
  }

  // Force enable Shadow Stack.
  if (ctx.arg.zShstk)
    ctx.arg.andFeatures |= GNU_PROPERTY_X86_FEATURE_1_SHSTK;

  // Force enable/disable GCS
  if (ctx.arg.zGcs == GcsPolicy::Always)
    ctx.arg.andFeatures |= GNU_PROPERTY_AARCH64_FEATURE_1_GCS;
  else if (ctx.arg.zGcs == GcsPolicy::Never)
    ctx.arg.andFeatures &= ~GNU_PROPERTY_AARCH64_FEATURE_1_GCS;
}

static void initSectionsAndLocalSyms(ELFFileBase *file, bool ignoreComdats) {
  switch (file->ekind) {
  case ELF32LEKind:
    cast<ObjFile<ELF32LE>>(file)->initSectionsAndLocalSyms(ignoreComdats);
    break;
  case ELF32BEKind:
    cast<ObjFile<ELF32BE>>(file)->initSectionsAndLocalSyms(ignoreComdats);
    break;
  case ELF64LEKind:
    cast<ObjFile<ELF64LE>>(file)->initSectionsAndLocalSyms(ignoreComdats);
    break;
  case ELF64BEKind:
    cast<ObjFile<ELF64BE>>(file)->initSectionsAndLocalSyms(ignoreComdats);
    break;
  default:
    llvm_unreachable("");
  }
}

static void postParseObjectFile(ELFFileBase *file) {
  switch (file->ekind) {
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
template <class ELFT> void LinkerDriver::link(opt::InputArgList &args) {
  llvm::TimeTraceScope timeScope("Link", StringRef("LinkerDriver::Link"));

  // Handle --trace-symbol.
  for (auto *arg : args.filtered(OPT_trace_symbol))
    ctx.symtab->insert(arg->getValue())->traced = true;

  ctx.internalFile = createInternalFile(ctx, "<internal>");

  // Handle -u/--undefined before input files. If both a.a and b.so define foo,
  // -u foo a.a b.so will extract a.a.
  for (StringRef name : ctx.arg.undefined)
    ctx.symtab->addUnusedUndefined(name)->referenced = true;

  parseFiles(ctx, files);

  // Dynamic linking is used if there is an input DSO,
  // or -shared or non-static pie is specified.
  ctx.hasDynsym = !ctx.sharedFiles.empty() || ctx.arg.shared ||
                  (ctx.arg.pie && !ctx.arg.noDynamicLinker);
  // Create dynamic sections for dynamic linking and static PIE.
  ctx.arg.hasDynSymTab = ctx.hasDynsym || ctx.arg.isPic;

  // If an entry symbol is in a static archive, pull out that file now.
  if (Symbol *sym = ctx.symtab->find(ctx.arg.entry))
    handleUndefined(ctx, sym, "--entry");

  // Handle the `--undefined-glob <pattern>` options.
  for (StringRef pat : args::getStrings(args, OPT_undefined_glob))
    handleUndefinedGlob(ctx, pat);

  // After potential archive member extraction involving ENTRY and
  // -u/--undefined-glob, check whether PROVIDE symbols should be defined (the
  // RHS may refer to definitions in just extracted object files).
  ctx.script->addScriptReferencedSymbolsToSymTable();

  // Prevent LTO from removing any definition referenced by -u.
  for (StringRef name : ctx.arg.undefined)
    if (Defined *sym = dyn_cast_or_null<Defined>(ctx.symtab->find(name)))
      sym->isUsedInRegularObj = true;

  // Mark -init and -fini symbols so that the LTO doesn't eliminate them.
  if (Symbol *sym = dyn_cast_or_null<Defined>(ctx.symtab->find(ctx.arg.init)))
    sym->isUsedInRegularObj = true;
  if (Symbol *sym = dyn_cast_or_null<Defined>(ctx.symtab->find(ctx.arg.fini)))
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
  if (!ctx.bitcodeFiles.empty()) {
    llvm::Triple TT(ctx.bitcodeFiles.front()->obj->getTargetTriple());
    for (auto *s : lto::LTO::getRuntimeLibcallSymbols(TT))
      handleLibcall(ctx, s);
  }

  // Archive members defining __wrap symbols may be extracted.
  std::vector<WrappedSymbol> wrapped = addWrappedSymbols(ctx, args);

  // No more lazy bitcode can be extracted at this point. Do post parse work
  // like checking duplicate symbols.
  parallelForEach(ctx.objectFiles, [](ELFFileBase *file) {
    initSectionsAndLocalSyms(file, /*ignoreComdats=*/false);
  });
  parallelForEach(ctx.objectFiles, postParseObjectFile);
  parallelForEach(ctx.bitcodeFiles,
                  [](BitcodeFile *file) { file->postParse(); });
  for (auto &it : ctx.nonPrevailingSyms) {
    Symbol &sym = *it.first;
    Undefined(sym.file, sym.getName(), sym.binding, sym.stOther, sym.type,
              it.second)
        .overwrite(sym);
    cast<Undefined>(sym).nonPrevailing = true;
  }
  ctx.nonPrevailingSyms.clear();
  for (const DuplicateSymbol &d : ctx.duplicates)
    reportDuplicate(ctx, *d.sym, d.file, d.section, d.value);
  ctx.duplicates.clear();

  // Return if there were name resolution errors.
  if (errCount(ctx))
    return;

  // We want to declare linker script's symbols early,
  // so that we can version them.
  // They also might be exported if referenced by DSOs.
  ctx.script->declareSymbols();

  // Handle --exclude-libs. This is before scanVersionScript() due to a
  // workaround for Android ndk: for a defined versioned symbol in an archive
  // without a version node in the version script, Android does not expect a
  // 'has undefined version' error in -shared --exclude-libs=ALL mode (PR36295).
  // GNU ld errors in this case.
  if (args.hasArg(OPT_exclude_libs))
    excludeLibs(ctx, args);

  // Create elfHeader early. We need a dummy section in
  // addReservedSymbols to mark the created symbols as not absolute.
  ctx.out.elfHeader = std::make_unique<OutputSection>(ctx, "", 0, SHF_ALLOC);

  // We need to create some reserved symbols such as _end. Create them.
  if (!ctx.arg.relocatable)
    addReservedSymbols(ctx);

  // Apply version scripts.
  //
  // For a relocatable output, version scripts don't make sense, and
  // parsing a symbol version string (e.g. dropping "@ver1" from a symbol
  // name "foo@ver1") rather do harm, so we don't call this if -r is given.
  if (!ctx.arg.relocatable) {
    llvm::TimeTraceScope timeScope("Process symbol versions");
    ctx.symtab->scanVersionScript();

    parseVersionAndComputeIsPreemptible(ctx);
  }

  // Skip the normal linked output if some LTO options are specified.
  //
  // For --thinlto-index-only, index file creation is performed in
  // compileBitcodeFiles, so we are done afterwards. --plugin-opt=emit-llvm and
  // --plugin-opt=emit-asm create output files in bitcode or assembly code,
  // respectively. When only certain thinLTO modules are specified for
  // compilation, the intermediate object file are the expected output.
  const bool skipLinkedOutput = ctx.arg.thinLTOIndexOnly || ctx.arg.emitLLVM ||
                                ctx.arg.ltoEmitAsm ||
                                !ctx.arg.thinLTOModulesToCompile.empty();

  // Handle --lto-validate-all-vtables-have-type-infos.
  if (ctx.arg.ltoValidateAllVtablesHaveTypeInfos)
    ltoValidateAllVtablesHaveTypeInfos<ELFT>(ctx, args);

  // Do link-time optimization if given files are LLVM bitcode files.
  // This compiles bitcode files into real object files.
  //
  // With this the symbol table should be complete. After this, no new names
  // except a few linker-synthesized ones will be added to the symbol table.
  const size_t numObjsBeforeLTO = ctx.objectFiles.size();
  const size_t numInputFilesBeforeLTO = ctx.driver.files.size();
  compileBitcodeFiles<ELFT>(skipLinkedOutput);

  // Symbol resolution finished. Report backward reference problems,
  // --print-archive-stats=, and --why-extract=.
  reportBackrefs(ctx);
  writeArchiveStats(ctx);
  writeWhyExtract(ctx);
  if (errCount(ctx))
    return;

  // Bail out if normal linked output is skipped due to LTO.
  if (skipLinkedOutput)
    return;

  // compileBitcodeFiles may have produced lto.tmp object files. After this, no
  // more file will be added.
  auto newObjectFiles = ArrayRef(ctx.objectFiles).slice(numObjsBeforeLTO);
  parallelForEach(newObjectFiles, [](ELFFileBase *file) {
    initSectionsAndLocalSyms(file, /*ignoreComdats=*/true);
  });
  parallelForEach(newObjectFiles, postParseObjectFile);
  for (const DuplicateSymbol &d : ctx.duplicates)
    reportDuplicate(ctx, *d.sym, d.file, d.section, d.value);

  // ELF dependent libraries may have introduced new input files after LTO has
  // completed. This is an error if the files haven't already been parsed, since
  // changing the symbol table could break the semantic assumptions of LTO.
  auto newInputFiles = ArrayRef(ctx.driver.files).slice(numInputFilesBeforeLTO);
  if (!newInputFiles.empty()) {
    DenseSet<StringRef> oldFilenames;
    for (auto &f : ArrayRef(ctx.driver.files).slice(0, numInputFilesBeforeLTO))
      oldFilenames.insert(f->getName());
    for (auto &newFile : newInputFiles)
      if (!oldFilenames.contains(newFile->getName()))
        Err(ctx) << "input file '" << newFile->getName() << "' added after LTO";
  }

  // Handle --exclude-libs again because lto.tmp may reference additional
  // libcalls symbols defined in an excluded archive. This may override
  // versionId set by scanVersionScript() and isExported.
  if (args.hasArg(OPT_exclude_libs))
    excludeLibs(ctx, args);

  // Record [__acle_se_<sym>, <sym>] pairs for later processing.
  processArmCmseSymbols(ctx);

  // Apply symbol renames for --wrap and combine foo@v1 and foo@@v1.
  redirectSymbols(ctx, wrapped);

  // Replace common symbols with regular symbols.
  replaceCommonSymbols(ctx);

  {
    llvm::TimeTraceScope timeScope("Aggregate sections");
    // Now that we have a complete list of input files.
    // Beyond this point, no new files are added.
    // Aggregate all input sections into one place.
    for (InputFile *f : ctx.objectFiles) {
      for (InputSectionBase *s : f->getSections()) {
        if (!s || s == &InputSection::discarded)
          continue;
        if (LLVM_UNLIKELY(isa<EhInputSection>(s)))
          ctx.ehInputSections.push_back(cast<EhInputSection>(s));
        else
          ctx.inputSections.push_back(s);
      }
    }
    for (BinaryFile *f : ctx.binaryFiles)
      for (InputSectionBase *s : f->getSections())
        ctx.inputSections.push_back(cast<InputSection>(s));
  }

  {
    llvm::TimeTraceScope timeScope("Strip sections");
    if (ctx.hasSympart.load(std::memory_order_relaxed)) {
      llvm::erase_if(ctx.inputSections, [&ctx = ctx](InputSectionBase *s) {
        if (s->type != SHT_LLVM_SYMPART)
          return false;
        readSymbolPartitionSection<ELFT>(ctx, s);
        return true;
      });
    }
    // We do not want to emit debug sections if --strip-all
    // or --strip-debug are given.
    if (ctx.arg.strip != StripPolicy::None) {
      llvm::erase_if(ctx.inputSections, [](InputSectionBase *s) {
        if (isDebugSection(*s))
          return true;
        if (auto *isec = dyn_cast<InputSection>(s))
          if (InputSectionBase *rel = isec->getRelocatedSection())
            if (isDebugSection(*rel))
              return true;

        return false;
      });
    }
  }

  // Since we now have a complete set of input files, we can create
  // a .d file to record build dependencies.
  if (!ctx.arg.dependencyFile.empty())
    writeDependencyFile(ctx);

  // Now that the number of partitions is fixed, save a pointer to the main
  // partition.
  ctx.mainPart = &ctx.partitions[0];

  // Read .note.gnu.property sections from input object files which
  // contain a hint to tweak linker's and loader's behaviors.
  readSecurityNotes(ctx);

  // The Target instance handles target-specific stuff, such as applying
  // relocations or writing a PLT section. It also contains target-dependent
  // values such as a default image base address.
  setTarget(ctx);

  ctx.arg.eflags = ctx.target->calcEFlags();
  // maxPageSize (sometimes called abi page size) is the maximum page size that
  // the output can be run on. For example if the OS can use 4k or 64k page
  // sizes then maxPageSize must be 64k for the output to be useable on both.
  // All important alignment decisions must use this value.
  ctx.arg.maxPageSize = getMaxPageSize(ctx, args);
  // commonPageSize is the most common page size that the output will be run on.
  // For example if an OS can use 4k or 64k page sizes and 4k is more common
  // than 64k then commonPageSize is set to 4k. commonPageSize can be used for
  // optimizations such as DATA_SEGMENT_ALIGN in linker scripts. LLD's use of it
  // is limited to writing trap instructions on the last executable segment.
  ctx.arg.commonPageSize = getCommonPageSize(ctx, args);

  ctx.arg.imageBase = getImageBase(ctx, args);

  // This adds a .comment section containing a version string.
  if (!ctx.arg.relocatable)
    ctx.inputSections.push_back(createCommentSection(ctx));

  // Split SHF_MERGE and .eh_frame sections into pieces in preparation for garbage collection.
  splitSections<ELFT>(ctx);

  // Garbage collection and removal of shared symbols from unused shared objects.
  markLive<ELFT>(ctx);

  // Make copies of any input sections that need to be copied into each
  // partition.
  copySectionsIntoPartitions(ctx);

  if (canHaveMemtagGlobals(ctx)) {
    llvm::TimeTraceScope timeScope("Process memory tagged symbols");
    createTaggedSymbols(ctx);
  }

  // Create synthesized sections such as .got and .plt. This is called before
  // processSectionCommands() so that they can be placed by SECTIONS commands.
  createSyntheticSections<ELFT>(ctx);

  // Some input sections that are used for exception handling need to be moved
  // into synthetic sections. Do that now so that they aren't assigned to
  // output sections in the usual way.
  if (!ctx.arg.relocatable)
    combineEhSections(ctx);

  // Merge .riscv.attributes sections.
  if (ctx.arg.emachine == EM_RISCV)
    mergeRISCVAttributesSections(ctx);

  {
    llvm::TimeTraceScope timeScope("Assign sections");

    // Create output sections described by SECTIONS commands.
    ctx.script->processSectionCommands();

    // Linker scripts control how input sections are assigned to output
    // sections. Input sections that were not handled by scripts are called
    // "orphans", and they are assigned to output sections by the default rule.
    // Process that.
    ctx.script->addOrphanSections();
  }

  {
    llvm::TimeTraceScope timeScope("Merge/finalize input sections");

    // Migrate InputSectionDescription::sectionBases to sections. This includes
    // merging MergeInputSections into a single MergeSyntheticSection. From this
    // point onwards InputSectionDescription::sections should be used instead of
    // sectionBases.
    for (SectionCommand *cmd : ctx.script->sectionCommands)
      if (auto *osd = dyn_cast<OutputDesc>(cmd))
        osd->osec.finalizeInputSections();
  }

  // Two input sections with different output sections should not be folded.
  // ICF runs after processSectionCommands() so that we know the output sections.
  if (ctx.arg.icf != ICFLevel::None) {
    findKeepUniqueSections<ELFT>(ctx, args);
    doIcf<ELFT>(ctx);
  }

  // Read the callgraph now that we know what was gced or icfed
  if (ctx.arg.callGraphProfileSort != CGProfileSortKind::None) {
    if (auto *arg = args.getLastArg(OPT_call_graph_ordering_file)) {
      if (std::optional<MemoryBufferRef> buffer =
              readFile(ctx, arg->getValue()))
        readCallGraph(ctx, *buffer);
    } else
      readCallGraphsFromObjectFiles<ELFT>(ctx);
  }

  // Write the result to the file.
  writeResult<ELFT>(ctx);
}
