//===- Driver.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Config.h"
#include "InputFiles.h"
#include "LTO.h"
#include "ObjC.h"
#include "OutputSection.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "Writer.h"

#include "lld/Common/Args.h"
#include "lld/Common/Driver.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/LLVM.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Reproduce.h"
#include "lld/Common/Version.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Config/config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TarWriter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/TextAPI/PackedVersion.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::object;
using namespace llvm::opt;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

Configuration *macho::config;
DependencyTracker *macho::depTracker;

static HeaderFileType getOutputType(const InputArgList &args) {
  // TODO: -r, -dylinker, -preload...
  Arg *outputArg = args.getLastArg(OPT_bundle, OPT_dylib, OPT_execute);
  if (outputArg == nullptr)
    return MH_EXECUTE;

  switch (outputArg->getOption().getID()) {
  case OPT_bundle:
    return MH_BUNDLE;
  case OPT_dylib:
    return MH_DYLIB;
  case OPT_execute:
    return MH_EXECUTE;
  default:
    llvm_unreachable("internal error");
  }
}

static Optional<StringRef> findLibrary(StringRef name) {
  if (config->searchDylibsFirst) {
    if (Optional<StringRef> path = findPathCombination(
            "lib" + name, config->librarySearchPaths, {".tbd", ".dylib"}))
      return path;
    return findPathCombination("lib" + name, config->librarySearchPaths,
                               {".a"});
  }
  return findPathCombination("lib" + name, config->librarySearchPaths,
                             {".tbd", ".dylib", ".a"});
}

static Optional<std::string> findFramework(StringRef name) {
  SmallString<260> symlink;
  StringRef suffix;
  std::tie(name, suffix) = name.split(",");
  for (StringRef dir : config->frameworkSearchPaths) {
    symlink = dir;
    path::append(symlink, name + ".framework", name);

    if (!suffix.empty()) {
      // NOTE: we must resolve the symlink before trying the suffixes, because
      // there are no symlinks for the suffixed paths.
      SmallString<260> location;
      if (!fs::real_path(symlink, location)) {
        // only append suffix if realpath() succeeds
        Twine suffixed = location + suffix;
        if (fs::exists(suffixed))
          return suffixed.str();
      }
      // Suffix lookup failed, fall through to the no-suffix case.
    }

    if (Optional<std::string> path = resolveDylibPath(symlink))
      return path;
  }
  return {};
}

static bool warnIfNotDirectory(StringRef option, StringRef path) {
  if (!fs::exists(path)) {
    warn("directory not found for option -" + option + path);
    return false;
  } else if (!fs::is_directory(path)) {
    warn("option -" + option + path + " references a non-directory path");
    return false;
  }
  return true;
}

static std::vector<StringRef>
getSearchPaths(unsigned optionCode, InputArgList &args,
               const std::vector<StringRef> &roots,
               const SmallVector<StringRef, 2> &systemPaths) {
  std::vector<StringRef> paths;
  StringRef optionLetter{optionCode == OPT_F ? "F" : "L"};
  for (StringRef path : args::getStrings(args, optionCode)) {
    // NOTE: only absolute paths are re-rooted to syslibroot(s)
    bool found = false;
    if (path::is_absolute(path, path::Style::posix)) {
      for (StringRef root : roots) {
        SmallString<261> buffer(root);
        path::append(buffer, path);
        // Do not warn about paths that are computed via the syslib roots
        if (fs::is_directory(buffer)) {
          paths.push_back(saver.save(buffer.str()));
          found = true;
        }
      }
    }
    if (!found && warnIfNotDirectory(optionLetter, path))
      paths.push_back(path);
  }

  // `-Z` suppresses the standard "system" search paths.
  if (args.hasArg(OPT_Z))
    return paths;

  for (const StringRef &path : systemPaths) {
    for (const StringRef &root : roots) {
      SmallString<261> buffer(root);
      path::append(buffer, path);
      if (fs::is_directory(buffer))
        paths.push_back(saver.save(buffer.str()));
    }
  }
  return paths;
}

static std::vector<StringRef> getSystemLibraryRoots(InputArgList &args) {
  std::vector<StringRef> roots;
  for (const Arg *arg : args.filtered(OPT_syslibroot))
    roots.push_back(arg->getValue());
  // NOTE: the final `-syslibroot` being `/` will ignore all roots
  if (roots.size() && roots.back() == "/")
    roots.clear();
  // NOTE: roots can never be empty - add an empty root to simplify the library
  // and framework search path computation.
  if (roots.empty())
    roots.emplace_back("");
  return roots;
}

static std::vector<StringRef>
getLibrarySearchPaths(InputArgList &args, const std::vector<StringRef> &roots) {
  return getSearchPaths(OPT_L, args, roots, {"/usr/lib", "/usr/local/lib"});
}

static std::vector<StringRef>
getFrameworkSearchPaths(InputArgList &args,
                        const std::vector<StringRef> &roots) {
  return getSearchPaths(OPT_F, args, roots,
                        {"/Library/Frameworks", "/System/Library/Frameworks"});
}

namespace {
struct ArchiveMember {
  MemoryBufferRef mbref;
  uint32_t modTime;
};
} // namespace

// Returns slices of MB by parsing MB as an archive file.
// Each slice consists of a member file in the archive.
static std::vector<ArchiveMember> getArchiveMembers(MemoryBufferRef mb) {
  std::unique_ptr<Archive> file =
      CHECK(Archive::create(mb),
            mb.getBufferIdentifier() + ": failed to parse archive");
  Archive *archive = file.get();
  make<std::unique_ptr<Archive>>(std::move(file)); // take ownership

  std::vector<ArchiveMember> v;
  Error err = Error::success();

  // Thin archives refer to .o files, so --reproduce needs the .o files too.
  bool addToTar = archive->isThin() && tar;

  for (const Archive::Child &c : archive->children(err)) {
    MemoryBufferRef mbref =
        CHECK(c.getMemoryBufferRef(),
              mb.getBufferIdentifier() +
                  ": could not get the buffer for a child of the archive");
    if (addToTar)
      tar->append(relativeToRoot(check(c.getFullName())), mbref.getBuffer());
    uint32_t modTime = toTimeT(
        CHECK(c.getLastModified(), mb.getBufferIdentifier() +
                                       ": could not get the modification "
                                       "time for a child of the archive"));
    v.push_back({mbref, modTime});
  }
  if (err)
    fatal(mb.getBufferIdentifier() +
          ": Archive::children failed: " + toString(std::move(err)));

  return v;
}

static InputFile *addFile(StringRef path, bool forceLoadArchive,
                          bool isBundleLoader = false) {
  Optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer)
    return nullptr;
  MemoryBufferRef mbref = *buffer;
  InputFile *newFile = nullptr;

  file_magic magic = identify_magic(mbref.getBuffer());
  switch (magic) {
  case file_magic::archive: {
    std::unique_ptr<object::Archive> file = CHECK(
        object::Archive::create(mbref), path + ": failed to parse archive");

    if (!file->isEmpty() && !file->hasSymbolTable())
      error(path + ": archive has no index; run ranlib to add one");

    if (config->allLoad || forceLoadArchive) {
      if (Optional<MemoryBufferRef> buffer = readFile(path)) {
        for (const ArchiveMember &member : getArchiveMembers(*buffer)) {
          if (Optional<InputFile *> file = loadArchiveMember(
                  member.mbref, member.modTime, path, /*objCOnly=*/false)) {
            inputFiles.insert(*file);
            printArchiveMemberLoad(
                (forceLoadArchive ? "-force_load" : "-all_load"),
                inputFiles.back());
          }
        }
      }
    } else if (config->forceLoadObjC) {
      for (const object::Archive::Symbol &sym : file->symbols())
        if (sym.getName().startswith(objc::klass))
          symtab->addUndefined(sym.getName(), /*file=*/nullptr,
                               /*isWeakRef=*/false);

      // TODO: no need to look for ObjC sections for a given archive member if
      // we already found that it contains an ObjC symbol. We should also
      // consider creating a LazyObjFile class in order to avoid double-loading
      // these files here and below (as part of the ArchiveFile).
      if (Optional<MemoryBufferRef> buffer = readFile(path)) {
        for (const ArchiveMember &member : getArchiveMembers(*buffer)) {
          if (Optional<InputFile *> file = loadArchiveMember(
                  member.mbref, member.modTime, path, /*objCOnly=*/true)) {
            inputFiles.insert(*file);
            printArchiveMemberLoad("-ObjC", inputFiles.back());
          }
        }
      }
    }

    newFile = make<ArchiveFile>(std::move(file));
    break;
  }
  case file_magic::macho_object:
    newFile = make<ObjFile>(mbref, getModTime(path), "");
    break;
  case file_magic::macho_dynamically_linked_shared_lib:
  case file_magic::macho_dynamically_linked_shared_lib_stub:
  case file_magic::tapi_file:
    if (Optional<DylibFile *> dylibFile = loadDylib(mbref))
      newFile = *dylibFile;
    break;
  case file_magic::bitcode:
    newFile = make<BitcodeFile>(mbref);
    break;
  case file_magic::macho_executable:
  case file_magic::macho_bundle:
    // We only allow executable and bundle type here if it is used
    // as a bundle loader.
    if (!isBundleLoader)
      error(path + ": unhandled file type");
    if (Optional<DylibFile *> dylibFile =
            loadDylib(mbref, nullptr, isBundleLoader))
      newFile = *dylibFile;
    break;
  default:
    error(path + ": unhandled file type");
  }
  if (newFile) {
    // printArchiveMemberLoad() prints both .a and .o names, so no need to
    // print the .a name here.
    if (config->printEachFile && magic != file_magic::archive)
      message(toString(newFile));
    inputFiles.insert(newFile);
  }
  return newFile;
}

static void addLibrary(StringRef name, bool isWeak) {
  if (Optional<StringRef> path = findLibrary(name)) {
    auto *dylibFile = dyn_cast_or_null<DylibFile>(addFile(*path, false));
    if (isWeak && dylibFile)
      dylibFile->forceWeakImport = true;
    return;
  }
  error("library not found for -l" + name);
}

static void addFramework(StringRef name, bool isWeak) {
  if (Optional<std::string> path = findFramework(name)) {
    auto *dylibFile = dyn_cast_or_null<DylibFile>(addFile(*path, false));
    if (isWeak && dylibFile)
      dylibFile->forceWeakImport = true;
    return;
  }
  error("framework not found for -framework " + name);
}

// Parses LC_LINKER_OPTION contents, which can add additional command line
// flags.
void macho::parseLCLinkerOption(InputFile *f, unsigned argc, StringRef data) {
  SmallVector<const char *, 4> argv;
  size_t offset = 0;
  for (unsigned i = 0; i < argc && offset < data.size(); ++i) {
    argv.push_back(data.data() + offset);
    offset += strlen(data.data() + offset) + 1;
  }
  if (argv.size() != argc || offset > data.size())
    fatal(toString(f) + ": invalid LC_LINKER_OPTION");

  MachOOptTable table;
  unsigned missingIndex, missingCount;
  InputArgList args = table.ParseArgs(argv, missingIndex, missingCount);
  if (missingCount)
    fatal(Twine(args.getArgString(missingIndex)) + ": missing argument");
  for (const Arg *arg : args.filtered(OPT_UNKNOWN))
    error("unknown argument: " + arg->getAsString(args));

  for (const Arg *arg : args) {
    switch (arg->getOption().getID()) {
    case OPT_l:
      addLibrary(arg->getValue(), false);
      break;
    case OPT_framework:
      addFramework(arg->getValue(), false);
      break;
    default:
      error(arg->getSpelling() + " is not allowed in LC_LINKER_OPTION");
    }
  }
}

static void addFileList(StringRef path) {
  Optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer)
    return;
  MemoryBufferRef mbref = *buffer;
  for (StringRef path : args::getLines(mbref))
    addFile(rerootPath(path), false);
}

// An order file has one entry per line, in the following format:
//
//   <cpu>:<object file>:<symbol name>
//
// <cpu> and <object file> are optional. If not specified, then that entry
// matches any symbol of that name. Parsing this format is not quite
// straightforward because the symbol name itself can contain colons, so when
// encountering a colon, we consider the preceding characters to decide if it
// can be a valid CPU type or file path.
//
// If a symbol is matched by multiple entries, then it takes the lowest-ordered
// entry (the one nearest to the front of the list.)
//
// The file can also have line comments that start with '#'.
static void parseOrderFile(StringRef path) {
  Optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer) {
    error("Could not read order file at " + path);
    return;
  }

  MemoryBufferRef mbref = *buffer;
  size_t priority = std::numeric_limits<size_t>::max();
  for (StringRef line : args::getLines(mbref)) {
    StringRef objectFile, symbol;
    line = line.take_until([](char c) { return c == '#'; }); // ignore comments
    line = line.ltrim();

    CPUType cpuType = StringSwitch<CPUType>(line)
                          .StartsWith("i386:", CPU_TYPE_I386)
                          .StartsWith("x86_64:", CPU_TYPE_X86_64)
                          .StartsWith("arm:", CPU_TYPE_ARM)
                          .StartsWith("arm64:", CPU_TYPE_ARM64)
                          .StartsWith("ppc:", CPU_TYPE_POWERPC)
                          .StartsWith("ppc64:", CPU_TYPE_POWERPC64)
                          .Default(CPU_TYPE_ANY);

    if (cpuType != CPU_TYPE_ANY && cpuType != target->cpuType)
      continue;

    // Drop the CPU type as well as the colon
    if (cpuType != CPU_TYPE_ANY)
      line = line.drop_until([](char c) { return c == ':'; }).drop_front();

    constexpr std::array<StringRef, 2> fileEnds = {".o:", ".o):"};
    for (StringRef fileEnd : fileEnds) {
      size_t pos = line.find(fileEnd);
      if (pos != StringRef::npos) {
        // Split the string around the colon
        objectFile = line.take_front(pos + fileEnd.size() - 1);
        line = line.drop_front(pos + fileEnd.size());
        break;
      }
    }
    symbol = line.trim();

    if (!symbol.empty()) {
      SymbolPriorityEntry &entry = config->priorities[symbol];
      if (!objectFile.empty())
        entry.objectFiles.insert(std::make_pair(objectFile, priority));
      else
        entry.anyObjectFile = std::max(entry.anyObjectFile, priority);
    }

    --priority;
  }
}

// We expect sub-library names of the form "libfoo", which will match a dylib
// with a path of .*/libfoo.{dylib, tbd}.
// XXX ld64 seems to ignore the extension entirely when matching sub-libraries;
// I'm not sure what the use case for that is.
static bool markReexport(StringRef searchName, ArrayRef<StringRef> extensions) {
  for (InputFile *file : inputFiles) {
    if (auto *dylibFile = dyn_cast<DylibFile>(file)) {
      StringRef filename = path::filename(dylibFile->getName());
      if (filename.consume_front(searchName) &&
          (filename.empty() ||
           find(extensions, filename) != extensions.end())) {
        dylibFile->reexport = true;
        return true;
      }
    }
  }
  return false;
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

static void compileBitcodeFiles() {
  TimeTraceScope timeScope("LTO");
  auto *lto = make<BitcodeCompiler>();
  for (InputFile *file : inputFiles)
    if (auto *bitcodeFile = dyn_cast<BitcodeFile>(file))
      lto->add(*bitcodeFile);

  for (ObjFile *file : lto->compile())
    inputFiles.insert(file);
}

// Replaces common symbols with defined symbols residing in __common sections.
// This function must be called after all symbol names are resolved (i.e. after
// all InputFiles have been loaded.) As a result, later operations won't see
// any CommonSymbols.
static void replaceCommonSymbols() {
  TimeTraceScope timeScope("Replace common symbols");
  for (Symbol *sym : symtab->getSymbols()) {
    auto *common = dyn_cast<CommonSymbol>(sym);
    if (common == nullptr)
      continue;

    auto *isec = make<InputSection>();
    isec->file = common->getFile();
    isec->name = section_names::common;
    isec->segname = segment_names::data;
    isec->align = common->align;
    // Casting to size_t will truncate large values on 32-bit architectures,
    // but it's not really worth supporting the linking of 64-bit programs on
    // 32-bit archs.
    isec->data = {nullptr, static_cast<size_t>(common->size)};
    isec->flags = S_ZEROFILL;
    inputSections.push_back(isec);

    replaceSymbol<Defined>(sym, sym->getName(), isec->file, isec, /*value=*/0,
                           /*size=*/0,
                           /*isWeakDef=*/false,
                           /*isExternal=*/true, common->privateExtern,
                           /*isThumb=*/false,
                           /*isReferencedDynamically=*/false);
  }
}

static void initializeSectionRenameMap() {
  if (config->dataConst) {
    SmallVector<StringRef> v{section_names::got,
                             section_names::authGot,
                             section_names::authPtr,
                             section_names::nonLazySymbolPtr,
                             section_names::const_,
                             section_names::cfString,
                             section_names::moduleInitFunc,
                             section_names::moduleTermFunc,
                             section_names::objcClassList,
                             section_names::objcNonLazyClassList,
                             section_names::objcCatList,
                             section_names::objcNonLazyCatList,
                             section_names::objcProtoList,
                             section_names::objcImageInfo};
    for (StringRef s : v)
      config->sectionRenameMap[{segment_names::data, s}] = {
          segment_names::dataConst, s};
  }
  config->sectionRenameMap[{segment_names::text, section_names::staticInit}] = {
      segment_names::text, section_names::text};
  config->sectionRenameMap[{segment_names::import, section_names::pointers}] = {
      config->dataConst ? segment_names::dataConst : segment_names::data,
      section_names::nonLazySymbolPtr};
}

static inline char toLowerDash(char x) {
  if (x >= 'A' && x <= 'Z')
    return x - 'A' + 'a';
  else if (x == ' ')
    return '-';
  return x;
}

static std::string lowerDash(StringRef s) {
  return std::string(map_iterator(s.begin(), toLowerDash),
                     map_iterator(s.end(), toLowerDash));
}

// Has the side-effect of setting Config::platformInfo.
static PlatformKind parsePlatformVersion(const ArgList &args) {
  const Arg *arg = args.getLastArg(OPT_platform_version);
  if (!arg) {
    error("must specify -platform_version");
    return PlatformKind::unknown;
  }

  StringRef platformStr = arg->getValue(0);
  StringRef minVersionStr = arg->getValue(1);
  StringRef sdkVersionStr = arg->getValue(2);

  // TODO(compnerd) see if we can generate this case list via XMACROS
  PlatformKind platform =
      StringSwitch<PlatformKind>(lowerDash(platformStr))
          .Cases("macos", "1", PlatformKind::macOS)
          .Cases("ios", "2", PlatformKind::iOS)
          .Cases("tvos", "3", PlatformKind::tvOS)
          .Cases("watchos", "4", PlatformKind::watchOS)
          .Cases("bridgeos", "5", PlatformKind::bridgeOS)
          .Cases("mac-catalyst", "6", PlatformKind::macCatalyst)
          .Cases("ios-simulator", "7", PlatformKind::iOSSimulator)
          .Cases("tvos-simulator", "8", PlatformKind::tvOSSimulator)
          .Cases("watchos-simulator", "9", PlatformKind::watchOSSimulator)
          .Cases("driverkit", "10", PlatformKind::driverKit)
          .Default(PlatformKind::unknown);
  if (platform == PlatformKind::unknown)
    error(Twine("malformed platform: ") + platformStr);
  // TODO: check validity of version strings, which varies by platform
  // NOTE: ld64 accepts version strings with 5 components
  // llvm::VersionTuple accepts no more than 4 components
  // Has Apple ever published version strings with 5 components?
  if (config->platformInfo.minimum.tryParse(minVersionStr))
    error(Twine("malformed minimum version: ") + minVersionStr);
  if (config->platformInfo.sdk.tryParse(sdkVersionStr))
    error(Twine("malformed sdk version: ") + sdkVersionStr);
  return platform;
}

// Has the side-effect of setting Config::target.
static TargetInfo *createTargetInfo(InputArgList &args) {
  StringRef archName = args.getLastArgValue(OPT_arch);
  if (archName.empty())
    fatal("must specify -arch");
  PlatformKind platform = parsePlatformVersion(args);

  config->platformInfo.target =
      MachO::Target(getArchitectureFromName(archName), platform);

  uint32_t cpuType;
  uint32_t cpuSubtype;
  std::tie(cpuType, cpuSubtype) = getCPUTypeFromArchitecture(config->arch());

  switch (cpuType) {
  case CPU_TYPE_X86_64:
    return createX86_64TargetInfo();
  case CPU_TYPE_ARM64:
    return createARM64TargetInfo();
  case CPU_TYPE_ARM64_32:
    return createARM64_32TargetInfo();
  case CPU_TYPE_ARM:
    return createARMTargetInfo(cpuSubtype);
  default:
    fatal("missing or unsupported -arch " + archName);
  }
}

static UndefinedSymbolTreatment
getUndefinedSymbolTreatment(const ArgList &args) {
  StringRef treatmentStr = args.getLastArgValue(OPT_undefined);
  auto treatment =
      StringSwitch<UndefinedSymbolTreatment>(treatmentStr)
          .Cases("error", "", UndefinedSymbolTreatment::error)
          .Case("warning", UndefinedSymbolTreatment::warning)
          .Case("suppress", UndefinedSymbolTreatment::suppress)
          .Case("dynamic_lookup", UndefinedSymbolTreatment::dynamic_lookup)
          .Default(UndefinedSymbolTreatment::unknown);
  if (treatment == UndefinedSymbolTreatment::unknown) {
    warn(Twine("unknown -undefined TREATMENT '") + treatmentStr +
         "', defaulting to 'error'");
    treatment = UndefinedSymbolTreatment::error;
  } else if (config->namespaceKind == NamespaceKind::twolevel &&
             (treatment == UndefinedSymbolTreatment::warning ||
              treatment == UndefinedSymbolTreatment::suppress)) {
    if (treatment == UndefinedSymbolTreatment::warning)
      error("'-undefined warning' only valid with '-flat_namespace'");
    else
      error("'-undefined suppress' only valid with '-flat_namespace'");
    treatment = UndefinedSymbolTreatment::error;
  }
  return treatment;
}

static void warnIfDeprecatedOption(const Option &opt) {
  if (!opt.getGroup().isValid())
    return;
  if (opt.getGroup().getID() == OPT_grp_deprecated) {
    warn("Option `" + opt.getPrefixedName() + "' is deprecated in ld64:");
    warn(opt.getHelpText());
  }
}

static void warnIfUnimplementedOption(const Option &opt) {
  if (!opt.getGroup().isValid() || !opt.hasFlag(DriverFlag::HelpHidden))
    return;
  switch (opt.getGroup().getID()) {
  case OPT_grp_deprecated:
    // warn about deprecated options elsewhere
    break;
  case OPT_grp_undocumented:
    warn("Option `" + opt.getPrefixedName() +
         "' is undocumented. Should lld implement it?");
    break;
  case OPT_grp_obsolete:
    warn("Option `" + opt.getPrefixedName() +
         "' is obsolete. Please modernize your usage.");
    break;
  case OPT_grp_ignored:
    warn("Option `" + opt.getPrefixedName() + "' is ignored.");
    break;
  default:
    warn("Option `" + opt.getPrefixedName() +
         "' is not yet implemented. Stay tuned...");
    break;
  }
}

static const char *getReproduceOption(InputArgList &args) {
  if (const Arg *arg = args.getLastArg(OPT_reproduce))
    return arg->getValue();
  return getenv("LLD_REPRODUCE");
}

static void parseClangOption(StringRef opt, const Twine &msg) {
  std::string err;
  raw_string_ostream os(err);

  const char *argv[] = {"lld", opt.data()};
  if (cl::ParseCommandLineOptions(2, argv, "", &os))
    return;
  os.flush();
  error(msg + ": " + StringRef(err).trim());
}

static uint32_t parseDylibVersion(const ArgList &args, unsigned id) {
  const Arg *arg = args.getLastArg(id);
  if (!arg)
    return 0;

  if (config->outputType != MH_DYLIB) {
    error(arg->getAsString(args) + ": only valid with -dylib");
    return 0;
  }

  PackedVersion version;
  if (!version.parse32(arg->getValue())) {
    error(arg->getAsString(args) + ": malformed version");
    return 0;
  }

  return version.rawValue();
}

static uint32_t parseProtection(StringRef protStr) {
  uint32_t prot = 0;
  for (char c : protStr) {
    switch (c) {
    case 'r':
      prot |= VM_PROT_READ;
      break;
    case 'w':
      prot |= VM_PROT_WRITE;
      break;
    case 'x':
      prot |= VM_PROT_EXECUTE;
      break;
    case '-':
      break;
    default:
      error("unknown -segprot letter '" + Twine(c) + "' in " + protStr);
      return 0;
    }
  }
  return prot;
}

static std::vector<SectionAlign> parseSectAlign(const opt::InputArgList &args) {
  std::vector<SectionAlign> sectAligns;
  for (const Arg *arg : args.filtered(OPT_sectalign)) {
    StringRef segName = arg->getValue(0);
    StringRef sectName = arg->getValue(1);
    StringRef alignStr = arg->getValue(2);
    if (alignStr.startswith("0x") || alignStr.startswith("0X"))
      alignStr = alignStr.drop_front(2);
    uint32_t align;
    if (alignStr.getAsInteger(16, align)) {
      error("-sectalign: failed to parse '" + StringRef(arg->getValue(2)) +
            "' as number");
      continue;
    }
    if (!isPowerOf2_32(align)) {
      error("-sectalign: '" + StringRef(arg->getValue(2)) +
            "' (in base 16) not a power of two");
      continue;
    }
    sectAligns.push_back({segName, sectName, align});
  }
  return sectAligns;
}

static bool dataConstDefault(const InputArgList &args) {
  switch (config->outputType) {
  case MH_EXECUTE:
    return !args.hasArg(OPT_no_pie);
  case MH_BUNDLE:
    // FIXME: return false when -final_name ...
    // has prefix "/System/Library/UserEventPlugins/"
    // or matches "/usr/libexec/locationd" "/usr/libexec/terminusd"
    return true;
  case MH_DYLIB:
    return true;
  case MH_OBJECT:
    return false;
  default:
    llvm_unreachable(
        "unsupported output type for determining data-const default");
  }
  return false;
}

void SymbolPatterns::clear() {
  literals.clear();
  globs.clear();
}

void SymbolPatterns::insert(StringRef symbolName) {
  if (symbolName.find_first_of("*?[]") == StringRef::npos)
    literals.insert(CachedHashStringRef(symbolName));
  else if (Expected<GlobPattern> pattern = GlobPattern::create(symbolName))
    globs.emplace_back(*pattern);
  else
    error("invalid symbol-name pattern: " + symbolName);
}

bool SymbolPatterns::matchLiteral(StringRef symbolName) const {
  return literals.contains(CachedHashStringRef(symbolName));
}

bool SymbolPatterns::matchGlob(StringRef symbolName) const {
  for (const llvm::GlobPattern &glob : globs)
    if (glob.match(symbolName))
      return true;
  return false;
}

bool SymbolPatterns::match(StringRef symbolName) const {
  return matchLiteral(symbolName) || matchGlob(symbolName);
}

static void handleSymbolPatterns(InputArgList &args,
                                 SymbolPatterns &symbolPatterns,
                                 unsigned singleOptionCode,
                                 unsigned listFileOptionCode) {
  for (const Arg *arg : args.filtered(singleOptionCode))
    symbolPatterns.insert(arg->getValue());
  for (const Arg *arg : args.filtered(listFileOptionCode)) {
    StringRef path = arg->getValue();
    Optional<MemoryBufferRef> buffer = readFile(path);
    if (!buffer) {
      error("Could not read symbol file: " + path);
      continue;
    }
    MemoryBufferRef mbref = *buffer;
    for (StringRef line : args::getLines(mbref)) {
      line = line.take_until([](char c) { return c == '#'; }).trim();
      if (!line.empty())
        symbolPatterns.insert(line);
    }
  }
}

void createFiles(const InputArgList &args) {
  TimeTraceScope timeScope("Load input files");
  // This loop should be reserved for options whose exact ordering matters.
  // Other options should be handled via filtered() and/or getLastArg().
  for (const Arg *arg : args) {
    const Option &opt = arg->getOption();
    warnIfDeprecatedOption(opt);
    warnIfUnimplementedOption(opt);

    switch (opt.getID()) {
    case OPT_INPUT:
      addFile(rerootPath(arg->getValue()), false);
      break;
    case OPT_weak_library:
      if (auto *dylibFile = dyn_cast_or_null<DylibFile>(
              addFile(rerootPath(arg->getValue()), false)))
        dylibFile->forceWeakImport = true;
      break;
    case OPT_filelist:
      addFileList(arg->getValue());
      break;
    case OPT_force_load:
      addFile(rerootPath(arg->getValue()), true);
      break;
    case OPT_l:
    case OPT_weak_l:
      addLibrary(arg->getValue(), opt.getID() == OPT_weak_l);
      break;
    case OPT_framework:
    case OPT_weak_framework:
      addFramework(arg->getValue(), opt.getID() == OPT_weak_framework);
      break;
    default:
      break;
    }
  }
}

bool macho::link(ArrayRef<const char *> argsArr, bool canExitEarly,
                 raw_ostream &stdoutOS, raw_ostream &stderrOS) {
  lld::stdoutOS = &stdoutOS;
  lld::stderrOS = &stderrOS;

  errorHandler().cleanupCallback = []() { freeArena(); };

  errorHandler().logName = args::getFilenameWithoutExe(argsArr[0]);
  stderrOS.enable_colors(stderrOS.has_colors());

  MachOOptTable parser;
  InputArgList args = parser.parse(argsArr.slice(1));

  errorHandler().errorLimitExceededMsg =
      "too many errors emitted, stopping now "
      "(use --error-limit=0 to see all errors)";
  errorHandler().errorLimit = args::getInteger(args, OPT_error_limit_eq, 20);
  errorHandler().verbose = args.hasArg(OPT_verbose);

  if (args.hasArg(OPT_help_hidden)) {
    parser.printHelp(argsArr[0], /*showHidden=*/true);
    return true;
  }
  if (args.hasArg(OPT_help)) {
    parser.printHelp(argsArr[0], /*showHidden=*/false);
    return true;
  }
  if (args.hasArg(OPT_version)) {
    message(getLLDVersion());
    return true;
  }

  config = make<Configuration>();
  symtab = make<SymbolTable>();
  target = createTargetInfo(args);
  depTracker =
      make<DependencyTracker>(args.getLastArgValue(OPT_dependency_info));

  config->systemLibraryRoots = getSystemLibraryRoots(args);
  if (const char *path = getReproduceOption(args)) {
    // Note that --reproduce is a debug option so you can ignore it
    // if you are trying to understand the whole picture of the code.
    Expected<std::unique_ptr<TarWriter>> errOrWriter =
        TarWriter::create(path, path::stem(path));
    if (errOrWriter) {
      tar = std::move(*errOrWriter);
      tar->append("response.txt", createResponseFile(args));
      tar->append("version.txt", getLLDVersion() + "\n");
    } else {
      error("--reproduce: " + toString(errOrWriter.takeError()));
    }
  }

  if (auto *arg = args.getLastArg(OPT_threads_eq)) {
    StringRef v(arg->getValue());
    unsigned threads = 0;
    if (!llvm::to_integer(v, threads, 0) || threads == 0)
      error(arg->getSpelling() + ": expected a positive integer, but got '" +
            arg->getValue() + "'");
    parallel::strategy = hardware_concurrency(threads);
    config->thinLTOJobs = v;
  }
  if (auto *arg = args.getLastArg(OPT_thinlto_jobs_eq))
    config->thinLTOJobs = arg->getValue();
  if (!get_threadpool_strategy(config->thinLTOJobs))
    error("--thinlto-jobs: invalid job count: " + config->thinLTOJobs);

  for (const Arg *arg : args.filtered(OPT_u)) {
    config->explicitUndefineds.push_back(symtab->addUndefined(
        arg->getValue(), /*file=*/nullptr, /*isWeakRef=*/false));
  }

  for (const Arg *arg : args.filtered(OPT_U))
    symtab->addDynamicLookup(arg->getValue());

  config->mapFile = args.getLastArgValue(OPT_map);
  config->outputFile = args.getLastArgValue(OPT_o, "a.out");
  config->astPaths = args.getAllArgValues(OPT_add_ast_path);
  config->headerPad = args::getHex(args, OPT_headerpad, /*Default=*/32);
  config->headerPadMaxInstallNames =
      args.hasArg(OPT_headerpad_max_install_names);
  config->printEachFile = args.hasArg(OPT_t);
  config->printWhyLoad = args.hasArg(OPT_why_load);
  config->outputType = getOutputType(args);
  if (const Arg *arg = args.getLastArg(OPT_bundle_loader)) {
    if (config->outputType != MH_BUNDLE)
      error("-bundle_loader can only be used with MachO bundle output");
    addFile(arg->getValue(), false, true);
  }
  config->ltoObjPath = args.getLastArgValue(OPT_object_path_lto);
  config->ltoNewPassManager =
      args.hasFlag(OPT_no_lto_legacy_pass_manager, OPT_lto_legacy_pass_manager,
                   LLVM_ENABLE_NEW_PASS_MANAGER);
  config->runtimePaths = args::getStrings(args, OPT_rpath);
  config->allLoad = args.hasArg(OPT_all_load);
  config->forceLoadObjC = args.hasArg(OPT_ObjC);
  config->demangle = args.hasArg(OPT_demangle);
  config->implicitDylibs = !args.hasArg(OPT_no_implicit_dylibs);
  config->emitFunctionStarts = !args.hasArg(OPT_no_function_starts);
  config->emitBitcodeBundle = args.hasArg(OPT_bitcode_bundle);

  std::array<PlatformKind, 3> encryptablePlatforms{
      PlatformKind::iOS, PlatformKind::watchOS, PlatformKind::tvOS};
  config->emitEncryptionInfo =
      args.hasFlag(OPT_encryptable, OPT_no_encryption,
                   is_contained(encryptablePlatforms, config->platform()));

#ifndef HAVE_LIBXAR
  if (config->emitBitcodeBundle)
    error("-bitcode_bundle unsupported because LLD wasn't built with libxar");
#endif

  if (const Arg *arg = args.getLastArg(OPT_install_name)) {
    if (config->outputType != MH_DYLIB)
      warn(arg->getAsString(args) + ": ignored, only has effect with -dylib");
    else
      config->installName = arg->getValue();
  } else if (config->outputType == MH_DYLIB) {
    config->installName = config->outputFile;
  }

  if (args.hasArg(OPT_mark_dead_strippable_dylib)) {
    if (config->outputType != MH_DYLIB)
      warn("-mark_dead_strippable_dylib: ignored, only has effect with -dylib");
    else
      config->markDeadStrippableDylib = true;
  }

  if (const Arg *arg = args.getLastArg(OPT_static, OPT_dynamic))
    config->staticLink = (arg->getOption().getID() == OPT_static);

  if (const Arg *arg =
          args.getLastArg(OPT_flat_namespace, OPT_twolevel_namespace))
    config->namespaceKind = arg->getOption().getID() == OPT_twolevel_namespace
                                ? NamespaceKind::twolevel
                                : NamespaceKind::flat;

  config->undefinedSymbolTreatment = getUndefinedSymbolTreatment(args);

  if (config->outputType == MH_EXECUTE)
    config->entry = symtab->addUndefined(args.getLastArgValue(OPT_e, "_main"),
                                         /*file=*/nullptr,
                                         /*isWeakRef=*/false);

  config->librarySearchPaths =
      getLibrarySearchPaths(args, config->systemLibraryRoots);
  config->frameworkSearchPaths =
      getFrameworkSearchPaths(args, config->systemLibraryRoots);
  if (const Arg *arg =
          args.getLastArg(OPT_search_paths_first, OPT_search_dylibs_first))
    config->searchDylibsFirst =
        arg->getOption().getID() == OPT_search_dylibs_first;

  config->dylibCompatibilityVersion =
      parseDylibVersion(args, OPT_compatibility_version);
  config->dylibCurrentVersion = parseDylibVersion(args, OPT_current_version);

  config->dataConst =
      args.hasFlag(OPT_data_const, OPT_no_data_const, dataConstDefault(args));
  // Populate config->sectionRenameMap with builtin default renames.
  // Options -rename_section and -rename_segment are able to override.
  initializeSectionRenameMap();
  // Reject every special character except '.' and '$'
  // TODO(gkm): verify that this is the proper set of invalid chars
  StringRef invalidNameChars("!\"#%&'()*+,-/:;<=>?@[\\]^`{|}~");
  auto validName = [invalidNameChars](StringRef s) {
    if (s.find_first_of(invalidNameChars) != StringRef::npos)
      error("invalid name for segment or section: " + s);
    return s;
  };
  for (const Arg *arg : args.filtered(OPT_rename_section)) {
    config->sectionRenameMap[{validName(arg->getValue(0)),
                              validName(arg->getValue(1))}] = {
        validName(arg->getValue(2)), validName(arg->getValue(3))};
  }
  for (const Arg *arg : args.filtered(OPT_rename_segment)) {
    config->segmentRenameMap[validName(arg->getValue(0))] =
        validName(arg->getValue(1));
  }

  config->sectionAlignments = parseSectAlign(args);

  for (const Arg *arg : args.filtered(OPT_segprot)) {
    StringRef segName = arg->getValue(0);
    uint32_t maxProt = parseProtection(arg->getValue(1));
    uint32_t initProt = parseProtection(arg->getValue(2));
    if (maxProt != initProt && config->arch() != AK_i386)
      error("invalid argument '" + arg->getAsString(args) +
            "': max and init must be the same for non-i386 archs");
    if (segName == segment_names::linkEdit)
      error("-segprot cannot be used to change __LINKEDIT's protections");
    config->segmentProtections.push_back({segName, maxProt, initProt});
  }

  handleSymbolPatterns(args, config->exportedSymbols, OPT_exported_symbol,
                       OPT_exported_symbols_list);
  handleSymbolPatterns(args, config->unexportedSymbols, OPT_unexported_symbol,
                       OPT_unexported_symbols_list);
  if (!config->exportedSymbols.empty() && !config->unexportedSymbols.empty()) {
    error("cannot use both -exported_symbol* and -unexported_symbol* options\n"
          ">>> ignoring unexports");
    config->unexportedSymbols.clear();
  }
  // Explicitly-exported literal symbols must be defined, but might
  // languish in an archive if unreferenced elsewhere. Light a fire
  // under those lazy symbols!
  for (const CachedHashStringRef &cachedName : config->exportedSymbols.literals)
    symtab->addUndefined(cachedName.val(), /*file=*/nullptr,
                         /*isWeakRef=*/false);

  config->saveTemps = args.hasArg(OPT_save_temps);

  config->adhocCodesign = args.hasFlag(
      OPT_adhoc_codesign, OPT_no_adhoc_codesign,
      (config->arch() == AK_arm64 || config->arch() == AK_arm64e) &&
          config->platform() == PlatformKind::macOS);

  if (args.hasArg(OPT_v)) {
    message(getLLDVersion());
    message(StringRef("Library search paths:") +
            (config->librarySearchPaths.empty()
                 ? ""
                 : "\n\t" + join(config->librarySearchPaths, "\n\t")));
    message(StringRef("Framework search paths:") +
            (config->frameworkSearchPaths.empty()
                 ? ""
                 : "\n\t" + join(config->frameworkSearchPaths, "\n\t")));
  }

  config->progName = argsArr[0];

  config->timeTraceEnabled = args.hasArg(
      OPT_time_trace, OPT_time_trace_granularity_eq, OPT_time_trace_file_eq);
  config->timeTraceGranularity =
      args::getInteger(args, OPT_time_trace_granularity_eq, 500);

  // Initialize time trace profiler.
  if (config->timeTraceEnabled)
    timeTraceProfilerInitialize(config->timeTraceGranularity, config->progName);

  {
    TimeTraceScope timeScope("ExecuteLinker");

    initLLVM(); // must be run before any call to addFile()
    createFiles(args);

    config->isPic = config->outputType == MH_DYLIB ||
                    config->outputType == MH_BUNDLE ||
                    (config->outputType == MH_EXECUTE &&
                     args.hasFlag(OPT_pie, OPT_no_pie, true));

    // Now that all dylibs have been loaded, search for those that should be
    // re-exported.
    {
      auto reexportHandler = [](const Arg *arg,
                                const std::vector<StringRef> &extensions) {
        config->hasReexports = true;
        StringRef searchName = arg->getValue();
        if (!markReexport(searchName, extensions))
          error(arg->getSpelling() + " " + searchName +
                " does not match a supplied dylib");
      };
      std::vector<StringRef> extensions = {".tbd"};
      for (const Arg *arg : args.filtered(OPT_sub_umbrella))
        reexportHandler(arg, extensions);

      extensions.push_back(".dylib");
      for (const Arg *arg : args.filtered(OPT_sub_library))
        reexportHandler(arg, extensions);
    }

    // Parse LTO options.
    if (const Arg *arg = args.getLastArg(OPT_mcpu))
      parseClangOption(saver.save("-mcpu=" + StringRef(arg->getValue())),
                       arg->getSpelling());

    for (const Arg *arg : args.filtered(OPT_mllvm))
      parseClangOption(arg->getValue(), arg->getSpelling());

    compileBitcodeFiles();
    replaceCommonSymbols();

    StringRef orderFile = args.getLastArgValue(OPT_order_file);
    if (!orderFile.empty())
      parseOrderFile(orderFile);

    if (config->entry)
      if (auto *undefined = dyn_cast<Undefined>(config->entry))
        treatUndefinedSymbol(*undefined, "the entry point");

    // FIXME: This prints symbols that are undefined both in input files and
    // via -u flag twice.
    for (const Symbol *sym : config->explicitUndefineds) {
      if (const auto *undefined = dyn_cast<Undefined>(sym))
        treatUndefinedSymbol(*undefined, "-u");
    }
    // Literal exported-symbol names must be defined, but glob
    // patterns need not match.
    for (const CachedHashStringRef &cachedName :
         config->exportedSymbols.literals) {
      if (const Symbol *sym = symtab->find(cachedName))
        if (const auto *undefined = dyn_cast<Undefined>(sym))
          treatUndefinedSymbol(*undefined, "-exported_symbol(s_list)");
    }

    // FIXME: should terminate the link early based on errors encountered so
    // far?

    createSyntheticSections();
    createSyntheticSymbols();

    for (const Arg *arg : args.filtered(OPT_sectcreate)) {
      StringRef segName = arg->getValue(0);
      StringRef sectName = arg->getValue(1);
      StringRef fileName = arg->getValue(2);
      Optional<MemoryBufferRef> buffer = readFile(fileName);
      if (buffer)
        inputFiles.insert(make<OpaqueFile>(*buffer, segName, sectName));
    }

    {
      TimeTraceScope timeScope("Gathering input sections");
      // Gather all InputSections into one vector.
      for (const InputFile *file : inputFiles) {
        for (const SubsectionMap &map : file->subsections)
          for (const SubsectionEntry &subsectionEntry : map)
            inputSections.push_back(subsectionEntry.isec);
      }
    }

    // Write to an output file.
    if (target->wordSize == 8)
      writeResult<LP64>();
    else
      writeResult<ILP32>();

    depTracker->write(getLLDVersion(), inputFiles, config->outputFile);
  }

  if (config->timeTraceEnabled) {
    if (auto E = timeTraceProfilerWrite(
            args.getLastArgValue(OPT_time_trace_file_eq).str(),
            config->outputFile)) {
      handleAllErrors(std::move(E),
                      [&](const StringError &SE) { error(SE.getMessage()); });
    }

    timeTraceProfilerCleanup();
  }

  if (canExitEarly)
    exitLld(errorCount() ? 1 : 0);

  return !errorCount();
}
