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
#include "lld/Common/Version.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::object;
using namespace llvm::opt;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

Configuration *lld::macho::config;

static HeaderFileType getOutputType(const opt::InputArgList &args) {
  // TODO: -r, -dylinker, -preload...
  opt::Arg *outputArg = args.getLastArg(OPT_bundle, OPT_dylib, OPT_execute);
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

static Optional<std::string>
findAlongPathsWithExtensions(StringRef name, ArrayRef<StringRef> extensions) {
  llvm::SmallString<261> base;
  for (StringRef dir : config->librarySearchPaths) {
    base = dir;
    path::append(base, Twine("lib") + name);
    for (StringRef ext : extensions) {
      Twine location = base + ext;
      if (fs::exists(location))
        return location.str();
    }
  }
  return {};
}

static Optional<std::string> findLibrary(StringRef name) {
  if (config->searchDylibsFirst) {
    if (Optional<std::string> path =
            findAlongPathsWithExtensions(name, {".tbd", ".dylib"}))
      return path;
    return findAlongPathsWithExtensions(name, {".a"});
  }
  return findAlongPathsWithExtensions(name, {".tbd", ".dylib", ".a"});
}

static Optional<std::string> findFramework(StringRef name) {
  llvm::SmallString<260> symlink;
  StringRef suffix;
  std::tie(name, suffix) = name.split(",");
  for (StringRef dir : config->frameworkSearchPaths) {
    symlink = dir;
    path::append(symlink, name + ".framework", name);

    if (!suffix.empty()) {
      // NOTE: we must resolve the symlink before trying the suffixes, because
      // there are no symlinks for the suffixed paths.
      llvm::SmallString<260> location;
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

static TargetInfo *createTargetInfo(opt::InputArgList &args) {
  StringRef arch = args.getLastArgValue(OPT_arch, "x86_64");
  config->arch = llvm::MachO::getArchitectureFromName(
      args.getLastArgValue(OPT_arch, arch));
  switch (config->arch) {
  case llvm::MachO::AK_x86_64:
  case llvm::MachO::AK_x86_64h:
    return createX86_64TargetInfo();
  default:
    fatal("missing or unsupported -arch " + arch);
  }
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
getSearchPaths(unsigned optionCode, opt::InputArgList &args,
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

  for (auto const &path : systemPaths) {
    for (auto root : roots) {
      SmallString<261> buffer(root);
      path::append(buffer, path);
      if (warnIfNotDirectory(optionLetter, buffer))
        paths.push_back(saver.save(buffer.str()));
    }
  }
  return paths;
}

static std::vector<StringRef> getSystemLibraryRoots(opt::InputArgList &args) {
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
getLibrarySearchPaths(opt::InputArgList &args,
                      const std::vector<StringRef> &roots) {
  return getSearchPaths(OPT_L, args, roots, {"/usr/lib", "/usr/local/lib"});
}

static std::vector<StringRef>
getFrameworkSearchPaths(opt::InputArgList &args,
                        const std::vector<StringRef> &roots) {
  return getSearchPaths(OPT_F, args, roots,
                        {"/Library/Frameworks", "/System/Library/Frameworks"});
}

// Returns slices of MB by parsing MB as an archive file.
// Each slice consists of a member file in the archive.
static std::vector<MemoryBufferRef> getArchiveMembers(MemoryBufferRef mb) {
  std::unique_ptr<Archive> file =
      CHECK(Archive::create(mb),
            mb.getBufferIdentifier() + ": failed to parse archive");

  std::vector<MemoryBufferRef> v;
  Error err = Error::success();
  for (const Archive::Child &c : file->children(err)) {
    MemoryBufferRef mbref =
        CHECK(c.getMemoryBufferRef(),
              mb.getBufferIdentifier() +
                  ": could not get the buffer for a child of the archive");
    v.push_back(mbref);
  }
  if (err)
    fatal(mb.getBufferIdentifier() +
          ": Archive::children failed: " + toString(std::move(err)));

  return v;
}

static InputFile *addFile(StringRef path) {
  Optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer)
    return nullptr;
  MemoryBufferRef mbref = *buffer;
  InputFile *newFile = nullptr;

  switch (identify_magic(mbref.getBuffer())) {
  case file_magic::archive: {
    std::unique_ptr<object::Archive> file = CHECK(
        object::Archive::create(mbref), path + ": failed to parse archive");

    if (!file->isEmpty() && !file->hasSymbolTable())
      error(path + ": archive has no index; run ranlib to add one");

    if (config->allLoad) {
      if (Optional<MemoryBufferRef> buffer = readFile(path))
        for (MemoryBufferRef member : getArchiveMembers(*buffer))
          inputFiles.push_back(make<ObjFile>(member));
    } else if (config->forceLoadObjC) {
      for (const object::Archive::Symbol &sym : file->symbols())
        if (sym.getName().startswith(objc::klass))
          symtab->addUndefined(sym.getName());

      // TODO: no need to look for ObjC sections for a given archive member if
      // we already found that it contains an ObjC symbol. We should also
      // consider creating a LazyObjFile class in order to avoid double-loading
      // these files here and below (as part of the ArchiveFile).
      if (Optional<MemoryBufferRef> buffer = readFile(path))
        for (MemoryBufferRef member : getArchiveMembers(*buffer))
          if (hasObjCSection(member))
            inputFiles.push_back(make<ObjFile>(member));
    }

    newFile = make<ArchiveFile>(std::move(file));
    break;
  }
  case file_magic::macho_object:
    newFile = make<ObjFile>(mbref);
    break;
  case file_magic::macho_dynamically_linked_shared_lib:
  case file_magic::macho_dynamically_linked_shared_lib_stub:
    newFile = make<DylibFile>(mbref);
    break;
  case file_magic::tapi_file: {
    if (Optional<DylibFile *> dylibFile = makeDylibFromTAPI(mbref))
      newFile = *dylibFile;
    break;
  }
  case file_magic::bitcode:
    newFile = make<BitcodeFile>(mbref);
    break;
  default:
    error(path + ": unhandled file type");
  }
  if (newFile)
    inputFiles.push_back(newFile);
  return newFile;
}

static void addFileList(StringRef path) {
  Optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer)
    return;
  MemoryBufferRef mbref = *buffer;
  for (StringRef path : args::getLines(mbref))
    addFile(path);
}

static void forceLoadArchive(StringRef path) {
  if (Optional<MemoryBufferRef> buffer = readFile(path))
    for (MemoryBufferRef member : getArchiveMembers(*buffer))
      inputFiles.push_back(make<ObjFile>(member));
}

static std::array<StringRef, 6> archNames{"arm",    "arm64", "i386",
                                          "x86_64", "ppc",   "ppc64"};
static bool isArchString(StringRef s) {
  static DenseSet<StringRef> archNamesSet(archNames.begin(), archNames.end());
  return archNamesSet.find(s) != archNamesSet.end();
}

// An order file has one entry per line, in the following format:
//
//   <arch>:<object file>:<symbol name>
//
// <arch> and <object file> are optional. If not specified, then that entry
// matches any symbol of that name.
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
  for (StringRef rest : args::getLines(mbref)) {
    StringRef arch, objectFile, symbol;

    std::array<StringRef, 3> fields;
    uint8_t fieldCount = 0;
    while (rest != "" && fieldCount < 3) {
      std::pair<StringRef, StringRef> p = getToken(rest, ": \t\n\v\f\r");
      StringRef tok = p.first;
      rest = p.second;

      // Check if we have a comment
      if (tok == "" || tok[0] == '#')
        break;

      fields[fieldCount++] = tok;
    }

    switch (fieldCount) {
    case 3:
      arch = fields[0];
      objectFile = fields[1];
      symbol = fields[2];
      break;
    case 2:
      (isArchString(fields[0]) ? arch : objectFile) = fields[0];
      symbol = fields[1];
      break;
    case 1:
      symbol = fields[0];
      break;
    case 0:
      break;
    default:
      llvm_unreachable("too many fields in order file");
    }

    if (!arch.empty()) {
      if (!isArchString(arch)) {
        error("invalid arch \"" + arch + "\" in order file: expected one of " +
              llvm::join(archNames, ", "));
        continue;
      }

      // TODO: Update when we extend support for other archs
      if (arch != "x86_64")
        continue;
    }

    if (!objectFile.empty() && !objectFile.endswith(".o")) {
      error("invalid object file name \"" + objectFile +
            "\" in order file: should end with .o");
      continue;
    }

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
static bool markSubLibrary(StringRef searchName) {
  for (InputFile *file : inputFiles) {
    if (auto *dylibFile = dyn_cast<DylibFile>(file)) {
      StringRef filename = path::filename(dylibFile->getName());
      if (filename.consume_front(searchName) &&
          (filename == ".dylib" || filename == ".tbd")) {
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
  auto lto = make<BitcodeCompiler>();
  for (InputFile *file : inputFiles)
    if (auto *bitcodeFile = dyn_cast<BitcodeFile>(file))
      lto->add(*bitcodeFile);

  for (ObjFile *file : lto->compile())
    inputFiles.push_back(file);
}

// Replaces common symbols with defined symbols residing in __common sections.
// This function must be called after all symbol names are resolved (i.e. after
// all InputFiles have been loaded.) As a result, later operations won't see
// any CommonSymbols.
static void replaceCommonSymbols() {
  for (macho::Symbol *sym : symtab->getSymbols()) {
    auto *common = dyn_cast<CommonSymbol>(sym);
    if (common == nullptr)
      continue;

    auto *isec = make<InputSection>();
    isec->file = common->file;
    isec->name = section_names::common;
    isec->segname = segment_names::data;
    isec->align = common->align;
    // Casting to size_t will truncate large values on 32-bit architectures,
    // but it's not really worth supporting the linking of 64-bit programs on
    // 32-bit archs.
    isec->data = {nullptr, static_cast<size_t>(common->size)};
    isec->flags = S_ZEROFILL;
    inputSections.push_back(isec);

    replaceSymbol<Defined>(sym, sym->getName(), isec, /*value=*/0,
                           /*isWeakDef=*/false,
                           /*isExternal=*/true);
  }
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

static void handlePlatformVersion(const opt::Arg *arg) {
  StringRef platformStr = arg->getValue(0);
  StringRef minVersionStr = arg->getValue(1);
  StringRef sdkVersionStr = arg->getValue(2);

  // TODO(compnerd) see if we can generate this case list via XMACROS
  config->platform.kind =
      llvm::StringSwitch<llvm::MachO::PlatformKind>(lowerDash(platformStr))
          .Cases("macos", "1", llvm::MachO::PlatformKind::macOS)
          .Cases("ios", "2", llvm::MachO::PlatformKind::iOS)
          .Cases("tvos", "3", llvm::MachO::PlatformKind::tvOS)
          .Cases("watchos", "4", llvm::MachO::PlatformKind::watchOS)
          .Cases("bridgeos", "5", llvm::MachO::PlatformKind::bridgeOS)
          .Cases("mac-catalyst", "6", llvm::MachO::PlatformKind::macCatalyst)
          .Cases("ios-simulator", "7", llvm::MachO::PlatformKind::iOSSimulator)
          .Cases("tvos-simulator", "8",
                 llvm::MachO::PlatformKind::tvOSSimulator)
          .Cases("watchos-simulator", "9",
                 llvm::MachO::PlatformKind::watchOSSimulator)
          .Default(llvm::MachO::PlatformKind::unknown);
  if (config->platform.kind == llvm::MachO::PlatformKind::unknown)
    error(Twine("malformed platform: ") + platformStr);
  // TODO: check validity of version strings, which varies by platform
  // NOTE: ld64 accepts version strings with 5 components
  // llvm::VersionTuple accepts no more than 4 components
  // Has Apple ever published version strings with 5 components?
  if (config->platform.minimum.tryParse(minVersionStr))
    error(Twine("malformed minimum version: ") + minVersionStr);
  if (config->platform.sdk.tryParse(sdkVersionStr))
    error(Twine("malformed sdk version: ") + sdkVersionStr);
}

static void warnIfDeprecatedOption(const opt::Option &opt) {
  if (!opt.getGroup().isValid())
    return;
  if (opt.getGroup().getID() == OPT_grp_deprecated) {
    warn("Option `" + opt.getPrefixedName() + "' is deprecated in ld64:");
    warn(opt.getHelpText());
  }
}

static void warnIfUnimplementedOption(const opt::Option &opt) {
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

bool macho::link(llvm::ArrayRef<const char *> argsArr, bool canExitEarly,
                 raw_ostream &stdoutOS, raw_ostream &stderrOS) {
  lld::stdoutOS = &stdoutOS;
  lld::stderrOS = &stderrOS;

  stderrOS.enable_colors(stderrOS.has_colors());
  // TODO: Set up error handler properly, e.g. the errorLimitExceededMsg

  errorHandler().cleanupCallback = []() { freeArena(); };

  MachOOptTable parser;
  opt::InputArgList args = parser.parse(argsArr.slice(1));

  if (args.hasArg(OPT_help_hidden)) {
    parser.printHelp(argsArr[0], /*showHidden=*/true);
    return true;
  } else if (args.hasArg(OPT_help)) {
    parser.printHelp(argsArr[0], /*showHidden=*/false);
    return true;
  }

  config = make<Configuration>();
  symtab = make<SymbolTable>();
  target = createTargetInfo(args);

  config->entry = symtab->addUndefined(args.getLastArgValue(OPT_e, "_main"));
  config->outputFile = args.getLastArgValue(OPT_o, "a.out");
  config->installName =
      args.getLastArgValue(OPT_install_name, config->outputFile);
  config->headerPad = args::getHex(args, OPT_headerpad, /*Default=*/32);
  config->headerPadMaxInstallNames =
      args.hasArg(OPT_headerpad_max_install_names);
  config->outputType = getOutputType(args);
  config->runtimePaths = args::getStrings(args, OPT_rpath);
  config->allLoad = args.hasArg(OPT_all_load);
  config->forceLoadObjC = args.hasArg(OPT_ObjC);

  if (const opt::Arg *arg = args.getLastArg(OPT_static, OPT_dynamic))
    config->staticLink = (arg->getOption().getID() == OPT_static);

  config->systemLibraryRoots = getSystemLibraryRoots(args);
  config->librarySearchPaths =
      getLibrarySearchPaths(args, config->systemLibraryRoots);
  config->frameworkSearchPaths =
      getFrameworkSearchPaths(args, config->systemLibraryRoots);
  if (const opt::Arg *arg =
          args.getLastArg(OPT_search_paths_first, OPT_search_dylibs_first))
    config->searchDylibsFirst =
        (arg && arg->getOption().getID() == OPT_search_dylibs_first);

  config->saveTemps = args.hasArg(OPT_save_temps);

  if (args.hasArg(OPT_v)) {
    message(getLLDVersion());
    message(StringRef("Library search paths:") +
            (config->librarySearchPaths.size()
                 ? "\n\t" + llvm::join(config->librarySearchPaths, "\n\t")
                 : ""));
    message(StringRef("Framework search paths:") +
            (config->frameworkSearchPaths.size()
                 ? "\n\t" + llvm::join(config->frameworkSearchPaths, "\n\t")
                 : ""));
    freeArena();
    return !errorCount();
  }

  for (const auto &arg : args) {
    const auto &opt = arg->getOption();
    warnIfDeprecatedOption(opt);
    warnIfUnimplementedOption(opt);
    // TODO: are any of these better handled via filtered() or getLastArg()?
    switch (opt.getID()) {
    case OPT_INPUT:
      addFile(arg->getValue());
      break;
    case OPT_weak_library: {
      auto *dylibFile = dyn_cast_or_null<DylibFile>(addFile(arg->getValue()));
      if (dylibFile)
        dylibFile->forceWeakImport = true;
      break;
    }
    case OPT_filelist:
      addFileList(arg->getValue());
      break;
    case OPT_force_load:
      forceLoadArchive(arg->getValue());
      break;
    case OPT_l:
    case OPT_weak_l: {
      StringRef name = arg->getValue();
      if (Optional<std::string> path = findLibrary(name)) {
        auto *dylibFile = dyn_cast_or_null<DylibFile>(addFile(*path));
        if (opt.getID() == OPT_weak_l && dylibFile)
          dylibFile->forceWeakImport = true;
        break;
      }
      error("library not found for -l" + name);
      break;
    }
    case OPT_framework:
    case OPT_weak_framework: {
      StringRef name = arg->getValue();
      if (Optional<std::string> path = findFramework(name)) {
        auto *dylibFile = dyn_cast_or_null<DylibFile>(addFile(*path));
        if (opt.getID() == OPT_weak_framework && dylibFile)
          dylibFile->forceWeakImport = true;
        break;
      }
      error("framework not found for -framework " + name);
      break;
    }
    case OPT_platform_version:
      handlePlatformVersion(arg);
      break;
    default:
      break;
    }
  }

  config->isPic = config->outputType == MH_DYLIB ||
                  config->outputType == MH_BUNDLE ||
                  (config->outputType == MH_EXECUTE && args.hasArg(OPT_pie));

  // Now that all dylibs have been loaded, search for those that should be
  // re-exported.
  for (opt::Arg *arg : args.filtered(OPT_sub_library)) {
    config->hasReexports = true;
    StringRef searchName = arg->getValue();
    if (!markSubLibrary(searchName))
      error("-sub_library " + searchName + " does not match a supplied dylib");
  }

  initLLVM();
  compileBitcodeFiles();
  replaceCommonSymbols();

  StringRef orderFile = args.getLastArgValue(OPT_order_file);
  if (!orderFile.empty())
    parseOrderFile(orderFile);

  if (config->outputType == MH_EXECUTE && isa<Undefined>(config->entry)) {
    error("undefined symbol: " + config->entry->getName());
    return false;
  }

  createSyntheticSections();
  symtab->addDSOHandle(in.header);

  for (opt::Arg *arg : args.filtered(OPT_sectcreate)) {
    StringRef segName = arg->getValue(0);
    StringRef sectName = arg->getValue(1);
    StringRef fileName = arg->getValue(2);
    Optional<MemoryBufferRef> buffer = readFile(fileName);
    if (buffer)
      inputFiles.push_back(make<OpaqueFile>(*buffer, segName, sectName));
  }

  // Initialize InputSections.
  for (InputFile *file : inputFiles) {
    for (SubsectionMap &map : file->subsections) {
      for (auto &p : map) {
        InputSection *isec = p.second;
        inputSections.push_back(isec);
      }
    }
  }

  // Write to an output file.
  writeResult();

  if (canExitEarly)
    exitLld(errorCount() ? 1 : 0);

  return !errorCount();
}
