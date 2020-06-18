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
#include "OutputSection.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
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
#include "llvm/Object/Archive.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::sys;
using namespace llvm::opt;
using namespace lld;
using namespace lld::macho;

Configuration *lld::macho::config;

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const opt::OptTable::Info optInfo[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X7, X8, X9, X10, X11, X12)      \
  {X1, X2, X10,         X11,         OPT_##ID, opt::Option::KIND##Class,       \
   X9, X8, OPT_##GROUP, OPT_##ALIAS, X7,       X12},
#include "Options.inc"
#undef OPTION
};

MachOOptTable::MachOOptTable() : OptTable(optInfo) {}

opt::InputArgList MachOOptTable::parse(ArrayRef<const char *> argv) {
  // Make InputArgList from string vectors.
  unsigned missingIndex;
  unsigned missingCount;
  SmallVector<const char *, 256> vec(argv.data(), argv.data() + argv.size());

  opt::InputArgList args = ParseArgs(vec, missingIndex, missingCount);

  if (missingCount)
    error(Twine(args.getArgString(missingIndex)) + ": missing argument");

  for (opt::Arg *arg : args.filtered(OPT_UNKNOWN))
    error("unknown argument: " + arg->getSpelling());
  return args;
}

void MachOOptTable::printHelp(const char *argv0, bool showHidden) const {
  PrintHelp(lld::outs(), (std::string(argv0) + " [options] file...").c_str(),
            "LLVM Linker", showHidden);
  lld::outs() << "\n";
}

static Optional<std::string> findLibrary(StringRef name) {
  std::string stub = (llvm::Twine("lib") + name + ".tbd").str();
  std::string shared = (llvm::Twine("lib") + name + ".dylib").str();
  std::string archive = (llvm::Twine("lib") + name + ".a").str();
  llvm::SmallString<260> location;

  for (StringRef dir : config->librarySearchPaths) {
    for (StringRef library : {stub, shared, archive}) {
      location = dir;
      llvm::sys::path::append(location, library);
      if (fs::exists(location))
        return location.str().str();
    }
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

static bool isDirectory(StringRef option, StringRef path) {
  if (!fs::exists(path)) {
    warn("directory not found for option -" + option + path);
    return false;
  } else if (!fs::is_directory(path)) {
    warn("option -" + option + path + " references a non-directory path");
    return false;
  }
  return true;
}

static void getSearchPaths(std::vector<StringRef> &paths, unsigned optionCode,
                           opt::InputArgList &args,
                           const SmallVector<StringRef, 2> &systemPaths) {
  StringRef optionLetter{(optionCode == OPT_F ? "F" : "L")};
  for (auto const &path : args::getStrings(args, optionCode)) {
    if (isDirectory(optionLetter, path))
      paths.push_back(path);
  }
  if (!args.hasArg(OPT_Z) && Triple(sys::getProcessTriple()).isOSDarwin()) {
    for (auto const &path : systemPaths) {
      if (isDirectory(optionLetter, path))
        paths.push_back(path);
    }
  }
}

static void getLibrarySearchPaths(std::vector<StringRef> &paths,
                                  opt::InputArgList &args) {
  getSearchPaths(paths, OPT_L, args, {"/usr/lib", "/usr/local/lib"});
}

static void getFrameworkSearchPaths(std::vector<StringRef> &paths,
                                    opt::InputArgList &args) {
  getSearchPaths(paths, OPT_F, args,
                 {"/Library/Frameworks", "/System/Library/Frameworks"});
}

static void addFile(StringRef path) {
  Optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer)
    return;
  MemoryBufferRef mbref = *buffer;

  switch (identify_magic(mbref.getBuffer())) {
  case file_magic::archive: {
    std::unique_ptr<object::Archive> file = CHECK(
        object::Archive::create(mbref), path + ": failed to parse archive");

    if (!file->isEmpty() && !file->hasSymbolTable())
      error(path + ": archive has no index; run ranlib to add one");

    inputFiles.push_back(make<ArchiveFile>(std::move(file)));
    break;
  }
  case file_magic::macho_object:
    inputFiles.push_back(make<ObjFile>(mbref));
    break;
  case file_magic::macho_dynamically_linked_shared_lib:
    inputFiles.push_back(make<DylibFile>(mbref));
    break;
  case file_magic::tapi_file: {
    llvm::Expected<std::unique_ptr<llvm::MachO::InterfaceFile>> result =
        TextAPIReader::get(mbref);
    if (!result)
      return;

    std::unique_ptr<llvm::MachO::InterfaceFile> interface{std::move(*result)};
    inputFiles.push_back(make<DylibFile>(std::move(interface)));
    break;
  }
  default:
    error(path + ": unhandled file type");
  }
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
void parseOrderFile(StringRef path) {
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
// with a path of .*/libfoo.dylib.
static bool markSubLibrary(StringRef searchName) {
  for (InputFile *file : inputFiles) {
    if (auto *dylibFile = dyn_cast<DylibFile>(file)) {
      StringRef filename = path::filename(dylibFile->getName());
      if (filename.consume_front(searchName) && filename == ".dylib") {
        dylibFile->reexport = true;
        return true;
      }
    }
  }
  return false;
}

static void handlePlatformVersion(const opt::Arg *arg) {
  // TODO: implementation coming very soon ...
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
  if (!opt.getGroup().isValid())
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
  getLibrarySearchPaths(config->librarySearchPaths, args);
  getFrameworkSearchPaths(config->frameworkSearchPaths, args);
  config->outputType = args.hasArg(OPT_dylib) ? MH_DYLIB : MH_EXECUTE;

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
    switch (arg->getOption().getID()) {
    case OPT_INPUT:
      addFile(arg->getValue());
      break;
    case OPT_l: {
      StringRef name = arg->getValue();
      if (Optional<std::string> path = findLibrary(name)) {
        addFile(*path);
        break;
      }
      error("library not found for -l" + name);
      break;
    }
    case OPT_platform_version:
      handlePlatformVersion(arg);
      break;
    case OPT_o:
    case OPT_dylib:
    case OPT_e:
    case OPT_L:
    case OPT_Z:
    case OPT_arch:
      // handled elsewhere
      break;
    default:
      warnIfUnimplementedOption(opt);
      break;
    }
  }

  // Now that all dylibs have been loaded, search for those that should be
  // re-exported.
  for (opt::Arg *arg : args.filtered(OPT_sub_library)) {
    config->hasReexports = true;
    StringRef searchName = arg->getValue();
    if (!markSubLibrary(searchName))
      error("-sub_library " + searchName + " does not match a supplied dylib");
  }

  StringRef orderFile = args.getLastArgValue(OPT_order_file);
  if (!orderFile.empty())
    parseOrderFile(orderFile);

  if (config->outputType == MH_EXECUTE && !isa<Defined>(config->entry)) {
    error("undefined symbol: " + config->entry->getName());
    return false;
  }

  createSyntheticSections();

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

  freeArena();
  return !errorCount();
}
