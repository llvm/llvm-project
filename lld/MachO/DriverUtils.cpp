//===- DriverUtils.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Driver.h"
#include "InputFiles.h"
#include "ObjC.h"
#include "Target.h"

#include "lld/Common/Args.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Reproduce.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/TextAPI/InterfaceFile.h"
#include "llvm/TextAPI/TextAPIReader.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::opt;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const OptTable::Info optInfo[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X7, X8, X9, X10, X11, X12)      \
  {X1, X2, X10,         X11,         OPT_##ID, Option::KIND##Class,            \
   X9, X8, OPT_##GROUP, OPT_##ALIAS, X7,       X12},
#include "Options.inc"
#undef OPTION
};

MachOOptTable::MachOOptTable() : OptTable(optInfo) {}

// Set color diagnostics according to --color-diagnostics={auto,always,never}
// or --no-color-diagnostics flags.
static void handleColorDiagnostics(InputArgList &args) {
  const Arg *arg =
      args.getLastArg(OPT_color_diagnostics, OPT_color_diagnostics_eq,
                      OPT_no_color_diagnostics);
  if (!arg)
    return;
  if (arg->getOption().getID() == OPT_color_diagnostics) {
    lld::errs().enable_colors(true);
  } else if (arg->getOption().getID() == OPT_no_color_diagnostics) {
    lld::errs().enable_colors(false);
  } else {
    StringRef s = arg->getValue();
    if (s == "always")
      lld::errs().enable_colors(true);
    else if (s == "never")
      lld::errs().enable_colors(false);
    else if (s != "auto")
      error("unknown option: --color-diagnostics=" + s);
  }
}

InputArgList MachOOptTable::parse(ArrayRef<const char *> argv) {
  // Make InputArgList from string vectors.
  unsigned missingIndex;
  unsigned missingCount;
  SmallVector<const char *, 256> vec(argv.data(), argv.data() + argv.size());

  // Expand response files (arguments in the form of @<filename>)
  // and then parse the argument again.
  cl::ExpandResponseFiles(saver, cl::TokenizeGNUCommandLine, vec);
  InputArgList args = ParseArgs(vec, missingIndex, missingCount);

  // Handle -fatal_warnings early since it converts missing argument warnings
  // to errors.
  errorHandler().fatalWarnings = args.hasArg(OPT_fatal_warnings);

  if (missingCount)
    error(Twine(args.getArgString(missingIndex)) + ": missing argument");

  handleColorDiagnostics(args);

  for (const Arg *arg : args.filtered(OPT_UNKNOWN)) {
    std::string nearest;
    if (findNearest(arg->getAsString(args), nearest) > 1)
      error("unknown argument '" + arg->getAsString(args) + "'");
    else
      error("unknown argument '" + arg->getAsString(args) +
            "', did you mean '" + nearest + "'");
  }
  return args;
}

void MachOOptTable::printHelp(const char *argv0, bool showHidden) const {
  PrintHelp(lld::outs(), (std::string(argv0) + " [options] file...").c_str(),
            "LLVM Linker", showHidden);
  lld::outs() << "\n";
}

static std::string rewritePath(StringRef s) {
  if (fs::exists(s))
    return relativeToRoot(s);
  return std::string(s);
}

// Reconstructs command line arguments so that so that you can re-run
// the same command with the same inputs. This is for --reproduce.
std::string macho::createResponseFile(const InputArgList &args) {
  SmallString<0> data;
  raw_svector_ostream os(data);

  // Copy the command line to the output while rewriting paths.
  for (const Arg *arg : args) {
    switch (arg->getOption().getID()) {
    case OPT_reproduce:
      break;
    case OPT_INPUT:
      os << quote(rewritePath(arg->getValue())) << "\n";
      break;
    case OPT_o:
      os << "-o " << quote(path::filename(arg->getValue())) << "\n";
      break;
    case OPT_filelist:
      if (Optional<MemoryBufferRef> buffer = readFile(arg->getValue()))
        for (StringRef path : args::getLines(*buffer))
          os << quote(rewritePath(path)) << "\n";
      break;
    case OPT_F:
    case OPT_L:
    case OPT_bundle_loader:
    case OPT_exported_symbols_list:
    case OPT_force_load:
    case OPT_order_file:
    case OPT_rpath:
    case OPT_syslibroot:
    case OPT_unexported_symbols_list:
      os << arg->getSpelling() << " " << quote(rewritePath(arg->getValue()))
         << "\n";
      break;
    case OPT_sectcreate:
      os << arg->getSpelling() << " " << quote(arg->getValue(0)) << " "
         << quote(arg->getValue(1)) << " "
         << quote(rewritePath(arg->getValue(2))) << "\n";
      break;
    default:
      os << toString(*arg) << "\n";
    }
  }
  return std::string(data.str());
}

Optional<std::string> macho::resolveDylibPath(StringRef path) {
  // TODO: if a tbd and dylib are both present, we should check to make sure
  // they are consistent.
  if (fs::exists(path))
    return std::string(path);
  else
    depTracker->logFileNotFound(path);

  SmallString<261> location = path;
  path::replace_extension(location, ".tbd");
  if (fs::exists(location))
    return std::string(location);
  else
    depTracker->logFileNotFound(location);
  return {};
}

// It's not uncommon to have multiple attempts to load a single dylib,
// especially if it's a commonly re-exported core library.
static DenseMap<CachedHashStringRef, DylibFile *> loadedDylibs;

Optional<DylibFile *> macho::loadDylib(MemoryBufferRef mbref,
                                       DylibFile *umbrella,
                                       bool isBundleLoader) {
  CachedHashStringRef path(mbref.getBufferIdentifier());
  DylibFile *file = loadedDylibs[path];
  if (file)
    return file;

  file_magic magic = identify_magic(mbref.getBuffer());
  if (magic == file_magic::tapi_file) {
    Expected<std::unique_ptr<InterfaceFile>> result = TextAPIReader::get(mbref);
    if (!result) {
      error("could not load TAPI file at " + mbref.getBufferIdentifier() +
            ": " + toString(result.takeError()));
      return {};
    }
    file = make<DylibFile>(**result, umbrella, isBundleLoader);
  } else {
    assert(magic == file_magic::macho_dynamically_linked_shared_lib ||
           magic == file_magic::macho_dynamically_linked_shared_lib_stub ||
           magic == file_magic::macho_executable ||
           magic == file_magic::macho_bundle);
    file = make<DylibFile>(mbref, umbrella, isBundleLoader);
  }
  // Note that DylibFile's ctor may recursively invoke loadDylib(), which can
  // cause loadedDylibs to get resized and its iterators invalidated. As such,
  // we redo the key lookup here instead of caching an iterator from our earlier
  // lookup at the start of the function.
  loadedDylibs[path] = file;
  return file;
}

Optional<InputFile *> macho::loadArchiveMember(MemoryBufferRef mb,
                                               uint32_t modTime,
                                               StringRef archiveName,
                                               bool objCOnly) {
  switch (identify_magic(mb.getBuffer())) {
  case file_magic::macho_object:
    if (!objCOnly || hasObjCSection(mb))
      return make<ObjFile>(mb, modTime, archiveName);
    return None;
  case file_magic::bitcode:
    if (!objCOnly || check(isBitcodeContainingObjCCategory(mb)))
      return make<BitcodeFile>(mb);
    return None;
  default:
    error(archiveName + ": archive member " + mb.getBufferIdentifier() +
          " has unhandled file type");
    return None;
  }
}

uint32_t macho::getModTime(StringRef path) {
  fs::file_status stat;
  if (!fs::status(path, stat))
    if (fs::exists(stat))
      return toTimeT(stat.getLastModificationTime());

  warn("failed to get modification time of " + path);
  return 0;
}

void macho::printArchiveMemberLoad(StringRef reason, const InputFile *f) {
  if (config->printEachFile)
    message(toString(f));
  if (config->printWhyLoad)
    message(reason + " forced load of " + toString(f));
}

macho::DependencyTracker::DependencyTracker(StringRef path)
    : path(path), active(!path.empty()) {
  if (active && fs::exists(path) && !fs::can_write(path)) {
    warn("Ignoring dependency_info option since specified path is not "
         "writeable.");
    active = false;
  }
}

void macho::DependencyTracker::write(llvm::StringRef version,
                                     const llvm::SetVector<InputFile *> &inputs,
                                     llvm::StringRef output) {
  if (!active)
    return;

  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    warn("Error writing dependency info to file");
    return;
  }

  auto addDep = [&os](DepOpCode opcode, const StringRef &path) {
    // XXX: Even though DepOpCode's underlying type is uint8_t,
    // this cast is still needed because Clang older than 10.x has a bug,
    // where it doesn't know to cast the enum to its underlying type.
    // Hence `<< DepOpCode` is ambiguous to it.
    os << static_cast<uint8_t>(opcode);
    os << path;
    os << '\0';
  };

  addDep(DepOpCode::Version, version);

  // Sort the input by its names.
  std::vector<StringRef> inputNames;
  inputNames.reserve(inputs.size());
  for (InputFile *f : inputs)
    inputNames.push_back(f->getName());
  llvm::sort(inputNames);

  for (const StringRef &in : inputNames)
    addDep(DepOpCode::Input, in);

  for (const std::string &f : notFounds)
    addDep(DepOpCode::NotFound, f);

  addDep(DepOpCode::Output, output);
}
