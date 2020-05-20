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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Archive.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::sys;
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

// This is for -lfoo. We'll look for libfoo.dylib from search paths.
static Optional<std::string> findDylib(StringRef name) {
  for (StringRef dir : config->searchPaths) {
    std::string path = (dir + "/lib" + name + ".dylib").str();
    if (fs::exists(path))
      return path;
  }
  error("library not found for -l" + name);
  return None;
}

static TargetInfo *createTargetInfo(opt::InputArgList &args) {
  StringRef s = args.getLastArgValue(OPT_arch, "x86_64");
  if (s != "x86_64")
    error("missing or unsupported -arch " + s);
  return createX86_64TargetInfo();
}

static std::vector<StringRef> getSearchPaths(opt::InputArgList &args) {
  std::vector<StringRef> ret{args::getStrings(args, OPT_L)};
  if (!args.hasArg(OPT_Z)) {
    ret.push_back("/usr/lib");
    ret.push_back("/usr/local/lib");
  }
  return ret;
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
  default:
    error(path + ": unhandled file type");
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

static void handlePlatformVersion(opt::ArgList::iterator &it,
                                  const opt::ArgList::iterator &end) {
  // -platform_version takes 3 args, which LLVM's option library doesn't
  // support directly.  So this explicitly handles that.
  // FIXME: stash skipped args for later use.
  for (int i = 0; i < 3; ++i) {
    ++it;
    if (it == end || (*it)->getOption().getID() != OPT_INPUT)
      fatal("usage: -platform_version platform min_version sdk_version");
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

  config = make<Configuration>();
  symtab = make<SymbolTable>();
  target = createTargetInfo(args);

  config->entry = symtab->addUndefined(args.getLastArgValue(OPT_e, "_main"));
  config->outputFile = args.getLastArgValue(OPT_o, "a.out");
  config->installName =
      args.getLastArgValue(OPT_install_name, config->outputFile);
  config->searchPaths = getSearchPaths(args);
  config->outputType = args.hasArg(OPT_dylib) ? MH_DYLIB : MH_EXECUTE;

  if (args.hasArg(OPT_v)) {
    message(getLLDVersion());
    std::vector<StringRef> &searchPaths = config->searchPaths;
    message("Library search paths:\n" +
            llvm::join(searchPaths.begin(), searchPaths.end(), "\n"));
    freeArena();
    return !errorCount();
  }

  for (opt::ArgList::iterator it = args.begin(), end = args.end(); it != end;
       ++it) {
    const opt::Arg *arg = *it;
    switch (arg->getOption().getID()) {
    case OPT_INPUT:
      addFile(arg->getValue());
      break;
    case OPT_l:
      if (Optional<std::string> path = findDylib(arg->getValue()))
        addFile(*path);
      break;
    case OPT_platform_version: {
      handlePlatformVersion(it, end); // Can advance "it".
      break;
    }
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

  // dyld requires us to load libSystem. Since we may run tests on non-OSX
  // systems which do not have libSystem, we mock it out here.
  // TODO: Replace this with a stub tbd file once we have TAPI support.
  if (StringRef(getenv("LLD_IN_TEST")) == "1" &&
      config->outputType == MH_EXECUTE) {
    inputFiles.push_back(DylibFile::createLibSystemMock());
  }

  if (config->outputType == MH_EXECUTE && !isa<Defined>(config->entry)) {
    error("undefined symbol: " + config->entry->getName());
    return false;
  }

  createSyntheticSections();

  // Initialize InputSections.
  for (InputFile *file : inputFiles)
    for (InputSection *sec : file->sections)
      inputSections.push_back(sec);

  // Write to an output file.
  writeResult();

  if (canExitEarly)
    exitLld(errorCount() ? 1 : 0);

  freeArena();
  return !errorCount();
}
