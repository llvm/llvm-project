//===- DriverUtils.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for the driver. Because there
// are so many small functions, we created this separate file to make
// Driver.cpp less cluttered.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Error.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;

using namespace lld;
using namespace lld::elf2;

// Create OptTable

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const opt::OptTable::Info infoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X6, X7, X8, X9, X10)            \
  {                                                                            \
    X1, X2, X9, X10, OPT_##ID, opt::Option::KIND##Class, X8, X7, OPT_##GROUP,  \
        OPT_##ALIAS, X6                                                        \
  }                                                                            \
  ,
#include "Options.inc"
#undef OPTION
};

class ELFOptTable : public opt::OptTable {
public:
  ELFOptTable() : OptTable(infoTable) {}
};

// Parses a given list of options.
opt::InputArgList elf2::parseArgs(llvm::BumpPtrAllocator *A,
                                  ArrayRef<const char *> Argv) {
  // Make InputArgList from string vectors.
  ELFOptTable Table;
  unsigned MissingIndex;
  unsigned MissingCount;

  // Expand response files. '@<filename>' is replaced by the file's contents.
  SmallVector<const char *, 256> Vec(Argv.data(), Argv.data() + Argv.size());
  StringSaver Saver(*A);
  llvm::cl::ExpandResponseFiles(Saver, llvm::cl::TokenizeGNUCommandLine, Vec);

  // Parse options and then do error checking.
  opt::InputArgList Args = Table.ParseArgs(Vec, MissingIndex, MissingCount);
  if (MissingCount)
    fatal(Twine("missing arg value for \"") + Args.getArgString(MissingIndex) +
          "\", expected " + Twine(MissingCount) +
          (MissingCount == 1 ? " argument.\n" : " arguments"));

  iterator_range<opt::arg_iterator> Unknowns = Args.filtered(OPT_UNKNOWN);
  for (auto *Arg : Unknowns)
    warning("warning: unknown argument: " + Arg->getSpelling());
  if (Unknowns.begin() != Unknowns.end())
    fatal("unknown argument(s) found");

  return Args;
}

std::string elf2::findFromSearchPaths(StringRef Path) {
  for (StringRef Dir : Config->SearchPaths) {
    std::string FullPath = buildSysrootedPath(Dir, Path);
    if (sys::fs::exists(FullPath))
      return FullPath;
  }
  return "";
}

// Searches a given library from input search paths, which are filled
// from -L command line switches. Returns a path to an existent library file.
std::string elf2::searchLibrary(StringRef Path) {
  std::vector<std::string> Names;
  if (Path[0] == ':') {
    Names.push_back(Path.drop_front());
  } else {
    if (!Config->Static)
      Names.push_back(("lib" + Path + ".so").str());
    Names.push_back(("lib" + Path + ".a").str());
  }
  for (const std::string &Name : Names) {
    std::string S = findFromSearchPaths(Name);
    if (!S.empty())
      return S;
  }
  fatal("Unable to find library -l" + Path);
}

// Makes a path by concatenating Dir and File.
// If Dir starts with '=' the result will be preceded by Sysroot,
// which can be set with --sysroot command line switch.
std::string elf2::buildSysrootedPath(StringRef Dir, StringRef File) {
  SmallString<128> Path;
  if (Dir.startswith("="))
    sys::path::append(Path, Config->Sysroot, Dir.substr(1), File);
  else
    sys::path::append(Path, Dir, File);
  return Path.str();
}
