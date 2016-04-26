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
#include "lld/Config/Version.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::sys;

using namespace lld;
using namespace lld::elf;

// Create OptTable

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const opt::OptTable::Info OptInfo[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X6, X7, X8, X9, X10)            \
  {                                                                            \
    X1, X2, X9, X10, OPT_##ID, opt::Option::KIND##Class, X8, X7, OPT_##GROUP,  \
        OPT_##ALIAS, X6                                                        \
  },
#include "Options.inc"
#undef OPTION
};

ELFOptTable::ELFOptTable() : OptTable(OptInfo) {}

// Parses a given list of options.
opt::InputArgList ELFOptTable::parse(ArrayRef<const char *> Argv) {
  // Make InputArgList from string vectors.
  unsigned MissingIndex;
  unsigned MissingCount;

  // Expand response files. '@<filename>' is replaced by the file's contents.
  SmallVector<const char *, 256> Vec(Argv.data(), Argv.data() + Argv.size());
  StringSaver Saver(Alloc);
  llvm::cl::ExpandResponseFiles(Saver, llvm::cl::TokenizeGNUCommandLine, Vec);

  // Parse options and then do error checking.
  opt::InputArgList Args = this->ParseArgs(Vec, MissingIndex, MissingCount);
  if (MissingCount)
    error(Twine("missing arg value for \"") + Args.getArgString(MissingIndex) +
          "\", expected " + Twine(MissingCount) +
          (MissingCount == 1 ? " argument.\n" : " arguments"));

  iterator_range<opt::arg_iterator> Unknowns = Args.filtered(OPT_UNKNOWN);
  for (auto *Arg : Unknowns)
    warning("warning: unknown argument: " + Arg->getSpelling());
  if (Unknowns.begin() != Unknowns.end())
    error("unknown argument(s) found");
  return Args;
}

void elf::printHelp(const char *Argv0) {
  ELFOptTable Table;
  Table.PrintHelp(outs(), Argv0, "lld", false);
}

void elf::printVersion() {
  outs() << "LLD " << getLLDVersion();
  std::string S = getLLDRepositoryVersion();
  if (!S.empty())
    outs() << " " << S;
  outs() << "\n";
}

// Concatenates S and T so that the resulting path becomes S/T.
// There are a few exceptions:
//
//  1. The result will never escape from S. Therefore, all ".."
//     are removed from T before concatenatig them.
//  2. Windows drive letters are removed from T before concatenation.
std::string elf::concat_paths(StringRef S, StringRef T) {
  // Remove leading '/' or a drive letter, and then remove "..".
  SmallString<128> T2(path::relative_path(T));
  path::remove_dots(T2, /*remove_dot_dot=*/true);

  SmallString<128> Res;
  path::append(Res, S, T2);
  return Res.str();
}

void elf::copyFile(StringRef Src, StringRef Dest) {
  SmallString<128> Dir(Dest);
  path::remove_filename(Dir);
  if (std::error_code EC = sys::fs::create_directories(Dir)) {
    error(EC, Dir + ": can't create directory");
    return;
  }
  if (std::error_code EC = sys::fs::copy_file(Src, Dest))
    error(EC, "failed to copy file: " + Dest);
}

std::string elf::findFromSearchPaths(StringRef Path) {
  for (StringRef Dir : Config->SearchPaths) {
    std::string FullPath = buildSysrootedPath(Dir, Path);
    if (sys::fs::exists(FullPath))
      return FullPath;
  }
  return "";
}

// Searches a given library from input search paths, which are filled
// from -L command line switches. Returns a path to an existent library file.
std::string elf::searchLibrary(StringRef Path) {
  if (Path.startswith(":"))
    return findFromSearchPaths(Path.substr(1));
  if (!Config->Static) {
    std::string S = findFromSearchPaths(("lib" + Path + ".so").str());
    if (!S.empty())
      return S;
  }
  return findFromSearchPaths(("lib" + Path + ".a").str());
}

// Makes a path by concatenating Dir and File.
// If Dir starts with '=' the result will be preceded by Sysroot,
// which can be set with --sysroot command line switch.
std::string elf::buildSysrootedPath(StringRef Dir, StringRef File) {
  SmallString<128> Path;
  if (Dir.startswith("="))
    sys::path::append(Path, Config->Sysroot, Dir.substr(1), File);
  else
    sys::path::append(Path, Dir, File);
  return Path.str();
}
