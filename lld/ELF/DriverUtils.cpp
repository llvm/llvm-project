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
  cl::ExpandResponseFiles(Saver, cl::TokenizeGNUCommandLine, Vec);

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

// Makes a given pathname an absolute path first, and then remove
// beginning /. For example, "../foo.o" is converted to "home/john/foo.o",
// assuming that the current directory is "/home/john/bar".
static std::string relativeToRoot(StringRef Path) {
  SmallString<128> Abs = Path;
  if (std::error_code EC = fs::make_absolute(Abs))
    fatal("make_absolute failed: " + EC.message());
  path::remove_dots(Abs, /*remove_dot_dot=*/true);

  // This is Windows specific. root_name() returns a drive letter
  // (e.g. "c:") or a UNC name (//net). We want to keep it as part
  // of the result.
  SmallString<128> Res;
  StringRef Root = path::root_name(Abs);
  if (Root.endswith(":"))
    Res = Root.drop_back();
  else if (Root.startswith("//"))
    Res = Root.substr(2);

  path::append(Res, path::relative_path(Abs));
  return Res.str();
}

static std::string getDestPath(StringRef Path) {
  std::string Relpath = relativeToRoot(Path);
  SmallString<128> Dest;
  path::append(Dest, path::filename(Config->Reproduce), Relpath);
  return Dest.str();
}

static void maybePrintCpioMemberAux(raw_fd_ostream &OS, StringRef Path,
                                    StringRef Data) {
  OS << "070707"; // c_magic

  // The c_dev/c_ino pair should be unique according to the spec, but no one
  // seems to care.
  OS << "000000"; // c_dev
  OS << "000000"; // c_ino

  OS << "100664";                        // c_mode: C_ISREG | rw-rw-r--
  OS << "000000";                        // c_uid
  OS << "000000";                        // c_gid
  OS << "000001";                        // c_nlink
  OS << "000000";                        // c_rdev
  OS << "00000000000";                   // c_mtime
  OS << format("%06o", Path.size() + 1); // c_namesize
  OS << format("%011o", Data.size());    // c_filesize
  OS << Path << '\0';                    // c_name
  OS << Data;                            // c_filedata
}

static void maybePrintCpioMember(StringRef Path, StringRef Data) {
  if (Config->Reproduce.empty())
    return;

  if (!Driver->IncludedFiles.insert(Path).second)
    return;
  raw_fd_ostream &OS = *Driver->ReproduceArchive;
  maybePrintCpioMemberAux(OS, Path, Data);

  // Print the trailer and seek back. This way we have a valid archive if we
  // crash.
  uint64_t Pos = OS.tell();
  maybePrintCpioMemberAux(OS, "TRAILER!!!", "");
  OS.seek(Pos);
}

// Write file Src with content Data to the archive.
void elf::maybeCopyInputFile(StringRef Src, StringRef Data) {
  std::string Dest = getDestPath(Src);
  maybePrintCpioMember(Dest, Data);
}

// Quote a given string if it contains a space character.
static std::string quote(StringRef S) {
  if (S.find(' ') == StringRef::npos)
    return S;
  return ("\"" + S + "\"").str();
}

static std::string rewritePath(StringRef S) {
  if (fs::exists(S))
    return relativeToRoot(S);
  return S;
}

// Copies all input files to Config->Reproduce directory and
// create a response file as "response.txt", so that you can re-run
// the same command with the same inputs just by executing
// "ld.lld @response.txt". Used by --reproduce. This feature is
// supposed to be used by users to report an issue to LLD developers.
void elf::createResponseFile(const opt::InputArgList &Args) {
  SmallString<0> Data;
  raw_svector_ostream OS(Data);
  // Copy the command line to response.txt while rewriting paths.
  for (auto *Arg : Args) {
    switch (Arg->getOption().getID()) {
    case OPT_reproduce:
      break;
    case OPT_INPUT:
      OS << quote(rewritePath(Arg->getValue())) << "\n";
      break;
    case OPT_L:
    case OPT_dynamic_list:
    case OPT_rpath:
    case OPT_script:
    case OPT_version_script:
      OS << Arg->getSpelling() << " "
         << quote(rewritePath(Arg->getValue())) << "\n";
      break;
    default:
      OS << Arg->getSpelling();
      if (Arg->getNumValues() > 0)
        OS << " " << quote(Arg->getValue());
      OS << "\n";
    }
  }

  SmallString<128> Dest;
  path::append(Dest, path::filename(Config->Reproduce), "response.txt");
  maybePrintCpioMember(Dest, Data);
}

std::string elf::findFromSearchPaths(StringRef Path) {
  for (StringRef Dir : Config->SearchPaths) {
    std::string FullPath = buildSysrootedPath(Dir, Path);
    if (fs::exists(FullPath))
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
    path::append(Path, Config->Sysroot, Dir.substr(1), File);
  else
    path::append(Path, Dir, File);
  return Path.str();
}
