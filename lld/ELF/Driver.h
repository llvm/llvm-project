//===- Driver.h -------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_DRIVER_H
#define LLD_ELF_DRIVER_H

#include "SymbolTable.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace elf {

extern class LinkerDriver *Driver;

class CpioFile;

class LinkerDriver {
public:
  void main(ArrayRef<const char *> Args);
  void addFile(StringRef Path);
  void addLibrary(StringRef Name);
  llvm::LLVMContext Context;      // to parse bitcode ifles
  std::unique_ptr<CpioFile> Cpio; // for reproduce

private:
  std::vector<MemoryBufferRef> getArchiveMembers(MemoryBufferRef MB);
  llvm::Optional<MemoryBufferRef> readFile(StringRef Path);
  void readConfigs(llvm::opt::InputArgList &Args);
  void createFiles(llvm::opt::InputArgList &Args);
  template <class ELFT> void link(llvm::opt::InputArgList &Args);

  // True if we are in --whole-archive and --no-whole-archive.
  bool WholeArchive = false;

  // True if we are in --start-lib and --end-lib.
  bool InLib = false;

  llvm::BumpPtrAllocator Alloc;
  std::vector<std::unique_ptr<InputFile>> Files;
  std::vector<std::unique_ptr<MemoryBuffer>> OwningMBs;
};

// Parses command line options.
class ELFOptTable : public llvm::opt::OptTable {
public:
  ELFOptTable();
  llvm::opt::InputArgList parse(ArrayRef<const char *> Argv);

private:
  llvm::BumpPtrAllocator Alloc;
};

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

// This is the class to create a .cpio file for --reproduce.
//
// If "--reproduce foo" is given, we create a file "foo.cpio" and
// copy all input files to the archive, along with a response file
// to re-run the same command with the same inputs.
// It is useful for reporting issues to LLD developers.
//
// Cpio as a file format is a deliberate choice. It's standardized in
// POSIX and very easy to create. cpio command is available virtually
// on all Unix systems. See
// http://pubs.opengroup.org/onlinepubs/9699919799/utilities/pax.html#tag_20_92_13_07
// for the format details.
class CpioFile {
public:
  static CpioFile *create(StringRef OutputPath);
  void append(StringRef Path, StringRef Data);

private:
  CpioFile(std::unique_ptr<llvm::raw_fd_ostream> OS, StringRef Basename);

  std::unique_ptr<llvm::raw_fd_ostream> OS;
  llvm::StringSet<> Seen;
  std::string Basename;
};

void printHelp(const char *Argv0);
std::string getVersionString();
std::vector<uint8_t> parseHexstring(StringRef S);

std::string createResponseFile(const llvm::opt::InputArgList &Args);
std::string relativeToRoot(StringRef Path);

std::string findFromSearchPaths(StringRef Path);
std::string searchLibrary(StringRef Path);
std::string buildSysrootedPath(llvm::StringRef Dir, llvm::StringRef File);

} // namespace elf
} // namespace lld

#endif
