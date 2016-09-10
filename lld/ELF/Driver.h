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
#include "lld/Core/Reproduce.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace elf {

extern class LinkerDriver *Driver;

class LinkerDriver {
public:
  void main(ArrayRef<const char *> Args);
  void addFile(StringRef Path, bool KnownScript = false);
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

void printHelp(const char *Argv0);
std::string getVersionString();
std::vector<uint8_t> parseHexstring(StringRef S);

std::string createResponseFile(const llvm::opt::InputArgList &Args);

std::string findFromSearchPaths(StringRef Path);
std::string searchLibrary(StringRef Path);
std::string buildSysrootedPath(llvm::StringRef Dir, llvm::StringRef File);

} // namespace elf
} // namespace lld

#endif
