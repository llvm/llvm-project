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
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"

namespace lld {
namespace elf2 {

extern class LinkerDriver *Driver;

// Entry point of the ELF linker. Returns true on success.
bool link(ArrayRef<const char *> Args);

class LinkerDriver {
public:
  void main(ArrayRef<const char *> Args);
  void addFile(StringRef Path);
  void addLibrary(StringRef Name);

private:
  void readConfigs(llvm::opt::InputArgList &Args);
  void createFiles(llvm::opt::InputArgList &Args);
  template <class ELFT> void link(llvm::opt::InputArgList &Args);

  llvm::BumpPtrAllocator Alloc;
  bool WholeArchive = false;
  std::vector<std::unique_ptr<InputFile>> Files;
  std::vector<std::unique_ptr<MemoryBuffer>> OwningMBs;
};

// Parses command line options.
llvm::opt::InputArgList parseArgs(llvm::BumpPtrAllocator *A,
                                  ArrayRef<const char *> Args);

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

// Parses a linker script. Calling this function updates the Symtab and Config.
void readLinkerScript(llvm::BumpPtrAllocator *A, MemoryBufferRef MB);

std::string findFromSearchPaths(StringRef Path);
std::string searchLibrary(StringRef Path);
std::string buildSysrootedPath(llvm::StringRef Dir, llvm::StringRef File);

} // namespace elf2
} // namespace lld

#endif
