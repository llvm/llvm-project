//===- Driver.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_DRIVER_H
#define LLD_ELF_DRIVER_H

#include "SymbolTable.h"
#include "lld/Common/LLVM.h"
#include "lld/Common/Reproduce.h"
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
  void addFile(StringRef Path, bool WithLOption);
  void addLibrary(StringRef Name);

private:
  void readConfigs(llvm::opt::InputArgList &Args);
  void createFiles(llvm::opt::InputArgList &Args);
  void inferMachineType();
  template <class ELFT> void link(llvm::opt::InputArgList &Args);

  // True if we are in --whole-archive and --no-whole-archive.
  bool InWholeArchive = false;

  // True if we are in --start-lib and --end-lib.
  bool InLib = false;

  std::vector<InputFile *> Files;
};

// Parses command line options.
class ELFOptTable : public llvm::opt::OptTable {
public:
  ELFOptTable();
  llvm::opt::InputArgList parse(ArrayRef<const char *> Argv);
};

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11, _12) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

void printHelp();
std::string createResponseFile(const llvm::opt::InputArgList &Args);

llvm::Optional<std::string> findFromSearchPaths(StringRef Path);
llvm::Optional<std::string> searchScript(StringRef Path);
llvm::Optional<std::string> searchLibrary(StringRef Path);

} // namespace elf
} // namespace lld

#endif
