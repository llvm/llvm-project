//===- Driver.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_DRIVER_H
#define LLD_ELF_DRIVER_H

#include "LTO.h"
#include "lld/Common/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"

namespace lld {
namespace elf {
class InputFile;
class Symbol;

extern std::unique_ptr<class LinkerDriver> driver;

class LinkerDriver {
public:
  void linkerMain(ArrayRef<const char *> args);
  void addFile(StringRef path, bool withLOption);
  void addLibrary(StringRef name);

private:
  void createFiles(llvm::opt::InputArgList &args);
  void inferMachineType();
  void link(llvm::opt::InputArgList &args);
  template <class ELFT> void compileBitcodeFiles(bool skipLinkedOutput);
  void writeArchiveStats() const;
  void writeWhyExtract() const;
  void reportBackrefs() const;

  // True if we are in --whole-archive and --no-whole-archive.
  bool inWholeArchive = false;

  // True if we are in --start-lib and --end-lib.
  bool inLib = false;

  // For LTO.
  std::unique_ptr<BitcodeCompiler> lto;

  std::vector<InputFile *> files;
  SmallVector<std::pair<StringRef, unsigned>, 0> archiveFiles;

public:
  // A tuple of (reference, extractedFile, sym). Used by --why-extract=.
  SmallVector<std::tuple<std::string, const InputFile *, const Symbol &>, 0>
      whyExtract;
  // A mapping from a symbol to an InputFile referencing it backward. Used by
  // --warn-backrefs.
  llvm::DenseMap<const Symbol *,
                 std::pair<const InputFile *, const InputFile *>>
      backwardReferences;
};

// Parses command line options.
class ELFOptTable : public llvm::opt::OptTable {
public:
  ELFOptTable();
  llvm::opt::InputArgList parse(ArrayRef<const char *> argv);
};

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11, _12) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

void printHelp();
std::string createResponseFile(const llvm::opt::InputArgList &args);

llvm::Optional<std::string> findFromSearchPaths(StringRef path);
llvm::Optional<std::string> searchScript(StringRef path);
llvm::Optional<std::string> searchLibraryBaseName(StringRef path);
llvm::Optional<std::string> searchLibrary(StringRef path);

} // namespace elf
} // namespace lld

#endif
