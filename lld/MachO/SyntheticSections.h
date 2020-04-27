//===- SyntheticSections.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_SYNTHETIC_SECTIONS_H
#define LLD_MACHO_SYNTHETIC_SECTIONS_H

#include "InputSection.h"
#include "Target.h"
#include "llvm/ADT/SetVector.h"

using namespace llvm::MachO;

namespace lld {
namespace macho {

namespace section_names {

constexpr const char *pageZero = "__pagezero";
constexpr const char *header = "__mach_header";
constexpr const char *binding = "__binding";

} // namespace section_names

class DylibSymbol;
class LoadCommand;

// The header of the Mach-O file, which must have a file offset of zero.
class MachHeaderSection : public InputSection {
public:
  MachHeaderSection();
  void addLoadCommand(LoadCommand *);
  bool isHidden() const override { return true; }
  size_t getSize() const override;
  void writeTo(uint8_t *buf) override;

private:
  std::vector<LoadCommand *> loadCommands;
  uint32_t sizeOfCmds = 0;
};

// A hidden section that exists solely for the purpose of creating the
// __PAGEZERO segment, which is used to catch null pointer dereferences.
class PageZeroSection : public InputSection {
public:
  PageZeroSection();
  bool isHidden() const override { return true; }
  size_t getSize() const override { return ImageBase; }
  uint64_t getFileSize() const override { return 0; }
};

// This section will be populated by dyld with addresses to non-lazily-loaded
// dylib symbols.
class GotSection : public InputSection {
public:
  GotSection();

  void addEntry(DylibSymbol &sym);
  const llvm::SetVector<const DylibSymbol *> &getEntries() const {
    return entries;
  }

  size_t getSize() const override { return entries.size() * WordSize; }

  bool isNeeded() const override { return !entries.empty(); }

  void writeTo(uint8_t *buf) override {
    // Nothing to write, GOT contains all zeros at link time; it's populated at
    // runtime by dyld.
  }

private:
  llvm::SetVector<const DylibSymbol *> entries;
};

// Stores bind opcodes for telling dyld which symbols to load non-lazily.
class BindingSection : public InputSection {
public:
  BindingSection();
  void finalizeContents();
  size_t getSize() const override { return contents.size(); }
  // Like other sections in __LINKEDIT, the binding section is special: its
  // offsets are recorded in the LC_DYLD_INFO_ONLY load command, instead of in
  // section headers.
  bool isHidden() const override { return true; }
  bool isNeeded() const override;
  void writeTo(uint8_t *buf) override;

  SmallVector<char, 128> contents;
};

struct InStruct {
  GotSection *got = nullptr;
};

extern InStruct in;

} // namespace macho
} // namespace lld

#endif
