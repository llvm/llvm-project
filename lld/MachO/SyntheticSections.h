//===- SyntheticSections.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_SYNTHETIC_SECTIONS_H
#define LLD_MACHO_SYNTHETIC_SECTIONS_H

#include "Config.h"
#include "ExportTrie.h"
#include "InputSection.h"
#include "OutputSection.h"
#include "Target.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace macho {

namespace section_names {

constexpr const char pageZero[] = "__pagezero";
constexpr const char header[] = "__mach_header";
constexpr const char binding[] = "__binding";
constexpr const char lazyBinding[] = "__lazy_binding";
constexpr const char export_[] = "__export";
constexpr const char symbolTable[] = "__symbol_table";
constexpr const char stringTable[] = "__string_table";
constexpr const char got[] = "__got";

} // namespace section_names

class DylibSymbol;
class LoadCommand;

class SyntheticSection : public OutputSection {
public:
  SyntheticSection(const char *segname, const char *name);
  virtual ~SyntheticSection() = default;

  static bool classof(const OutputSection *sec) {
    return sec->kind() == SyntheticKind;
  }

  const StringRef segname;
};

// The header of the Mach-O file, which must have a file offset of zero.
class MachHeaderSection : public SyntheticSection {
public:
  MachHeaderSection();
  void addLoadCommand(LoadCommand *);
  bool isHidden() const override { return true; }
  uint64_t getSize() const override;
  void writeTo(uint8_t *buf) const override;

private:
  std::vector<LoadCommand *> loadCommands;
  uint32_t sizeOfCmds = 0;
};

// A hidden section that exists solely for the purpose of creating the
// __PAGEZERO segment, which is used to catch null pointer dereferences.
class PageZeroSection : public SyntheticSection {
public:
  PageZeroSection();
  bool isHidden() const override { return true; }
  uint64_t getSize() const override { return PageZeroSize; }
  uint64_t getFileSize() const override { return 0; }
  void writeTo(uint8_t *buf) const override {}
};

// This section will be populated by dyld with addresses to non-lazily-loaded
// dylib symbols.
class GotSection : public SyntheticSection {
public:
  GotSection();

  const llvm::SetVector<const Symbol *> &getEntries() const { return entries; }

  bool isNeeded() const override { return !entries.empty(); }

  uint64_t getSize() const override { return entries.size() * WordSize; }

  void writeTo(uint8_t *buf) const override;

  void addEntry(Symbol &sym);

private:
  llvm::SetVector<const Symbol *> entries;
};

// Stores bind opcodes for telling dyld which symbols to load non-lazily.
class BindingSection : public SyntheticSection {
public:
  BindingSection();
  void finalizeContents();
  uint64_t getSize() const override { return contents.size(); }
  // Like other sections in __LINKEDIT, the binding section is special: its
  // offsets are recorded in the LC_DYLD_INFO_ONLY load command, instead of in
  // section headers.
  bool isHidden() const override { return true; }
  bool isNeeded() const override;
  void writeTo(uint8_t *buf) const override;

  SmallVector<char, 128> contents;
};

// The following sections implement lazy symbol binding -- very similar to the
// PLT mechanism in ELF.
//
// ELF's .plt section is broken up into two sections in Mach-O: StubsSection and
// StubHelperSection. Calls to functions in dylibs will end up calling into
// StubsSection, which contains indirect jumps to addresses stored in the
// LazyPointerSection (the counterpart to ELF's .plt.got).
//
// Initially, the LazyPointerSection contains addresses that point into one of
// the entry points in the middle of the StubHelperSection. The code in
// StubHelperSection will push on the stack an offset into the
// LazyBindingSection. The push is followed by a jump to the beginning of the
// StubHelperSection (similar to PLT0), which then calls into dyld_stub_binder.
// dyld_stub_binder is a non-lazily-bound symbol, so this call looks it up in
// the GOT.
//
// The stub binder will look up the bind opcodes in the LazyBindingSection at
// the given offset. The bind opcodes will tell the binder to update the address
// in the LazyPointerSection to point to the symbol, so that subsequent calls
// don't have to redo the symbol resolution. The binder will then jump to the
// resolved symbol.

class StubsSection : public SyntheticSection {
public:
  StubsSection();
  uint64_t getSize() const override;
  bool isNeeded() const override { return !entries.empty(); }
  void writeTo(uint8_t *buf) const override;

  const llvm::SetVector<DylibSymbol *> &getEntries() const { return entries; }

  void addEntry(DylibSymbol &sym);

private:
  llvm::SetVector<DylibSymbol *> entries;
};

class StubHelperSection : public SyntheticSection {
public:
  StubHelperSection();
  uint64_t getSize() const override;
  bool isNeeded() const override;
  void writeTo(uint8_t *buf) const override;

  void setup();

  DylibSymbol *stubBinder = nullptr;
};

// This section contains space for just a single word, and will be used by dyld
// to cache an address to the image loader it uses. Note that unlike the other
// synthetic sections, which are OutputSections, the ImageLoaderCacheSection is
// an InputSection that gets merged into the __data OutputSection.
class ImageLoaderCacheSection : public InputSection {
public:
  ImageLoaderCacheSection();
  uint64_t getSize() const override { return WordSize; }
};

class LazyPointerSection : public SyntheticSection {
public:
  LazyPointerSection();
  uint64_t getSize() const override;
  bool isNeeded() const override;
  void writeTo(uint8_t *buf) const override;
};

class LazyBindingSection : public SyntheticSection {
public:
  LazyBindingSection();
  void finalizeContents();
  uint64_t getSize() const override { return contents.size(); }
  uint32_t encode(const DylibSymbol &);
  // Like other sections in __LINKEDIT, the lazy binding section is special: its
  // offsets are recorded in the LC_DYLD_INFO_ONLY load command, instead of in
  // section headers.
  bool isHidden() const override { return true; }
  bool isNeeded() const override;
  void writeTo(uint8_t *buf) const override;

private:
  SmallVector<char, 128> contents;
  llvm::raw_svector_ostream os{contents};
};

// Stores a trie that describes the set of exported symbols.
class ExportSection : public SyntheticSection {
public:
  ExportSection();
  void finalizeContents();
  uint64_t getSize() const override { return size; }
  // Like other sections in __LINKEDIT, the export section is special: its
  // offsets are recorded in the LC_DYLD_INFO_ONLY load command, instead of in
  // section headers.
  bool isHidden() const override { return true; }
  void writeTo(uint8_t *buf) const override;

private:
  TrieBuilder trieBuilder;
  size_t size = 0;
};

// Stores the strings referenced by the symbol table.
class StringTableSection : public SyntheticSection {
public:
  StringTableSection();
  // Returns the start offset of the added string.
  uint32_t addString(StringRef);
  uint64_t getSize() const override { return size; }
  // Like other sections in __LINKEDIT, the string table section is special: its
  // offsets are recorded in the LC_SYMTAB load command, instead of in section
  // headers.
  bool isHidden() const override { return true; }
  void writeTo(uint8_t *buf) const override;

private:
  // An n_strx value of 0 always indicates the empty string, so we must locate
  // our non-empty string values at positive offsets in the string table.
  // Therefore we insert a dummy value at position zero.
  std::vector<StringRef> strings{"\0"};
  size_t size = 1;
};

struct SymtabEntry {
  Symbol *sym;
  size_t strx;
};

class SymtabSection : public SyntheticSection {
public:
  SymtabSection(StringTableSection &);
  void finalizeContents();
  size_t getNumSymbols() const { return symbols.size(); }
  uint64_t getSize() const override;
  // Like other sections in __LINKEDIT, the symtab section is special: its
  // offsets are recorded in the LC_SYMTAB load command, instead of in section
  // headers.
  bool isHidden() const override { return true; }
  void writeTo(uint8_t *buf) const override;

private:
  StringTableSection &stringTableSection;
  std::vector<SymtabEntry> symbols;
};

struct InStruct {
  GotSection *got = nullptr;
  LazyPointerSection *lazyPointers = nullptr;
  StubsSection *stubs = nullptr;
  StubHelperSection *stubHelper = nullptr;
  ImageLoaderCacheSection *imageLoaderCache = nullptr;
};

extern InStruct in;
extern std::vector<SyntheticSection *> syntheticSections;

} // namespace macho
} // namespace lld

#endif
