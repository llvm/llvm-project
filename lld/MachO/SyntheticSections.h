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
#include "OutputSegment.h"
#include "Target.h"

#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace macho {

namespace section_names {

constexpr const char pageZero[] = "__pagezero";
constexpr const char header[] = "__mach_header";
constexpr const char binding[] = "__binding";
constexpr const char weakBinding[] = "__weak_binding";
constexpr const char lazyBinding[] = "__lazy_binding";
constexpr const char export_[] = "__export";
constexpr const char symbolTable[] = "__symbol_table";
constexpr const char stringTable[] = "__string_table";
constexpr const char got[] = "__got";
constexpr const char threadPtrs[] = "__thread_ptrs";

} // namespace section_names

class Defined;
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

// All sections in __LINKEDIT should inherit from this.
class LinkEditSection : public SyntheticSection {
public:
  LinkEditSection(const char *segname, const char *name)
      : SyntheticSection(segname, name) {
    align = WordSize;
  }

  // Sections in __LINKEDIT are special: their offsets are recorded in the
  // load commands like LC_DYLD_INFO_ONLY and LC_SYMTAB, instead of in section
  // headers.
  bool isHidden() const override final { return true; }

  virtual uint64_t getRawSize() const = 0;

  // codesign (or more specifically libstuff) checks that each section in
  // __LINKEDIT ends where the next one starts -- no gaps are permitted. We
  // therefore align every section's start and end points to WordSize.
  //
  // NOTE: This assumes that the extra bytes required for alignment can be
  // zero-valued bytes.
  uint64_t getSize() const override final {
    return llvm::alignTo(getRawSize(), WordSize);
  }
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

// This is the base class for the GOT and TLVPointer sections, which are nearly
// functionally identical -- they will both be populated by dyld with addresses
// to non-lazily-loaded dylib symbols. The main difference is that the
// TLVPointerSection stores references to thread-local variables.
class NonLazyPointerSectionBase : public SyntheticSection {
public:
  NonLazyPointerSectionBase(const char *segname, const char *name);

  const llvm::SetVector<const Symbol *> &getEntries() const { return entries; }

  bool isNeeded() const override { return !entries.empty(); }

  uint64_t getSize() const override { return entries.size() * WordSize; }

  void writeTo(uint8_t *buf) const override;

  void addEntry(Symbol *sym);

private:
  llvm::SetVector<const Symbol *> entries;
};

class GotSection : public NonLazyPointerSectionBase {
public:
  GotSection()
      : NonLazyPointerSectionBase(segment_names::dataConst,
                                  section_names::got) {
    // TODO: section_64::reserved1 should be an index into the indirect symbol
    // table, which we do not currently emit
  }
};

class TlvPointerSection : public NonLazyPointerSectionBase {
public:
  TlvPointerSection()
      : NonLazyPointerSectionBase(segment_names::data,
                                  section_names::threadPtrs) {}
};

using SectionPointerUnion =
    llvm::PointerUnion<const InputSection *, const OutputSection *>;

struct BindingTarget {
  SectionPointerUnion section;
  uint64_t offset;
  int64_t addend;

  BindingTarget(SectionPointerUnion section, uint64_t offset, int64_t addend)
      : section(section), offset(offset), addend(addend) {}

  uint64_t getVA() const;
};

struct BindingEntry {
  const DylibSymbol *dysym;
  BindingTarget target;
  BindingEntry(const DylibSymbol *dysym, BindingTarget target)
      : dysym(dysym), target(std::move(target)) {}
};

// Stores bind opcodes for telling dyld which symbols to load non-lazily.
class BindingSection : public LinkEditSection {
public:
  BindingSection();
  void finalizeContents();
  uint64_t getRawSize() const override { return contents.size(); }
  bool isNeeded() const override { return !bindings.empty(); }
  void writeTo(uint8_t *buf) const override;

  void addEntry(const DylibSymbol *dysym, SectionPointerUnion section,
                uint64_t offset, int64_t addend = 0) {
    bindings.emplace_back(dysym, BindingTarget(section, offset, addend));
  }

private:
  std::vector<BindingEntry> bindings;
  SmallVector<char, 128> contents;
};

struct WeakBindingEntry {
  const Symbol *symbol;
  BindingTarget target;
  WeakBindingEntry(const Symbol *symbol, BindingTarget target)
      : symbol(symbol), target(std::move(target)) {}
};

// Stores bind opcodes for telling dyld which weak symbols need coalescing.
// There are two types of entries in this section:
//
//   1) Non-weak definitions: This is a symbol definition that weak symbols in
//   other dylibs should coalesce to.
//
//   2) Weak bindings: These tell dyld that a given symbol reference should
//   coalesce to a non-weak definition if one is found. Note that unlike in the
//   entries in the BindingSection, the bindings here only refer to these
//   symbols by name, but do not specify which dylib to load them from.
class WeakBindingSection : public LinkEditSection {
public:
  WeakBindingSection();
  void finalizeContents();
  uint64_t getRawSize() const override { return contents.size(); }
  bool isNeeded() const override {
    return !bindings.empty() || !definitions.empty();
  }

  void writeTo(uint8_t *buf) const override;

  void addEntry(const Symbol *symbol, SectionPointerUnion section,
                uint64_t offset, int64_t addend = 0) {
    bindings.emplace_back(symbol, BindingTarget(section, offset, addend));
  }

  bool hasEntry() const { return !bindings.empty(); }

  void addNonWeakDefinition(const Defined *defined) {
    definitions.emplace_back(defined);
  }

  bool hasNonWeakDefinition() const { return !definitions.empty(); }

private:
  std::vector<WeakBindingEntry> bindings;
  std::vector<const Defined *> definitions;
  SmallVector<char, 128> contents;
};

// Whether a given symbol's address can only be resolved at runtime.
bool needsBinding(const Symbol *);

// Add bindings for symbols that need weak or non-lazy bindings.
void addNonLazyBindingEntries(const Symbol *, SectionPointerUnion,
                              uint64_t offset, int64_t addend = 0);

// The following sections implement lazy symbol binding -- very similar to the
// PLT mechanism in ELF.
//
// ELF's .plt section is broken up into two sections in Mach-O: StubsSection
// and StubHelperSection. Calls to functions in dylibs will end up calling into
// StubsSection, which contains indirect jumps to addresses stored in the
// LazyPointerSection (the counterpart to ELF's .plt.got).
//
// We will first describe how non-weak symbols are handled.
//
// At program start, the LazyPointerSection contains addresses that point into
// one of the entry points in the middle of the StubHelperSection. The code in
// StubHelperSection will push on the stack an offset into the
// LazyBindingSection. The push is followed by a jump to the beginning of the
// StubHelperSection (similar to PLT0), which then calls into dyld_stub_binder.
// dyld_stub_binder is a non-lazily-bound symbol, so this call looks it up in
// the GOT.
//
// The stub binder will look up the bind opcodes in the LazyBindingSection at
// the given offset. The bind opcodes will tell the binder to update the
// address in the LazyPointerSection to point to the symbol, so that subsequent
// calls don't have to redo the symbol resolution. The binder will then jump to
// the resolved symbol.
//
// With weak symbols, the situation is slightly different. Since there is no
// "weak lazy" lookup, function calls to weak symbols are always non-lazily
// bound. We emit both regular non-lazy bindings as well as weak bindings, in
// order that the weak bindings may overwrite the non-lazy bindings if an
// appropriate symbol is found at runtime. However, the bound addresses will
// still be written (non-lazily) into the LazyPointerSection.

class StubsSection : public SyntheticSection {
public:
  StubsSection();
  uint64_t getSize() const override;
  bool isNeeded() const override { return !entries.empty(); }
  void writeTo(uint8_t *buf) const override;
  const llvm::SetVector<Symbol *> &getEntries() const { return entries; }
  // Returns whether the symbol was added. Note that every stubs entry will
  // have a corresponding entry in the LazyPointerSection.
  bool addEntry(Symbol *);

private:
  llvm::SetVector<Symbol *> entries;
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

// Note that this section may also be targeted by non-lazy bindings. In
// particular, this happens when branch relocations target weak symbols.
class LazyPointerSection : public SyntheticSection {
public:
  LazyPointerSection();
  uint64_t getSize() const override;
  bool isNeeded() const override;
  void writeTo(uint8_t *buf) const override;
};

class LazyBindingSection : public LinkEditSection {
public:
  LazyBindingSection();
  void finalizeContents();
  uint64_t getRawSize() const override { return contents.size(); }
  bool isNeeded() const override { return !entries.empty(); }
  void writeTo(uint8_t *buf) const override;
  // Note that every entry here will by referenced by a corresponding entry in
  // the StubHelperSection.
  void addEntry(DylibSymbol *dysym);
  const llvm::SetVector<DylibSymbol *> &getEntries() const { return entries; }

private:
  uint32_t encode(const DylibSymbol &);

  llvm::SetVector<DylibSymbol *> entries;
  SmallVector<char, 128> contents;
  llvm::raw_svector_ostream os{contents};
};

// Stores a trie that describes the set of exported symbols.
class ExportSection : public LinkEditSection {
public:
  ExportSection();
  void finalizeContents();
  uint64_t getRawSize() const override { return size; }
  void writeTo(uint8_t *buf) const override;

  bool hasWeakSymbol = false;

private:
  TrieBuilder trieBuilder;
  size_t size = 0;
};

// Stores the strings referenced by the symbol table.
class StringTableSection : public LinkEditSection {
public:
  StringTableSection();
  // Returns the start offset of the added string.
  uint32_t addString(StringRef);
  uint64_t getRawSize() const override { return size; }
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

class SymtabSection : public LinkEditSection {
public:
  SymtabSection(StringTableSection &);
  void finalizeContents();
  size_t getNumSymbols() const { return symbols.size(); }
  uint64_t getRawSize() const override;
  void writeTo(uint8_t *buf) const override;

private:
  StringTableSection &stringTableSection;
  std::vector<SymtabEntry> symbols;
};

struct InStruct {
  MachHeaderSection *header = nullptr;
  BindingSection *binding = nullptr;
  WeakBindingSection *weakBinding = nullptr;
  LazyBindingSection *lazyBinding = nullptr;
  ExportSection *exports = nullptr;
  GotSection *got = nullptr;
  TlvPointerSection *tlvPointers = nullptr;
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
