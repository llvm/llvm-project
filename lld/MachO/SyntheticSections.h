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
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
class DWARFUnit;
} // namespace llvm

namespace lld {
namespace macho {

namespace section_names {

constexpr const char pageZero[] = "__pagezero";
constexpr const char common[] = "__common";
constexpr const char header[] = "__mach_header";
constexpr const char rebase[] = "__rebase";
constexpr const char binding[] = "__binding";
constexpr const char weakBinding[] = "__weak_binding";
constexpr const char lazyBinding[] = "__lazy_binding";
constexpr const char export_[] = "__export";
constexpr const char functionStarts_[] = "__functionStarts";
constexpr const char symbolTable[] = "__symbol_table";
constexpr const char indirectSymbolTable[] = "__ind_sym_tab";
constexpr const char stringTable[] = "__string_table";
constexpr const char codeSignature[] = "__code_signature";
constexpr const char got[] = "__got";
constexpr const char threadPtrs[] = "__thread_ptrs";
constexpr const char unwindInfo[] = "__unwind_info";
// these are not synthetic, but in service of synthetic __unwind_info
constexpr const char compactUnwind[] = "__compact_unwind";
constexpr const char ehFrame[] = "__eh_frame";
// these are not synthetic, but need to be sorted
constexpr const char text[] = "__text";
constexpr const char stubs[] = "__stubs";
constexpr const char stubHelper[] = "__stub_helper";
constexpr const char laSymbolPtr[] = "__la_symbol_ptr";
constexpr const char data[] = "__data";

} // namespace section_names

class Defined;
class DylibSymbol;
class LoadCommand;
class ObjFile;

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
    align = WordSize; // mimic ld64
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
    return llvm::alignTo(getRawSize(), align);
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

struct Location {
  SectionPointerUnion section = nullptr;
  uint64_t offset = 0;

  Location(SectionPointerUnion section, uint64_t offset)
      : section(section), offset(offset) {}
  uint64_t getVA() const;
};

// Stores rebase opcodes, which tell dyld where absolute addresses have been
// encoded in the binary. If the binary is not loaded at its preferred address,
// dyld has to rebase these addresses by adding an offset to them.
class RebaseSection : public LinkEditSection {
public:
  RebaseSection();
  void finalizeContents();
  uint64_t getRawSize() const override { return contents.size(); }
  bool isNeeded() const override { return !locations.empty(); }
  void writeTo(uint8_t *buf) const override;

  void addEntry(SectionPointerUnion section, uint64_t offset) {
    if (config->isPic)
      locations.push_back({section, offset});
  }

private:
  std::vector<Location> locations;
  SmallVector<char, 128> contents;
};

struct BindingEntry {
  const DylibSymbol *dysym;
  int64_t addend;
  Location target;
  BindingEntry(const DylibSymbol *dysym, int64_t addend, Location target)
      : dysym(dysym), addend(addend), target(std::move(target)) {}
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
    bindings.emplace_back(dysym, addend, Location(section, offset));
  }

private:
  std::vector<BindingEntry> bindings;
  SmallVector<char, 128> contents;
};

struct WeakBindingEntry {
  const Symbol *symbol;
  int64_t addend;
  Location target;
  WeakBindingEntry(const Symbol *symbol, int64_t addend, Location target)
      : symbol(symbol), addend(addend), target(std::move(target)) {}
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
    bindings.emplace_back(symbol, addend, Location(section, offset));
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
  Defined *dyldPrivate = nullptr;
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

// Adds stubs and bindings where necessary (e.g. if the symbol is a
// DylibSymbol.)
void prepareBranchTarget(Symbol *);

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

class FunctionStartsSection : public LinkEditSection {
public:
  FunctionStartsSection();
  void finalizeContents();
  uint64_t getRawSize() const override { return contents.size(); }
  void writeTo(uint8_t *buf) const override;

private:
  SmallVector<char, 128> contents;
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
  // ld64 emits string tables which start with a space and a zero byte. We
  // match its behavior here since some tools depend on it.
  std::vector<StringRef> strings{" "};
  size_t size = 2;
};

struct SymtabEntry {
  Symbol *sym;
  size_t strx;
};

struct StabsEntry {
  uint8_t type = 0;
  uint32_t strx = 0;
  uint8_t sect = 0;
  uint16_t desc = 0;
  uint64_t value = 0;

  StabsEntry() = default;
  explicit StabsEntry(uint8_t type) : type(type) {}
};

// Symbols of the same type must be laid out contiguously: we choose to emit
// all local symbols first, then external symbols, and finally undefined
// symbols. For each symbol type, the LC_DYSYMTAB load command will record the
// range (start index and total number) of those symbols in the symbol table.
class SymtabSection : public LinkEditSection {
public:
  SymtabSection(StringTableSection &);
  void finalizeContents();
  uint32_t getNumSymbols() const;
  uint32_t getNumLocalSymbols() const {
    return stabs.size() + localSymbols.size();
  }
  uint32_t getNumExternalSymbols() const { return externalSymbols.size(); }
  uint32_t getNumUndefinedSymbols() const { return undefinedSymbols.size(); }
  uint64_t getRawSize() const override;
  void writeTo(uint8_t *buf) const override;

private:
  void emitBeginSourceStab(llvm::DWARFUnit *compileUnit);
  void emitEndSourceStab();
  void emitObjectFileStab(ObjFile *);
  void emitEndFunStab(Defined *);
  void emitStabs();

  StringTableSection &stringTableSection;
  // STABS symbols are always local symbols, but we represent them with special
  // entries because they may use fields like n_sect and n_desc differently.
  std::vector<StabsEntry> stabs;
  std::vector<SymtabEntry> localSymbols;
  std::vector<SymtabEntry> externalSymbols;
  std::vector<SymtabEntry> undefinedSymbols;
};

// The indirect symbol table is a list of 32-bit integers that serve as indices
// into the (actual) symbol table. The indirect symbol table is a
// concatenation of several sub-arrays of indices, each sub-array belonging to
// a separate section. The starting offset of each sub-array is stored in the
// reserved1 header field of the respective section.
//
// These sub-arrays provide symbol information for sections that store
// contiguous sequences of symbol references. These references can be pointers
// (e.g. those in the GOT and TLVP sections) or assembly sequences (e.g.
// function stubs).
class IndirectSymtabSection : public LinkEditSection {
public:
  IndirectSymtabSection();
  void finalizeContents();
  uint32_t getNumSymbols() const;
  uint64_t getRawSize() const override {
    return getNumSymbols() * sizeof(uint32_t);
  }
  bool isNeeded() const override;
  void writeTo(uint8_t *buf) const override;
};

// The code signature comes at the very end of the linked output file.
class CodeSignatureSection : public LinkEditSection {
public:
  static constexpr uint8_t blockSizeShift = 12;
  static constexpr size_t blockSize = (1 << blockSizeShift); // 4 KiB
  static constexpr size_t hashSize = 256 / 8;
  static constexpr size_t blobHeadersSize = llvm::alignTo<8>(
      sizeof(llvm::MachO::CS_SuperBlob) + sizeof(llvm::MachO::CS_BlobIndex));
  static constexpr uint32_t fixedHeadersSize =
      blobHeadersSize + sizeof(llvm::MachO::CS_CodeDirectory);

  uint32_t fileNamePad = 0;
  uint32_t allHeadersSize = 0;
  StringRef fileName;

  CodeSignatureSection();
  uint64_t getRawSize() const override;
  bool isNeeded() const override { return true; }
  void writeTo(uint8_t *buf) const override;
  uint32_t getBlockCount() const;
  void writeHashes(uint8_t *buf) const;
};

static_assert((CodeSignatureSection::blobHeadersSize % 8) == 0, "");
static_assert((CodeSignatureSection::fixedHeadersSize % 8) == 0, "");

struct InStruct {
  MachHeaderSection *header = nullptr;
  RebaseSection *rebase = nullptr;
  BindingSection *binding = nullptr;
  WeakBindingSection *weakBinding = nullptr;
  LazyBindingSection *lazyBinding = nullptr;
  ExportSection *exports = nullptr;
  FunctionStartsSection *functionStarts = nullptr;
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
