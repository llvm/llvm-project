//===- SyntheticSections.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SyntheticSections.h"
#include "Config.h"
#include "ExportTrie.h"
#include "InputFiles.h"
#include "MachOStructs.h"
#include "MergedOutputSection.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "Writer.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#if defined(__APPLE__)
#include <sys/mman.h>
#endif

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::support;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::macho;

InStruct macho::in;
std::vector<SyntheticSection *> macho::syntheticSections;

SyntheticSection::SyntheticSection(const char *segname, const char *name)
    : OutputSection(SyntheticKind, name), segname(segname) {
  isec = make<InputSection>();
  isec->segname = segname;
  isec->name = name;
  isec->parent = this;
  isec->outSecOff = 0;
  syntheticSections.push_back(this);
}

// dyld3's MachOLoaded::getSlide() assumes that the __TEXT segment starts
// from the beginning of the file (i.e. the header).
MachHeaderSection::MachHeaderSection()
    : SyntheticSection(segment_names::text, section_names::header) {
  // XXX: This is a hack. (See D97007)
  // Setting the index to 1 to pretend that this section is the text
  // section.
  index = 1;
}

void MachHeaderSection::addLoadCommand(LoadCommand *lc) {
  loadCommands.push_back(lc);
  sizeOfCmds += lc->getSize();
}

uint64_t MachHeaderSection::getSize() const {
  return sizeof(mach_header_64) + sizeOfCmds + config->headerPad;
}

static uint32_t cpuSubtype() {
  uint32_t subtype = target->cpuSubtype;

  if (config->outputType == MH_EXECUTE && !config->staticLink &&
      target->cpuSubtype == CPU_SUBTYPE_X86_64_ALL &&
      config->target.Platform == PlatformKind::macOS &&
      config->platformInfo.minimum >= VersionTuple(10, 5))
    subtype |= CPU_SUBTYPE_LIB64;

  return subtype;
}

void MachHeaderSection::writeTo(uint8_t *buf) const {
  auto *hdr = reinterpret_cast<mach_header_64 *>(buf);
  hdr->magic = MH_MAGIC_64;
  hdr->cputype = target->cpuType;
  hdr->cpusubtype = cpuSubtype();
  hdr->filetype = config->outputType;
  hdr->ncmds = loadCommands.size();
  hdr->sizeofcmds = sizeOfCmds;
  hdr->flags = MH_DYLDLINK;

  if (config->namespaceKind == NamespaceKind::twolevel)
    hdr->flags |= MH_NOUNDEFS | MH_TWOLEVEL;

  if (config->outputType == MH_DYLIB && !config->hasReexports)
    hdr->flags |= MH_NO_REEXPORTED_DYLIBS;

  if (config->markDeadStrippableDylib)
    hdr->flags |= MH_DEAD_STRIPPABLE_DYLIB;

  if (config->outputType == MH_EXECUTE && config->isPic)
    hdr->flags |= MH_PIE;

  if (in.exports->hasWeakSymbol || in.weakBinding->hasNonWeakDefinition())
    hdr->flags |= MH_WEAK_DEFINES;

  if (in.exports->hasWeakSymbol || in.weakBinding->hasEntry())
    hdr->flags |= MH_BINDS_TO_WEAK;

  for (const OutputSegment *seg : outputSegments) {
    for (const OutputSection *osec : seg->getSections()) {
      if (isThreadLocalVariables(osec->flags)) {
        hdr->flags |= MH_HAS_TLV_DESCRIPTORS;
        break;
      }
    }
  }

  uint8_t *p = reinterpret_cast<uint8_t *>(hdr + 1);
  for (const LoadCommand *lc : loadCommands) {
    lc->writeTo(p);
    p += lc->getSize();
  }
}

PageZeroSection::PageZeroSection()
    : SyntheticSection(segment_names::pageZero, section_names::pageZero) {}

RebaseSection::RebaseSection()
    : LinkEditSection(segment_names::linkEdit, section_names::rebase) {}

namespace {
struct Rebase {
  OutputSegment *segment = nullptr;
  uint64_t offset = 0;
  uint64_t consecutiveCount = 0;
};
} // namespace

// Rebase opcodes allow us to describe a contiguous sequence of rebase location
// using a single DO_REBASE opcode. To take advantage of it, we delay emitting
// `DO_REBASE` until we have reached the end of a contiguous sequence.
static void encodeDoRebase(Rebase &rebase, raw_svector_ostream &os) {
  assert(rebase.consecutiveCount != 0);
  if (rebase.consecutiveCount <= REBASE_IMMEDIATE_MASK) {
    os << static_cast<uint8_t>(REBASE_OPCODE_DO_REBASE_IMM_TIMES |
                               rebase.consecutiveCount);
  } else {
    os << static_cast<uint8_t>(REBASE_OPCODE_DO_REBASE_ULEB_TIMES);
    encodeULEB128(rebase.consecutiveCount, os);
  }
  rebase.consecutiveCount = 0;
}

static void encodeRebase(const OutputSection *osec, uint64_t outSecOff,
                         Rebase &lastRebase, raw_svector_ostream &os) {
  OutputSegment *seg = osec->parent;
  uint64_t offset = osec->getSegmentOffset() + outSecOff;
  if (lastRebase.segment != seg || lastRebase.offset != offset) {
    if (lastRebase.consecutiveCount != 0)
      encodeDoRebase(lastRebase, os);

    if (lastRebase.segment != seg) {
      os << static_cast<uint8_t>(REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB |
                                 seg->index);
      encodeULEB128(offset, os);
      lastRebase.segment = seg;
      lastRebase.offset = offset;
    } else {
      assert(lastRebase.offset != offset);
      os << static_cast<uint8_t>(REBASE_OPCODE_ADD_ADDR_ULEB);
      encodeULEB128(offset - lastRebase.offset, os);
      lastRebase.offset = offset;
    }
  }
  ++lastRebase.consecutiveCount;
  // DO_REBASE causes dyld to both perform the binding and increment the offset
  lastRebase.offset += WordSize;
}

void RebaseSection::finalizeContents() {
  if (locations.empty())
    return;

  raw_svector_ostream os{contents};
  Rebase lastRebase;

  os << static_cast<uint8_t>(REBASE_OPCODE_SET_TYPE_IMM | REBASE_TYPE_POINTER);

  llvm::sort(locations, [](const Location &a, const Location &b) {
    return a.isec->getVA() < b.isec->getVA();
  });
  for (const Location &loc : locations)
    encodeRebase(loc.isec->parent, loc.isec->outSecOff + loc.offset, lastRebase,
                 os);
  if (lastRebase.consecutiveCount != 0)
    encodeDoRebase(lastRebase, os);

  os << static_cast<uint8_t>(REBASE_OPCODE_DONE);
}

void RebaseSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
}

NonLazyPointerSectionBase::NonLazyPointerSectionBase(const char *segname,
                                                     const char *name)
    : SyntheticSection(segname, name) {
  align = WordSize;
  flags = S_NON_LAZY_SYMBOL_POINTERS;
}

void macho::addNonLazyBindingEntries(const Symbol *sym,
                                     const InputSection *isec, uint64_t offset,
                                     int64_t addend) {
  if (const auto *dysym = dyn_cast<DylibSymbol>(sym)) {
    in.binding->addEntry(dysym, isec, offset, addend);
    if (dysym->isWeakDef())
      in.weakBinding->addEntry(sym, isec, offset, addend);
  } else if (const auto *defined = dyn_cast<Defined>(sym)) {
    in.rebase->addEntry(isec, offset);
    if (defined->isExternalWeakDef())
      in.weakBinding->addEntry(sym, isec, offset, addend);
  } else {
    // Undefined symbols are filtered out in scanRelocations(); we should never
    // get here
    llvm_unreachable("cannot bind to an undefined symbol");
  }
}

void NonLazyPointerSectionBase::addEntry(Symbol *sym) {
  if (entries.insert(sym)) {
    assert(!sym->isInGot());
    sym->gotIndex = entries.size() - 1;

    addNonLazyBindingEntries(sym, isec, sym->gotIndex * WordSize);
  }
}

void NonLazyPointerSectionBase::writeTo(uint8_t *buf) const {
  for (size_t i = 0, n = entries.size(); i < n; ++i)
    if (auto *defined = dyn_cast<Defined>(entries[i]))
      write64le(&buf[i * WordSize], defined->getVA());
}

BindingSection::BindingSection()
    : LinkEditSection(segment_names::linkEdit, section_names::binding) {}

namespace {
struct Binding {
  OutputSegment *segment = nullptr;
  uint64_t offset = 0;
  int64_t addend = 0;
  int16_t ordinal = 0;
};
} // namespace

// Encode a sequence of opcodes that tell dyld to write the address of symbol +
// addend at osec->addr + outSecOff.
//
// The bind opcode "interpreter" remembers the values of each binding field, so
// we only need to encode the differences between bindings. Hence the use of
// lastBinding.
static void encodeBinding(const Symbol *sym, const OutputSection *osec,
                          uint64_t outSecOff, int64_t addend,
                          bool isWeakBinding, Binding &lastBinding,
                          raw_svector_ostream &os) {
  OutputSegment *seg = osec->parent;
  uint64_t offset = osec->getSegmentOffset() + outSecOff;
  if (lastBinding.segment != seg) {
    os << static_cast<uint8_t>(BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB |
                               seg->index);
    encodeULEB128(offset, os);
    lastBinding.segment = seg;
    lastBinding.offset = offset;
  } else if (lastBinding.offset != offset) {
    os << static_cast<uint8_t>(BIND_OPCODE_ADD_ADDR_ULEB);
    encodeULEB128(offset - lastBinding.offset, os);
    lastBinding.offset = offset;
  }

  if (lastBinding.addend != addend) {
    os << static_cast<uint8_t>(BIND_OPCODE_SET_ADDEND_SLEB);
    encodeSLEB128(addend, os);
    lastBinding.addend = addend;
  }

  uint8_t flags = BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM;
  if (!isWeakBinding && sym->isWeakRef())
    flags |= BIND_SYMBOL_FLAGS_WEAK_IMPORT;

  os << flags << sym->getName() << '\0'
     << static_cast<uint8_t>(BIND_OPCODE_SET_TYPE_IMM | BIND_TYPE_POINTER)
     << static_cast<uint8_t>(BIND_OPCODE_DO_BIND);
  // DO_BIND causes dyld to both perform the binding and increment the offset
  lastBinding.offset += WordSize;
}

// Non-weak bindings need to have their dylib ordinal encoded as well.
static int16_t ordinalForDylibSymbol(const DylibSymbol &dysym) {
  return config->namespaceKind == NamespaceKind::flat || dysym.isDynamicLookup()
             ? static_cast<int16_t>(BIND_SPECIAL_DYLIB_FLAT_LOOKUP)
             : dysym.getFile()->ordinal;
}

static void encodeDylibOrdinal(int16_t ordinal, raw_svector_ostream &os) {
  if (ordinal <= 0) {
    os << static_cast<uint8_t>(BIND_OPCODE_SET_DYLIB_SPECIAL_IMM |
                               (ordinal & BIND_IMMEDIATE_MASK));
  } else if (ordinal <= BIND_IMMEDIATE_MASK) {
    os << static_cast<uint8_t>(BIND_OPCODE_SET_DYLIB_ORDINAL_IMM | ordinal);
  } else {
    os << static_cast<uint8_t>(BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB);
    encodeULEB128(ordinal, os);
  }
}

static void encodeWeakOverride(const Defined *defined,
                               raw_svector_ostream &os) {
  os << static_cast<uint8_t>(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM |
                             BIND_SYMBOL_FLAGS_NON_WEAK_DEFINITION)
     << defined->getName() << '\0';
}

// Emit bind opcodes, which are a stream of byte-sized opcodes that dyld
// interprets to update a record with the following fields:
//  * segment index (of the segment to write the symbol addresses to, typically
//    the __DATA_CONST segment which contains the GOT)
//  * offset within the segment, indicating the next location to write a binding
//  * symbol type
//  * symbol library ordinal (the index of its library's LC_LOAD_DYLIB command)
//  * symbol name
//  * addend
// When dyld sees BIND_OPCODE_DO_BIND, it uses the current record state to bind
// a symbol in the GOT, and increments the segment offset to point to the next
// entry. It does *not* clear the record state after doing the bind, so
// subsequent opcodes only need to encode the differences between bindings.
void BindingSection::finalizeContents() {
  raw_svector_ostream os{contents};
  Binding lastBinding;

  // Since bindings are delta-encoded, sorting them allows for a more compact
  // result. Note that sorting by address alone ensures that bindings for the
  // same segment / section are located together.
  llvm::sort(bindings, [](const BindingEntry &a, const BindingEntry &b) {
    return a.target.getVA() < b.target.getVA();
  });
  for (const BindingEntry &b : bindings) {
    int16_t ordinal = ordinalForDylibSymbol(*b.dysym);
    if (ordinal != lastBinding.ordinal) {
      encodeDylibOrdinal(ordinal, os);
      lastBinding.ordinal = ordinal;
    }
    encodeBinding(b.dysym, b.target.isec->parent,
                  b.target.isec->outSecOff + b.target.offset, b.addend,
                  /*isWeakBinding=*/false, lastBinding, os);
  }
  if (!bindings.empty())
    os << static_cast<uint8_t>(BIND_OPCODE_DONE);
}

void BindingSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
}

WeakBindingSection::WeakBindingSection()
    : LinkEditSection(segment_names::linkEdit, section_names::weakBinding) {}

void WeakBindingSection::finalizeContents() {
  raw_svector_ostream os{contents};
  Binding lastBinding;

  for (const Defined *defined : definitions)
    encodeWeakOverride(defined, os);

  // Since bindings are delta-encoded, sorting them allows for a more compact
  // result.
  llvm::sort(bindings,
             [](const WeakBindingEntry &a, const WeakBindingEntry &b) {
               return a.target.getVA() < b.target.getVA();
             });
  for (const WeakBindingEntry &b : bindings)
    encodeBinding(b.symbol, b.target.isec->parent,
                  b.target.isec->outSecOff + b.target.offset, b.addend,
                  /*isWeakBinding=*/true, lastBinding, os);
  if (!bindings.empty() || !definitions.empty())
    os << static_cast<uint8_t>(BIND_OPCODE_DONE);
}

void WeakBindingSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
}

StubsSection::StubsSection()
    : SyntheticSection(segment_names::text, "__stubs") {
  flags = S_SYMBOL_STUBS | S_ATTR_SOME_INSTRUCTIONS | S_ATTR_PURE_INSTRUCTIONS;
  // The stubs section comprises machine instructions, which are aligned to
  // 4 bytes on the archs we care about.
  align = 4;
  reserved2 = target->stubSize;
}

uint64_t StubsSection::getSize() const {
  return entries.size() * target->stubSize;
}

void StubsSection::writeTo(uint8_t *buf) const {
  size_t off = 0;
  for (const Symbol *sym : entries) {
    target->writeStub(buf + off, *sym);
    off += target->stubSize;
  }
}

bool StubsSection::addEntry(Symbol *sym) {
  bool inserted = entries.insert(sym);
  if (inserted)
    sym->stubsIndex = entries.size() - 1;
  return inserted;
}

StubHelperSection::StubHelperSection()
    : SyntheticSection(segment_names::text, "__stub_helper") {
  flags = S_ATTR_SOME_INSTRUCTIONS | S_ATTR_PURE_INSTRUCTIONS;
  align = 4; // This section comprises machine instructions
}

uint64_t StubHelperSection::getSize() const {
  return target->stubHelperHeaderSize +
         in.lazyBinding->getEntries().size() * target->stubHelperEntrySize;
}

bool StubHelperSection::isNeeded() const { return in.lazyBinding->isNeeded(); }

void StubHelperSection::writeTo(uint8_t *buf) const {
  target->writeStubHelperHeader(buf);
  size_t off = target->stubHelperHeaderSize;
  for (const DylibSymbol *sym : in.lazyBinding->getEntries()) {
    target->writeStubHelperEntry(buf + off, *sym, addr + off);
    off += target->stubHelperEntrySize;
  }
}

void StubHelperSection::setup() {
  stubBinder = dyn_cast_or_null<DylibSymbol>(symtab->find("dyld_stub_binder"));
  if (stubBinder == nullptr) {
    error("symbol dyld_stub_binder not found (normally in libSystem.dylib). "
          "Needed to perform lazy binding.");
    return;
  }
  stubBinder->refState = RefState::Strong;
  in.got->addEntry(stubBinder);

  inputSections.push_back(in.imageLoaderCache);
  dyldPrivate = make<Defined>("__dyld_private", nullptr, in.imageLoaderCache, 0,
                              /*isWeakDef=*/false,
                              /*isExternal=*/false, /*isPrivateExtern=*/false);
}

ImageLoaderCacheSection::ImageLoaderCacheSection() {
  segname = segment_names::data;
  name = "__data";
  uint8_t *arr = bAlloc.Allocate<uint8_t>(WordSize);
  memset(arr, 0, WordSize);
  data = {arr, WordSize};
  align = WordSize;
}

LazyPointerSection::LazyPointerSection()
    : SyntheticSection(segment_names::data, "__la_symbol_ptr") {
  align = WordSize;
  flags = S_LAZY_SYMBOL_POINTERS;
}

uint64_t LazyPointerSection::getSize() const {
  return in.stubs->getEntries().size() * WordSize;
}

bool LazyPointerSection::isNeeded() const {
  return !in.stubs->getEntries().empty();
}

void LazyPointerSection::writeTo(uint8_t *buf) const {
  size_t off = 0;
  for (const Symbol *sym : in.stubs->getEntries()) {
    if (const auto *dysym = dyn_cast<DylibSymbol>(sym)) {
      if (dysym->hasStubsHelper()) {
        uint64_t stubHelperOffset =
            target->stubHelperHeaderSize +
            dysym->stubsHelperIndex * target->stubHelperEntrySize;
        write64le(buf + off, in.stubHelper->addr + stubHelperOffset);
      }
    } else {
      write64le(buf + off, sym->getVA());
    }
    off += WordSize;
  }
}

LazyBindingSection::LazyBindingSection()
    : LinkEditSection(segment_names::linkEdit, section_names::lazyBinding) {}

void LazyBindingSection::finalizeContents() {
  // TODO: Just precompute output size here instead of writing to a temporary
  // buffer
  for (DylibSymbol *sym : entries)
    sym->lazyBindOffset = encode(*sym);
}

void LazyBindingSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
}

void LazyBindingSection::addEntry(DylibSymbol *dysym) {
  if (entries.insert(dysym)) {
    dysym->stubsHelperIndex = entries.size() - 1;
    in.rebase->addEntry(in.lazyPointers->isec, dysym->stubsIndex * WordSize);
  }
}

// Unlike the non-lazy binding section, the bind opcodes in this section aren't
// interpreted all at once. Rather, dyld will start interpreting opcodes at a
// given offset, typically only binding a single symbol before it finds a
// BIND_OPCODE_DONE terminator. As such, unlike in the non-lazy-binding case,
// we cannot encode just the differences between symbols; we have to emit the
// complete bind information for each symbol.
uint32_t LazyBindingSection::encode(const DylibSymbol &sym) {
  uint32_t opstreamOffset = contents.size();
  OutputSegment *dataSeg = in.lazyPointers->parent;
  os << static_cast<uint8_t>(BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB |
                             dataSeg->index);
  uint64_t offset = in.lazyPointers->addr - dataSeg->firstSection()->addr +
                    sym.stubsIndex * WordSize;
  encodeULEB128(offset, os);
  encodeDylibOrdinal(ordinalForDylibSymbol(sym), os);

  uint8_t flags = BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM;
  if (sym.isWeakRef())
    flags |= BIND_SYMBOL_FLAGS_WEAK_IMPORT;

  os << flags << sym.getName() << '\0'
     << static_cast<uint8_t>(BIND_OPCODE_DO_BIND)
     << static_cast<uint8_t>(BIND_OPCODE_DONE);
  return opstreamOffset;
}

ExportSection::ExportSection()
    : LinkEditSection(segment_names::linkEdit, section_names::export_) {}

static void validateExportSymbol(const Defined *defined) {
  StringRef symbolName = defined->getName();
  if (defined->privateExtern && config->exportedSymbols.match(symbolName))
    error("cannot export hidden symbol " + symbolName + "\n>>> defined in " +
          toString(defined->getFile()));
}

static bool shouldExportSymbol(const Defined *defined) {
  if (defined->privateExtern)
    return false;
  // TODO: Is this a performance bottleneck? If a build has mostly
  // global symbols in the input but uses -exported_symbols to filter
  // out most of them, then it would be better to set the value of
  // privateExtern at parse time instead of calling
  // exportedSymbols.match() more than once.
  //
  // Measurements show that symbol ordering (which again looks up
  // every symbol in a hashmap) is the biggest bottleneck when linking
  // chromium_framework, so this will likely be worth optimizing.
  return config->exportedSymbols.empty()
             ? !config->unexportedSymbols.match(defined->getName())
             : config->exportedSymbols.match(defined->getName());
}

void ExportSection::finalizeContents() {
  trieBuilder.setImageBase(in.header->addr);
  for (const Symbol *sym : symtab->getSymbols()) {
    if (const auto *defined = dyn_cast<Defined>(sym)) {
      validateExportSymbol(defined);
      if (!shouldExportSymbol(defined))
        continue;
      trieBuilder.addSymbol(*defined);
      hasWeakSymbol = hasWeakSymbol || sym->isWeakDef();
    }
  }
  size = trieBuilder.build();
}

void ExportSection::writeTo(uint8_t *buf) const { trieBuilder.writeTo(buf); }

FunctionStartsSection::FunctionStartsSection()
    : LinkEditSection(segment_names::linkEdit, section_names::functionStarts) {}

void FunctionStartsSection::finalizeContents() {
  raw_svector_ostream os{contents};
  uint64_t addr = in.header->addr;
  for (const Symbol *sym : symtab->getSymbols()) {
    if (const auto *defined = dyn_cast<Defined>(sym)) {
      if (!defined->isec || !isCodeSection(defined->isec))
        continue;
      // TODO: Add support for thumbs, in that case
      // the lowest bit of nextAddr needs to be set to 1.
      uint64_t nextAddr = defined->getVA();
      uint64_t delta = nextAddr - addr;
      if (delta == 0)
        continue;
      encodeULEB128(delta, os);
      addr = nextAddr;
    }
  }
  os << '\0';
}

void FunctionStartsSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
}

SymtabSection::SymtabSection(StringTableSection &stringTableSection)
    : LinkEditSection(segment_names::linkEdit, section_names::symbolTable),
      stringTableSection(stringTableSection) {}

uint64_t SymtabSection::getRawSize() const {
  return getNumSymbols() * sizeof(structs::nlist_64);
}

void SymtabSection::emitBeginSourceStab(DWARFUnit *compileUnit) {
  StabsEntry stab(N_SO);
  SmallString<261> dir(compileUnit->getCompilationDir());
  StringRef sep = sys::path::get_separator();
  // We don't use `path::append` here because we want an empty `dir` to result
  // in an absolute path. `append` would give us a relative path for that case.
  if (!dir.endswith(sep))
    dir += sep;
  stab.strx = stringTableSection.addString(
      saver.save(dir + compileUnit->getUnitDIE().getShortName()));
  stabs.emplace_back(std::move(stab));
}

void SymtabSection::emitEndSourceStab() {
  StabsEntry stab(N_SO);
  stab.sect = 1;
  stabs.emplace_back(std::move(stab));
}

void SymtabSection::emitObjectFileStab(ObjFile *file) {
  StabsEntry stab(N_OSO);
  stab.sect = target->cpuSubtype;
  SmallString<261> path(!file->archiveName.empty() ? file->archiveName
                                                   : file->getName());
  std::error_code ec = sys::fs::make_absolute(path);
  if (ec)
    fatal("failed to get absolute path for " + path);

  if (!file->archiveName.empty())
    path.append({"(", file->getName(), ")"});

  stab.strx = stringTableSection.addString(saver.save(path.str()));
  stab.desc = 1;
  stab.value = file->modTime;
  stabs.emplace_back(std::move(stab));
}

void SymtabSection::emitEndFunStab(Defined *defined) {
  StabsEntry stab(N_FUN);
  // FIXME this should be the size of the symbol. Using the section size in
  // lieu is only correct if .subsections_via_symbols is set.
  stab.value = defined->isec->getSize();
  stabs.emplace_back(std::move(stab));
}

void SymtabSection::emitStabs() {
  std::vector<Defined *> symbolsNeedingStabs;
  for (const SymtabEntry &entry :
       concat<SymtabEntry>(localSymbols, externalSymbols)) {
    Symbol *sym = entry.sym;
    if (auto *defined = dyn_cast<Defined>(sym)) {
      if (defined->isAbsolute())
        continue;
      InputSection *isec = defined->isec;
      ObjFile *file = dyn_cast_or_null<ObjFile>(isec->file);
      if (!file || !file->compileUnit)
        continue;
      symbolsNeedingStabs.push_back(defined);
    }
  }

  llvm::stable_sort(symbolsNeedingStabs, [&](Defined *a, Defined *b) {
    return a->isec->file->id < b->isec->file->id;
  });

  // Emit STABS symbols so that dsymutil and/or the debugger can map address
  // regions in the final binary to the source and object files from which they
  // originated.
  InputFile *lastFile = nullptr;
  for (Defined *defined : symbolsNeedingStabs) {
    InputSection *isec = defined->isec;
    ObjFile *file = dyn_cast<ObjFile>(isec->file);
    assert(file);

    if (lastFile == nullptr || lastFile != file) {
      if (lastFile != nullptr)
        emitEndSourceStab();
      lastFile = file;

      emitBeginSourceStab(file->compileUnit);
      emitObjectFileStab(file);
    }

    StabsEntry symStab;
    symStab.sect = defined->isec->parent->index;
    symStab.strx = stringTableSection.addString(defined->getName());
    symStab.value = defined->getVA();

    if (isCodeSection(isec)) {
      symStab.type = N_FUN;
      stabs.emplace_back(std::move(symStab));
      emitEndFunStab(defined);
    } else {
      symStab.type = defined->isExternal() ? N_GSYM : N_STSYM;
      stabs.emplace_back(std::move(symStab));
    }
  }

  if (!stabs.empty())
    emitEndSourceStab();
}

void SymtabSection::finalizeContents() {
  auto addSymbol = [&](std::vector<SymtabEntry> &symbols, Symbol *sym) {
    uint32_t strx = stringTableSection.addString(sym->getName());
    symbols.push_back({sym, strx});
  };

  // Local symbols aren't in the SymbolTable, so we walk the list of object
  // files to gather them.
  for (const InputFile *file : inputFiles) {
    if (auto *objFile = dyn_cast<ObjFile>(file)) {
      for (Symbol *sym : objFile->symbols) {
        if (sym == nullptr)
          continue;
        // TODO: when we implement -dead_strip, we should filter out symbols
        // that belong to dead sections.
        if (auto *defined = dyn_cast<Defined>(sym)) {
          if (!defined->isExternal()) {
            StringRef name = defined->getName();
            if (!name.startswith("l") && !name.startswith("L"))
              addSymbol(localSymbols, sym);
          }
        }
      }
    }
  }

  // __dyld_private is a local symbol too. It's linker-created and doesn't
  // exist in any object file.
  if (Defined* dyldPrivate = in.stubHelper->dyldPrivate)
    addSymbol(localSymbols, dyldPrivate);

  for (Symbol *sym : symtab->getSymbols()) {
    if (auto *defined = dyn_cast<Defined>(sym)) {
      if (!defined->includeInSymtab)
        continue;
      assert(defined->isExternal());
      addSymbol(externalSymbols, defined);
    } else if (auto *dysym = dyn_cast<DylibSymbol>(sym)) {
      if (dysym->isReferenced())
        addSymbol(undefinedSymbols, sym);
    }
  }

  emitStabs();
  uint32_t symtabIndex = stabs.size();
  for (const SymtabEntry &entry :
       concat<SymtabEntry>(localSymbols, externalSymbols, undefinedSymbols)) {
    entry.sym->symtabIndex = symtabIndex++;
  }
}

uint32_t SymtabSection::getNumSymbols() const {
  return stabs.size() + localSymbols.size() + externalSymbols.size() +
         undefinedSymbols.size();
}

void SymtabSection::writeTo(uint8_t *buf) const {
  auto *nList = reinterpret_cast<structs::nlist_64 *>(buf);
  // Emit the stabs entries before the "real" symbols. We cannot emit them
  // after as that would render Symbol::symtabIndex inaccurate.
  for (const StabsEntry &entry : stabs) {
    nList->n_strx = entry.strx;
    nList->n_type = entry.type;
    nList->n_sect = entry.sect;
    nList->n_desc = entry.desc;
    nList->n_value = entry.value;
    ++nList;
  }

  for (const SymtabEntry &entry : concat<const SymtabEntry>(
           localSymbols, externalSymbols, undefinedSymbols)) {
    nList->n_strx = entry.strx;
    // TODO populate n_desc with more flags
    if (auto *defined = dyn_cast<Defined>(entry.sym)) {
      uint8_t scope = 0;
      if (!shouldExportSymbol(defined)) {
        // Private external -- dylib scoped symbol.
        // Promote to non-external at link time.
        assert(defined->isExternal() && "invalid input file");
        scope = N_PEXT;
      } else if (defined->isExternal()) {
        // Normal global symbol.
        scope = N_EXT;
      } else {
        // TU-local symbol from localSymbols.
        scope = 0;
      }

      if (defined->isAbsolute()) {
        nList->n_type = scope | N_ABS;
        nList->n_sect = NO_SECT;
        nList->n_value = defined->value;
      } else {
        nList->n_type = scope | N_SECT;
        nList->n_sect = defined->isec->parent->index;
        // For the N_SECT symbol type, n_value is the address of the symbol
        nList->n_value = defined->getVA();
      }
      nList->n_desc |= defined->isExternalWeakDef() ? N_WEAK_DEF : 0;
    } else if (auto *dysym = dyn_cast<DylibSymbol>(entry.sym)) {
      uint16_t n_desc = nList->n_desc;
      int16_t ordinal = ordinalForDylibSymbol(*dysym);
      if (ordinal == BIND_SPECIAL_DYLIB_FLAT_LOOKUP)
        SET_LIBRARY_ORDINAL(n_desc, DYNAMIC_LOOKUP_ORDINAL);
      else if (ordinal == BIND_SPECIAL_DYLIB_MAIN_EXECUTABLE)
        SET_LIBRARY_ORDINAL(n_desc, EXECUTABLE_ORDINAL);
      else {
        assert(ordinal > 0);
        SET_LIBRARY_ORDINAL(n_desc, static_cast<uint8_t>(ordinal));
      }

      nList->n_type = N_EXT;
      n_desc |= dysym->isWeakDef() ? N_WEAK_DEF : 0;
      n_desc |= dysym->isWeakRef() ? N_WEAK_REF : 0;
      nList->n_desc = n_desc;
    }
    ++nList;
  }
}

IndirectSymtabSection::IndirectSymtabSection()
    : LinkEditSection(segment_names::linkEdit,
                      section_names::indirectSymbolTable) {}

uint32_t IndirectSymtabSection::getNumSymbols() const {
  return in.got->getEntries().size() + in.tlvPointers->getEntries().size() +
         in.stubs->getEntries().size();
}

bool IndirectSymtabSection::isNeeded() const {
  return in.got->isNeeded() || in.tlvPointers->isNeeded() ||
         in.stubs->isNeeded();
}

void IndirectSymtabSection::finalizeContents() {
  uint32_t off = 0;
  in.got->reserved1 = off;
  off += in.got->getEntries().size();
  in.tlvPointers->reserved1 = off;
  off += in.tlvPointers->getEntries().size();
  // There is a 1:1 correspondence between stubs and LazyPointerSection
  // entries, so they can share the same sub-array in the table.
  in.stubs->reserved1 = in.lazyPointers->reserved1 = off;
}

static uint32_t indirectValue(const Symbol *sym) {
  return sym->symtabIndex != UINT32_MAX ? sym->symtabIndex
                                        : INDIRECT_SYMBOL_LOCAL;
}

void IndirectSymtabSection::writeTo(uint8_t *buf) const {
  uint32_t off = 0;
  for (const Symbol *sym : in.got->getEntries()) {
    write32le(buf + off * sizeof(uint32_t), indirectValue(sym));
    ++off;
  }
  for (const Symbol *sym : in.tlvPointers->getEntries()) {
    write32le(buf + off * sizeof(uint32_t), indirectValue(sym));
    ++off;
  }
  for (const Symbol *sym : in.stubs->getEntries()) {
    write32le(buf + off * sizeof(uint32_t), indirectValue(sym));
    ++off;
  }
}

StringTableSection::StringTableSection()
    : LinkEditSection(segment_names::linkEdit, section_names::stringTable) {}

uint32_t StringTableSection::addString(StringRef str) {
  uint32_t strx = size;
  strings.push_back(str); // TODO: consider deduplicating strings
  size += str.size() + 1; // account for null terminator
  return strx;
}

void StringTableSection::writeTo(uint8_t *buf) const {
  uint32_t off = 0;
  for (StringRef str : strings) {
    memcpy(buf + off, str.data(), str.size());
    off += str.size() + 1; // account for null terminator
  }
}

CodeSignatureSection::CodeSignatureSection()
    : LinkEditSection(segment_names::linkEdit, section_names::codeSignature) {
  align = 16; // required by libstuff
  fileName = config->outputFile;
  size_t slashIndex = fileName.rfind("/");
  if (slashIndex != std::string::npos)
    fileName = fileName.drop_front(slashIndex + 1);
  allHeadersSize = alignTo<16>(fixedHeadersSize + fileName.size() + 1);
  fileNamePad = allHeadersSize - fixedHeadersSize - fileName.size();
}

uint32_t CodeSignatureSection::getBlockCount() const {
  return (fileOff + blockSize - 1) / blockSize;
}

uint64_t CodeSignatureSection::getRawSize() const {
  return allHeadersSize + getBlockCount() * hashSize;
}

void CodeSignatureSection::writeHashes(uint8_t *buf) const {
  uint8_t *code = buf;
  uint8_t *codeEnd = buf + fileOff;
  uint8_t *hashes = codeEnd + allHeadersSize;
  while (code < codeEnd) {
    StringRef block(reinterpret_cast<char *>(code),
                    std::min(codeEnd - code, static_cast<ssize_t>(blockSize)));
    SHA256 hasher;
    hasher.update(block);
    StringRef hash = hasher.final();
    assert(hash.size() == hashSize);
    memcpy(hashes, hash.data(), hashSize);
    code += blockSize;
    hashes += hashSize;
  }
#if defined(__APPLE__)
  // This is macOS-specific work-around and makes no sense for any
  // other host OS. See https://openradar.appspot.com/FB8914231
  //
  // The macOS kernel maintains a signature-verification cache to
  // quickly validate applications at time of execve(2).  The trouble
  // is that for the kernel creates the cache entry at the time of the
  // mmap(2) call, before we have a chance to write either the code to
  // sign or the signature header+hashes.  The fix is to invalidate
  // all cached data associated with the output file, thus discarding
  // the bogus prematurely-cached signature.
  msync(buf, fileOff + getSize(), MS_INVALIDATE);
#endif
}

void CodeSignatureSection::writeTo(uint8_t *buf) const {
  uint32_t signatureSize = static_cast<uint32_t>(getSize());
  auto *superBlob = reinterpret_cast<CS_SuperBlob *>(buf);
  write32be(&superBlob->magic, CSMAGIC_EMBEDDED_SIGNATURE);
  write32be(&superBlob->length, signatureSize);
  write32be(&superBlob->count, 1);
  auto *blobIndex = reinterpret_cast<CS_BlobIndex *>(&superBlob[1]);
  write32be(&blobIndex->type, CSSLOT_CODEDIRECTORY);
  write32be(&blobIndex->offset, blobHeadersSize);
  auto *codeDirectory =
      reinterpret_cast<CS_CodeDirectory *>(buf + blobHeadersSize);
  write32be(&codeDirectory->magic, CSMAGIC_CODEDIRECTORY);
  write32be(&codeDirectory->length, signatureSize - blobHeadersSize);
  write32be(&codeDirectory->version, CS_SUPPORTSEXECSEG);
  write32be(&codeDirectory->flags, CS_ADHOC | CS_LINKER_SIGNED);
  write32be(&codeDirectory->hashOffset,
            sizeof(CS_CodeDirectory) + fileName.size() + fileNamePad);
  write32be(&codeDirectory->identOffset, sizeof(CS_CodeDirectory));
  codeDirectory->nSpecialSlots = 0;
  write32be(&codeDirectory->nCodeSlots, getBlockCount());
  write32be(&codeDirectory->codeLimit, fileOff);
  codeDirectory->hashSize = static_cast<uint8_t>(hashSize);
  codeDirectory->hashType = kSecCodeSignatureHashSHA256;
  codeDirectory->platform = 0;
  codeDirectory->pageSize = blockSizeShift;
  codeDirectory->spare2 = 0;
  codeDirectory->scatterOffset = 0;
  codeDirectory->teamOffset = 0;
  codeDirectory->spare3 = 0;
  codeDirectory->codeLimit64 = 0;
  OutputSegment *textSeg = getOrCreateOutputSegment(segment_names::text);
  write64be(&codeDirectory->execSegBase, textSeg->fileOff);
  write64be(&codeDirectory->execSegLimit, textSeg->fileSize);
  write64be(&codeDirectory->execSegFlags,
            config->outputType == MH_EXECUTE ? CS_EXECSEG_MAIN_BINARY : 0);
  auto *id = reinterpret_cast<char *>(&codeDirectory[1]);
  memcpy(id, fileName.begin(), fileName.size());
  memset(id + fileName.size(), 0, fileNamePad);
}

void macho::createSyntheticSymbols() {
  auto addHeaderSymbol = [](const char *name) {
    symtab->addSynthetic(name, in.header->isec, 0,
                         /*privateExtern=*/true,
                         /*includeInSymtab*/ false);
  };

  switch (config->outputType) {
    // FIXME: Assign the right addresse value for these symbols
    // (rather than 0). But we need to do that after assignAddresses().
  case MH_EXECUTE:
    // If linking PIE, __mh_execute_header is a defined symbol in
    //  __TEXT, __text)
    // Otherwise, it's an absolute symbol.
    if (config->isPic)
      symtab->addSynthetic("__mh_execute_header", in.header->isec, 0,
                           /*privateExtern*/ false,
                           /*includeInSymbtab*/ true);
    else
      symtab->addSynthetic("__mh_execute_header",
                           /*isec*/ nullptr, 0,
                           /*privateExtern*/ false,
                           /*includeInSymbtab*/ true);
    break;

    // The following symbols are  N_SECT symbols, even though the header is not
    // part of any section and that they are private to the bundle/dylib/object
    // they are part of.
  case MH_BUNDLE:
    addHeaderSymbol("__mh_bundle_header");
    break;
  case MH_DYLIB:
    addHeaderSymbol("__mh_dylib_header");
    break;
  case MH_DYLINKER:
    addHeaderSymbol("__mh_dylinker_header");
    break;
  case MH_OBJECT:
    addHeaderSymbol("__mh_object_header");
    break;
  default:
    llvm_unreachable("unexpected outputType");
    break;
  }

  // The Itanium C++ ABI requires dylibs to pass a pointer to __cxa_atexit
  // which does e.g. cleanup of static global variables. The ABI document
  // says that the pointer can point to any address in one of the dylib's
  // segments, but in practice ld64 seems to set it to point to the header,
  // so that's what's implemented here.
  addHeaderSymbol("___dso_handle");
}
