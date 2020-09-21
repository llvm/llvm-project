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
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;
using namespace llvm::support;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::macho;

InStruct macho::in;
std::vector<SyntheticSection *> macho::syntheticSections;

SyntheticSection::SyntheticSection(const char *segname, const char *name)
    : OutputSection(SyntheticKind, name), segname(segname) {
  syntheticSections.push_back(this);
}

// dyld3's MachOLoaded::getSlide() assumes that the __TEXT segment starts
// from the beginning of the file (i.e. the header).
MachHeaderSection::MachHeaderSection()
    : SyntheticSection(segment_names::text, section_names::header) {}

void MachHeaderSection::addLoadCommand(LoadCommand *lc) {
  loadCommands.push_back(lc);
  sizeOfCmds += lc->getSize();
}

uint64_t MachHeaderSection::getSize() const {
  return sizeof(MachO::mach_header_64) + sizeOfCmds + config->headerPad;
}

void MachHeaderSection::writeTo(uint8_t *buf) const {
  auto *hdr = reinterpret_cast<MachO::mach_header_64 *>(buf);
  hdr->magic = MachO::MH_MAGIC_64;
  hdr->cputype = MachO::CPU_TYPE_X86_64;
  hdr->cpusubtype = MachO::CPU_SUBTYPE_X86_64_ALL | MachO::CPU_SUBTYPE_LIB64;
  hdr->filetype = config->outputType;
  hdr->ncmds = loadCommands.size();
  hdr->sizeofcmds = sizeOfCmds;
  hdr->flags = MachO::MH_NOUNDEFS | MachO::MH_DYLDLINK | MachO::MH_TWOLEVEL;

  if (config->outputType == MachO::MH_DYLIB && !config->hasReexports)
    hdr->flags |= MachO::MH_NO_REEXPORTED_DYLIBS;

  if (in.exports->hasWeakSymbol || in.weakBinding->hasNonWeakDefinition())
    hdr->flags |= MachO::MH_WEAK_DEFINES;

  if (in.exports->hasWeakSymbol || in.weakBinding->hasEntry())
    hdr->flags |= MachO::MH_BINDS_TO_WEAK;

  for (OutputSegment *seg : outputSegments) {
    for (OutputSection *osec : seg->getSections()) {
      if (isThreadLocalVariables(osec->flags)) {
        hdr->flags |= MachO::MH_HAS_TLV_DESCRIPTORS;
        break;
      }
    }
  }

  uint8_t *p = reinterpret_cast<uint8_t *>(hdr + 1);
  for (LoadCommand *lc : loadCommands) {
    lc->writeTo(p);
    p += lc->getSize();
  }
}

PageZeroSection::PageZeroSection()
    : SyntheticSection(segment_names::pageZero, section_names::pageZero) {}

NonLazyPointerSectionBase::NonLazyPointerSectionBase(const char *segname,
                                                     const char *name)
    : SyntheticSection(segname, name) {
  align = 8;
  flags = MachO::S_NON_LAZY_SYMBOL_POINTERS;
}

void NonLazyPointerSectionBase::addEntry(Symbol *sym) {
  if (entries.insert(sym)) {
    assert(!sym->isInGot());
    sym->gotIndex = entries.size() - 1;

    addNonLazyBindingEntries(sym, this, sym->gotIndex * WordSize);
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
  uint8_t ordinal = 0;
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
                          Binding &lastBinding, raw_svector_ostream &os) {
  using namespace llvm::MachO;
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

  os << static_cast<uint8_t>(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM)
     << sym->getName() << '\0'
     << static_cast<uint8_t>(BIND_OPCODE_SET_TYPE_IMM | BIND_TYPE_POINTER)
     << static_cast<uint8_t>(BIND_OPCODE_DO_BIND);
  // DO_BIND causes dyld to both perform the binding and increment the offset
  lastBinding.offset += WordSize;
}

// Non-weak bindings need to have their dylib ordinal encoded as well.
static void encodeDylibOrdinal(const DylibSymbol *dysym, Binding &lastBinding,
                               raw_svector_ostream &os) {
  using namespace llvm::MachO;
  if (lastBinding.ordinal != dysym->file->ordinal) {
    if (dysym->file->ordinal <= BIND_IMMEDIATE_MASK) {
      os << static_cast<uint8_t>(BIND_OPCODE_SET_DYLIB_ORDINAL_IMM |
                                 dysym->file->ordinal);
    } else {
      os << static_cast<uint8_t>(BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB);
      encodeULEB128(dysym->file->ordinal, os);
    }
    lastBinding.ordinal = dysym->file->ordinal;
  }
}

static void encodeWeakOverride(const Defined *defined,
                               raw_svector_ostream &os) {
  using namespace llvm::MachO;
  os << static_cast<uint8_t>(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM |
                             BIND_SYMBOL_FLAGS_NON_WEAK_DEFINITION)
     << defined->getName() << '\0';
}

uint64_t BindingTarget::getVA() const {
  if (auto *isec = section.dyn_cast<const InputSection *>())
    return isec->getVA() + offset;
  return section.get<const OutputSection *>()->addr + offset;
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
    encodeDylibOrdinal(b.dysym, lastBinding, os);
    if (auto *isec = b.target.section.dyn_cast<const InputSection *>()) {
      encodeBinding(b.dysym, isec->parent, isec->outSecOff + b.target.offset,
                    b.target.addend, lastBinding, os);
    } else {
      auto *osec = b.target.section.get<const OutputSection *>();
      encodeBinding(b.dysym, osec, b.target.offset, b.target.addend,
                    lastBinding, os);
    }
  }
  if (!bindings.empty())
    os << static_cast<uint8_t>(MachO::BIND_OPCODE_DONE);
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
  for (const WeakBindingEntry &b : bindings) {
    if (auto *isec = b.target.section.dyn_cast<const InputSection *>()) {
      encodeBinding(b.symbol, isec->parent, isec->outSecOff + b.target.offset,
                    b.target.addend, lastBinding, os);
    } else {
      auto *osec = b.target.section.get<const OutputSection *>();
      encodeBinding(b.symbol, osec, b.target.offset, b.target.addend,
                    lastBinding, os);
    }
  }
  if (!bindings.empty() || !definitions.empty())
    os << static_cast<uint8_t>(MachO::BIND_OPCODE_DONE);
}

void WeakBindingSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
}

bool macho::needsBinding(const Symbol *sym) {
  if (isa<DylibSymbol>(sym)) {
    return true;
  } else if (const auto *defined = dyn_cast<Defined>(sym)) {
    if (defined->isWeakDef() && defined->isExternal())
      return true;
  }
  return false;
}

void macho::addNonLazyBindingEntries(const Symbol *sym,
                                     SectionPointerUnion section,
                                     uint64_t offset, int64_t addend) {
  if (auto *dysym = dyn_cast<DylibSymbol>(sym)) {
    in.binding->addEntry(dysym, section, offset, addend);
    if (dysym->isWeakDef())
      in.weakBinding->addEntry(sym, section, offset, addend);
  } else if (auto *defined = dyn_cast<Defined>(sym)) {
    if (defined->isWeakDef() && defined->isExternal())
      in.weakBinding->addEntry(sym, section, offset, addend);
  } else if (isa<DSOHandle>(sym)) {
    error("cannot bind to " + DSOHandle::name);
  } else {
    // Undefined symbols are filtered out in scanRelocations(); we should never
    // get here
    llvm_unreachable("cannot bind to an undefined symbol");
  }
}

StubsSection::StubsSection()
    : SyntheticSection(segment_names::text, "__stubs") {}

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
    : SyntheticSection(segment_names::text, "__stub_helper") {}

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
  in.got->addEntry(stubBinder);

  inputSections.push_back(in.imageLoaderCache);
  symtab->addDefined("__dyld_private", in.imageLoaderCache, 0,
                     /*isWeakDef=*/false);
}

ImageLoaderCacheSection::ImageLoaderCacheSection() {
  segname = segment_names::data;
  name = "__data";
  uint8_t *arr = bAlloc.Allocate<uint8_t>(WordSize);
  memset(arr, 0, WordSize);
  data = {arr, WordSize};
}

LazyPointerSection::LazyPointerSection()
    : SyntheticSection(segment_names::data, "__la_symbol_ptr") {
  align = 8;
  flags = MachO::S_LAZY_SYMBOL_POINTERS;
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
  if (entries.insert(dysym))
    dysym->stubsHelperIndex = entries.size() - 1;
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
  os << static_cast<uint8_t>(MachO::BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB |
                             dataSeg->index);
  uint64_t offset = in.lazyPointers->addr - dataSeg->firstSection()->addr +
                    sym.stubsIndex * WordSize;
  encodeULEB128(offset, os);
  if (sym.file->ordinal <= MachO::BIND_IMMEDIATE_MASK) {
    os << static_cast<uint8_t>(MachO::BIND_OPCODE_SET_DYLIB_ORDINAL_IMM |
                               sym.file->ordinal);
  } else {
    os << static_cast<uint8_t>(MachO::BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB);
    encodeULEB128(sym.file->ordinal, os);
  }

  os << static_cast<uint8_t>(MachO::BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM)
     << sym.getName() << '\0'
     << static_cast<uint8_t>(MachO::BIND_OPCODE_DO_BIND)
     << static_cast<uint8_t>(MachO::BIND_OPCODE_DONE);
  return opstreamOffset;
}

ExportSection::ExportSection()
    : LinkEditSection(segment_names::linkEdit, section_names::export_) {}

void ExportSection::finalizeContents() {
  trieBuilder.setImageBase(in.header->addr);
  // TODO: We should check symbol visibility.
  for (const Symbol *sym : symtab->getSymbols()) {
    if (const auto *defined = dyn_cast<Defined>(sym)) {
      trieBuilder.addSymbol(*defined);
      hasWeakSymbol = hasWeakSymbol || sym->isWeakDef();
    }
  }
  size = trieBuilder.build();
}

void ExportSection::writeTo(uint8_t *buf) const { trieBuilder.writeTo(buf); }

SymtabSection::SymtabSection(StringTableSection &stringTableSection)
    : LinkEditSection(segment_names::linkEdit, section_names::symbolTable),
      stringTableSection(stringTableSection) {}

uint64_t SymtabSection::getRawSize() const {
  return symbols.size() * sizeof(structs::nlist_64);
}

void SymtabSection::finalizeContents() {
  // TODO support other symbol types
  for (Symbol *sym : symtab->getSymbols())
    if (isa<Defined>(sym))
      symbols.push_back({sym, stringTableSection.addString(sym->getName())});
}

void SymtabSection::writeTo(uint8_t *buf) const {
  auto *nList = reinterpret_cast<structs::nlist_64 *>(buf);
  for (const SymtabEntry &entry : symbols) {
    nList->n_strx = entry.strx;
    // TODO support other symbol types
    // TODO populate n_desc with more flags
    if (auto *defined = dyn_cast<Defined>(entry.sym)) {
      nList->n_type = MachO::N_EXT | MachO::N_SECT;
      nList->n_sect = defined->isec->parent->index;
      nList->n_desc |= defined->isWeakDef() ? MachO::N_WEAK_DEF : 0;
      // For the N_SECT symbol type, n_value is the address of the symbol
      nList->n_value = defined->value + defined->isec->getVA();
    }
    ++nList;
  }
}

StringTableSection::StringTableSection()
    : LinkEditSection(segment_names::linkEdit, section_names::stringTable) {}

uint32_t StringTableSection::addString(StringRef str) {
  uint32_t strx = size;
  strings.push_back(str);
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
