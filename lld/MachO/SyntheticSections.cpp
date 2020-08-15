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

GotSection::GotSection()
    : SyntheticSection(segment_names::dataConst, section_names::got) {
  align = 8;
  flags = MachO::S_NON_LAZY_SYMBOL_POINTERS;

  // TODO: section_64::reserved1 should be an index into the indirect symbol
  // table, which we do not currently emit
}

void GotSection::addEntry(Symbol &sym) {
  if (entries.insert(&sym)) {
    sym.gotIndex = entries.size() - 1;
  }
}

void GotSection::writeTo(uint8_t *buf) const {
  for (size_t i = 0, n = entries.size(); i < n; ++i)
    if (auto *defined = dyn_cast<Defined>(entries[i]))
      write64le(&buf[i * WordSize], defined->getVA());
}

BindingSection::BindingSection()
    : LinkEditSection(segment_names::linkEdit, section_names::binding) {}

bool BindingSection::isNeeded() const {
  return bindings.size() != 0 || in.got->isNeeded();
}

namespace {
struct Binding {
  OutputSegment *segment = nullptr;
  uint64_t offset = 0;
  int64_t addend = 0;
  uint8_t ordinal = 0;
};
} // namespace

// Encode a sequence of opcodes that tell dyld to write the address of dysym +
// addend at osec->addr + outSecOff.
//
// The bind opcode "interpreter" remembers the values of each binding field, so
// we only need to encode the differences between bindings. Hence the use of
// lastBinding.
static void encodeBinding(const DylibSymbol &dysym, const OutputSection *osec,
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
    assert(lastBinding.offset <= offset);
    os << static_cast<uint8_t>(BIND_OPCODE_ADD_ADDR_ULEB);
    encodeULEB128(offset - lastBinding.offset, os);
    lastBinding.offset = offset;
  }

  if (lastBinding.ordinal != dysym.file->ordinal) {
    if (dysym.file->ordinal <= BIND_IMMEDIATE_MASK) {
      os << static_cast<uint8_t>(BIND_OPCODE_SET_DYLIB_ORDINAL_IMM |
                                 dysym.file->ordinal);
    } else {
      error("TODO: Support larger dylib symbol ordinals");
      return;
    }
    lastBinding.ordinal = dysym.file->ordinal;
  }

  if (lastBinding.addend != addend) {
    os << static_cast<uint8_t>(BIND_OPCODE_SET_ADDEND_SLEB);
    encodeSLEB128(addend, os);
    lastBinding.addend = addend;
  }

  os << static_cast<uint8_t>(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM)
     << dysym.getName() << '\0'
     << static_cast<uint8_t>(BIND_OPCODE_SET_TYPE_IMM | BIND_TYPE_POINTER)
     << static_cast<uint8_t>(BIND_OPCODE_DO_BIND);
  // DO_BIND causes dyld to both perform the binding and increment the offset
  lastBinding.offset += WordSize;
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
  bool didEncode = false;
  size_t gotIdx = 0;
  for (const Symbol *sym : in.got->getEntries()) {
    if (const auto *dysym = dyn_cast<DylibSymbol>(sym)) {
      didEncode = true;
      encodeBinding(*dysym, in.got, gotIdx * WordSize, 0, lastBinding, os);
    }
    ++gotIdx;
  }

  // Sorting the relocations by segment and address allows us to encode them
  // more compactly.
  llvm::sort(bindings, [](const BindingEntry &a, const BindingEntry &b) {
    OutputSegment *segA = a.isec->parent->parent;
    OutputSegment *segB = b.isec->parent->parent;
    if (segA != segB)
      return segA->fileOff < segB->fileOff;
    OutputSection *osecA = a.isec->parent;
    OutputSection *osecB = b.isec->parent;
    if (osecA != osecB)
      return osecA->addr < osecB->addr;
    if (a.isec != b.isec)
      return a.isec->outSecOff < b.isec->outSecOff;
    return a.offset < b.offset;
  });
  for (const BindingEntry &b : bindings) {
    didEncode = true;
    encodeBinding(*b.dysym, b.isec->parent, b.isec->outSecOff + b.offset,
                  b.addend, lastBinding, os);
  }
  if (didEncode)
    os << static_cast<uint8_t>(MachO::BIND_OPCODE_DONE);
}

void BindingSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
}

StubsSection::StubsSection()
    : SyntheticSection(segment_names::text, "__stubs") {}

uint64_t StubsSection::getSize() const {
  return entries.size() * target->stubSize;
}

void StubsSection::writeTo(uint8_t *buf) const {
  size_t off = 0;
  for (const DylibSymbol *sym : in.stubs->getEntries()) {
    target->writeStub(buf + off, *sym);
    off += target->stubSize;
  }
}

void StubsSection::addEntry(DylibSymbol &sym) {
  if (entries.insert(&sym))
    sym.stubsIndex = entries.size() - 1;
}

StubHelperSection::StubHelperSection()
    : SyntheticSection(segment_names::text, "__stub_helper") {}

uint64_t StubHelperSection::getSize() const {
  return target->stubHelperHeaderSize +
         in.stubs->getEntries().size() * target->stubHelperEntrySize;
}

bool StubHelperSection::isNeeded() const {
  return !in.stubs->getEntries().empty();
}

void StubHelperSection::writeTo(uint8_t *buf) const {
  target->writeStubHelperHeader(buf);
  size_t off = target->stubHelperHeaderSize;
  for (const DylibSymbol *sym : in.stubs->getEntries()) {
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
  in.got->addEntry(*stubBinder);

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
  for (const DylibSymbol *sym : in.stubs->getEntries()) {
    uint64_t stubHelperOffset = target->stubHelperHeaderSize +
                                sym->stubsIndex * target->stubHelperEntrySize;
    write64le(buf + off, in.stubHelper->addr + stubHelperOffset);
    off += WordSize;
  }
}

LazyBindingSection::LazyBindingSection()
    : LinkEditSection(segment_names::linkEdit, section_names::lazyBinding) {}

bool LazyBindingSection::isNeeded() const { return in.stubs->isNeeded(); }

void LazyBindingSection::finalizeContents() {
  // TODO: Just precompute output size here instead of writing to a temporary
  // buffer
  for (DylibSymbol *sym : in.stubs->getEntries())
    sym->lazyBindOffset = encode(*sym);
}

void LazyBindingSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
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
  if (sym.file->ordinal <= MachO::BIND_IMMEDIATE_MASK)
    os << static_cast<uint8_t>(MachO::BIND_OPCODE_SET_DYLIB_ORDINAL_IMM |
                               sym.file->ordinal);
  else
    fatal("TODO: Support larger dylib symbol ordinals");

  os << static_cast<uint8_t>(MachO::BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM)
     << sym.getName() << '\0'
     << static_cast<uint8_t>(MachO::BIND_OPCODE_DO_BIND)
     << static_cast<uint8_t>(MachO::BIND_OPCODE_DONE);
  return opstreamOffset;
}

ExportSection::ExportSection()
    : LinkEditSection(segment_names::linkEdit, section_names::export_) {}

void ExportSection::finalizeContents() {
  // TODO: We should check symbol visibility.
  for (const Symbol *sym : symtab->getSymbols())
    if (auto *defined = dyn_cast<Defined>(sym))
      trieBuilder.addSymbol(*defined);
  size = trieBuilder.build();
}

void ExportSection::writeTo(uint8_t *buf) const { trieBuilder.writeTo(buf); }

SymtabSection::SymtabSection(StringTableSection &stringTableSection)
    : SyntheticSection(segment_names::linkEdit, section_names::symbolTable),
      stringTableSection(stringTableSection) {}

uint64_t SymtabSection::getSize() const {
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
    // TODO populate n_desc
    if (auto *defined = dyn_cast<Defined>(entry.sym)) {
      nList->n_type = MachO::N_EXT | MachO::N_SECT;
      nList->n_sect = defined->isec->parent->index;
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
