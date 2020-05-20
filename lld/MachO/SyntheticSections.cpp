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
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "Writer.h"

#include "lld/Common/ErrorHandler.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::support;
using namespace llvm::support::endian;

namespace lld {
namespace macho {

SyntheticSection::SyntheticSection(const char *segname, const char *name)
    : OutputSection(SyntheticKind, name) {
  // Synthetic sections always know which segment they belong to so hook
  // them up when they're made
  getOrCreateOutputSegment(segname)->addOutputSection(this);
}

// dyld3's MachOLoaded::getSlide() assumes that the __TEXT segment starts
// from the beginning of the file (i.e. the header).
MachHeaderSection::MachHeaderSection()
    : SyntheticSection(segment_names::text, section_names::header) {}

void MachHeaderSection::addLoadCommand(LoadCommand *lc) {
  loadCommands.push_back(lc);
  sizeOfCmds += lc->getSize();
}

size_t MachHeaderSection::getSize() const {
  return sizeof(mach_header_64) + sizeOfCmds;
}

void MachHeaderSection::writeTo(uint8_t *buf) const {
  auto *hdr = reinterpret_cast<mach_header_64 *>(buf);
  hdr->magic = MH_MAGIC_64;
  hdr->cputype = CPU_TYPE_X86_64;
  hdr->cpusubtype = CPU_SUBTYPE_X86_64_ALL | CPU_SUBTYPE_LIB64;
  hdr->filetype = config->outputType;
  hdr->ncmds = loadCommands.size();
  hdr->sizeofcmds = sizeOfCmds;
  hdr->flags = MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL;
  if (config->outputType == MH_DYLIB && !config->hasReexports)
    hdr->flags |= MH_NO_REEXPORTED_DYLIBS;

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
  flags = S_NON_LAZY_SYMBOL_POINTERS;

  // TODO: section_64::reserved1 should be an index into the indirect symbol
  // table, which we do not currently emit
}

void GotSection::addEntry(DylibSymbol &sym) {
  if (entries.insert(&sym)) {
    sym.gotIndex = entries.size() - 1;
  }
}

BindingSection::BindingSection()
    : SyntheticSection(segment_names::linkEdit, section_names::binding) {}

bool BindingSection::isNeeded() const { return in.got->isNeeded(); }

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
  if (!isNeeded())
    return;

  raw_svector_ostream os{contents};
  os << static_cast<uint8_t>(BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB |
                             in.got->parent->index);
  encodeULEB128(in.got->getSegmentOffset(), os);
  for (const DylibSymbol *sym : in.got->getEntries()) {
    // TODO: Implement compact encoding -- we only need to encode the
    // differences between consecutive symbol entries.
    if (sym->file->ordinal <= BIND_IMMEDIATE_MASK) {
      os << static_cast<uint8_t>(BIND_OPCODE_SET_DYLIB_ORDINAL_IMM |
                                 sym->file->ordinal);
    } else {
      error("TODO: Support larger dylib symbol ordinals");
      continue;
    }
    os << static_cast<uint8_t>(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM)
       << sym->getName() << '\0'
       << static_cast<uint8_t>(BIND_OPCODE_SET_TYPE_IMM | BIND_TYPE_POINTER)
       << static_cast<uint8_t>(BIND_OPCODE_DO_BIND);
  }

  os << static_cast<uint8_t>(BIND_OPCODE_DONE);
}

void BindingSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
}

StubsSection::StubsSection()
    : SyntheticSection(segment_names::text, "__stubs") {}

size_t StubsSection::getSize() const {
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

size_t StubHelperSection::getSize() const {
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
  symtab->addDefined("__dyld_private", in.imageLoaderCache, 0);
}

ImageLoaderCacheSection::ImageLoaderCacheSection() {
  segname = segment_names::data;
  name = "__data";
}

LazyPointerSection::LazyPointerSection()
    : SyntheticSection(segment_names::data, "__la_symbol_ptr") {
  align = 8;
  flags = S_LAZY_SYMBOL_POINTERS;
}

size_t LazyPointerSection::getSize() const {
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
    : SyntheticSection(segment_names::linkEdit, section_names::lazyBinding) {}

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
  os << static_cast<uint8_t>(BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB |
                             dataSeg->index);
  uint64_t offset = in.lazyPointers->addr - dataSeg->firstSection()->addr +
                    sym.stubsIndex * WordSize;
  encodeULEB128(offset, os);
  if (sym.file->ordinal <= BIND_IMMEDIATE_MASK)
    os << static_cast<uint8_t>(BIND_OPCODE_SET_DYLIB_ORDINAL_IMM |
                               sym.file->ordinal);
  else
    fatal("TODO: Support larger dylib symbol ordinals");

  os << static_cast<uint8_t>(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM)
     << sym.getName() << '\0' << static_cast<uint8_t>(BIND_OPCODE_DO_BIND)
     << static_cast<uint8_t>(BIND_OPCODE_DONE);
  return opstreamOffset;
}

ExportSection::ExportSection()
    : SyntheticSection(segment_names::linkEdit, section_names::export_) {}

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
      stringTableSection(stringTableSection) {
  // TODO: When we introduce the SyntheticSections superclass, we should make
  // all synthetic sections aligned to WordSize by default.
  align = WordSize;
}

size_t SymtabSection::getSize() const {
  return symbols.size() * sizeof(nlist_64);
}

void SymtabSection::finalizeContents() {
  // TODO support other symbol types
  for (Symbol *sym : symtab->getSymbols())
    if (isa<Defined>(sym))
      symbols.push_back({sym, stringTableSection.addString(sym->getName())});
}

void SymtabSection::writeTo(uint8_t *buf) const {
  auto *nList = reinterpret_cast<nlist_64 *>(buf);
  for (const SymtabEntry &entry : symbols) {
    nList->n_strx = entry.strx;
    // TODO support other symbol types
    // TODO populate n_desc
    if (auto *defined = dyn_cast<Defined>(entry.sym)) {
      nList->n_type = N_EXT | N_SECT;
      nList->n_sect = defined->isec->parent->index;
      // For the N_SECT symbol type, n_value is the address of the symbol
      nList->n_value = defined->value + defined->isec->getVA();
    }
    ++nList;
  }
}

StringTableSection::StringTableSection()
    : SyntheticSection(segment_names::linkEdit, section_names::stringTable) {}

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

InStruct in;

} // namespace macho
} // namespace lld
