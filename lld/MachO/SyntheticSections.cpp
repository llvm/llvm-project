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

static uint32_t cpuSubtype() {
  uint32_t subtype = target->cpuSubtype;

  if (config->outputType == MachO::MH_EXECUTE && !config->staticLink &&
      target->cpuSubtype == MachO::CPU_SUBTYPE_X86_64_ALL &&
      config->platform.kind == MachO::PlatformKind::macOS &&
      config->platform.minimum >= VersionTuple(10, 5))
    subtype |= MachO::CPU_SUBTYPE_LIB64;

  return subtype;
}

void MachHeaderSection::writeTo(uint8_t *buf) const {
  auto *hdr = reinterpret_cast<MachO::mach_header_64 *>(buf);
  hdr->magic = MachO::MH_MAGIC_64;
  hdr->cputype = target->cpuType;
  hdr->cpusubtype = cpuSubtype();
  hdr->filetype = config->outputType;
  hdr->ncmds = loadCommands.size();
  hdr->sizeofcmds = sizeOfCmds;
  hdr->flags = MachO::MH_NOUNDEFS | MachO::MH_DYLDLINK | MachO::MH_TWOLEVEL;

  if (config->outputType == MachO::MH_DYLIB && !config->hasReexports)
    hdr->flags |= MachO::MH_NO_REEXPORTED_DYLIBS;

  if (config->outputType == MachO::MH_EXECUTE && config->isPic)
    hdr->flags |= MachO::MH_PIE;

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

uint64_t Location::getVA() const {
  if (const auto *isec = section.dyn_cast<const InputSection *>())
    return isec->getVA() + offset;
  return section.get<const OutputSection *>()->addr + offset;
}

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
  using namespace llvm::MachO;
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
  using namespace llvm::MachO;
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
  using namespace llvm::MachO;
  if (locations.empty())
    return;

  raw_svector_ostream os{contents};
  Rebase lastRebase;

  os << static_cast<uint8_t>(REBASE_OPCODE_SET_TYPE_IMM | REBASE_TYPE_POINTER);

  llvm::sort(locations, [](const Location &a, const Location &b) {
    return a.getVA() < b.getVA();
  });
  for (const Location &loc : locations) {
    if (const auto *isec = loc.section.dyn_cast<const InputSection *>()) {
      encodeRebase(isec->parent, isec->outSecOff + loc.offset, lastRebase, os);
    } else {
      const auto *osec = loc.section.get<const OutputSection *>();
      encodeRebase(osec, loc.offset, lastRebase, os);
    }
  }
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
  align = WordSize; // vector of pointers / mimic ld64
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
                          bool isWeakBinding, Binding &lastBinding,
                          raw_svector_ostream &os) {
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
static void encodeDylibOrdinal(const DylibSymbol *dysym, Binding &lastBinding,
                               raw_svector_ostream &os) {
  using namespace llvm::MachO;
  if (lastBinding.ordinal != dysym->getFile()->ordinal) {
    if (dysym->getFile()->ordinal <= BIND_IMMEDIATE_MASK) {
      os << static_cast<uint8_t>(BIND_OPCODE_SET_DYLIB_ORDINAL_IMM |
                                 dysym->getFile()->ordinal);
    } else {
      os << static_cast<uint8_t>(BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB);
      encodeULEB128(dysym->getFile()->ordinal, os);
    }
    lastBinding.ordinal = dysym->getFile()->ordinal;
  }
}

static void encodeWeakOverride(const Defined *defined,
                               raw_svector_ostream &os) {
  using namespace llvm::MachO;
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
    encodeDylibOrdinal(b.dysym, lastBinding, os);
    if (auto *isec = b.target.section.dyn_cast<const InputSection *>()) {
      encodeBinding(b.dysym, isec->parent, isec->outSecOff + b.target.offset,
                    b.addend, /*isWeakBinding=*/false, lastBinding, os);
    } else {
      auto *osec = b.target.section.get<const OutputSection *>();
      encodeBinding(b.dysym, osec, b.target.offset, b.addend,
                    /*isWeakBinding=*/false, lastBinding, os);
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
                    b.addend, /*isWeakBinding=*/true, lastBinding, os);
    } else {
      auto *osec = b.target.section.get<const OutputSection *>();
      encodeBinding(b.symbol, osec, b.target.offset, b.addend,
                    /*isWeakBinding=*/true, lastBinding, os);
    }
  }
  if (!bindings.empty() || !definitions.empty())
    os << static_cast<uint8_t>(MachO::BIND_OPCODE_DONE);
}

void WeakBindingSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
}

bool macho::needsBinding(const Symbol *sym) {
  if (isa<DylibSymbol>(sym))
    return true;
  if (const auto *defined = dyn_cast<Defined>(sym))
    return defined->isExternalWeakDef();
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
    in.rebase->addEntry(section, offset);
    if (defined->isExternalWeakDef())
      in.weakBinding->addEntry(sym, section, offset, addend);
  } else if (!isa<DSOHandle>(sym)) {
    // Undefined symbols are filtered out in scanRelocations(); we should never
    // get here
    llvm_unreachable("cannot bind to an undefined symbol");
  }
  // TODO: understand the DSOHandle case better.
  // Is it bindable?  Add a new test?
}

StubsSection::StubsSection()
    : SyntheticSection(segment_names::text, "__stubs") {
  flags = MachO::S_SYMBOL_STUBS | MachO::S_ATTR_SOME_INSTRUCTIONS |
          MachO::S_ATTR_PURE_INSTRUCTIONS;
  align = 4; // machine instructions / mimic ld64
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
  flags = MachO::S_ATTR_SOME_INSTRUCTIONS | MachO::S_ATTR_PURE_INSTRUCTIONS;
  align = 4; // machine instructions / mimic ld64
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
  align = WordSize; // pointer / mimic ld64
}

LazyPointerSection::LazyPointerSection()
    : SyntheticSection(segment_names::data, "__la_symbol_ptr") {
  align = WordSize; // vector of pointers / mimic ld64
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
  if (entries.insert(dysym)) {
    dysym->stubsHelperIndex = entries.size() - 1;
    in.rebase->addEntry(in.lazyPointers, dysym->stubsIndex * WordSize);
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
  os << static_cast<uint8_t>(MachO::BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB |
                             dataSeg->index);
  uint64_t offset = in.lazyPointers->addr - dataSeg->firstSection()->addr +
                    sym.stubsIndex * WordSize;
  encodeULEB128(offset, os);
  if (sym.getFile()->ordinal <= MachO::BIND_IMMEDIATE_MASK) {
    os << static_cast<uint8_t>(MachO::BIND_OPCODE_SET_DYLIB_ORDINAL_IMM |
                               sym.getFile()->ordinal);
  } else {
    os << static_cast<uint8_t>(MachO::BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB);
    encodeULEB128(sym.getFile()->ordinal, os);
  }

  uint8_t flags = MachO::BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM;
  if (sym.isWeakRef())
    flags |= MachO::BIND_SYMBOL_FLAGS_WEAK_IMPORT;

  os << flags << sym.getName() << '\0'
     << static_cast<uint8_t>(MachO::BIND_OPCODE_DO_BIND)
     << static_cast<uint8_t>(MachO::BIND_OPCODE_DONE);
  return opstreamOffset;
}

void macho::prepareBranchTarget(Symbol *sym) {
  if (auto *dysym = dyn_cast<DylibSymbol>(sym)) {
    if (in.stubs->addEntry(dysym)) {
      if (sym->isWeakDef()) {
        in.binding->addEntry(dysym, in.lazyPointers,
                             sym->stubsIndex * WordSize);
        in.weakBinding->addEntry(sym, in.lazyPointers,
                                 sym->stubsIndex * WordSize);
      } else {
        in.lazyBinding->addEntry(dysym);
      }
    }
  } else if (auto *defined = dyn_cast<Defined>(sym)) {
    if (defined->isExternalWeakDef()) {
      if (in.stubs->addEntry(sym)) {
        in.rebase->addEntry(in.lazyPointers, sym->stubsIndex * WordSize);
        in.weakBinding->addEntry(sym, in.lazyPointers,
                                 sym->stubsIndex * WordSize);
      }
    }
  }
}

ExportSection::ExportSection()
    : LinkEditSection(segment_names::linkEdit, section_names::export_) {}

void ExportSection::finalizeContents() {
  trieBuilder.setImageBase(in.header->addr);
  for (const Symbol *sym : symtab->getSymbols()) {
    if (const auto *defined = dyn_cast<Defined>(sym)) {
      if (defined->privateExtern)
        continue;
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
  return getNumSymbols() * sizeof(structs::nlist_64);
}

void SymtabSection::emitBeginSourceStab(DWARFUnit *compileUnit) {
  StabsEntry stab(MachO::N_SO);
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
  StabsEntry stab(MachO::N_SO);
  stab.sect = 1;
  stabs.emplace_back(std::move(stab));
}

void SymtabSection::emitObjectFileStab(ObjFile *file) {
  StabsEntry stab(MachO::N_OSO);
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
  StabsEntry stab(MachO::N_FUN);
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
      symStab.type = MachO::N_FUN;
      stabs.emplace_back(std::move(symStab));
      emitEndFunStab(defined);
    } else {
      symStab.type = defined->isExternal() ? MachO::N_GSYM : MachO::N_STSYM;
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
  for (InputFile *file : inputFiles) {
    if (auto *objFile = dyn_cast<ObjFile>(file)) {
      for (Symbol *sym : objFile->symbols) {
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
      assert(defined->isExternal());
      (void)defined;
      addSymbol(externalSymbols, sym);
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
      if (defined->privateExtern) {
        // Private external -- dylib scoped symbol.
        // Promote to non-external at link time.
        assert(defined->isExternal() && "invalid input file");
        scope = MachO::N_PEXT;
      } else if (defined->isExternal()) {
        // Normal global symbol.
        scope = MachO::N_EXT;
      } else {
        // TU-local symbol from localSymbols.
        scope = 0;
      }

      if (defined->isAbsolute()) {
        nList->n_type = scope | MachO::N_ABS;
        nList->n_sect = MachO::NO_SECT;
        nList->n_value = defined->value;
      } else {
        nList->n_type = scope | MachO::N_SECT;
        nList->n_sect = defined->isec->parent->index;
        // For the N_SECT symbol type, n_value is the address of the symbol
        nList->n_value = defined->getVA();
      }
      nList->n_desc |= defined->isExternalWeakDef() ? MachO::N_WEAK_DEF : 0;
    } else if (auto *dysym = dyn_cast<DylibSymbol>(entry.sym)) {
      uint16_t n_desc = nList->n_desc;
      MachO::SET_LIBRARY_ORDINAL(n_desc, dysym->getFile()->ordinal);
      nList->n_type = MachO::N_EXT;
      n_desc |= dysym->isWeakRef() ? MachO::N_WEAK_REF : 0;
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
                                        : MachO::INDIRECT_SYMBOL_LOCAL;
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
