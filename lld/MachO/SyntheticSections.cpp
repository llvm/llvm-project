//===- SyntheticSections.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SyntheticSections.h"
#include "ConcatOutputSection.h"
#include "Config.h"
#include "ExportTrie.h"
#include "InputFiles.h"
#include "MachOStructs.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#if defined(__APPLE__)
#include <sys/mman.h>
#endif

#ifdef LLVM_HAVE_LIBXAR
#include <fcntl.h>
extern "C" {
#include <xar/xar.h>
}
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
    : OutputSection(SyntheticKind, name) {
  std::tie(this->segname, this->name) = maybeRenameSection({segname, name});
  isec = make<ConcatInputSection>(segname, name);
  isec->parent = this;
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
  isec->isFinal = true;
}

void MachHeaderSection::addLoadCommand(LoadCommand *lc) {
  loadCommands.push_back(lc);
  sizeOfCmds += lc->getSize();
}

uint64_t MachHeaderSection::getSize() const {
  uint64_t size = target->headerSize + sizeOfCmds + config->headerPad;
  // If we are emitting an encryptable binary, our load commands must have a
  // separate (non-encrypted) page to themselves.
  if (config->emitEncryptionInfo)
    size = alignTo(size, target->getPageSize());
  return size;
}

static uint32_t cpuSubtype() {
  uint32_t subtype = target->cpuSubtype;

  if (config->outputType == MH_EXECUTE && !config->staticLink &&
      target->cpuSubtype == CPU_SUBTYPE_X86_64_ALL &&
      config->platform() == PlatformKind::macOS &&
      config->platformInfo.minimum >= VersionTuple(10, 5))
    subtype |= CPU_SUBTYPE_LIB64;

  return subtype;
}

void MachHeaderSection::writeTo(uint8_t *buf) const {
  auto *hdr = reinterpret_cast<mach_header *>(buf);
  hdr->magic = target->magic;
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

  if (config->outputType == MH_DYLIB && config->applicationExtension)
    hdr->flags |= MH_APP_EXTENSION_SAFE;

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

  uint8_t *p = reinterpret_cast<uint8_t *>(hdr) + target->headerSize;
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
  lastRebase.offset += target->wordSize;
}

void RebaseSection::finalizeContents() {
  if (locations.empty())
    return;

  raw_svector_ostream os{contents};
  Rebase lastRebase;

  os << static_cast<uint8_t>(REBASE_OPCODE_SET_TYPE_IMM | REBASE_TYPE_POINTER);

  llvm::sort(locations, [](const Location &a, const Location &b) {
    return a.isec->getVA(a.offset) < b.isec->getVA(b.offset);
  });
  for (const Location &loc : locations)
    encodeRebase(loc.isec->parent, loc.isec->getOffset(loc.offset), lastRebase,
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
  align = target->wordSize;
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

    addNonLazyBindingEntries(sym, isec, sym->gotIndex * target->wordSize);
  }
}

void NonLazyPointerSectionBase::writeTo(uint8_t *buf) const {
  for (size_t i = 0, n = entries.size(); i < n; ++i)
    if (auto *defined = dyn_cast<Defined>(entries[i]))
      write64le(&buf[i * target->wordSize], defined->getVA());
}

GotSection::GotSection()
    : NonLazyPointerSectionBase(segment_names::dataConst, section_names::got) {
  flags = S_NON_LAZY_SYMBOL_POINTERS;
}

TlvPointerSection::TlvPointerSection()
    : NonLazyPointerSectionBase(segment_names::data,
                                section_names::threadPtrs) {
  flags = S_THREAD_LOCAL_VARIABLE_POINTERS;
}

BindingSection::BindingSection()
    : LinkEditSection(segment_names::linkEdit, section_names::binding) {}

namespace {
struct Binding {
  OutputSegment *segment = nullptr;
  uint64_t offset = 0;
  int64_t addend = 0;
};
struct BindIR {
  // Default value of 0xF0 is not valid opcode and should make the program
  // scream instead of accidentally writing "valid" values.
  uint8_t opcode = 0xF0;
  uint64_t data = 0;
  uint64_t consecutiveCount = 0;
};
} // namespace

// Encode a sequence of opcodes that tell dyld to write the address of symbol +
// addend at osec->addr + outSecOff.
//
// The bind opcode "interpreter" remembers the values of each binding field, so
// we only need to encode the differences between bindings. Hence the use of
// lastBinding.
static void encodeBinding(const OutputSection *osec, uint64_t outSecOff,
                          int64_t addend, Binding &lastBinding,
                          std::vector<BindIR> &opcodes) {
  OutputSegment *seg = osec->parent;
  uint64_t offset = osec->getSegmentOffset() + outSecOff;
  if (lastBinding.segment != seg) {
    opcodes.push_back(
        {static_cast<uint8_t>(BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB |
                              seg->index),
         offset});
    lastBinding.segment = seg;
    lastBinding.offset = offset;
  } else if (lastBinding.offset != offset) {
    opcodes.push_back({BIND_OPCODE_ADD_ADDR_ULEB, offset - lastBinding.offset});
    lastBinding.offset = offset;
  }

  if (lastBinding.addend != addend) {
    opcodes.push_back(
        {BIND_OPCODE_SET_ADDEND_SLEB, static_cast<uint64_t>(addend)});
    lastBinding.addend = addend;
  }

  opcodes.push_back({BIND_OPCODE_DO_BIND, 0});
  // DO_BIND causes dyld to both perform the binding and increment the offset
  lastBinding.offset += target->wordSize;
}

static void optimizeOpcodes(std::vector<BindIR> &opcodes) {
  // Pass 1: Combine bind/add pairs
  size_t i;
  int pWrite = 0;
  for (i = 1; i < opcodes.size(); ++i, ++pWrite) {
    if ((opcodes[i].opcode == BIND_OPCODE_ADD_ADDR_ULEB) &&
        (opcodes[i - 1].opcode == BIND_OPCODE_DO_BIND)) {
      opcodes[pWrite].opcode = BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB;
      opcodes[pWrite].data = opcodes[i].data;
      ++i;
    } else {
      opcodes[pWrite] = opcodes[i - 1];
    }
  }
  if (i == opcodes.size())
    opcodes[pWrite] = opcodes[i - 1];
  opcodes.resize(pWrite + 1);

  // Pass 2: Compress two or more bind_add opcodes
  pWrite = 0;
  for (i = 1; i < opcodes.size(); ++i, ++pWrite) {
    if ((opcodes[i].opcode == BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB) &&
        (opcodes[i - 1].opcode == BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB) &&
        (opcodes[i].data == opcodes[i - 1].data)) {
      opcodes[pWrite].opcode = BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB;
      opcodes[pWrite].consecutiveCount = 2;
      opcodes[pWrite].data = opcodes[i].data;
      ++i;
      while (i < opcodes.size() &&
             (opcodes[i].opcode == BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB) &&
             (opcodes[i].data == opcodes[i - 1].data)) {
        opcodes[pWrite].consecutiveCount++;
        ++i;
      }
    } else {
      opcodes[pWrite] = opcodes[i - 1];
    }
  }
  if (i == opcodes.size())
    opcodes[pWrite] = opcodes[i - 1];
  opcodes.resize(pWrite + 1);

  // Pass 3: Use immediate encodings
  // Every binding is the size of one pointer. If the next binding is a
  // multiple of wordSize away that is within BIND_IMMEDIATE_MASK, the
  // opcode can be scaled by wordSize into a single byte and dyld will
  // expand it to the correct address.
  for (auto &p : opcodes) {
    // It's unclear why the check needs to be less than BIND_IMMEDIATE_MASK,
    // but ld64 currently does this. This could be a potential bug, but
    // for now, perform the same behavior to prevent mysterious bugs.
    if ((p.opcode == BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB) &&
        ((p.data / target->wordSize) < BIND_IMMEDIATE_MASK) &&
        ((p.data % target->wordSize) == 0)) {
      p.opcode = BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED;
      p.data /= target->wordSize;
    }
  }
}

static void flushOpcodes(const BindIR &op, raw_svector_ostream &os) {
  uint8_t opcode = op.opcode & BIND_OPCODE_MASK;
  switch (opcode) {
  case BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB:
  case BIND_OPCODE_ADD_ADDR_ULEB:
  case BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB:
    os << op.opcode;
    encodeULEB128(op.data, os);
    break;
  case BIND_OPCODE_SET_ADDEND_SLEB:
    os << op.opcode;
    encodeSLEB128(static_cast<int64_t>(op.data), os);
    break;
  case BIND_OPCODE_DO_BIND:
    os << op.opcode;
    break;
  case BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB:
    os << op.opcode;
    encodeULEB128(op.consecutiveCount, os);
    encodeULEB128(op.data, os);
    break;
  case BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED:
    os << static_cast<uint8_t>(op.opcode | op.data);
    break;
  default:
    llvm_unreachable("cannot bind to an unrecognized symbol");
  }
}

// Non-weak bindings need to have their dylib ordinal encoded as well.
static int16_t ordinalForDylibSymbol(const DylibSymbol &dysym) {
  if (config->namespaceKind == NamespaceKind::flat || dysym.isDynamicLookup())
    return static_cast<int16_t>(BIND_SPECIAL_DYLIB_FLAT_LOOKUP);
  assert(dysym.getFile()->isReferenced());
  return dysym.getFile()->ordinal;
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

// Organize the bindings so we can encoded them with fewer opcodes.
//
// First, all bindings for a given symbol should be grouped together.
// BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM is the largest opcode (since it
// has an associated symbol string), so we only want to emit it once per symbol.
//
// Within each group, we sort the bindings by address. Since bindings are
// delta-encoded, sorting them allows for a more compact result. Note that
// sorting by address alone ensures that bindings for the same segment / section
// are located together, minimizing the number of times we have to emit
// BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB.
//
// Finally, we sort the symbols by the address of their first binding, again
// to facilitate the delta-encoding process.
template <class Sym>
std::vector<std::pair<const Sym *, std::vector<BindingEntry>>>
sortBindings(const BindingsMap<const Sym *> &bindingsMap) {
  std::vector<std::pair<const Sym *, std::vector<BindingEntry>>> bindingsVec(
      bindingsMap.begin(), bindingsMap.end());
  for (auto &p : bindingsVec) {
    std::vector<BindingEntry> &bindings = p.second;
    llvm::sort(bindings, [](const BindingEntry &a, const BindingEntry &b) {
      return a.target.getVA() < b.target.getVA();
    });
  }
  llvm::sort(bindingsVec, [](const auto &a, const auto &b) {
    return a.second[0].target.getVA() < b.second[0].target.getVA();
  });
  return bindingsVec;
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
  int16_t lastOrdinal = 0;

  for (auto &p : sortBindings(bindingsMap)) {
    const DylibSymbol *sym = p.first;
    std::vector<BindingEntry> &bindings = p.second;
    uint8_t flags = BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM;
    if (sym->isWeakRef())
      flags |= BIND_SYMBOL_FLAGS_WEAK_IMPORT;
    os << flags << sym->getName() << '\0'
       << static_cast<uint8_t>(BIND_OPCODE_SET_TYPE_IMM | BIND_TYPE_POINTER);
    int16_t ordinal = ordinalForDylibSymbol(*sym);
    if (ordinal != lastOrdinal) {
      encodeDylibOrdinal(ordinal, os);
      lastOrdinal = ordinal;
    }
    std::vector<BindIR> opcodes;
    for (const BindingEntry &b : bindings)
      encodeBinding(b.target.isec->parent,
                    b.target.isec->getOffset(b.target.offset), b.addend,
                    lastBinding, opcodes);
    if (config->optimize > 1)
      optimizeOpcodes(opcodes);
    for (const auto &op : opcodes)
      flushOpcodes(op, os);
  }
  if (!bindingsMap.empty())
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

  for (auto &p : sortBindings(bindingsMap)) {
    const Symbol *sym = p.first;
    std::vector<BindingEntry> &bindings = p.second;
    os << static_cast<uint8_t>(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM)
       << sym->getName() << '\0'
       << static_cast<uint8_t>(BIND_OPCODE_SET_TYPE_IMM | BIND_TYPE_POINTER);
    std::vector<BindIR> opcodes;
    for (const BindingEntry &b : bindings)
      encodeBinding(b.target.isec->parent,
                    b.target.isec->getOffset(b.target.offset), b.addend,
                    lastBinding, opcodes);
    if (config->optimize > 1)
      optimizeOpcodes(opcodes);
    for (const auto &op : opcodes)
      flushOpcodes(op, os);
  }
  if (!bindingsMap.empty() || !definitions.empty())
    os << static_cast<uint8_t>(BIND_OPCODE_DONE);
}

void WeakBindingSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
}

StubsSection::StubsSection()
    : SyntheticSection(segment_names::text, section_names::stubs) {
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

void StubsSection::finalize() { isFinal = true; }

bool StubsSection::addEntry(Symbol *sym) {
  bool inserted = entries.insert(sym);
  if (inserted)
    sym->stubsIndex = entries.size() - 1;
  return inserted;
}

StubHelperSection::StubHelperSection()
    : SyntheticSection(segment_names::text, section_names::stubHelper) {
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
  Symbol *binder = symtab->addUndefined("dyld_stub_binder", /*file=*/nullptr,
                                        /*isWeakRef=*/false);
  if (auto *undefined = dyn_cast<Undefined>(binder))
    treatUndefinedSymbol(*undefined,
                         "lazy binding (normally in libSystem.dylib)");

  // treatUndefinedSymbol() can replace binder with a DylibSymbol; re-check.
  stubBinder = dyn_cast_or_null<DylibSymbol>(binder);
  if (stubBinder == nullptr)
    return;

  in.got->addEntry(stubBinder);

  in.imageLoaderCache->parent =
      ConcatOutputSection::getOrCreateForInput(in.imageLoaderCache);
  inputSections.push_back(in.imageLoaderCache);
  // Since this isn't in the symbol table or in any input file, the noDeadStrip
  // argument doesn't matter. It's kept alive by ImageLoaderCacheSection()
  // setting `live` to true on the backing InputSection.
  dyldPrivate =
      make<Defined>("__dyld_private", nullptr, in.imageLoaderCache, 0, 0,
                    /*isWeakDef=*/false,
                    /*isExternal=*/false, /*isPrivateExtern=*/false,
                    /*isThumb=*/false, /*isReferencedDynamically=*/false,
                    /*noDeadStrip=*/false);
}

LazyPointerSection::LazyPointerSection()
    : SyntheticSection(segment_names::data, section_names::lazySymbolPtr) {
  align = target->wordSize;
  flags = S_LAZY_SYMBOL_POINTERS;
}

uint64_t LazyPointerSection::getSize() const {
  return in.stubs->getEntries().size() * target->wordSize;
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
    off += target->wordSize;
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
    in.rebase->addEntry(in.lazyPointers->isec,
                        dysym->stubsIndex * target->wordSize);
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
  uint64_t offset = in.lazyPointers->addr - dataSeg->addr +
                    sym.stubsIndex * target->wordSize;
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

void ExportSection::finalizeContents() {
  trieBuilder.setImageBase(in.header->addr);
  for (const Symbol *sym : symtab->getSymbols()) {
    if (const auto *defined = dyn_cast<Defined>(sym)) {
      if (defined->privateExtern || !defined->isLive())
        continue;
      trieBuilder.addSymbol(*defined);
      hasWeakSymbol = hasWeakSymbol || sym->isWeakDef();
    }
  }
  size = trieBuilder.build();
}

void ExportSection::writeTo(uint8_t *buf) const { trieBuilder.writeTo(buf); }

DataInCodeSection::DataInCodeSection()
    : LinkEditSection(segment_names::linkEdit, section_names::dataInCode) {}

template <class LP>
static std::vector<MachO::data_in_code_entry> collectDataInCodeEntries() {
  using SegmentCommand = typename LP::segment_command;
  using Section = typename LP::section;

  std::vector<MachO::data_in_code_entry> dataInCodeEntries;
  for (const InputFile *inputFile : inputFiles) {
    if (!isa<ObjFile>(inputFile))
      continue;
    const ObjFile *objFile = cast<ObjFile>(inputFile);
    const auto *c = reinterpret_cast<const SegmentCommand *>(
        findCommand(objFile->mb.getBufferStart(), LP::segmentLCType));
    if (!c)
      continue;
    ArrayRef<Section> sections{reinterpret_cast<const Section *>(c + 1),
                               c->nsects};

    ArrayRef<MachO::data_in_code_entry> entries = objFile->dataInCodeEntries;
    if (entries.empty())
      continue;
    // For each code subsection find 'data in code' entries residing in it.
    // Compute the new offset values as
    // <offset within subsection> + <subsection address> - <__TEXT address>.
    for (size_t i = 0, n = sections.size(); i < n; ++i) {
      const SubsectionMap &subsecMap = objFile->subsections[i];
      for (const SubsectionEntry &subsecEntry : subsecMap) {
        const InputSection *isec = subsecEntry.isec;
        if (!isCodeSection(isec))
          continue;
        if (cast<ConcatInputSection>(isec)->shouldOmitFromOutput())
          continue;
        const uint64_t beginAddr = sections[i].addr + subsecEntry.offset;
        auto it = llvm::lower_bound(
            entries, beginAddr,
            [](const MachO::data_in_code_entry &entry, uint64_t addr) {
              return entry.offset < addr;
            });
        const uint64_t endAddr = beginAddr + isec->getFileSize();
        for (const auto end = entries.end();
             it != end && it->offset + it->length <= endAddr; ++it)
          dataInCodeEntries.push_back(
              {static_cast<uint32_t>(isec->getVA(it->offset - beginAddr) -
                                     in.header->addr),
               it->length, it->kind});
      }
    }
  }
  return dataInCodeEntries;
}

void DataInCodeSection::finalizeContents() {
  entries = target->wordSize == 8 ? collectDataInCodeEntries<LP64>()
                                  : collectDataInCodeEntries<ILP32>();
}

void DataInCodeSection::writeTo(uint8_t *buf) const {
  if (!entries.empty())
    memcpy(buf, entries.data(), getRawSize());
}

FunctionStartsSection::FunctionStartsSection()
    : LinkEditSection(segment_names::linkEdit, section_names::functionStarts) {}

void FunctionStartsSection::finalizeContents() {
  raw_svector_ostream os{contents};
  std::vector<uint64_t> addrs;
  for (const Symbol *sym : symtab->getSymbols()) {
    if (const auto *defined = dyn_cast<Defined>(sym)) {
      if (!defined->isec || !isCodeSection(defined->isec) || !defined->isLive())
        continue;
      if (const auto *concatIsec = dyn_cast<ConcatInputSection>(defined->isec))
        if (concatIsec->shouldOmitFromOutput())
          continue;
      // TODO: Add support for thumbs, in that case
      // the lowest bit of nextAddr needs to be set to 1.
      addrs.push_back(defined->getVA());
    }
  }
  llvm::sort(addrs);
  uint64_t addr = in.header->addr;
  for (uint64_t nextAddr : addrs) {
    uint64_t delta = nextAddr - addr;
    if (delta == 0)
      continue;
    encodeULEB128(delta, os);
    addr = nextAddr;
  }
  os << '\0';
}

void FunctionStartsSection::writeTo(uint8_t *buf) const {
  memcpy(buf, contents.data(), contents.size());
}

SymtabSection::SymtabSection(StringTableSection &stringTableSection)
    : LinkEditSection(segment_names::linkEdit, section_names::symbolTable),
      stringTableSection(stringTableSection) {}

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
  stab.value = defined->size;
  stabs.emplace_back(std::move(stab));
}

void SymtabSection::emitStabs() {
  for (const std::string &s : config->astPaths) {
    StabsEntry astStab(N_AST);
    astStab.strx = stringTableSection.addString(s);
    stabs.emplace_back(std::move(astStab));
  }

  std::vector<Defined *> symbolsNeedingStabs;
  for (const SymtabEntry &entry :
       concat<SymtabEntry>(localSymbols, externalSymbols)) {
    Symbol *sym = entry.sym;
    assert(sym->isLive() &&
           "dead symbols should not be in localSymbols, externalSymbols");
    if (auto *defined = dyn_cast<Defined>(sym)) {
      if (defined->isAbsolute())
        continue;
      InputSection *isec = defined->isec;
      ObjFile *file = dyn_cast_or_null<ObjFile>(isec->getFile());
      if (!file || !file->compileUnit)
        continue;
      symbolsNeedingStabs.push_back(defined);
    }
  }

  llvm::stable_sort(symbolsNeedingStabs, [&](Defined *a, Defined *b) {
    return a->isec->getFile()->id < b->isec->getFile()->id;
  });

  // Emit STABS symbols so that dsymutil and/or the debugger can map address
  // regions in the final binary to the source and object files from which they
  // originated.
  InputFile *lastFile = nullptr;
  for (Defined *defined : symbolsNeedingStabs) {
    InputSection *isec = defined->isec;
    ObjFile *file = cast<ObjFile>(isec->getFile());

    if (lastFile == nullptr || lastFile != file) {
      if (lastFile != nullptr)
        emitEndSourceStab();
      lastFile = file;

      emitBeginSourceStab(file->compileUnit);
      emitObjectFileStab(file);
    }

    StabsEntry symStab;
    symStab.sect = defined->isec->canonical()->parent->index;
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
        if (auto *defined = dyn_cast_or_null<Defined>(sym)) {
          if (!defined->isExternal() && defined->isLive()) {
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
  if (Defined *dyldPrivate = in.stubHelper->dyldPrivate)
    addSymbol(localSymbols, dyldPrivate);

  for (Symbol *sym : symtab->getSymbols()) {
    if (!sym->isLive())
      continue;
    if (auto *defined = dyn_cast<Defined>(sym)) {
      if (!defined->includeInSymtab)
        continue;
      assert(defined->isExternal());
      if (defined->privateExtern)
        addSymbol(localSymbols, defined);
      else
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

// This serves to hide (type-erase) the template parameter from SymtabSection.
template <class LP> class SymtabSectionImpl final : public SymtabSection {
public:
  SymtabSectionImpl(StringTableSection &stringTableSection)
      : SymtabSection(stringTableSection) {}
  uint64_t getRawSize() const override;
  void writeTo(uint8_t *buf) const override;
};

template <class LP> uint64_t SymtabSectionImpl<LP>::getRawSize() const {
  return getNumSymbols() * sizeof(typename LP::nlist);
}

template <class LP> void SymtabSectionImpl<LP>::writeTo(uint8_t *buf) const {
  auto *nList = reinterpret_cast<typename LP::nlist *>(buf);
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
        nList->n_sect = defined->isec->canonical()->parent->index;
        // For the N_SECT symbol type, n_value is the address of the symbol
        nList->n_value = defined->getVA();
      }
      nList->n_desc |= defined->thumb ? N_ARM_THUMB_DEF : 0;
      nList->n_desc |= defined->isExternalWeakDef() ? N_WEAK_DEF : 0;
      nList->n_desc |=
          defined->referencedDynamically ? REFERENCED_DYNAMICALLY : 0;
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

template <class LP>
SymtabSection *
macho::makeSymtabSection(StringTableSection &stringTableSection) {
  return make<SymtabSectionImpl<LP>>(stringTableSection);
}

IndirectSymtabSection::IndirectSymtabSection()
    : LinkEditSection(segment_names::linkEdit,
                      section_names::indirectSymbolTable) {}

uint32_t IndirectSymtabSection::getNumSymbols() const {
  return in.got->getEntries().size() + in.tlvPointers->getEntries().size() +
         2 * in.stubs->getEntries().size();
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
  in.stubs->reserved1 = off;
  off += in.stubs->getEntries().size();
  in.lazyPointers->reserved1 = off;
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
  // There is a 1:1 correspondence between stubs and LazyPointerSection
  // entries. But giving __stubs and __la_symbol_ptr the same reserved1
  // (the offset into the indirect symbol table) so that they both refer
  // to the same range of offsets confuses `strip`, so write the stubs
  // symbol table offsets a second time.
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

static_assert((CodeSignatureSection::blobHeadersSize % 8) == 0, "");
static_assert((CodeSignatureSection::fixedHeadersSize % 8) == 0, "");

CodeSignatureSection::CodeSignatureSection()
    : LinkEditSection(segment_names::linkEdit, section_names::codeSignature) {
  align = 16; // required by libstuff
  // FIXME: Consider using finalOutput instead of outputFile.
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

BitcodeBundleSection::BitcodeBundleSection()
    : SyntheticSection(segment_names::llvm, section_names::bitcodeBundle) {}

class ErrorCodeWrapper {
public:
  explicit ErrorCodeWrapper(std::error_code ec) : errorCode(ec.value()) {}
  explicit ErrorCodeWrapper(int ec) : errorCode(ec) {}
  operator int() const { return errorCode; }

private:
  int errorCode;
};

#define CHECK_EC(exp)                                                          \
  do {                                                                         \
    ErrorCodeWrapper ec(exp);                                                  \
    if (ec)                                                                    \
      fatal(Twine("operation failed with error code ") + Twine(ec) + ": " +    \
            #exp);                                                             \
  } while (0);

void BitcodeBundleSection::finalize() {
#ifdef LLVM_HAVE_LIBXAR
  using namespace llvm::sys::fs;
  CHECK_EC(createTemporaryFile("bitcode-bundle", "xar", xarPath));

  xar_t xar(xar_open(xarPath.data(), O_RDWR));
  if (!xar)
    fatal("failed to open XAR temporary file at " + xarPath);
  CHECK_EC(xar_opt_set(xar, XAR_OPT_COMPRESSION, XAR_OPT_VAL_NONE));
  // FIXME: add more data to XAR
  CHECK_EC(xar_close(xar));

  file_size(xarPath, xarSize);
#endif // defined(LLVM_HAVE_LIBXAR)
}

void BitcodeBundleSection::writeTo(uint8_t *buf) const {
  using namespace llvm::sys::fs;
  file_t handle =
      CHECK(openNativeFile(xarPath, CD_OpenExisting, FA_Read, OF_None),
            "failed to open XAR file");
  std::error_code ec;
  mapped_file_region xarMap(handle, mapped_file_region::mapmode::readonly,
                            xarSize, 0, ec);
  if (ec)
    fatal("failed to map XAR file");
  memcpy(buf, xarMap.const_data(), xarSize);

  closeFile(handle);
  remove(xarPath);
}

CStringSection::CStringSection()
    : SyntheticSection(segment_names::text, section_names::cString) {
  flags = S_CSTRING_LITERALS;
}

void CStringSection::addInput(CStringInputSection *isec) {
  isec->parent = this;
  inputs.push_back(isec);
  if (isec->align > align)
    align = isec->align;
}

void CStringSection::writeTo(uint8_t *buf) const {
  for (const CStringInputSection *isec : inputs) {
    for (size_t i = 0, e = isec->pieces.size(); i != e; ++i) {
      if (!isec->pieces[i].live)
        continue;
      StringRef string = isec->getStringRef(i);
      memcpy(buf + isec->pieces[i].outSecOff, string.data(), string.size());
    }
  }
}

void CStringSection::finalizeContents() {
  uint64_t offset = 0;
  for (CStringInputSection *isec : inputs) {
    for (size_t i = 0, e = isec->pieces.size(); i != e; ++i) {
      if (!isec->pieces[i].live)
        continue;
      uint32_t pieceAlign = MinAlign(isec->pieces[i].inSecOff, align);
      offset = alignTo(offset, pieceAlign);
      isec->pieces[i].outSecOff = offset;
      isec->isFinal = true;
      StringRef string = isec->getStringRef(i);
      offset += string.size();
    }
  }
  size = offset;
}
// Mergeable cstring literals are found under the __TEXT,__cstring section. In
// contrast to ELF, which puts strings that need different alignments into
// different sections, clang's Mach-O backend puts them all in one section.
// Strings that need to be aligned have the .p2align directive emitted before
// them, which simply translates into zero padding in the object file.
//
// I *think* ld64 extracts the desired per-string alignment from this data by
// preserving each string's offset from the last section-aligned address. I'm
// not entirely certain since it doesn't seem consistent about doing this, and
// in fact doesn't seem to be correct in general: we can in fact can induce ld64
// to produce a crashing binary just by linking in an additional object file
// that only contains a duplicate cstring at a different alignment. See PR50563
// for details.
//
// On x86_64, the cstrings we've seen so far that require special alignment are
// all accessed by SIMD operations -- x86_64 requires SIMD accesses to be
// 16-byte-aligned. arm64 also seems to require 16-byte-alignment in some cases
// (PR50791), but I haven't tracked down the root cause. So for now, I'm just
// aligning all strings to 16 bytes.  This is indeed wasteful, but
// implementation-wise it's simpler than preserving per-string
// alignment+offsets. It also avoids the aforementioned crash after
// deduplication of differently-aligned strings.  Finally, the overhead is not
// huge: using 16-byte alignment (vs no alignment) is only a 0.5% size overhead
// when linking chromium_framework on x86_64.
DeduplicatedCStringSection::DeduplicatedCStringSection()
    : builder(StringTableBuilder::RAW, /*Alignment=*/16) {}

void DeduplicatedCStringSection::finalizeContents() {
  // Add all string pieces to the string table builder to create section
  // contents.
  for (const CStringInputSection *isec : inputs)
    for (size_t i = 0, e = isec->pieces.size(); i != e; ++i)
      if (isec->pieces[i].live)
        builder.add(isec->getCachedHashStringRef(i));

  // Fix the string table content. After this, the contents will never change.
  builder.finalizeInOrder();

  // finalize() fixed tail-optimized strings, so we can now get
  // offsets of strings. Get an offset for each string and save it
  // to a corresponding SectionPiece for easy access.
  for (CStringInputSection *isec : inputs) {
    for (size_t i = 0, e = isec->pieces.size(); i != e; ++i) {
      if (!isec->pieces[i].live)
        continue;
      isec->pieces[i].outSecOff =
          builder.getOffset(isec->getCachedHashStringRef(i));
      isec->isFinal = true;
    }
  }
}

// This section is actually emitted as __TEXT,__const by ld64, but clang may
// emit input sections of that name, and LLD doesn't currently support mixing
// synthetic and concat-type OutputSections. To work around this, I've given
// our merged-literals section a different name.
WordLiteralSection::WordLiteralSection()
    : SyntheticSection(segment_names::text, section_names::literals) {
  align = 16;
}

void WordLiteralSection::addInput(WordLiteralInputSection *isec) {
  isec->parent = this;
  inputs.push_back(isec);
}

void WordLiteralSection::finalizeContents() {
  for (WordLiteralInputSection *isec : inputs) {
    // We do all processing of the InputSection here, so it will be effectively
    // finalized.
    isec->isFinal = true;
    const uint8_t *buf = isec->data.data();
    switch (sectionType(isec->getFlags())) {
    case S_4BYTE_LITERALS: {
      for (size_t off = 0, e = isec->data.size(); off < e; off += 4) {
        if (!isec->isLive(off))
          continue;
        uint32_t value = *reinterpret_cast<const uint32_t *>(buf + off);
        literal4Map.emplace(value, literal4Map.size());
      }
      break;
    }
    case S_8BYTE_LITERALS: {
      for (size_t off = 0, e = isec->data.size(); off < e; off += 8) {
        if (!isec->isLive(off))
          continue;
        uint64_t value = *reinterpret_cast<const uint64_t *>(buf + off);
        literal8Map.emplace(value, literal8Map.size());
      }
      break;
    }
    case S_16BYTE_LITERALS: {
      for (size_t off = 0, e = isec->data.size(); off < e; off += 16) {
        if (!isec->isLive(off))
          continue;
        UInt128 value = *reinterpret_cast<const UInt128 *>(buf + off);
        literal16Map.emplace(value, literal16Map.size());
      }
      break;
    }
    default:
      llvm_unreachable("invalid literal section type");
    }
  }
}

void WordLiteralSection::writeTo(uint8_t *buf) const {
  // Note that we don't attempt to do any endianness conversion in addInput(),
  // so we don't do it here either -- just write out the original value,
  // byte-for-byte.
  for (const auto &p : literal16Map)
    memcpy(buf + p.second * 16, &p.first, 16);
  buf += literal16Map.size() * 16;

  for (const auto &p : literal8Map)
    memcpy(buf + p.second * 8, &p.first, 8);
  buf += literal8Map.size() * 8;

  for (const auto &p : literal4Map)
    memcpy(buf + p.second * 4, &p.first, 4);
}

void macho::createSyntheticSymbols() {
  auto addHeaderSymbol = [](const char *name) {
    symtab->addSynthetic(name, in.header->isec, /*value=*/0,
                         /*privateExtern=*/true, /*includeInSymtab=*/false,
                         /*referencedDynamically=*/false);
  };

  switch (config->outputType) {
    // FIXME: Assign the right address value for these symbols
    // (rather than 0). But we need to do that after assignAddresses().
  case MH_EXECUTE:
    // If linking PIE, __mh_execute_header is a defined symbol in
    //  __TEXT, __text)
    // Otherwise, it's an absolute symbol.
    if (config->isPic)
      symtab->addSynthetic("__mh_execute_header", in.header->isec, /*value=*/0,
                           /*privateExtern=*/false, /*includeInSymtab=*/true,
                           /*referencedDynamically=*/true);
    else
      symtab->addSynthetic("__mh_execute_header", /*isec=*/nullptr, /*value=*/0,
                           /*privateExtern=*/false, /*includeInSymtab=*/true,
                           /*referencedDynamically=*/true);
    break;

    // The following symbols are N_SECT symbols, even though the header is not
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

template SymtabSection *macho::makeSymtabSection<LP64>(StringTableSection &);
template SymtabSection *macho::makeSymtabSection<ILP32>(StringTableSection &);
