//===- Symbols.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "Config.h"
#include "InputChunks.h"
#include "InputEvent.h"
#include "InputFiles.h"
#include "InputGlobal.h"
#include "OutputSections.h"
#include "OutputSegment.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Strings.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::wasm;

namespace lld {
std::string toString(const wasm::Symbol &sym) {
  return maybeDemangleSymbol(sym.getName());
}

std::string maybeDemangleSymbol(StringRef name) {
  // WebAssembly requires caller and callee signatures to match, so we mangle
  // `main` in the case where we need to pass it arguments.
  if (name == "__main_argc_argv")
    return "main";
  if (wasm::config->demangle)
    return demangleItanium(name);
  return std::string(name);
}

std::string toString(wasm::Symbol::Kind kind) {
  switch (kind) {
  case wasm::Symbol::DefinedFunctionKind:
    return "DefinedFunction";
  case wasm::Symbol::DefinedDataKind:
    return "DefinedData";
  case wasm::Symbol::DefinedGlobalKind:
    return "DefinedGlobal";
  case wasm::Symbol::DefinedEventKind:
    return "DefinedEvent";
  case wasm::Symbol::UndefinedFunctionKind:
    return "UndefinedFunction";
  case wasm::Symbol::UndefinedDataKind:
    return "UndefinedData";
  case wasm::Symbol::UndefinedGlobalKind:
    return "UndefinedGlobal";
  case wasm::Symbol::LazyKind:
    return "LazyKind";
  case wasm::Symbol::SectionKind:
    return "SectionKind";
  case wasm::Symbol::OutputSectionKind:
    return "OutputSectionKind";
  }
  llvm_unreachable("invalid symbol kind");
}

namespace wasm {
DefinedFunction *WasmSym::callCtors;
DefinedFunction *WasmSym::callDtors;
DefinedFunction *WasmSym::initMemory;
DefinedFunction *WasmSym::applyDataRelocs;
DefinedFunction *WasmSym::applyGlobalRelocs;
DefinedFunction *WasmSym::initTLS;
DefinedFunction *WasmSym::startFunction;
DefinedData *WasmSym::dsoHandle;
DefinedData *WasmSym::dataEnd;
DefinedData *WasmSym::globalBase;
DefinedData *WasmSym::heapBase;
DefinedData *WasmSym::initMemoryFlag;
GlobalSymbol *WasmSym::stackPointer;
GlobalSymbol *WasmSym::tlsBase;
GlobalSymbol *WasmSym::tlsSize;
GlobalSymbol *WasmSym::tlsAlign;
UndefinedGlobal *WasmSym::tableBase;
DefinedData *WasmSym::definedTableBase;
UndefinedGlobal *WasmSym::memoryBase;
DefinedData *WasmSym::definedMemoryBase;

WasmSymbolType Symbol::getWasmType() const {
  if (isa<FunctionSymbol>(this))
    return WASM_SYMBOL_TYPE_FUNCTION;
  if (isa<DataSymbol>(this))
    return WASM_SYMBOL_TYPE_DATA;
  if (isa<GlobalSymbol>(this))
    return WASM_SYMBOL_TYPE_GLOBAL;
  if (isa<EventSymbol>(this))
    return WASM_SYMBOL_TYPE_EVENT;
  if (isa<SectionSymbol>(this) || isa<OutputSectionSymbol>(this))
    return WASM_SYMBOL_TYPE_SECTION;
  llvm_unreachable("invalid symbol kind");
}

const WasmSignature *Symbol::getSignature() const {
  if (auto* f = dyn_cast<FunctionSymbol>(this))
    return f->signature;
  if (auto *l = dyn_cast<LazySymbol>(this))
    return l->signature;
  return nullptr;
}

InputChunk *Symbol::getChunk() const {
  if (auto *f = dyn_cast<DefinedFunction>(this))
    return f->function;
  if (auto *f = dyn_cast<UndefinedFunction>(this))
    if (f->stubFunction)
      return f->stubFunction->function;
  if (auto *d = dyn_cast<DefinedData>(this))
    return d->segment;
  return nullptr;
}

bool Symbol::isDiscarded() const {
  if (InputChunk *c = getChunk())
    return c->discarded;
  return false;
}

bool Symbol::isLive() const {
  if (auto *g = dyn_cast<DefinedGlobal>(this))
    return g->global->live;
  if (auto *e = dyn_cast<DefinedEvent>(this))
    return e->event->live;
  if (InputChunk *c = getChunk())
    return c->live;
  return referenced;
}

void Symbol::markLive() {
  assert(!isDiscarded());
  if (file != NULL && isDefined())
    file->markLive();
  if (auto *g = dyn_cast<DefinedGlobal>(this))
    g->global->live = true;
  if (auto *e = dyn_cast<DefinedEvent>(this))
    e->event->live = true;
  if (InputChunk *c = getChunk())
    c->live = true;
  referenced = true;
}

uint32_t Symbol::getOutputSymbolIndex() const {
  assert(outputSymbolIndex != INVALID_INDEX);
  return outputSymbolIndex;
}

void Symbol::setOutputSymbolIndex(uint32_t index) {
  LLVM_DEBUG(dbgs() << "setOutputSymbolIndex " << name << " -> " << index
                    << "\n");
  assert(outputSymbolIndex == INVALID_INDEX);
  outputSymbolIndex = index;
}

void Symbol::setGOTIndex(uint32_t index) {
  LLVM_DEBUG(dbgs() << "setGOTIndex " << name << " -> " << index << "\n");
  assert(gotIndex == INVALID_INDEX);
  if (config->isPic) {
    // Any symbol that is assigned a GOT entry must be exported otherwise the
    // dynamic linker won't be able create the entry that contains it.
    forceExport = true;
  }
  gotIndex = index;
}

bool Symbol::isWeak() const {
  return (flags & WASM_SYMBOL_BINDING_MASK) == WASM_SYMBOL_BINDING_WEAK;
}

bool Symbol::isLocal() const {
  return (flags & WASM_SYMBOL_BINDING_MASK) == WASM_SYMBOL_BINDING_LOCAL;
}

bool Symbol::isHidden() const {
  return (flags & WASM_SYMBOL_VISIBILITY_MASK) == WASM_SYMBOL_VISIBILITY_HIDDEN;
}

void Symbol::setHidden(bool isHidden) {
  LLVM_DEBUG(dbgs() << "setHidden: " << name << " -> " << isHidden << "\n");
  flags &= ~WASM_SYMBOL_VISIBILITY_MASK;
  if (isHidden)
    flags |= WASM_SYMBOL_VISIBILITY_HIDDEN;
  else
    flags |= WASM_SYMBOL_VISIBILITY_DEFAULT;
}

bool Symbol::isExported() const {
  if (!isDefined() || isLocal())
    return false;

  if (forceExport || config->exportAll)
    return true;

  if (config->exportDynamic && !isHidden())
    return true;

  return flags & WASM_SYMBOL_EXPORTED;
}

bool Symbol::isNoStrip() const {
  return flags & WASM_SYMBOL_NO_STRIP;
}

uint32_t FunctionSymbol::getFunctionIndex() const {
  if (auto *f = dyn_cast<DefinedFunction>(this))
    return f->function->getFunctionIndex();
  if (const auto *u = dyn_cast<UndefinedFunction>(this)) {
    if (u->stubFunction) {
      return u->stubFunction->getFunctionIndex();
    }
  }
  assert(functionIndex != INVALID_INDEX);
  return functionIndex;
}

void FunctionSymbol::setFunctionIndex(uint32_t index) {
  LLVM_DEBUG(dbgs() << "setFunctionIndex " << name << " -> " << index << "\n");
  assert(functionIndex == INVALID_INDEX);
  functionIndex = index;
}

bool FunctionSymbol::hasFunctionIndex() const {
  if (auto *f = dyn_cast<DefinedFunction>(this))
    return f->function->hasFunctionIndex();
  return functionIndex != INVALID_INDEX;
}

uint32_t FunctionSymbol::getTableIndex() const {
  if (auto *f = dyn_cast<DefinedFunction>(this))
    return f->function->getTableIndex();
  assert(tableIndex != INVALID_INDEX);
  return tableIndex;
}

bool FunctionSymbol::hasTableIndex() const {
  if (auto *f = dyn_cast<DefinedFunction>(this))
    return f->function->hasTableIndex();
  return tableIndex != INVALID_INDEX;
}

void FunctionSymbol::setTableIndex(uint32_t index) {
  // For imports, we set the table index here on the Symbol; for defined
  // functions we set the index on the InputFunction so that we don't export
  // the same thing twice (keeps the table size down).
  if (auto *f = dyn_cast<DefinedFunction>(this)) {
    f->function->setTableIndex(index);
    return;
  }
  LLVM_DEBUG(dbgs() << "setTableIndex " << name << " -> " << index << "\n");
  assert(tableIndex == INVALID_INDEX);
  tableIndex = index;
}

DefinedFunction::DefinedFunction(StringRef name, uint32_t flags, InputFile *f,
                                 InputFunction *function)
    : FunctionSymbol(name, DefinedFunctionKind, flags, f,
                     function ? &function->signature : nullptr),
      function(function) {}

uint64_t DefinedData::getVirtualAddress() const {
  LLVM_DEBUG(dbgs() << "getVirtualAddress: " << getName() << "\n");
  if (segment)
    return segment->outputSeg->startVA + segment->outputSegmentOffset + offset;
  return offset;
}

void DefinedData::setVirtualAddress(uint64_t value) {
  LLVM_DEBUG(dbgs() << "setVirtualAddress " << name << " -> " << value << "\n");
  assert(!segment);
  offset = value;
}

uint64_t DefinedData::getOutputSegmentOffset() const {
  LLVM_DEBUG(dbgs() << "getOutputSegmentOffset: " << getName() << "\n");
  return segment->outputSegmentOffset + offset;
}

uint64_t DefinedData::getOutputSegmentIndex() const {
  LLVM_DEBUG(dbgs() << "getOutputSegmentIndex: " << getName() << "\n");
  return segment->outputSeg->index;
}

uint32_t GlobalSymbol::getGlobalIndex() const {
  if (auto *f = dyn_cast<DefinedGlobal>(this))
    return f->global->getGlobalIndex();
  assert(globalIndex != INVALID_INDEX);
  return globalIndex;
}

void GlobalSymbol::setGlobalIndex(uint32_t index) {
  LLVM_DEBUG(dbgs() << "setGlobalIndex " << name << " -> " << index << "\n");
  assert(globalIndex == INVALID_INDEX);
  globalIndex = index;
}

bool GlobalSymbol::hasGlobalIndex() const {
  if (auto *f = dyn_cast<DefinedGlobal>(this))
    return f->global->hasGlobalIndex();
  return globalIndex != INVALID_INDEX;
}

DefinedGlobal::DefinedGlobal(StringRef name, uint32_t flags, InputFile *file,
                             InputGlobal *global)
    : GlobalSymbol(name, DefinedGlobalKind, flags, file,
                   global ? &global->getType() : nullptr),
      global(global) {}

uint32_t EventSymbol::getEventIndex() const {
  if (auto *f = dyn_cast<DefinedEvent>(this))
    return f->event->getEventIndex();
  assert(eventIndex != INVALID_INDEX);
  return eventIndex;
}

void EventSymbol::setEventIndex(uint32_t index) {
  LLVM_DEBUG(dbgs() << "setEventIndex " << name << " -> " << index << "\n");
  assert(eventIndex == INVALID_INDEX);
  eventIndex = index;
}

bool EventSymbol::hasEventIndex() const {
  if (auto *f = dyn_cast<DefinedEvent>(this))
    return f->event->hasEventIndex();
  return eventIndex != INVALID_INDEX;
}

DefinedEvent::DefinedEvent(StringRef name, uint32_t flags, InputFile *file,
                           InputEvent *event)
    : EventSymbol(name, DefinedEventKind, flags, file,
                  event ? &event->getType() : nullptr,
                  event ? &event->signature : nullptr),
      event(event) {}

const OutputSectionSymbol *SectionSymbol::getOutputSectionSymbol() const {
  assert(section->outputSec && section->outputSec->sectionSym);
  return section->outputSec->sectionSym;
}

void LazySymbol::fetch() { cast<ArchiveFile>(file)->addMember(&archiveSymbol); }

void LazySymbol::setWeak() {
  flags |= (flags & ~WASM_SYMBOL_BINDING_MASK) | WASM_SYMBOL_BINDING_WEAK;
}

MemoryBufferRef LazySymbol::getMemberBuffer() {
  Archive::Child c =
      CHECK(archiveSymbol.getMember(),
            "could not get the member for symbol " + toString(*this));

  return CHECK(c.getMemoryBufferRef(),
               "could not get the buffer for the member defining symbol " +
                   toString(*this));
}

void printTraceSymbolUndefined(StringRef name, const InputFile* file) {
  message(toString(file) + ": reference to " + name);
}

// Print out a log message for --trace-symbol.
void printTraceSymbol(Symbol *sym) {
  // Undefined symbols are traced via printTraceSymbolUndefined
  if (sym->isUndefined())
    return;

  std::string s;
  if (sym->isLazy())
    s = ": lazy definition of ";
  else
    s = ": definition of ";

  message(toString(sym->getFile()) + s + sym->getName());
}

const char *defaultModule = "env";
const char *functionTableName = "__indirect_function_table";

} // namespace wasm
} // namespace lld
