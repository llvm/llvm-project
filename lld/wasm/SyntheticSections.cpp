//===- SyntheticSections.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains linker-synthesized sections.
//
//===----------------------------------------------------------------------===//

#include "SyntheticSections.h"

#include "InputChunks.h"
#include "InputEvent.h"
#include "InputGlobal.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::wasm;

namespace lld {
namespace wasm {

OutStruct out;

namespace {

// Some synthetic sections (e.g. "name" and "linking") have subsections.
// Just like the synthetic sections themselves these need to be created before
// they can be written out (since they are preceded by their length). This
// class is used to create subsections and then write them into the stream
// of the parent section.
class SubSection {
public:
  explicit SubSection(uint32_t type) : type(type) {}

  void writeTo(raw_ostream &to) {
    os.flush();
    writeUleb128(to, type, "subsection type");
    writeUleb128(to, body.size(), "subsection size");
    to.write(body.data(), body.size());
  }

private:
  uint32_t type;
  std::string body;

public:
  raw_string_ostream os{body};
};

} // namespace

void DylinkSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, memSize, "MemSize");
  writeUleb128(os, memAlign, "MemAlign");
  writeUleb128(os, out.elemSec->numEntries(), "TableSize");
  writeUleb128(os, 0, "TableAlign");
  writeUleb128(os, symtab->sharedFiles.size(), "Needed");
  for (auto *so : symtab->sharedFiles)
    writeStr(os, llvm::sys::path::filename(so->getName()), "so name");
}

uint32_t TypeSection::registerType(const WasmSignature &sig) {
  auto pair = typeIndices.insert(std::make_pair(sig, types.size()));
  if (pair.second) {
    LLVM_DEBUG(llvm::dbgs() << "type " << toString(sig) << "\n");
    types.push_back(&sig);
  }
  return pair.first->second;
}

uint32_t TypeSection::lookupType(const WasmSignature &sig) {
  auto it = typeIndices.find(sig);
  if (it == typeIndices.end()) {
    error("type not found: " + toString(sig));
    return 0;
  }
  return it->second;
}

void TypeSection::writeBody() {
  writeUleb128(bodyOutputStream, types.size(), "type count");
  for (const WasmSignature *sig : types)
    writeSig(bodyOutputStream, *sig);
}

uint32_t ImportSection::getNumImports() const {
  assert(isSealed);
  uint32_t numImports = importedSymbols.size() + gotSymbols.size();
  if (config->importMemory)
    ++numImports;
  if (config->importTable)
    ++numImports;
  return numImports;
}

void ImportSection::addGOTEntry(Symbol *sym) {
  assert(!isSealed);
  if (sym->hasGOTIndex())
    return;
  LLVM_DEBUG(dbgs() << "addGOTEntry: " << toString(*sym) << "\n");
  sym->setGOTIndex(numImportedGlobals++);
  gotSymbols.push_back(sym);
}

void ImportSection::addImport(Symbol *sym) {
  assert(!isSealed);
  importedSymbols.emplace_back(sym);
  if (auto *f = dyn_cast<FunctionSymbol>(sym))
    f->setFunctionIndex(numImportedFunctions++);
  else if (auto *g = dyn_cast<GlobalSymbol>(sym))
    g->setGlobalIndex(numImportedGlobals++);
  else
    cast<EventSymbol>(sym)->setEventIndex(numImportedEvents++);
}

void ImportSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, getNumImports(), "import count");

  if (config->importMemory) {
    WasmImport import;
    import.Module = defaultModule;
    import.Field = "memory";
    import.Kind = WASM_EXTERNAL_MEMORY;
    import.Memory.Flags = 0;
    import.Memory.Initial = out.memorySec->numMemoryPages;
    if (out.memorySec->maxMemoryPages != 0 || config->sharedMemory) {
      import.Memory.Flags |= WASM_LIMITS_FLAG_HAS_MAX;
      import.Memory.Maximum = out.memorySec->maxMemoryPages;
    }
    if (config->sharedMemory)
      import.Memory.Flags |= WASM_LIMITS_FLAG_IS_SHARED;
    if (config->is64.getValueOr(false))
      import.Memory.Flags |= WASM_LIMITS_FLAG_IS_64;
    writeImport(os, import);
  }

  if (config->importTable) {
    uint32_t tableSize = config->tableBase + out.elemSec->numEntries();
    WasmImport import;
    import.Module = defaultModule;
    import.Field = functionTableName;
    import.Kind = WASM_EXTERNAL_TABLE;
    import.Table.ElemType = WASM_TYPE_FUNCREF;
    import.Table.Limits = {0, tableSize, 0};
    writeImport(os, import);
  }

  for (const Symbol *sym : importedSymbols) {
    WasmImport import;
    if (auto *f = dyn_cast<UndefinedFunction>(sym)) {
      import.Field = f->importName ? *f->importName : sym->getName();
      import.Module = f->importModule ? *f->importModule : defaultModule;
    } else if (auto *g = dyn_cast<UndefinedGlobal>(sym)) {
      import.Field = g->importName ? *g->importName : sym->getName();
      import.Module = g->importModule ? *g->importModule : defaultModule;
    } else {
      import.Field = sym->getName();
      import.Module = defaultModule;
    }

    if (auto *functionSym = dyn_cast<FunctionSymbol>(sym)) {
      import.Kind = WASM_EXTERNAL_FUNCTION;
      import.SigIndex = out.typeSec->lookupType(*functionSym->signature);
    } else if (auto *globalSym = dyn_cast<GlobalSymbol>(sym)) {
      import.Kind = WASM_EXTERNAL_GLOBAL;
      import.Global = *globalSym->getGlobalType();
    } else {
      auto *eventSym = cast<EventSymbol>(sym);
      import.Kind = WASM_EXTERNAL_EVENT;
      import.Event.Attribute = eventSym->getEventType()->Attribute;
      import.Event.SigIndex = out.typeSec->lookupType(*eventSym->signature);
    }
    writeImport(os, import);
  }

  for (const Symbol *sym : gotSymbols) {
    WasmImport import;
    import.Kind = WASM_EXTERNAL_GLOBAL;
    import.Global = {WASM_TYPE_I32, true};
    if (isa<DataSymbol>(sym))
      import.Module = "GOT.mem";
    else
      import.Module = "GOT.func";
    import.Field = sym->getName();
    writeImport(os, import);
  }
}

void FunctionSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, inputFunctions.size(), "function count");
  for (const InputFunction *func : inputFunctions)
    writeUleb128(os, out.typeSec->lookupType(func->signature), "sig index");
}

void FunctionSection::addFunction(InputFunction *func) {
  if (!func->live)
    return;
  uint32_t functionIndex =
      out.importSec->getNumImportedFunctions() + inputFunctions.size();
  inputFunctions.emplace_back(func);
  func->setFunctionIndex(functionIndex);
}

void TableSection::writeBody() {
  uint32_t tableSize = config->tableBase + out.elemSec->numEntries();

  raw_ostream &os = bodyOutputStream;
  writeUleb128(os, 1, "table count");
  WasmLimits limits;
  if (config->growableTable)
    limits = {0, tableSize, 0};
  else
    limits = {WASM_LIMITS_FLAG_HAS_MAX, tableSize, tableSize};
  writeTableType(os, WasmTable{0, WASM_TYPE_FUNCREF, limits});
}

void MemorySection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  bool hasMax = maxMemoryPages != 0 || config->sharedMemory;
  writeUleb128(os, 1, "memory count");
  unsigned flags = 0;
  if (hasMax)
    flags |= WASM_LIMITS_FLAG_HAS_MAX;
  if (config->sharedMemory)
    flags |= WASM_LIMITS_FLAG_IS_SHARED;
  if (config->is64.getValueOr(false))
    flags |= WASM_LIMITS_FLAG_IS_64;
  writeUleb128(os, flags, "memory limits flags");
  writeUleb128(os, numMemoryPages, "initial pages");
  if (hasMax)
    writeUleb128(os, maxMemoryPages, "max pages");
}

void EventSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, inputEvents.size(), "event count");
  for (InputEvent *e : inputEvents) {
    e->event.Type.SigIndex = out.typeSec->lookupType(e->signature);
    writeEvent(os, e->event);
  }
}

void EventSection::addEvent(InputEvent *event) {
  if (!event->live)
    return;
  uint32_t eventIndex =
      out.importSec->getNumImportedEvents() + inputEvents.size();
  LLVM_DEBUG(dbgs() << "addEvent: " << eventIndex << "\n");
  event->setEventIndex(eventIndex);
  inputEvents.push_back(event);
}

void GlobalSection::assignIndexes() {
  uint32_t globalIndex = out.importSec->getNumImportedGlobals();
  for (InputGlobal *g : inputGlobals)
    g->setGlobalIndex(globalIndex++);
  for (Symbol *sym : internalGotSymbols)
    sym->setGOTIndex(globalIndex++);
  isSealed = true;
}

void GlobalSection::addInternalGOTEntry(Symbol *sym) {
  assert(!isSealed);
  if (sym->requiresGOT)
    return;
  LLVM_DEBUG(dbgs() << "addInternalGOTEntry: " << sym->getName() << " "
                    << toString(sym->kind()) << "\n");
  sym->requiresGOT = true;
  if (auto *F = dyn_cast<FunctionSymbol>(sym))
    out.elemSec->addEntry(F);
  internalGotSymbols.push_back(sym);
}

void GlobalSection::generateRelocationCode(raw_ostream &os) const {
  unsigned opcode_ptr_const = config->is64.getValueOr(false)
                                  ? WASM_OPCODE_I64_CONST
                                  : WASM_OPCODE_I32_CONST;
  unsigned opcode_ptr_add = config->is64.getValueOr(false)
                                ? WASM_OPCODE_I64_ADD
                                : WASM_OPCODE_I32_ADD;

  for (const Symbol *sym : internalGotSymbols) {
    if (auto *d = dyn_cast<DefinedData>(sym)) {
      // Get __memory_base
      writeU8(os, WASM_OPCODE_GLOBAL_GET, "GLOBAL_GET");
      writeUleb128(os, WasmSym::memoryBase->getGlobalIndex(), "__memory_base");

      // Add the virtual address of the data symbol
      writeU8(os, opcode_ptr_const, "CONST");
      writeSleb128(os, d->getVirtualAddress(), "offset");
    } else if (auto *f = dyn_cast<FunctionSymbol>(sym)) {
      // Get __table_base
      writeU8(os, WASM_OPCODE_GLOBAL_GET, "GLOBAL_GET");
      writeUleb128(os, WasmSym::tableBase->getGlobalIndex(), "__table_base");

      // Add the table index to __table_base
      writeU8(os, opcode_ptr_const, "CONST");
      writeSleb128(os, f->getTableIndex(), "offset");
    } else {
      assert(isa<UndefinedData>(sym));
      continue;
    }
    writeU8(os, opcode_ptr_add, "ADD");
    writeU8(os, WASM_OPCODE_GLOBAL_SET, "GLOBAL_SET");
    writeUleb128(os, sym->getGOTIndex(), "got_entry");
  }
}

void GlobalSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, numGlobals(), "global count");
  for (InputGlobal *g : inputGlobals)
    writeGlobal(os, g->global);
  // TODO(wvo): when do these need I64_CONST?
  for (const Symbol *sym : internalGotSymbols) {
    WasmGlobal global;
    global.Type = {WASM_TYPE_I32, config->isPic};
    global.InitExpr.Opcode = WASM_OPCODE_I32_CONST;
    if (auto *d = dyn_cast<DefinedData>(sym))
      global.InitExpr.Value.Int32 = d->getVirtualAddress();
    else if (auto *f = dyn_cast<FunctionSymbol>(sym))
      global.InitExpr.Value.Int32 = f->getTableIndex();
    else {
      assert(isa<UndefinedData>(sym));
      global.InitExpr.Value.Int32 = 0;
    }
    writeGlobal(os, global);
  }
  for (const DefinedData *sym : dataAddressGlobals) {
    WasmGlobal global;
    global.Type = {WASM_TYPE_I32, false};
    global.InitExpr.Opcode = WASM_OPCODE_I32_CONST;
    global.InitExpr.Value.Int32 = sym->getVirtualAddress();
    writeGlobal(os, global);
  }
}

void GlobalSection::addGlobal(InputGlobal *global) {
  assert(!isSealed);
  if (!global->live)
    return;
  inputGlobals.push_back(global);
}

void ExportSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, exports.size(), "export count");
  for (const WasmExport &export_ : exports)
    writeExport(os, export_);
}

bool StartSection::isNeeded() const {
  return !config->relocatable && hasInitializedSegments && config->sharedMemory;
}

void StartSection::writeBody() {
  raw_ostream &os = bodyOutputStream;
  writeUleb128(os, WasmSym::initMemory->getFunctionIndex(), "function index");
}

void ElemSection::addEntry(FunctionSymbol *sym) {
  if (sym->hasTableIndex())
    return;
  sym->setTableIndex(config->tableBase + indirectFunctions.size());
  indirectFunctions.emplace_back(sym);
}

void ElemSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, 1, "segment count");
  writeUleb128(os, 0, "table index");
  WasmInitExpr initExpr;
  if (config->isPic) {
    initExpr.Opcode = WASM_OPCODE_GLOBAL_GET;
    initExpr.Value.Global = WasmSym::tableBase->getGlobalIndex();
  } else {
    initExpr.Opcode = WASM_OPCODE_I32_CONST;
    initExpr.Value.Int32 = config->tableBase;
  }
  writeInitExpr(os, initExpr);
  writeUleb128(os, indirectFunctions.size(), "elem count");

  uint32_t tableIndex = config->tableBase;
  for (const FunctionSymbol *sym : indirectFunctions) {
    assert(sym->getTableIndex() == tableIndex);
    writeUleb128(os, sym->getFunctionIndex(), "function index");
    ++tableIndex;
  }
}

DataCountSection::DataCountSection(ArrayRef<OutputSegment *> segments)
    : SyntheticSection(llvm::wasm::WASM_SEC_DATACOUNT),
      numSegments(std::count_if(
          segments.begin(), segments.end(),
          [](OutputSegment *const segment) { return !segment->isBss; })) {}

void DataCountSection::writeBody() {
  writeUleb128(bodyOutputStream, numSegments, "data count");
}

bool DataCountSection::isNeeded() const {
  return numSegments && config->sharedMemory;
}

void LinkingSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, WasmMetadataVersion, "Version");

  if (!symtabEntries.empty()) {
    SubSection sub(WASM_SYMBOL_TABLE);
    writeUleb128(sub.os, symtabEntries.size(), "num symbols");

    for (const Symbol *sym : symtabEntries) {
      assert(sym->isDefined() || sym->isUndefined());
      WasmSymbolType kind = sym->getWasmType();
      uint32_t flags = sym->flags;

      writeU8(sub.os, kind, "sym kind");
      writeUleb128(sub.os, flags, "sym flags");

      if (auto *f = dyn_cast<FunctionSymbol>(sym)) {
        writeUleb128(sub.os, f->getFunctionIndex(), "index");
        if (sym->isDefined() || (flags & WASM_SYMBOL_EXPLICIT_NAME) != 0)
          writeStr(sub.os, sym->getName(), "sym name");
      } else if (auto *g = dyn_cast<GlobalSymbol>(sym)) {
        writeUleb128(sub.os, g->getGlobalIndex(), "index");
        if (sym->isDefined() || (flags & WASM_SYMBOL_EXPLICIT_NAME) != 0)
          writeStr(sub.os, sym->getName(), "sym name");
      } else if (auto *e = dyn_cast<EventSymbol>(sym)) {
        writeUleb128(sub.os, e->getEventIndex(), "index");
        if (sym->isDefined() || (flags & WASM_SYMBOL_EXPLICIT_NAME) != 0)
          writeStr(sub.os, sym->getName(), "sym name");
      } else if (isa<DataSymbol>(sym)) {
        writeStr(sub.os, sym->getName(), "sym name");
        if (auto *dataSym = dyn_cast<DefinedData>(sym)) {
          writeUleb128(sub.os, dataSym->getOutputSegmentIndex(), "index");
          writeUleb128(sub.os, dataSym->getOutputSegmentOffset(),
                       "data offset");
          writeUleb128(sub.os, dataSym->getSize(), "data size");
        }
      } else {
        auto *s = cast<OutputSectionSymbol>(sym);
        writeUleb128(sub.os, s->section->sectionIndex, "sym section index");
      }
    }

    sub.writeTo(os);
  }

  if (dataSegments.size()) {
    SubSection sub(WASM_SEGMENT_INFO);
    writeUleb128(sub.os, dataSegments.size(), "num data segments");
    for (const OutputSegment *s : dataSegments) {
      writeStr(sub.os, s->name, "segment name");
      writeUleb128(sub.os, s->alignment, "alignment");
      writeUleb128(sub.os, 0, "flags");
    }
    sub.writeTo(os);
  }

  if (!initFunctions.empty()) {
    SubSection sub(WASM_INIT_FUNCS);
    writeUleb128(sub.os, initFunctions.size(), "num init functions");
    for (const WasmInitEntry &f : initFunctions) {
      writeUleb128(sub.os, f.priority, "priority");
      writeUleb128(sub.os, f.sym->getOutputSymbolIndex(), "function index");
    }
    sub.writeTo(os);
  }

  struct ComdatEntry {
    unsigned kind;
    uint32_t index;
  };
  std::map<StringRef, std::vector<ComdatEntry>> comdats;

  for (const InputFunction *f : out.functionSec->inputFunctions) {
    StringRef comdat = f->getComdatName();
    if (!comdat.empty())
      comdats[comdat].emplace_back(
          ComdatEntry{WASM_COMDAT_FUNCTION, f->getFunctionIndex()});
  }
  for (uint32_t i = 0; i < dataSegments.size(); ++i) {
    const auto &inputSegments = dataSegments[i]->inputSegments;
    if (inputSegments.empty())
      continue;
    StringRef comdat = inputSegments[0]->getComdatName();
#ifndef NDEBUG
    for (const InputSegment *isec : inputSegments)
      assert(isec->getComdatName() == comdat);
#endif
    if (!comdat.empty())
      comdats[comdat].emplace_back(ComdatEntry{WASM_COMDAT_DATA, i});
  }

  if (!comdats.empty()) {
    SubSection sub(WASM_COMDAT_INFO);
    writeUleb128(sub.os, comdats.size(), "num comdats");
    for (const auto &c : comdats) {
      writeStr(sub.os, c.first, "comdat name");
      writeUleb128(sub.os, 0, "comdat flags"); // flags for future use
      writeUleb128(sub.os, c.second.size(), "num entries");
      for (const ComdatEntry &entry : c.second) {
        writeU8(sub.os, entry.kind, "entry kind");
        writeUleb128(sub.os, entry.index, "entry index");
      }
    }
    sub.writeTo(os);
  }
}

void LinkingSection::addToSymtab(Symbol *sym) {
  sym->setOutputSymbolIndex(symtabEntries.size());
  symtabEntries.emplace_back(sym);
}

unsigned NameSection::numNamedFunctions() const {
  unsigned numNames = out.importSec->getNumImportedFunctions();

  for (const InputFunction *f : out.functionSec->inputFunctions)
    if (!f->getName().empty() || !f->getDebugName().empty())
      ++numNames;

  return numNames;
}

unsigned NameSection::numNamedGlobals() const {
  unsigned numNames = out.importSec->getNumImportedGlobals();

  for (const InputGlobal *g : out.globalSec->inputGlobals)
    if (!g->getName().empty())
      ++numNames;

  numNames += out.globalSec->internalGotSymbols.size();
  return numNames;
}

// Create the custom "name" section containing debug symbol names.
void NameSection::writeBody() {
  unsigned count = numNamedFunctions();
  if (count) {
    SubSection sub(WASM_NAMES_FUNCTION);
    writeUleb128(sub.os, count, "name count");

    // Function names appear in function index order.  As it happens
    // importedSymbols and inputFunctions are numbered in order with imported
    // functions coming first.
    for (const Symbol *s : out.importSec->importedSymbols) {
      if (auto *f = dyn_cast<FunctionSymbol>(s)) {
        writeUleb128(sub.os, f->getFunctionIndex(), "func index");
        writeStr(sub.os, toString(*s), "symbol name");
      }
    }
    for (const InputFunction *f : out.functionSec->inputFunctions) {
      if (!f->getName().empty()) {
        writeUleb128(sub.os, f->getFunctionIndex(), "func index");
        if (!f->getDebugName().empty()) {
          writeStr(sub.os, f->getDebugName(), "symbol name");
        } else {
          writeStr(sub.os, maybeDemangleSymbol(f->getName()), "symbol name");
        }
      }
    }
    sub.writeTo(bodyOutputStream);
  }

  count = numNamedGlobals();
  if (count) {
    SubSection sub(WASM_NAMES_GLOBAL);
    writeUleb128(sub.os, count, "name count");

    for (const Symbol *s : out.importSec->importedSymbols) {
      if (auto *g = dyn_cast<GlobalSymbol>(s)) {
        writeUleb128(sub.os, g->getGlobalIndex(), "global index");
        writeStr(sub.os, toString(*s), "symbol name");
      }
    }
    for (const Symbol *s : out.importSec->gotSymbols) {
      writeUleb128(sub.os, s->getGOTIndex(), "global index");
      writeStr(sub.os, toString(*s), "symbol name");
    }
    for (const InputGlobal *g : out.globalSec->inputGlobals) {
      if (!g->getName().empty()) {
        writeUleb128(sub.os, g->getGlobalIndex(), "global index");
        writeStr(sub.os, maybeDemangleSymbol(g->getName()), "symbol name");
      }
    }
    for (Symbol *s : out.globalSec->internalGotSymbols) {
      writeUleb128(sub.os, s->getGOTIndex(), "global index");
      writeStr(sub.os, toString(*s), "symbol name");
    }

    sub.writeTo(bodyOutputStream);
  }
}

void ProducersSection::addInfo(const WasmProducerInfo &info) {
  for (auto &producers :
       {std::make_pair(&info.Languages, &languages),
        std::make_pair(&info.Tools, &tools), std::make_pair(&info.SDKs, &sDKs)})
    for (auto &producer : *producers.first)
      if (producers.second->end() ==
          llvm::find_if(*producers.second,
                        [&](std::pair<std::string, std::string> seen) {
                          return seen.first == producer.first;
                        }))
        producers.second->push_back(producer);
}

void ProducersSection::writeBody() {
  auto &os = bodyOutputStream;
  writeUleb128(os, fieldCount(), "field count");
  for (auto &field :
       {std::make_pair("language", languages),
        std::make_pair("processed-by", tools), std::make_pair("sdk", sDKs)}) {
    if (field.second.empty())
      continue;
    writeStr(os, field.first, "field name");
    writeUleb128(os, field.second.size(), "number of entries");
    for (auto &entry : field.second) {
      writeStr(os, entry.first, "producer name");
      writeStr(os, entry.second, "producer version");
    }
  }
}

void TargetFeaturesSection::writeBody() {
  SmallVector<std::string, 8> emitted(features.begin(), features.end());
  llvm::sort(emitted);
  auto &os = bodyOutputStream;
  writeUleb128(os, emitted.size(), "feature count");
  for (auto &feature : emitted) {
    writeU8(os, WASM_FEATURE_PREFIX_USED, "feature used prefix");
    writeStr(os, feature, "feature name");
  }
}

void RelocSection::writeBody() {
  uint32_t count = sec->getNumRelocations();
  assert(sec->sectionIndex != UINT32_MAX);
  writeUleb128(bodyOutputStream, sec->sectionIndex, "reloc section");
  writeUleb128(bodyOutputStream, count, "reloc count");
  sec->writeRelocations(bodyOutputStream);
}

} // namespace wasm
} // namespace lld
