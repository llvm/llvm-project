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

using namespace lld;
using namespace lld::wasm;

OutStruct lld::wasm::Out;

namespace {

// Some synthetic sections (e.g. "name" and "linking") have subsections.
// Just like the synthetic sections themselves these need to be created before
// they can be written out (since they are preceded by their length). This
// class is used to create subsections and then write them into the stream
// of the parent section.
class SubSection {
public:
  explicit SubSection(uint32_t Type) : Type(Type) {}

  void writeTo(raw_ostream &To) {
    OS.flush();
    writeUleb128(To, Type, "subsection type");
    writeUleb128(To, Body.size(), "subsection size");
    To.write(Body.data(), Body.size());
  }

private:
  uint32_t Type;
  std::string Body;

public:
  raw_string_ostream OS{Body};
};

} // namespace

void DylinkSection::writeBody() {
  raw_ostream &OS = BodyOutputStream;

  writeUleb128(OS, MemSize, "MemSize");
  writeUleb128(OS, MemAlign, "MemAlign");
  writeUleb128(OS, Out.ElemSec->numEntries(), "TableSize");
  writeUleb128(OS, 0, "TableAlign");
  writeUleb128(OS, Symtab->SharedFiles.size(), "Needed");
  for (auto *SO : Symtab->SharedFiles)
    writeStr(OS, llvm::sys::path::filename(SO->getName()), "so name");
}

uint32_t TypeSection::registerType(const WasmSignature &Sig) {
  auto Pair = TypeIndices.insert(std::make_pair(Sig, Types.size()));
  if (Pair.second) {
    LLVM_DEBUG(llvm::dbgs() << "type " << toString(Sig) << "\n");
    Types.push_back(&Sig);
  }
  return Pair.first->second;
}

uint32_t TypeSection::lookupType(const WasmSignature &Sig) {
  auto It = TypeIndices.find(Sig);
  if (It == TypeIndices.end()) {
    error("type not found: " + toString(Sig));
    return 0;
  }
  return It->second;
}

void TypeSection::writeBody() {
  writeUleb128(BodyOutputStream, Types.size(), "type count");
  for (const WasmSignature *Sig : Types)
    writeSig(BodyOutputStream, *Sig);
}

uint32_t ImportSection::numImports() const {
  assert(IsSealed);
  uint32_t NumImports = ImportedSymbols.size() + GOTSymbols.size();
  if (Config->ImportMemory)
    ++NumImports;
  if (Config->ImportTable)
    ++NumImports;
  return NumImports;
}

void ImportSection::addGOTEntry(Symbol *Sym) {
  assert(!IsSealed);
  if (Sym->hasGOTIndex())
    return;
  Sym->setGOTIndex(NumImportedGlobals++);
  GOTSymbols.push_back(Sym);
}

void ImportSection::addImport(Symbol *Sym) {
  assert(!IsSealed);
  ImportedSymbols.emplace_back(Sym);
  if (auto *F = dyn_cast<FunctionSymbol>(Sym))
    F->setFunctionIndex(NumImportedFunctions++);
  else if (auto *G = dyn_cast<GlobalSymbol>(Sym))
    G->setGlobalIndex(NumImportedGlobals++);
  else
    cast<EventSymbol>(Sym)->setEventIndex(NumImportedEvents++);
}

void ImportSection::writeBody() {
  raw_ostream &OS = BodyOutputStream;

  writeUleb128(OS, numImports(), "import count");

  if (Config->ImportMemory) {
    WasmImport Import;
    Import.Module = DefaultModule;
    Import.Field = "memory";
    Import.Kind = WASM_EXTERNAL_MEMORY;
    Import.Memory.Flags = 0;
    Import.Memory.Initial = Out.MemorySec->NumMemoryPages;
    if (Out.MemorySec->MaxMemoryPages != 0 || Config->SharedMemory) {
      Import.Memory.Flags |= WASM_LIMITS_FLAG_HAS_MAX;
      Import.Memory.Maximum = Out.MemorySec->MaxMemoryPages;
    }
    if (Config->SharedMemory)
      Import.Memory.Flags |= WASM_LIMITS_FLAG_IS_SHARED;
    writeImport(OS, Import);
  }

  if (Config->ImportTable) {
    uint32_t TableSize = Out.ElemSec->ElemOffset + Out.ElemSec->numEntries();
    WasmImport Import;
    Import.Module = DefaultModule;
    Import.Field = FunctionTableName;
    Import.Kind = WASM_EXTERNAL_TABLE;
    Import.Table.ElemType = WASM_TYPE_FUNCREF;
    Import.Table.Limits = {0, TableSize, 0};
    writeImport(OS, Import);
  }

  for (const Symbol *Sym : ImportedSymbols) {
    WasmImport Import;
    if (auto *F = dyn_cast<UndefinedFunction>(Sym)) {
      Import.Field = F->ImportName;
      Import.Module = F->ImportModule;
    } else if (auto *G = dyn_cast<UndefinedGlobal>(Sym)) {
      Import.Field = G->ImportName;
      Import.Module = G->ImportModule;
    } else {
      Import.Field = Sym->getName();
      Import.Module = DefaultModule;
    }

    if (auto *FunctionSym = dyn_cast<FunctionSymbol>(Sym)) {
      Import.Kind = WASM_EXTERNAL_FUNCTION;
      Import.SigIndex = Out.TypeSec->lookupType(*FunctionSym->Signature);
    } else if (auto *GlobalSym = dyn_cast<GlobalSymbol>(Sym)) {
      Import.Kind = WASM_EXTERNAL_GLOBAL;
      Import.Global = *GlobalSym->getGlobalType();
    } else {
      auto *EventSym = cast<EventSymbol>(Sym);
      Import.Kind = WASM_EXTERNAL_EVENT;
      Import.Event.Attribute = EventSym->getEventType()->Attribute;
      Import.Event.SigIndex = Out.TypeSec->lookupType(*EventSym->Signature);
    }
    writeImport(OS, Import);
  }

  for (const Symbol *Sym : GOTSymbols) {
    WasmImport Import;
    Import.Kind = WASM_EXTERNAL_GLOBAL;
    Import.Global = {WASM_TYPE_I32, true};
    if (isa<DataSymbol>(Sym))
      Import.Module = "GOT.mem";
    else
      Import.Module = "GOT.func";
    Import.Field = Sym->getName();
    writeImport(OS, Import);
  }
}

void FunctionSection::writeBody() {
  raw_ostream &OS = BodyOutputStream;

  writeUleb128(OS, InputFunctions.size(), "function count");
  for (const InputFunction *Func : InputFunctions)
    writeUleb128(OS, Out.TypeSec->lookupType(Func->Signature), "sig index");
}

void FunctionSection::addFunction(InputFunction *Func) {
  if (!Func->Live)
    return;
  uint32_t FunctionIndex =
      Out.ImportSec->numImportedFunctions() + InputFunctions.size();
  InputFunctions.emplace_back(Func);
  Func->setFunctionIndex(FunctionIndex);
}

void TableSection::writeBody() {
  uint32_t TableSize = Out.ElemSec->ElemOffset + Out.ElemSec->numEntries();

  raw_ostream &OS = BodyOutputStream;
  writeUleb128(OS, 1, "table count");
  WasmLimits Limits = {WASM_LIMITS_FLAG_HAS_MAX, TableSize, TableSize};
  writeTableType(OS, WasmTable{WASM_TYPE_FUNCREF, Limits});
}

void MemorySection::writeBody() {
  raw_ostream &OS = BodyOutputStream;

  bool HasMax = MaxMemoryPages != 0 || Config->SharedMemory;
  writeUleb128(OS, 1, "memory count");
  unsigned Flags = 0;
  if (HasMax)
    Flags |= WASM_LIMITS_FLAG_HAS_MAX;
  if (Config->SharedMemory)
    Flags |= WASM_LIMITS_FLAG_IS_SHARED;
  writeUleb128(OS, Flags, "memory limits flags");
  writeUleb128(OS, NumMemoryPages, "initial pages");
  if (HasMax)
    writeUleb128(OS, MaxMemoryPages, "max pages");
}

void GlobalSection::writeBody() {
  raw_ostream &OS = BodyOutputStream;

  writeUleb128(OS, numGlobals(), "global count");
  for (const InputGlobal *G : InputGlobals)
    writeGlobal(OS, G->Global);
  for (const DefinedData *Sym : DefinedFakeGlobals) {
    WasmGlobal Global;
    Global.Type = {WASM_TYPE_I32, false};
    Global.InitExpr.Opcode = WASM_OPCODE_I32_CONST;
    Global.InitExpr.Value.Int32 = Sym->getVirtualAddress();
    writeGlobal(OS, Global);
  }
}

void GlobalSection::addGlobal(InputGlobal *Global) {
  if (!Global->Live)
    return;
  uint32_t GlobalIndex =
      Out.ImportSec->numImportedGlobals() + InputGlobals.size();
  LLVM_DEBUG(dbgs() << "addGlobal: " << GlobalIndex << "\n");
  Global->setGlobalIndex(GlobalIndex);
  Out.GlobalSec->InputGlobals.push_back(Global);
}

void EventSection::writeBody() {
  raw_ostream &OS = BodyOutputStream;

  writeUleb128(OS, InputEvents.size(), "event count");
  for (InputEvent *E : InputEvents) {
    E->Event.Type.SigIndex = Out.TypeSec->lookupType(E->Signature);
    writeEvent(OS, E->Event);
  }
}

void EventSection::addEvent(InputEvent *Event) {
  if (!Event->Live)
    return;
  uint32_t EventIndex = Out.ImportSec->numImportedEvents() + InputEvents.size();
  LLVM_DEBUG(dbgs() << "addEvent: " << EventIndex << "\n");
  Event->setEventIndex(EventIndex);
  InputEvents.push_back(Event);
}

void ExportSection::writeBody() {
  raw_ostream &OS = BodyOutputStream;

  writeUleb128(OS, Exports.size(), "export count");
  for (const WasmExport &Export : Exports)
    writeExport(OS, Export);
}

void ElemSection::addEntry(FunctionSymbol *Sym) {
  if (Sym->hasTableIndex())
    return;
  Sym->setTableIndex(ElemOffset + IndirectFunctions.size());
  IndirectFunctions.emplace_back(Sym);
}

void ElemSection::writeBody() {
  raw_ostream &OS = BodyOutputStream;

  writeUleb128(OS, 1, "segment count");
  writeUleb128(OS, 0, "table index");
  WasmInitExpr InitExpr;
  if (Config->Pic) {
    InitExpr.Opcode = WASM_OPCODE_GLOBAL_GET;
    InitExpr.Value.Global = WasmSym::TableBase->getGlobalIndex();
  } else {
    InitExpr.Opcode = WASM_OPCODE_I32_CONST;
    InitExpr.Value.Int32 = ElemOffset;
  }
  writeInitExpr(OS, InitExpr);
  writeUleb128(OS, IndirectFunctions.size(), "elem count");

  uint32_t TableIndex = ElemOffset;
  for (const FunctionSymbol *Sym : IndirectFunctions) {
    assert(Sym->getTableIndex() == TableIndex);
    writeUleb128(OS, Sym->getFunctionIndex(), "function index");
    ++TableIndex;
  }
}

void DataCountSection::writeBody() {
  writeUleb128(BodyOutputStream, NumSegments, "data count");
}

bool DataCountSection::isNeeded() const {
  return NumSegments && Out.TargetFeaturesSec->Features.count("bulk-memory");
}

static uint32_t getWasmFlags(const Symbol *Sym) {
  uint32_t Flags = 0;
  if (Sym->isLocal())
    Flags |= WASM_SYMBOL_BINDING_LOCAL;
  if (Sym->isWeak())
    Flags |= WASM_SYMBOL_BINDING_WEAK;
  if (Sym->isHidden())
    Flags |= WASM_SYMBOL_VISIBILITY_HIDDEN;
  if (Sym->isUndefined())
    Flags |= WASM_SYMBOL_UNDEFINED;
  if (auto *F = dyn_cast<UndefinedFunction>(Sym)) {
    if (F->getName() != F->ImportName)
      Flags |= WASM_SYMBOL_EXPLICIT_NAME;
  } else if (auto *G = dyn_cast<UndefinedGlobal>(Sym)) {
    if (G->getName() != G->ImportName)
      Flags |= WASM_SYMBOL_EXPLICIT_NAME;
  }
  return Flags;
}

void LinkingSection::writeBody() {
  raw_ostream &OS = BodyOutputStream;

  writeUleb128(OS, WasmMetadataVersion, "Version");

  if (!SymtabEntries.empty()) {
    SubSection Sub(WASM_SYMBOL_TABLE);
    writeUleb128(Sub.OS, SymtabEntries.size(), "num symbols");

    for (const Symbol *Sym : SymtabEntries) {
      assert(Sym->isDefined() || Sym->isUndefined());
      WasmSymbolType Kind = Sym->getWasmType();
      uint32_t Flags = getWasmFlags(Sym);

      writeU8(Sub.OS, Kind, "sym kind");
      writeUleb128(Sub.OS, Flags, "sym flags");

      if (auto *F = dyn_cast<FunctionSymbol>(Sym)) {
        writeUleb128(Sub.OS, F->getFunctionIndex(), "index");
        if (Sym->isDefined() || (Flags & WASM_SYMBOL_EXPLICIT_NAME) != 0)
          writeStr(Sub.OS, Sym->getName(), "sym name");
      } else if (auto *G = dyn_cast<GlobalSymbol>(Sym)) {
        writeUleb128(Sub.OS, G->getGlobalIndex(), "index");
        if (Sym->isDefined() || (Flags & WASM_SYMBOL_EXPLICIT_NAME) != 0)
          writeStr(Sub.OS, Sym->getName(), "sym name");
      } else if (auto *E = dyn_cast<EventSymbol>(Sym)) {
        writeUleb128(Sub.OS, E->getEventIndex(), "index");
        if (Sym->isDefined() || (Flags & WASM_SYMBOL_EXPLICIT_NAME) != 0)
          writeStr(Sub.OS, Sym->getName(), "sym name");
      } else if (isa<DataSymbol>(Sym)) {
        writeStr(Sub.OS, Sym->getName(), "sym name");
        if (auto *DataSym = dyn_cast<DefinedData>(Sym)) {
          writeUleb128(Sub.OS, DataSym->getOutputSegmentIndex(), "index");
          writeUleb128(Sub.OS, DataSym->getOutputSegmentOffset(),
                       "data offset");
          writeUleb128(Sub.OS, DataSym->getSize(), "data size");
        }
      } else {
        auto *S = cast<OutputSectionSymbol>(Sym);
        writeUleb128(Sub.OS, S->Section->SectionIndex, "sym section index");
      }
    }

    Sub.writeTo(OS);
  }

  if (DataSegments.size()) {
    SubSection Sub(WASM_SEGMENT_INFO);
    writeUleb128(Sub.OS, DataSegments.size(), "num data segments");
    for (const OutputSegment *S : DataSegments) {
      writeStr(Sub.OS, S->Name, "segment name");
      writeUleb128(Sub.OS, S->Alignment, "alignment");
      writeUleb128(Sub.OS, 0, "flags");
    }
    Sub.writeTo(OS);
  }

  if (!InitFunctions.empty()) {
    SubSection Sub(WASM_INIT_FUNCS);
    writeUleb128(Sub.OS, InitFunctions.size(), "num init functions");
    for (const WasmInitEntry &F : InitFunctions) {
      writeUleb128(Sub.OS, F.Priority, "priority");
      writeUleb128(Sub.OS, F.Sym->getOutputSymbolIndex(), "function index");
    }
    Sub.writeTo(OS);
  }

  struct ComdatEntry {
    unsigned Kind;
    uint32_t Index;
  };
  std::map<StringRef, std::vector<ComdatEntry>> Comdats;

  for (const InputFunction *F : Out.FunctionSec->InputFunctions) {
    StringRef Comdat = F->getComdatName();
    if (!Comdat.empty())
      Comdats[Comdat].emplace_back(
          ComdatEntry{WASM_COMDAT_FUNCTION, F->getFunctionIndex()});
  }
  for (uint32_t I = 0; I < DataSegments.size(); ++I) {
    const auto &InputSegments = DataSegments[I]->InputSegments;
    if (InputSegments.empty())
      continue;
    StringRef Comdat = InputSegments[0]->getComdatName();
#ifndef NDEBUG
    for (const InputSegment *IS : InputSegments)
      assert(IS->getComdatName() == Comdat);
#endif
    if (!Comdat.empty())
      Comdats[Comdat].emplace_back(ComdatEntry{WASM_COMDAT_DATA, I});
  }

  if (!Comdats.empty()) {
    SubSection Sub(WASM_COMDAT_INFO);
    writeUleb128(Sub.OS, Comdats.size(), "num comdats");
    for (const auto &C : Comdats) {
      writeStr(Sub.OS, C.first, "comdat name");
      writeUleb128(Sub.OS, 0, "comdat flags"); // flags for future use
      writeUleb128(Sub.OS, C.second.size(), "num entries");
      for (const ComdatEntry &Entry : C.second) {
        writeU8(Sub.OS, Entry.Kind, "entry kind");
        writeUleb128(Sub.OS, Entry.Index, "entry index");
      }
    }
    Sub.writeTo(OS);
  }
}

void LinkingSection::addToSymtab(Symbol *Sym) {
  Sym->setOutputSymbolIndex(SymtabEntries.size());
  SymtabEntries.emplace_back(Sym);
}

unsigned NameSection::numNames() const {
  unsigned NumNames = Out.ImportSec->numImportedFunctions();
  for (const InputFunction *F : Out.FunctionSec->InputFunctions)
    if (!F->getName().empty() || !F->getDebugName().empty())
      ++NumNames;

  return NumNames;
}

// Create the custom "name" section containing debug symbol names.
void NameSection::writeBody() {
  SubSection Sub(WASM_NAMES_FUNCTION);
  writeUleb128(Sub.OS, numNames(), "name count");

  // Names must appear in function index order.  As it happens ImportedSymbols
  // and InputFunctions are numbered in order with imported functions coming
  // first.
  for (const Symbol *S : Out.ImportSec->ImportedSymbols) {
    if (auto *F = dyn_cast<FunctionSymbol>(S)) {
      writeUleb128(Sub.OS, F->getFunctionIndex(), "func index");
      writeStr(Sub.OS, toString(*S), "symbol name");
    }
  }
  for (const InputFunction *F : Out.FunctionSec->InputFunctions) {
    if (!F->getName().empty()) {
      writeUleb128(Sub.OS, F->getFunctionIndex(), "func index");
      if (!F->getDebugName().empty()) {
        writeStr(Sub.OS, F->getDebugName(), "symbol name");
      } else {
        writeStr(Sub.OS, maybeDemangleSymbol(F->getName()), "symbol name");
      }
    }
  }

  Sub.writeTo(BodyOutputStream);
}

void ProducersSection::addInfo(const WasmProducerInfo &Info) {
  for (auto &Producers :
       {std::make_pair(&Info.Languages, &Languages),
        std::make_pair(&Info.Tools, &Tools), std::make_pair(&Info.SDKs, &SDKs)})
    for (auto &Producer : *Producers.first)
      if (Producers.second->end() ==
          llvm::find_if(*Producers.second,
                        [&](std::pair<std::string, std::string> Seen) {
                          return Seen.first == Producer.first;
                        }))
        Producers.second->push_back(Producer);
}

void ProducersSection::writeBody() {
  auto &OS = BodyOutputStream;
  writeUleb128(OS, fieldCount(), "field count");
  for (auto &Field :
       {std::make_pair("language", Languages),
        std::make_pair("processed-by", Tools), std::make_pair("sdk", SDKs)}) {
    if (Field.second.empty())
      continue;
    writeStr(OS, Field.first, "field name");
    writeUleb128(OS, Field.second.size(), "number of entries");
    for (auto &Entry : Field.second) {
      writeStr(OS, Entry.first, "producer name");
      writeStr(OS, Entry.second, "producer version");
    }
  }
}

void TargetFeaturesSection::writeBody() {
  SmallVector<std::string, 8> Emitted(Features.begin(), Features.end());
  llvm::sort(Emitted);
  auto &OS = BodyOutputStream;
  writeUleb128(OS, Emitted.size(), "feature count");
  for (auto &Feature : Emitted) {
    writeU8(OS, WASM_FEATURE_PREFIX_USED, "feature used prefix");
    writeStr(OS, Feature, "feature name");
  }
}

void RelocSection::writeBody() {
  uint32_t Count = Sec->numRelocations();
  assert(Sec->SectionIndex != UINT32_MAX);
  writeUleb128(BodyOutputStream, Sec->SectionIndex, "reloc section");
  writeUleb128(BodyOutputStream, Count, "reloc count");
  Sec->writeRelocations(BodyOutputStream);
}
