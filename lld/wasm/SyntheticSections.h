//===- SyntheticSection.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Synthetic sections represent chunks of linker-created data. If you
// need to create a chunk of data that to be included in some section
// in the result, you probably want to create that as a synthetic section.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_SYNTHETIC_SECTIONS_H
#define LLD_WASM_SYNTHETIC_SECTIONS_H

#include "OutputSections.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Object/WasmTraits.h"

#define DEBUG_TYPE "lld"

namespace lld {
namespace wasm {

// An init entry to be written to either the synthetic init func or the
// linking metadata.
struct WasmInitEntry {
  const FunctionSymbol *Sym;
  uint32_t Priority;
};

class SyntheticSection : public OutputSection {
public:
  SyntheticSection(uint32_t Type, std::string Name = "")
      : OutputSection(Type, Name), BodyOutputStream(Body) {
    if (!Name.empty())
      writeStr(BodyOutputStream, Name, "section name");
  }

  void writeTo(uint8_t *Buf) override {
    assert(Offset);
    log("writing " + toString(*this));
    memcpy(Buf + Offset, Header.data(), Header.size());
    memcpy(Buf + Offset + Header.size(), Body.data(), Body.size());
  }

  size_t getSize() const override { return Header.size() + Body.size(); }

  virtual void writeBody() {}

  void finalizeContents() override {
    writeBody();
    BodyOutputStream.flush();
    createHeader(Body.size());
  }

  raw_ostream &getStream() { return BodyOutputStream; }

  std::string Body;

protected:
  llvm::raw_string_ostream BodyOutputStream;
};

// Create the custom "dylink" section containing information for the dynamic
// linker.
// See
// https://github.com/WebAssembly/tool-conventions/blob/master/DynamicLinking.md
class DylinkSection : public SyntheticSection {
public:
  DylinkSection() : SyntheticSection(llvm::wasm::WASM_SEC_CUSTOM, "dylink") {}
  bool isNeeded() const override { return Config->Pic; }
  void writeBody() override;

  uint32_t MemAlign = 0;
  uint32_t MemSize = 0;
};

class TypeSection : public SyntheticSection {
public:
  TypeSection() : SyntheticSection(llvm::wasm::WASM_SEC_TYPE) {}

  bool isNeeded() const override { return Types.size() > 0; };
  void writeBody() override;
  uint32_t registerType(const WasmSignature &Sig);
  uint32_t lookupType(const WasmSignature &Sig);

protected:
  std::vector<const WasmSignature *> Types;
  llvm::DenseMap<WasmSignature, int32_t> TypeIndices;
};

class ImportSection : public SyntheticSection {
public:
  ImportSection() : SyntheticSection(llvm::wasm::WASM_SEC_IMPORT) {}
  bool isNeeded() const override { return numImports() > 0; }
  void writeBody() override;
  void addImport(Symbol *Sym);
  void addGOTEntry(Symbol *Sym);
  void seal() { IsSealed = true; }
  uint32_t numImports() const;
  uint32_t numImportedGlobals() const {
    assert(IsSealed);
    return NumImportedGlobals;
  }
  uint32_t numImportedFunctions() const {
    assert(IsSealed);
    return NumImportedFunctions;
  }
  uint32_t numImportedEvents() const {
    assert(IsSealed);
    return NumImportedEvents;
  }

  std::vector<const Symbol *> ImportedSymbols;

protected:
  bool IsSealed = false;
  unsigned NumImportedGlobals = 0;
  unsigned NumImportedFunctions = 0;
  unsigned NumImportedEvents = 0;
  std::vector<const Symbol *> GOTSymbols;
};

class FunctionSection : public SyntheticSection {
public:
  FunctionSection() : SyntheticSection(llvm::wasm::WASM_SEC_FUNCTION) {}

  bool isNeeded() const override { return InputFunctions.size() > 0; };
  void writeBody() override;
  void addFunction(InputFunction *Func);

  std::vector<InputFunction *> InputFunctions;

protected:
};

class MemorySection : public SyntheticSection {
public:
  MemorySection() : SyntheticSection(llvm::wasm::WASM_SEC_MEMORY) {}

  bool isNeeded() const override { return !Config->ImportMemory; }
  void writeBody() override;

  uint32_t NumMemoryPages = 0;
  uint32_t MaxMemoryPages = 0;
};

class TableSection : public SyntheticSection {
public:
  TableSection() : SyntheticSection(llvm::wasm::WASM_SEC_TABLE) {}

  bool isNeeded() const override {
    // Always output a table section (or table import), even if there are no
    // indirect calls.  There are two reasons for this:
    //  1. For executables it is useful to have an empty table slot at 0
    //     which can be filled with a null function call handler.
    //  2. If we don't do this, any program that contains a call_indirect but
    //     no address-taken function will fail at validation time since it is
    //     a validation error to include a call_indirect instruction if there
    //     is not table.
    return !Config->ImportTable;
  }

  void writeBody() override;
};

class GlobalSection : public SyntheticSection {
public:
  GlobalSection() : SyntheticSection(llvm::wasm::WASM_SEC_GLOBAL) {}
  uint32_t numGlobals() const {
    return InputGlobals.size() + DefinedFakeGlobals.size();
  }
  bool isNeeded() const override { return numGlobals() > 0; }
  void writeBody() override;
  void addGlobal(InputGlobal *Global);

  std::vector<const DefinedData *> DefinedFakeGlobals;
  std::vector<InputGlobal *> InputGlobals;
};

// The event section contains a list of declared wasm events associated with the
// module. Currently the only supported event kind is exceptions. A single event
// entry represents a single event with an event tag. All C++ exceptions are
// represented by a single event. An event entry in this section contains
// information on what kind of event it is (e.g. exception) and the type of
// values contained in a single event object. (In wasm, an event can contain
// multiple values of primitive types. But for C++ exceptions, we just throw a
// pointer which is an i32 value (for wasm32 architecture), so the signature of
// C++ exception is (i32)->(void), because all event types are assumed to have
// void return type to share WasmSignature with functions.)
class EventSection : public SyntheticSection {
public:
  EventSection() : SyntheticSection(llvm::wasm::WASM_SEC_EVENT) {}
  void writeBody() override;
  bool isNeeded() const override { return InputEvents.size() > 0; }
  void addEvent(InputEvent *Event);

  std::vector<InputEvent *> InputEvents;
};

class ExportSection : public SyntheticSection {
public:
  ExportSection() : SyntheticSection(llvm::wasm::WASM_SEC_EXPORT) {}
  bool isNeeded() const override { return Exports.size() > 0; }
  void writeBody() override;

  std::vector<llvm::wasm::WasmExport> Exports;
};

class ElemSection : public SyntheticSection {
public:
  ElemSection(uint32_t Offset)
      : SyntheticSection(llvm::wasm::WASM_SEC_ELEM), ElemOffset(Offset) {}
  bool isNeeded() const override { return IndirectFunctions.size() > 0; };
  void writeBody() override;
  void addEntry(FunctionSymbol *Sym);
  uint32_t numEntries() const { return IndirectFunctions.size(); }
  uint32_t ElemOffset;

protected:
  std::vector<const FunctionSymbol *> IndirectFunctions;
};

class DataCountSection : public SyntheticSection {
public:
  DataCountSection(uint32_t NumSegments)
      : SyntheticSection(llvm::wasm::WASM_SEC_DATACOUNT),
        NumSegments(NumSegments) {}
  bool isNeeded() const override;
  void writeBody() override;

protected:
  uint32_t NumSegments;
};

// Create the custom "linking" section containing linker metadata.
// This is only created when relocatable output is requested.
class LinkingSection : public SyntheticSection {
public:
  LinkingSection(const std::vector<WasmInitEntry> &InitFunctions,
                 const std::vector<OutputSegment *> &DataSegments)
      : SyntheticSection(llvm::wasm::WASM_SEC_CUSTOM, "linking"),
        InitFunctions(InitFunctions), DataSegments(DataSegments) {}
  bool isNeeded() const override {
    return Config->Relocatable || Config->EmitRelocs;
  }
  void writeBody() override;
  void addToSymtab(Symbol *Sym);

protected:
  std::vector<const Symbol *> SymtabEntries;
  llvm::StringMap<uint32_t> SectionSymbolIndices;
  const std::vector<WasmInitEntry> &InitFunctions;
  const std::vector<OutputSegment *> &DataSegments;
};

// Create the custom "name" section containing debug symbol names.
class NameSection : public SyntheticSection {
public:
  NameSection() : SyntheticSection(llvm::wasm::WASM_SEC_CUSTOM, "name") {}
  bool isNeeded() const override {
    return !Config->StripDebug && !Config->StripAll && numNames() > 0;
  }
  void writeBody() override;
  unsigned numNames() const;
};

class ProducersSection : public SyntheticSection {
public:
  ProducersSection()
      : SyntheticSection(llvm::wasm::WASM_SEC_CUSTOM, "producers") {}
  bool isNeeded() const override {
    return !Config->StripAll && fieldCount() > 0;
  }
  void writeBody() override;
  void addInfo(const llvm::wasm::WasmProducerInfo &Info);

protected:
  int fieldCount() const {
    return int(!Languages.empty()) + int(!Tools.empty()) + int(!SDKs.empty());
  }
  SmallVector<std::pair<std::string, std::string>, 8> Languages;
  SmallVector<std::pair<std::string, std::string>, 8> Tools;
  SmallVector<std::pair<std::string, std::string>, 8> SDKs;
};

class TargetFeaturesSection : public SyntheticSection {
public:
  TargetFeaturesSection()
      : SyntheticSection(llvm::wasm::WASM_SEC_CUSTOM, "target_features") {}
  bool isNeeded() const override {
    return !Config->StripAll && Features.size() > 0;
  }
  void writeBody() override;

  llvm::SmallSet<std::string, 8> Features;
};

class RelocSection : public SyntheticSection {
public:
  RelocSection(StringRef Name, OutputSection *Sec)
      : SyntheticSection(llvm::wasm::WASM_SEC_CUSTOM, Name), Sec(Sec) {}
  void writeBody() override;
  bool isNeeded() const override { return Sec->numRelocations() > 0; };

protected:
  OutputSection *Sec;
};

// Linker generated output sections
struct OutStruct {
  DylinkSection *DylinkSec;
  TypeSection *TypeSec;
  FunctionSection *FunctionSec;
  ImportSection *ImportSec;
  TableSection *TableSec;
  MemorySection *MemorySec;
  GlobalSection *GlobalSec;
  EventSection *EventSec;
  ExportSection *ExportSec;
  ElemSection *ElemSec;
  DataCountSection *DataCountSec;
  LinkingSection *LinkingSec;
  NameSection *NameSec;
  ProducersSection *ProducersSec;
  TargetFeaturesSection *TargetFeaturesSec;
};

extern OutStruct Out;

} // namespace wasm
} // namespace lld

#endif
