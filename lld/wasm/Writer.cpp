//===- Writer.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Writer.h"
#include "Config.h"
#include "InputChunks.h"
#include "InputEvent.h"
#include "InputGlobal.h"
#include "OutputSections.h"
#include "OutputSegment.h"
#include "Relocations.h"
#include "SymbolTable.h"
#include "SyntheticSections.h"
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Strings.h"
#include "lld/Common/Threads.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/Object/WasmTraits.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LEB128.h"

#include <cstdarg>
#include <map>

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;
using namespace lld;
using namespace lld::wasm;

static constexpr int StackAlignment = 16;

namespace {

// The writer writes a SymbolTable result to a file.
class Writer {
public:
  void run();

private:
  void openFile();

  void createApplyRelocationsFunction();
  void createCallCtorsFunction();

  void assignIndexes();
  void populateSymtab();
  void populateProducers();
  void populateTargetFeatures();
  void calculateInitFunctions();
  void calculateImports();
  void calculateExports();
  void calculateCustomSections();
  void calculateTypes();
  void createOutputSegments();
  void layoutMemory();
  void createHeader();

  void addSection(OutputSection *Sec);

  void addSections();
  void addStartStopSymbols(const InputSegment *Seg);

  void createCustomSections();
  void createSyntheticSections();
  void finalizeSections();

  // Custom sections
  void createRelocSections();

  void writeHeader();
  void writeSections();

  uint64_t FileSize = 0;
  uint32_t TableBase = 0;

  std::vector<WasmInitEntry> InitFunctions;
  llvm::StringMap<std::vector<InputSection *>> CustomSectionMapping;

  // Elements that are used to construct the final output
  std::string Header;
  std::vector<OutputSection *> OutputSections;

  std::unique_ptr<FileOutputBuffer> Buffer;

  std::vector<OutputSegment *> Segments;
  llvm::SmallDenseMap<StringRef, OutputSegment *> SegmentMap;
};

} // anonymous namespace

void Writer::calculateCustomSections() {
  log("calculateCustomSections");
  bool StripDebug = Config->StripDebug || Config->StripAll;
  for (ObjFile *File : Symtab->ObjectFiles) {
    for (InputSection *Section : File->CustomSections) {
      StringRef Name = Section->getName();
      // These custom sections are known the linker and synthesized rather than
      // blindly copied
      if (Name == "linking" || Name == "name" || Name == "producers" ||
          Name == "target_features" || Name.startswith("reloc."))
        continue;
      // .. or it is a debug section
      if (StripDebug && Name.startswith(".debug_"))
        continue;
      CustomSectionMapping[Name].push_back(Section);
    }
  }
}

void Writer::createCustomSections() {
  log("createCustomSections");
  for (auto &Pair : CustomSectionMapping) {
    StringRef Name = Pair.first();
    LLVM_DEBUG(dbgs() << "createCustomSection: " << Name << "\n");

    OutputSection *Sec = make<CustomSection>(Name, Pair.second);
    if (Config->Relocatable || Config->EmitRelocs) {
      auto *Sym = make<OutputSectionSymbol>(Sec);
      Out.LinkingSec->addToSymtab(Sym);
      Sec->SectionSym = Sym;
    }
    addSection(Sec);
  }
}

// Create relocations sections in the final output.
// These are only created when relocatable output is requested.
void Writer::createRelocSections() {
  log("createRelocSections");
  // Don't use iterator here since we are adding to OutputSection
  size_t OrigSize = OutputSections.size();
  for (size_t I = 0; I < OrigSize; I++) {
    LLVM_DEBUG(dbgs() << "check section " << I << "\n");
    OutputSection *Sec = OutputSections[I];

    // Count the number of needed sections.
    uint32_t Count = Sec->numRelocations();
    if (!Count)
      continue;

    StringRef Name;
    if (Sec->Type == WASM_SEC_DATA)
      Name = "reloc.DATA";
    else if (Sec->Type == WASM_SEC_CODE)
      Name = "reloc.CODE";
    else if (Sec->Type == WASM_SEC_CUSTOM)
      Name = Saver.save("reloc." + Sec->Name);
    else
      llvm_unreachable(
          "relocations only supported for code, data, or custom sections");

    addSection(make<RelocSection>(Name, Sec));
  }
}

void Writer::populateProducers() {
  for (ObjFile *File : Symtab->ObjectFiles) {
    const WasmProducerInfo &Info = File->getWasmObj()->getProducerInfo();
    Out.ProducersSec->addInfo(Info);
  }
}

void Writer::writeHeader() {
  memcpy(Buffer->getBufferStart(), Header.data(), Header.size());
}

void Writer::writeSections() {
  uint8_t *Buf = Buffer->getBufferStart();
  parallelForEach(OutputSections, [Buf](OutputSection *S) {
    assert(S->isNeeded());
    S->writeTo(Buf);
  });
}

// Fix the memory layout of the output binary.  This assigns memory offsets
// to each of the input data sections as well as the explicit stack region.
// The default memory layout is as follows, from low to high.
//
//  - initialized data (starting at Config->GlobalBase)
//  - BSS data (not currently implemented in llvm)
//  - explicit stack (Config->ZStackSize)
//  - heap start / unallocated
//
// The --stack-first option means that stack is placed before any static data.
// This can be useful since it means that stack overflow traps immediately
// rather than overwriting global data, but also increases code size since all
// static data loads and stores requires larger offsets.
void Writer::layoutMemory() {
  uint32_t MemoryPtr = 0;

  auto PlaceStack = [&]() {
    if (Config->Relocatable || Config->Shared)
      return;
    MemoryPtr = alignTo(MemoryPtr, StackAlignment);
    if (Config->ZStackSize != alignTo(Config->ZStackSize, StackAlignment))
      error("stack size must be " + Twine(StackAlignment) + "-byte aligned");
    log("mem: stack size  = " + Twine(Config->ZStackSize));
    log("mem: stack base  = " + Twine(MemoryPtr));
    MemoryPtr += Config->ZStackSize;
    auto *SP = cast<DefinedGlobal>(WasmSym::StackPointer);
    SP->Global->Global.InitExpr.Value.Int32 = MemoryPtr;
    log("mem: stack top   = " + Twine(MemoryPtr));
  };

  if (Config->StackFirst) {
    PlaceStack();
  } else {
    MemoryPtr = Config->GlobalBase;
    log("mem: global base = " + Twine(Config->GlobalBase));
  }

  uint32_t DataStart = MemoryPtr;

  // Arbitrarily set __dso_handle handle to point to the start of the data
  // segments.
  if (WasmSym::DsoHandle)
    WasmSym::DsoHandle->setVirtualAddress(DataStart);

  Out.DylinkSec->MemAlign = 0;
  for (OutputSegment *Seg : Segments) {
    Out.DylinkSec->MemAlign = std::max(Out.DylinkSec->MemAlign, Seg->Alignment);
    MemoryPtr = alignTo(MemoryPtr, 1ULL << Seg->Alignment);
    Seg->StartVA = MemoryPtr;
    log(formatv("mem: {0,-15} offset={1,-8} size={2,-8} align={3}", Seg->Name,
                MemoryPtr, Seg->Size, Seg->Alignment));
    MemoryPtr += Seg->Size;
  }

  // TODO: Add .bss space here.
  if (WasmSym::DataEnd)
    WasmSym::DataEnd->setVirtualAddress(MemoryPtr);

  log("mem: static data = " + Twine(MemoryPtr - DataStart));

  if (Config->Shared) {
    Out.DylinkSec->MemSize = MemoryPtr;
    return;
  }

  if (!Config->StackFirst)
    PlaceStack();

  // Set `__heap_base` to directly follow the end of the stack or global data.
  // The fact that this comes last means that a malloc/brk implementation
  // can grow the heap at runtime.
  log("mem: heap base   = " + Twine(MemoryPtr));
  if (WasmSym::HeapBase)
    WasmSym::HeapBase->setVirtualAddress(MemoryPtr);

  if (Config->InitialMemory != 0) {
    if (Config->InitialMemory != alignTo(Config->InitialMemory, WasmPageSize))
      error("initial memory must be " + Twine(WasmPageSize) + "-byte aligned");
    if (MemoryPtr > Config->InitialMemory)
      error("initial memory too small, " + Twine(MemoryPtr) + " bytes needed");
    else
      MemoryPtr = Config->InitialMemory;
  }
  Out.DylinkSec->MemSize = MemoryPtr;
  Out.MemorySec->NumMemoryPages =
      alignTo(MemoryPtr, WasmPageSize) / WasmPageSize;
  log("mem: total pages = " + Twine(Out.MemorySec->NumMemoryPages));

  // Check max if explicitly supplied or required by shared memory
  if (Config->MaxMemory != 0 || Config->SharedMemory) {
    if (Config->MaxMemory != alignTo(Config->MaxMemory, WasmPageSize))
      error("maximum memory must be " + Twine(WasmPageSize) + "-byte aligned");
    if (MemoryPtr > Config->MaxMemory)
      error("maximum memory too small, " + Twine(MemoryPtr) + " bytes needed");
    Out.MemorySec->MaxMemoryPages = Config->MaxMemory / WasmPageSize;
    log("mem: max pages   = " + Twine(Out.MemorySec->MaxMemoryPages));
  }
}

void Writer::addSection(OutputSection *Sec) {
  if (!Sec->isNeeded())
    return;
  log("addSection: " + toString(*Sec));
  Sec->SectionIndex = OutputSections.size();
  OutputSections.push_back(Sec);
}

// If a section name is valid as a C identifier (which is rare because of
// the leading '.'), linkers are expected to define __start_<secname> and
// __stop_<secname> symbols. They are at beginning and end of the section,
// respectively. This is not requested by the ELF standard, but GNU ld and
// gold provide the feature, and used by many programs.
void Writer::addStartStopSymbols(const InputSegment *Seg) {
  StringRef S = Seg->getName();
  LLVM_DEBUG(dbgs() << "addStartStopSymbols: " << S << "\n");
  if (!isValidCIdentifier(S))
    return;
  uint32_t Start = Seg->OutputSeg->StartVA + Seg->OutputSegmentOffset;
  uint32_t Stop = Start + Seg->getSize();
  Symtab->addOptionalDataSymbol(Saver.save("__start_" + S), Start);
  Symtab->addOptionalDataSymbol(Saver.save("__stop_" + S), Stop);
}

void Writer::addSections() {
  addSection(Out.DylinkSec);
  addSection(Out.TypeSec);
  addSection(Out.ImportSec);
  addSection(Out.FunctionSec);
  addSection(Out.TableSec);
  addSection(Out.MemorySec);
  addSection(Out.GlobalSec);
  addSection(Out.EventSec);
  addSection(Out.ExportSec);
  addSection(Out.ElemSec);
  addSection(Out.DataCountSec);

  addSection(make<CodeSection>(Out.FunctionSec->InputFunctions));
  addSection(make<DataSection>(Segments));

  createCustomSections();

  addSection(Out.LinkingSec);
  if (Config->EmitRelocs || Config->Relocatable) {
    createRelocSections();
  }

  addSection(Out.NameSec);
  addSection(Out.ProducersSec);
  addSection(Out.TargetFeaturesSec);
}

void Writer::finalizeSections() {
  for (OutputSection *S : OutputSections) {
    S->setOffset(FileSize);
    S->finalizeContents();
    FileSize += S->getSize();
  }
}

void Writer::populateTargetFeatures() {
  StringMap<std::string> Used;
  StringMap<std::string> Required;
  StringMap<std::string> Disallowed;

  // Only infer used features if user did not specify features
  bool InferFeatures = !Config->Features.hasValue();

  if (!InferFeatures) {
    for (auto &Feature : Config->Features.getValue())
      Out.TargetFeaturesSec->Features.insert(Feature);
    // No need to read or check features
    if (!Config->CheckFeatures)
      return;
  }

  // Find the sets of used, required, and disallowed features
  for (ObjFile *File : Symtab->ObjectFiles) {
    StringRef FileName(File->getName());
    for (auto &Feature : File->getWasmObj()->getTargetFeatures()) {
      switch (Feature.Prefix) {
      case WASM_FEATURE_PREFIX_USED:
        Used.insert({Feature.Name, FileName});
        break;
      case WASM_FEATURE_PREFIX_REQUIRED:
        Used.insert({Feature.Name, FileName});
        Required.insert({Feature.Name, FileName});
        break;
      case WASM_FEATURE_PREFIX_DISALLOWED:
        Disallowed.insert({Feature.Name, FileName});
        break;
      default:
        error("Unrecognized feature policy prefix " +
              std::to_string(Feature.Prefix));
      }
    }
  }

  if (InferFeatures)
    Out.TargetFeaturesSec->Features.insert(Used.keys().begin(),
                                           Used.keys().end());

  if (Out.TargetFeaturesSec->Features.count("atomics") &&
      !Config->SharedMemory) {
    if (InferFeatures)
      error(Twine("'atomics' feature is used by ") + Used["atomics"] +
            ", so --shared-memory must be used");
    else
      error("'atomics' feature is used, so --shared-memory must be used");
  }

  if (!Config->CheckFeatures)
    return;

  if (Disallowed.count("atomics") && Config->SharedMemory)
    error("'atomics' feature is disallowed by " + Disallowed["atomics"] +
          ", so --shared-memory must not be used");

  // Validate that used features are allowed in output
  if (!InferFeatures) {
    for (auto &Feature : Used.keys()) {
      if (!Out.TargetFeaturesSec->Features.count(Feature))
        error(Twine("Target feature '") + Feature + "' used by " +
              Used[Feature] + " is not allowed.");
    }
  }

  // Validate the required and disallowed constraints for each file
  for (ObjFile *File : Symtab->ObjectFiles) {
    StringRef FileName(File->getName());
    SmallSet<std::string, 8> ObjectFeatures;
    for (auto &Feature : File->getWasmObj()->getTargetFeatures()) {
      if (Feature.Prefix == WASM_FEATURE_PREFIX_DISALLOWED)
        continue;
      ObjectFeatures.insert(Feature.Name);
      if (Disallowed.count(Feature.Name))
        error(Twine("Target feature '") + Feature.Name + "' used in " +
              FileName + " is disallowed by " + Disallowed[Feature.Name] +
              ". Use --no-check-features to suppress.");
    }
    for (auto &Feature : Required.keys()) {
      if (!ObjectFeatures.count(Feature))
        error(Twine("Missing target feature '") + Feature + "' in " + FileName +
              ", required by " + Required[Feature] +
              ". Use --no-check-features to suppress.");
    }
  }
}

void Writer::calculateImports() {
  for (Symbol *Sym : Symtab->getSymbols()) {
    if (!Sym->isUndefined())
      continue;
    if (Sym->isWeak() && !Config->Relocatable)
      continue;
    if (!Sym->isLive())
      continue;
    if (!Sym->IsUsedInRegularObj)
      continue;
    // We don't generate imports for data symbols. They however can be imported
    // as GOT entries.
    if (isa<DataSymbol>(Sym))
      continue;

    LLVM_DEBUG(dbgs() << "import: " << Sym->getName() << "\n");
    Out.ImportSec->addImport(Sym);
  }
}

void Writer::calculateExports() {
  if (Config->Relocatable)
    return;

  if (!Config->Relocatable && !Config->ImportMemory)
    Out.ExportSec->Exports.push_back(
        WasmExport{"memory", WASM_EXTERNAL_MEMORY, 0});

  if (!Config->Relocatable && Config->ExportTable)
    Out.ExportSec->Exports.push_back(
        WasmExport{FunctionTableName, WASM_EXTERNAL_TABLE, 0});

  unsigned FakeGlobalIndex =
      Out.ImportSec->numImportedGlobals() + Out.GlobalSec->InputGlobals.size();

  for (Symbol *Sym : Symtab->getSymbols()) {
    if (!Sym->isExported())
      continue;
    if (!Sym->isLive())
      continue;

    StringRef Name = Sym->getName();
    WasmExport Export;
    if (auto *F = dyn_cast<DefinedFunction>(Sym)) {
      Export = {Name, WASM_EXTERNAL_FUNCTION, F->getFunctionIndex()};
    } else if (auto *G = dyn_cast<DefinedGlobal>(Sym)) {
      // TODO(sbc): Remove this check once to mutable global proposal is
      // implement in all major browsers.
      // See: https://github.com/WebAssembly/mutable-global
      if (G->getGlobalType()->Mutable) {
        // Only the __stack_pointer should ever be create as mutable.
        assert(G == WasmSym::StackPointer);
        continue;
      }
      Export = {Name, WASM_EXTERNAL_GLOBAL, G->getGlobalIndex()};
    } else if (auto *E = dyn_cast<DefinedEvent>(Sym)) {
      Export = {Name, WASM_EXTERNAL_EVENT, E->getEventIndex()};
    } else {
      auto *D = cast<DefinedData>(Sym);
      Out.GlobalSec->DefinedFakeGlobals.emplace_back(D);
      Export = {Name, WASM_EXTERNAL_GLOBAL, FakeGlobalIndex++};
    }

    LLVM_DEBUG(dbgs() << "Export: " << Name << "\n");
    Out.ExportSec->Exports.push_back(Export);
  }
}

void Writer::populateSymtab() {
  if (!Config->Relocatable && !Config->EmitRelocs)
    return;

  for (Symbol *Sym : Symtab->getSymbols())
    if (Sym->IsUsedInRegularObj && Sym->isLive())
      Out.LinkingSec->addToSymtab(Sym);

  for (ObjFile *File : Symtab->ObjectFiles) {
    LLVM_DEBUG(dbgs() << "Local symtab entries: " << File->getName() << "\n");
    for (Symbol *Sym : File->getSymbols())
      if (Sym->isLocal() && !isa<SectionSymbol>(Sym) && Sym->isLive())
        Out.LinkingSec->addToSymtab(Sym);
  }
}

void Writer::calculateTypes() {
  // The output type section is the union of the following sets:
  // 1. Any signature used in the TYPE relocation
  // 2. The signatures of all imported functions
  // 3. The signatures of all defined functions
  // 4. The signatures of all imported events
  // 5. The signatures of all defined events

  for (ObjFile *File : Symtab->ObjectFiles) {
    ArrayRef<WasmSignature> Types = File->getWasmObj()->types();
    for (uint32_t I = 0; I < Types.size(); I++)
      if (File->TypeIsUsed[I])
        File->TypeMap[I] = Out.TypeSec->registerType(Types[I]);
  }

  for (const Symbol *Sym : Out.ImportSec->ImportedSymbols) {
    if (auto *F = dyn_cast<FunctionSymbol>(Sym))
      Out.TypeSec->registerType(*F->Signature);
    else if (auto *E = dyn_cast<EventSymbol>(Sym))
      Out.TypeSec->registerType(*E->Signature);
  }

  for (const InputFunction *F : Out.FunctionSec->InputFunctions)
    Out.TypeSec->registerType(F->Signature);

  for (const InputEvent *E : Out.EventSec->InputEvents)
    Out.TypeSec->registerType(E->Signature);
}

static void scanRelocations() {
  for (ObjFile *File : Symtab->ObjectFiles) {
    LLVM_DEBUG(dbgs() << "scanRelocations: " << File->getName() << "\n");
    for (InputChunk *Chunk : File->Functions)
      scanRelocations(Chunk);
    for (InputChunk *Chunk : File->Segments)
      scanRelocations(Chunk);
    for (auto &P : File->CustomSections)
      scanRelocations(P);
  }
}

void Writer::assignIndexes() {
  // Seal the import section, since other index spaces such as function and
  // global are effected by the number of imports.
  Out.ImportSec->seal();

  for (InputFunction *Func : Symtab->SyntheticFunctions)
    Out.FunctionSec->addFunction(Func);

  for (ObjFile *File : Symtab->ObjectFiles) {
    LLVM_DEBUG(dbgs() << "Functions: " << File->getName() << "\n");
    for (InputFunction *Func : File->Functions)
      Out.FunctionSec->addFunction(Func);
  }

  for (InputGlobal *Global : Symtab->SyntheticGlobals)
    Out.GlobalSec->addGlobal(Global);

  for (ObjFile *File : Symtab->ObjectFiles) {
    LLVM_DEBUG(dbgs() << "Globals: " << File->getName() << "\n");
    for (InputGlobal *Global : File->Globals)
      Out.GlobalSec->addGlobal(Global);
  }

  for (ObjFile *File : Symtab->ObjectFiles) {
    LLVM_DEBUG(dbgs() << "Events: " << File->getName() << "\n");
    for (InputEvent *Event : File->Events)
      Out.EventSec->addEvent(Event);
  }
}

static StringRef getOutputDataSegmentName(StringRef Name) {
  // With PIC code we currently only support a single data segment since
  // we only have a single __memory_base to use as our base address.
  if (Config->Pic)
    return "data";
  if (!Config->MergeDataSegments)
    return Name;
  if (Name.startswith(".text."))
    return ".text";
  if (Name.startswith(".data."))
    return ".data";
  if (Name.startswith(".bss."))
    return ".bss";
  if (Name.startswith(".rodata."))
    return ".rodata";
  return Name;
}

void Writer::createOutputSegments() {
  for (ObjFile *File : Symtab->ObjectFiles) {
    for (InputSegment *Segment : File->Segments) {
      if (!Segment->Live)
        continue;
      StringRef Name = getOutputDataSegmentName(Segment->getName());
      OutputSegment *&S = SegmentMap[Name];
      if (S == nullptr) {
        LLVM_DEBUG(dbgs() << "new segment: " << Name << "\n");
        S = make<OutputSegment>(Name, Segments.size());
        Segments.push_back(S);
      }
      S->addInputSegment(Segment);
      LLVM_DEBUG(dbgs() << "added data: " << Name << ": " << S->Size << "\n");
    }
  }
}

// For -shared (PIC) output, we create create a synthetic function which will
// apply any relocations to the data segments on startup.  This function is
// called __wasm_apply_relocs and is added at the very beginning of
// __wasm_call_ctors before any of the constructors run.
void Writer::createApplyRelocationsFunction() {
  LLVM_DEBUG(dbgs() << "createApplyRelocationsFunction\n");
  // First write the body's contents to a string.
  std::string BodyContent;
  {
    raw_string_ostream OS(BodyContent);
    writeUleb128(OS, 0, "num locals");
    for (const OutputSegment *Seg : Segments)
      for (const InputSegment *InSeg : Seg->InputSegments)
        InSeg->generateRelocationCode(OS);
    writeU8(OS, WASM_OPCODE_END, "END");
  }

  // Once we know the size of the body we can create the final function body
  std::string FunctionBody;
  {
    raw_string_ostream OS(FunctionBody);
    writeUleb128(OS, BodyContent.size(), "function size");
    OS << BodyContent;
  }

  ArrayRef<uint8_t> Body = arrayRefFromStringRef(Saver.save(FunctionBody));
  cast<SyntheticFunction>(WasmSym::ApplyRelocs->Function)->setBody(Body);
}

// Create synthetic "__wasm_call_ctors" function based on ctor functions
// in input object.
void Writer::createCallCtorsFunction() {
  if (!WasmSym::CallCtors->isLive())
    return;

  // First write the body's contents to a string.
  std::string BodyContent;
  {
    raw_string_ostream OS(BodyContent);
    writeUleb128(OS, 0, "num locals");
    if (Config->Pic) {
      writeU8(OS, WASM_OPCODE_CALL, "CALL");
      writeUleb128(OS, WasmSym::ApplyRelocs->getFunctionIndex(),
                   "function index");
    }
    for (const WasmInitEntry &F : InitFunctions) {
      writeU8(OS, WASM_OPCODE_CALL, "CALL");
      writeUleb128(OS, F.Sym->getFunctionIndex(), "function index");
    }
    writeU8(OS, WASM_OPCODE_END, "END");
  }

  // Once we know the size of the body we can create the final function body
  std::string FunctionBody;
  {
    raw_string_ostream OS(FunctionBody);
    writeUleb128(OS, BodyContent.size(), "function size");
    OS << BodyContent;
  }

  ArrayRef<uint8_t> Body = arrayRefFromStringRef(Saver.save(FunctionBody));
  cast<SyntheticFunction>(WasmSym::CallCtors->Function)->setBody(Body);
}

// Populate InitFunctions vector with init functions from all input objects.
// This is then used either when creating the output linking section or to
// synthesize the "__wasm_call_ctors" function.
void Writer::calculateInitFunctions() {
  if (!Config->Relocatable && !WasmSym::CallCtors->isLive())
    return;

  for (ObjFile *File : Symtab->ObjectFiles) {
    const WasmLinkingData &L = File->getWasmObj()->linkingData();
    for (const WasmInitFunc &F : L.InitFunctions) {
      FunctionSymbol *Sym = File->getFunctionSymbol(F.Symbol);
      // comdat exclusions can cause init functions be discarded.
      if (Sym->isDiscarded())
        continue;
      assert(Sym->isLive());
      if (*Sym->Signature != WasmSignature{{}, {}})
        error("invalid signature for init func: " + toString(*Sym));
      InitFunctions.emplace_back(WasmInitEntry{Sym, F.Priority});
    }
  }

  // Sort in order of priority (lowest first) so that they are called
  // in the correct order.
  llvm::stable_sort(InitFunctions,
                    [](const WasmInitEntry &L, const WasmInitEntry &R) {
                      return L.Priority < R.Priority;
                    });
}

void Writer::createSyntheticSections() {
  Out.DylinkSec = make<DylinkSection>();
  Out.TypeSec = make<TypeSection>();
  Out.ImportSec = make<ImportSection>();
  Out.FunctionSec = make<FunctionSection>();
  Out.TableSec = make<TableSection>();
  Out.MemorySec = make<MemorySection>();
  Out.GlobalSec = make<GlobalSection>();
  Out.EventSec = make<EventSection>();
  Out.ExportSec = make<ExportSection>();
  Out.ElemSec = make<ElemSection>(TableBase);
  Out.DataCountSec = make<DataCountSection>(Segments.size());
  Out.LinkingSec = make<LinkingSection>(InitFunctions, Segments);
  Out.NameSec = make<NameSection>();
  Out.ProducersSec = make<ProducersSection>();
  Out.TargetFeaturesSec = make<TargetFeaturesSection>();
}

void Writer::run() {
  if (Config->Relocatable || Config->Pic)
    Config->GlobalBase = 0;

  // For PIC code the table base is assigned dynamically by the loader.
  // For non-PIC, we start at 1 so that accessing table index 0 always traps.
  if (!Config->Pic)
    TableBase = 1;

  log("-- createOutputSegments");
  createOutputSegments();
  log("-- createSyntheticSections");
  createSyntheticSections();
  log("-- populateProducers");
  populateProducers();
  log("-- populateTargetFeatures");
  populateTargetFeatures();
  log("-- calculateImports");
  calculateImports();
  log("-- layoutMemory");
  layoutMemory();

  if (!Config->Relocatable) {
    // Create linker synthesized __start_SECNAME/__stop_SECNAME symbols
    // This has to be done after memory layout is performed.
    for (const OutputSegment *Seg : Segments)
      for (const InputSegment *S : Seg->InputSegments)
        addStartStopSymbols(S);
  }

  log("-- scanRelocations");
  scanRelocations();
  log("-- assignIndexes");
  assignIndexes();
  log("-- calculateInitFunctions");
  calculateInitFunctions();

  if (!Config->Relocatable) {
    // Create linker synthesized functions
    if (Config->Pic)
      createApplyRelocationsFunction();
    createCallCtorsFunction();

    // Make sure we have resolved all symbols.
    if (!Config->AllowUndefined)
      Symtab->reportRemainingUndefines();

    if (errorCount())
      return;
  }

  log("-- calculateTypes");
  calculateTypes();
  log("-- calculateExports");
  calculateExports();
  log("-- calculateCustomSections");
  calculateCustomSections();
  log("-- populateSymtab");
  populateSymtab();
  log("-- addSections");
  addSections();

  if (errorHandler().Verbose) {
    log("Defined Functions: " + Twine(Out.FunctionSec->InputFunctions.size()));
    log("Defined Globals  : " + Twine(Out.GlobalSec->InputGlobals.size()));
    log("Defined Events   : " + Twine(Out.EventSec->InputEvents.size()));
    log("Function Imports : " + Twine(Out.ImportSec->numImportedFunctions()));
    log("Global Imports   : " + Twine(Out.ImportSec->numImportedGlobals()));
    log("Event Imports    : " + Twine(Out.ImportSec->numImportedEvents()));
    for (ObjFile *File : Symtab->ObjectFiles)
      File->dumpInfo();
  }

  createHeader();
  log("-- finalizeSections");
  finalizeSections();

  log("-- openFile");
  openFile();
  if (errorCount())
    return;

  writeHeader();

  log("-- writeSections");
  writeSections();
  if (errorCount())
    return;

  if (Error E = Buffer->commit())
    fatal("failed to write the output file: " + toString(std::move(E)));
}

// Open a result file.
void Writer::openFile() {
  log("writing: " + Config->OutputFile);

  Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(Config->OutputFile, FileSize,
                               FileOutputBuffer::F_executable);

  if (!BufferOrErr)
    error("failed to open " + Config->OutputFile + ": " +
          toString(BufferOrErr.takeError()));
  else
    Buffer = std::move(*BufferOrErr);
}

void Writer::createHeader() {
  raw_string_ostream OS(Header);
  writeBytes(OS, WasmMagic, sizeof(WasmMagic), "wasm magic");
  writeU32(OS, WasmVersion, "wasm version");
  OS.flush();
  FileSize += Header.size();
}

void lld::wasm::writeResult() { Writer().run(); }
