//===- Writer.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Writer.h"
#include "Config.h"
#include "InputChunks.h"
#include "InputGlobal.h"
#include "OutputSections.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Strings.h"
#include "lld/Common/Threads.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/BinaryFormat/Wasm.h"
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

static constexpr int kStackAlignment = 16;
static constexpr int kInitialTableOffset = 1;

namespace {

// Traits for using WasmSignature in a DenseMap.
struct WasmSignatureDenseMapInfo {
  static WasmSignature getEmptyKey() {
    WasmSignature Sig;
    Sig.ReturnType = 1;
    return Sig;
  }
  static WasmSignature getTombstoneKey() {
    WasmSignature Sig;
    Sig.ReturnType = 2;
    return Sig;
  }
  static unsigned getHashValue(const WasmSignature &Sig) {
    unsigned H = hash_value(Sig.ReturnType);
    for (int32_t Param : Sig.ParamTypes)
      H = hash_combine(H, Param);
    return H;
  }
  static bool isEqual(const WasmSignature &LHS, const WasmSignature &RHS) {
    return LHS == RHS;
  }
};

// An init entry to be written to either the synthetic init func or the
// linking metadata.
struct WasmInitEntry {
  const Symbol *Sym;
  uint32_t Priority;
};

// The writer writes a SymbolTable result to a file.
class Writer {
public:
  void run();

private:
  void openFile();

  uint32_t lookupType(const WasmSignature &Sig);
  uint32_t registerType(const WasmSignature &Sig);

  void createCtorFunction();
  void calculateInitFunctions();
  void assignIndexes();
  void calculateImports();
  void calculateExports();
  void assignSymtab();
  void calculateTypes();
  void createOutputSegments();
  void layoutMemory();
  void createHeader();
  void createSections();
  SyntheticSection *createSyntheticSection(uint32_t Type,
                                           StringRef Name = "");

  // Builtin sections
  void createTypeSection();
  void createFunctionSection();
  void createTableSection();
  void createGlobalSection();
  void createExportSection();
  void createImportSection();
  void createMemorySection();
  void createElemSection();
  void createCodeSection();
  void createDataSection();

  // Custom sections
  void createRelocSections();
  void createLinkingSection();
  void createNameSection();

  void writeHeader();
  void writeSections();

  uint64_t FileSize = 0;
  uint32_t NumMemoryPages = 0;

  std::vector<const WasmSignature *> Types;
  DenseMap<WasmSignature, int32_t, WasmSignatureDenseMapInfo> TypeIndices;
  std::vector<const Symbol *> ImportedSymbols;
  unsigned NumImportedFunctions = 0;
  unsigned NumImportedGlobals = 0;
  std::vector<Symbol *> ExportedSymbols;
  std::vector<const DefinedData *> DefinedFakeGlobals;
  std::vector<InputGlobal *> InputGlobals;
  std::vector<InputFunction *> InputFunctions;
  std::vector<const FunctionSymbol *> IndirectFunctions;
  std::vector<const Symbol *> SymtabEntries;
  std::vector<WasmInitEntry> InitFunctions;

  // Elements that are used to construct the final output
  std::string Header;
  std::vector<OutputSection *> OutputSections;

  std::unique_ptr<FileOutputBuffer> Buffer;
  std::string CtorFunctionBody;

  std::vector<OutputSegment *> Segments;
  llvm::SmallDenseMap<StringRef, OutputSegment *> SegmentMap;
};

} // anonymous namespace

static void debugPrint(const char *fmt, ...) {
  if (!errorHandler().Verbose)
    return;
  fprintf(stderr, "lld: ");
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

void Writer::createImportSection() {
  uint32_t NumImports = ImportedSymbols.size();
  if (Config->ImportMemory)
    ++NumImports;

  if (NumImports == 0)
    return;

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_IMPORT);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, NumImports, "import count");

  if (Config->ImportMemory) {
    WasmImport Import;
    Import.Module = "env";
    Import.Field = "memory";
    Import.Kind = WASM_EXTERNAL_MEMORY;
    Import.Memory.Flags = 0;
    Import.Memory.Initial = NumMemoryPages;
    writeImport(OS, Import);
  }

  for (const Symbol *Sym : ImportedSymbols) {
    WasmImport Import;
    Import.Module = "env";
    Import.Field = Sym->getName();
    if (auto *FunctionSym = dyn_cast<FunctionSymbol>(Sym)) {
      Import.Kind = WASM_EXTERNAL_FUNCTION;
      Import.SigIndex = lookupType(*FunctionSym->getFunctionType());
    } else {
      auto *GlobalSym = cast<GlobalSymbol>(Sym);
      Import.Kind = WASM_EXTERNAL_GLOBAL;
      Import.Global = *GlobalSym->getGlobalType();
    }
    writeImport(OS, Import);
  }
}

void Writer::createTypeSection() {
  SyntheticSection *Section = createSyntheticSection(WASM_SEC_TYPE);
  raw_ostream &OS = Section->getStream();
  writeUleb128(OS, Types.size(), "type count");
  for (const WasmSignature *Sig : Types)
    writeSig(OS, *Sig);
}

void Writer::createFunctionSection() {
  if (InputFunctions.empty())
    return;

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_FUNCTION);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, InputFunctions.size(), "function count");
  for (const InputFunction *Func : InputFunctions)
    writeUleb128(OS, lookupType(Func->Signature), "sig index");
}

void Writer::createMemorySection() {
  if (Config->ImportMemory)
    return;

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_MEMORY);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, 1, "memory count");
  writeUleb128(OS, 0, "memory limits flags");
  writeUleb128(OS, NumMemoryPages, "initial pages");
}

void Writer::createGlobalSection() {
  unsigned NumGlobals = InputGlobals.size() + DefinedFakeGlobals.size();
  if (NumGlobals == 0)
    return;

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_GLOBAL);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, NumGlobals, "global count");
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

void Writer::createTableSection() {
  // Always output a table section, even if there are no indirect calls.
  // There are two reasons for this:
  //  1. For executables it is useful to have an empty table slot at 0
  //     which can be filled with a null function call handler.
  //  2. If we don't do this, any program that contains a call_indirect but
  //     no address-taken function will fail at validation time since it is
  //     a validation error to include a call_indirect instruction if there
  //     is not table.
  uint32_t TableSize = kInitialTableOffset + IndirectFunctions.size();

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_TABLE);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, 1, "table count");
  writeU8(OS, WASM_TYPE_ANYFUNC, "table type");
  writeUleb128(OS, WASM_LIMITS_FLAG_HAS_MAX, "table flags");
  writeUleb128(OS, TableSize, "table initial size");
  writeUleb128(OS, TableSize, "table max size");
}

void Writer::createExportSection() {
  bool ExportMemory = !Config->Relocatable && !Config->ImportMemory;

  uint32_t NumExports = (ExportMemory ? 1 : 0) + ExportedSymbols.size();
  if (!NumExports)
    return;

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_EXPORT);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, NumExports, "export count");

  if (ExportMemory)
    writeExport(OS, {"memory", WASM_EXTERNAL_MEMORY, 0});

  unsigned FakeGlobalIndex = NumImportedGlobals + InputGlobals.size();

  for (const Symbol *Sym : ExportedSymbols) {
    StringRef Name = Sym->getName();
    WasmExport Export;
    DEBUG(dbgs() << "Export: " << Name << "\n");

    if (isa<DefinedFunction>(Sym))
      Export = {Name, WASM_EXTERNAL_FUNCTION, Sym->getOutputIndex()};
    else if (isa<DefinedGlobal>(Sym))
      Export = {Name, WASM_EXTERNAL_GLOBAL, Sym->getOutputIndex()};
    else if (isa<DefinedData>(Sym))
      Export = {Name, WASM_EXTERNAL_GLOBAL, FakeGlobalIndex++};
    else
      llvm_unreachable("unexpected symbol type");
    writeExport(OS, Export);
  }
}

void Writer::createElemSection() {
  if (IndirectFunctions.empty())
    return;

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_ELEM);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, 1, "segment count");
  writeUleb128(OS, 0, "table index");
  WasmInitExpr InitExpr;
  InitExpr.Opcode = WASM_OPCODE_I32_CONST;
  InitExpr.Value.Int32 = kInitialTableOffset;
  writeInitExpr(OS, InitExpr);
  writeUleb128(OS, IndirectFunctions.size(), "elem count");

  uint32_t TableIndex = kInitialTableOffset;
  for (const FunctionSymbol *Sym : IndirectFunctions) {
    assert(Sym->getTableIndex() == TableIndex);
    writeUleb128(OS, Sym->getOutputIndex(), "function index");
    ++TableIndex;
  }
}

void Writer::createCodeSection() {
  if (InputFunctions.empty())
    return;

  log("createCodeSection");

  auto Section = make<CodeSection>(InputFunctions);
  OutputSections.push_back(Section);
}

void Writer::createDataSection() {
  if (!Segments.size())
    return;

  log("createDataSection");
  auto Section = make<DataSection>(Segments);
  OutputSections.push_back(Section);
}

// Create relocations sections in the final output.
// These are only created when relocatable output is requested.
void Writer::createRelocSections() {
  log("createRelocSections");
  // Don't use iterator here since we are adding to OutputSection
  size_t OrigSize = OutputSections.size();
  for (size_t i = 0; i < OrigSize; i++) {
    OutputSection *OSec = OutputSections[i];
    uint32_t Count = OSec->numRelocations();
    if (!Count)
      continue;

    StringRef Name;
    if (OSec->Type == WASM_SEC_DATA)
      Name = "reloc.DATA";
    else if (OSec->Type == WASM_SEC_CODE)
      Name = "reloc.CODE";
    else
      llvm_unreachable("relocations only supported for code and data");

    SyntheticSection *Section = createSyntheticSection(WASM_SEC_CUSTOM, Name);
    raw_ostream &OS = Section->getStream();
    writeUleb128(OS, OSec->Type, "reloc section");
    writeUleb128(OS, Count, "reloc count");
    OSec->writeRelocations(OS);
  }
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
  return Flags;
}

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

// Create the custom "linking" section containing linker metadata.
// This is only created when relocatable output is requested.
void Writer::createLinkingSection() {
  SyntheticSection *Section =
      createSyntheticSection(WASM_SEC_CUSTOM, "linking");
  raw_ostream &OS = Section->getStream();

  if (!Config->Relocatable)
    return;

  if (!SymtabEntries.empty()) {
    SubSection Sub(WASM_SYMBOL_TABLE);
    writeUleb128(Sub.OS, SymtabEntries.size(), "num symbols");

    for (const Symbol *Sym : SymtabEntries) {
      assert(Sym->isDefined() || Sym->isUndefined());
      WasmSymbolType Kind = Sym->getWasmType();
      uint32_t Flags = getWasmFlags(Sym);

      writeU8(Sub.OS, Kind, "sym kind");
      writeUleb128(Sub.OS, Flags, "sym flags");

      switch (Kind) {
      case llvm::wasm::WASM_SYMBOL_TYPE_FUNCTION:
      case llvm::wasm::WASM_SYMBOL_TYPE_GLOBAL:
        writeUleb128(Sub.OS, Sym->getOutputIndex(), "index");
        if (Sym->isDefined())
          writeStr(Sub.OS, Sym->getName(), "sym name");
        break;
      case llvm::wasm::WASM_SYMBOL_TYPE_DATA:
        writeStr(Sub.OS, Sym->getName(), "sym name");
        if (auto *DataSym = dyn_cast<DefinedData>(Sym)) {
          writeUleb128(Sub.OS, DataSym->getOutputSegmentIndex(), "index");
          writeUleb128(Sub.OS, DataSym->getOutputSegmentOffset(),
                       "data offset");
          writeUleb128(Sub.OS, DataSym->getSize(), "data size");
        }
        break;
      }
    }

    Sub.writeTo(OS);
  }

  if (Segments.size()) {
    SubSection Sub(WASM_SEGMENT_INFO);
    writeUleb128(Sub.OS, Segments.size(), "num data segments");
    for (const OutputSegment *S : Segments) {
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

  struct ComdatEntry { unsigned Kind; uint32_t Index; };
  std::map<StringRef,std::vector<ComdatEntry>> Comdats;

  for (const InputFunction *F : InputFunctions) {
    StringRef Comdat = F->getComdat();
    if (!Comdat.empty())
      Comdats[Comdat].emplace_back(
          ComdatEntry{WASM_COMDAT_FUNCTION, F->getOutputIndex()});
  }
  for (uint32_t I = 0; I < Segments.size(); ++I) {
    const auto &InputSegments = Segments[I]->InputSegments;
    if (InputSegments.empty())
      continue;
    StringRef Comdat = InputSegments[0]->getComdat();
#ifndef NDEBUG
    for (const InputSegment *IS : InputSegments)
      assert(IS->getComdat() == Comdat);
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

// Create the custom "name" section containing debug symbol names.
void Writer::createNameSection() {
  unsigned NumNames = NumImportedFunctions;
  for (const InputFunction *F : InputFunctions)
    if (!F->getName().empty())
      ++NumNames;

  if (NumNames == 0)
    return;

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_CUSTOM, "name");

  SubSection Sub(WASM_NAMES_FUNCTION);
  writeUleb128(Sub.OS, NumNames, "name count");

  // Names must appear in function index order.  As it happens ImportedSymbols
  // and InputFunctions are numbered in order with imported functions coming
  // first.
  for (const Symbol *S : ImportedSymbols) {
    if (!isa<FunctionSymbol>(S))
      continue;
    writeUleb128(Sub.OS, S->getOutputIndex(), "import index");
    writeStr(Sub.OS, S->getName(), "symbol name");
  }
  for (const InputFunction *F : InputFunctions) {
    if (!F->getName().empty()) {
      writeUleb128(Sub.OS, F->getOutputIndex(), "func index");
      writeStr(Sub.OS, F->getName(), "symbol name");
    }
  }

  Sub.writeTo(Section->getStream());
}

void Writer::writeHeader() {
  memcpy(Buffer->getBufferStart(), Header.data(), Header.size());
}

void Writer::writeSections() {
  uint8_t *Buf = Buffer->getBufferStart();
  parallelForEach(OutputSections, [Buf](OutputSection *S) { S->writeTo(Buf); });
}

// Fix the memory layout of the output binary.  This assigns memory offsets
// to each of the input data sections as well as the explicit stack region.
// The memory layout is as follows, from low to high.
//  - initialized data (starting at Config->GlobalBase)
//  - BSS data (not currently implemented in llvm)
//  - explicit stack (Config->ZStackSize)
//  - heap start / unallocated
void Writer::layoutMemory() {
  uint32_t MemoryPtr = 0;
  MemoryPtr = Config->GlobalBase;
  debugPrint("mem: global base = %d\n", Config->GlobalBase);

  createOutputSegments();

  // Arbitrarily set __dso_handle handle to point to the start of the data
  // segments.
  if (WasmSym::DsoHandle)
    WasmSym::DsoHandle->setVirtualAddress(MemoryPtr);

  for (OutputSegment *Seg : Segments) {
    MemoryPtr = alignTo(MemoryPtr, Seg->Alignment);
    Seg->StartVA = MemoryPtr;
    debugPrint("mem: %-15s offset=%-8d size=%-8d align=%d\n",
               Seg->Name.str().c_str(), MemoryPtr, Seg->Size, Seg->Alignment);
    MemoryPtr += Seg->Size;
  }

  // TODO: Add .bss space here.
  if (WasmSym::DataEnd)
    WasmSym::DataEnd->setVirtualAddress(MemoryPtr);

  debugPrint("mem: static data = %d\n", MemoryPtr - Config->GlobalBase);

  // Stack comes after static data and bss
  if (!Config->Relocatable) {
    MemoryPtr = alignTo(MemoryPtr, kStackAlignment);
    if (Config->ZStackSize != alignTo(Config->ZStackSize, kStackAlignment))
      error("stack size must be " + Twine(kStackAlignment) + "-byte aligned");
    debugPrint("mem: stack size  = %d\n", Config->ZStackSize);
    debugPrint("mem: stack base  = %d\n", MemoryPtr);
    MemoryPtr += Config->ZStackSize;
    WasmSym::StackPointer->Global->Global.InitExpr.Value.Int32 = MemoryPtr;
    debugPrint("mem: stack top   = %d\n", MemoryPtr);

    // Set `__heap_base` to directly follow the end of the stack.  We don't
    // allocate any heap memory up front, but instead really on the malloc/brk
    // implementation growing the memory at runtime.
    WasmSym::HeapBase->setVirtualAddress(MemoryPtr);
    debugPrint("mem: heap base   = %d\n", MemoryPtr);
  }

  uint32_t MemSize = alignTo(MemoryPtr, WasmPageSize);
  NumMemoryPages = MemSize / WasmPageSize;
  debugPrint("mem: total pages = %d\n", NumMemoryPages);
}

SyntheticSection *Writer::createSyntheticSection(uint32_t Type,
                                                 StringRef Name) {
  auto Sec = make<SyntheticSection>(Type, Name);
  log("createSection: " + toString(*Sec));
  OutputSections.push_back(Sec);
  return Sec;
}

void Writer::createSections() {
  // Known sections
  createTypeSection();
  createImportSection();
  createFunctionSection();
  createTableSection();
  createMemorySection();
  createGlobalSection();
  createExportSection();
  createElemSection();
  createCodeSection();
  createDataSection();

  // Custom sections
  if (Config->Relocatable) {
    createRelocSections();
    createLinkingSection();
  }
  if (!Config->StripDebug && !Config->StripAll)
    createNameSection();

  for (OutputSection *S : OutputSections) {
    S->setOffset(FileSize);
    S->finalizeContents();
    FileSize += S->getSize();
  }
}

void Writer::calculateImports() {
  for (Symbol *Sym : Symtab->getSymbols()) {
    if (!Sym->isUndefined())
      continue;
    if (isa<DataSymbol>(Sym))
      continue;
    if (Sym->isWeak() && !Config->Relocatable)
      continue;

    DEBUG(dbgs() << "import: " << Sym->getName() << "\n");
    Sym->setOutputIndex(ImportedSymbols.size());
    ImportedSymbols.emplace_back(Sym);
    if (isa<FunctionSymbol>(Sym))
      ++NumImportedFunctions;
    else
      ++NumImportedGlobals;
  }
}

void Writer::calculateExports() {
  if (Config->Relocatable)
    return;

  for (Symbol *Sym : Symtab->getSymbols()) {
    if (!Sym->isDefined())
      continue;
    if (Sym->isHidden() || Sym->isLocal())
      continue;
    if (!Sym->isLive())
      continue;

    DEBUG(dbgs() << "exporting sym: " << Sym->getName() << "\n");

    if (auto *D = dyn_cast<DefinedData>(Sym)) {
      // TODO Remove this check here; for non-relocatable output we actually
      // used only to create fake-global exports for the synthetic symbols.  Fix
      // this in a future commit
      if (Sym != WasmSym::DataEnd && Sym != WasmSym::HeapBase)
        continue;
      DefinedFakeGlobals.emplace_back(D);
    }
    ExportedSymbols.emplace_back(Sym);
  }
}

void Writer::assignSymtab() {
  if (!Config->Relocatable)
    return;

  unsigned SymbolIndex = SymtabEntries.size();
  for (ObjFile *File : Symtab->ObjectFiles) {
    DEBUG(dbgs() << "Symtab entries: " << File->getName() << "\n");
    for (Symbol *Sym : File->getSymbols()) {
      if (Sym->getFile() != File)
        continue;
      if (!Sym->isLive())
        return;
      Sym->setOutputSymbolIndex(SymbolIndex++);
      SymtabEntries.emplace_back(Sym);
    }
  }

  // For the moment, relocatable output doesn't contain any synthetic functions,
  // so no need to look through the Symtab for symbols not referenced by
  // Symtab->ObjectFiles.
}

uint32_t Writer::lookupType(const WasmSignature &Sig) {
  auto It = TypeIndices.find(Sig);
  if (It == TypeIndices.end()) {
    error("type not found: " + toString(Sig));
    return 0;
  }
  return It->second;
}

uint32_t Writer::registerType(const WasmSignature &Sig) {
  auto Pair = TypeIndices.insert(std::make_pair(Sig, Types.size()));
  if (Pair.second) {
    DEBUG(dbgs() << "type " << toString(Sig) << "\n");
    Types.push_back(&Sig);
  }
  return Pair.first->second;
}

void Writer::calculateTypes() {
  // The output type section is the union of the following sets:
  // 1. Any signature used in the TYPE relocation
  // 2. The signatures of all imported functions
  // 3. The signatures of all defined functions

  for (ObjFile *File : Symtab->ObjectFiles) {
    ArrayRef<WasmSignature> Types = File->getWasmObj()->types();
    for (uint32_t I = 0; I < Types.size(); I++)
      if (File->TypeIsUsed[I])
        File->TypeMap[I] = registerType(Types[I]);
  }

  for (const Symbol *Sym : ImportedSymbols)
    if (auto *F = dyn_cast<FunctionSymbol>(Sym))
      registerType(*F->getFunctionType());

  for (const InputFunction *F : InputFunctions)
    registerType(F->Signature);
}

void Writer::assignIndexes() {
  uint32_t FunctionIndex = NumImportedFunctions + InputFunctions.size();
  for (ObjFile *File : Symtab->ObjectFiles) {
    DEBUG(dbgs() << "Functions: " << File->getName() << "\n");
    for (InputFunction *Func : File->Functions) {
      if (!Func->Live)
        continue;
      InputFunctions.emplace_back(Func);
      Func->setOutputIndex(FunctionIndex++);
    }
  }

  uint32_t TableIndex = kInitialTableOffset;
  auto HandleRelocs = [&](InputChunk *Chunk) {
    if (!Chunk->Live)
      return;
    ObjFile *File = Chunk->File;
    ArrayRef<WasmSignature> Types = File->getWasmObj()->types();
    for (const WasmRelocation &Reloc : Chunk->getRelocations()) {
      if (Reloc.Type == R_WEBASSEMBLY_TABLE_INDEX_I32 ||
          Reloc.Type == R_WEBASSEMBLY_TABLE_INDEX_SLEB) {
        FunctionSymbol *Sym = File->getFunctionSymbol(Reloc.Index);
        if (Sym->hasTableIndex() || !Sym->hasOutputIndex())
          continue;
        Sym->setTableIndex(TableIndex++);
        IndirectFunctions.emplace_back(Sym);
      } else if (Reloc.Type == R_WEBASSEMBLY_TYPE_INDEX_LEB) {
        // Mark target type as live
        File->TypeMap[Reloc.Index] = registerType(Types[Reloc.Index]);
        File->TypeIsUsed[Reloc.Index] = true;
      } else if (Reloc.Type == R_WEBASSEMBLY_GLOBAL_INDEX_LEB) {
        // Mark target global as live
        GlobalSymbol *Sym = File->getGlobalSymbol(Reloc.Index);
        if (auto *G = dyn_cast<DefinedGlobal>(Sym)) {
          DEBUG(dbgs() << "marking global live: " << Sym->getName() << "\n");
          G->Global->Live = true;
        }
      }
    }
  };

  for (ObjFile *File : Symtab->ObjectFiles) {
    DEBUG(dbgs() << "Handle relocs: " << File->getName() << "\n");
    for (InputChunk *Chunk : File->Functions)
      HandleRelocs(Chunk);
    for (InputChunk *Chunk : File->Segments)
      HandleRelocs(Chunk);
  }

  uint32_t GlobalIndex = NumImportedGlobals + InputGlobals.size();
  auto AddDefinedGlobal = [&](InputGlobal *Global) {
    if (Global->Live) {
      DEBUG(dbgs() << "AddDefinedGlobal: " << GlobalIndex << "\n");
      Global->setOutputIndex(GlobalIndex++);
      InputGlobals.push_back(Global);
    }
  };

  if (WasmSym::StackPointer)
    AddDefinedGlobal(WasmSym::StackPointer->Global);

  for (ObjFile *File : Symtab->ObjectFiles) {
    DEBUG(dbgs() << "Globals: " << File->getName() << "\n");
    for (InputGlobal *Global : File->Globals)
      AddDefinedGlobal(Global);
  }
}

static StringRef getOutputDataSegmentName(StringRef Name) {
  if (Config->Relocatable)
    return Name;
  if (Name.startswith(".text."))
    return ".text";
  if (Name.startswith(".data."))
    return ".data";
  if (Name.startswith(".bss."))
    return ".bss";
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
        DEBUG(dbgs() << "new segment: " << Name << "\n");
        S = make<OutputSegment>(Name, Segments.size());
        Segments.push_back(S);
      }
      S->addInputSegment(Segment);
      DEBUG(dbgs() << "added data: " << Name << ": " << S->Size << "\n");
    }
  }
}

static const int OPCODE_CALL = 0x10;
static const int OPCODE_END = 0xb;

// Create synthetic "__wasm_call_ctors" function based on ctor functions
// in input object.
void Writer::createCtorFunction() {
  uint32_t FunctionIndex = NumImportedFunctions + InputFunctions.size();
  WasmSym::CallCtors->setOutputIndex(FunctionIndex);

  // First write the body bytes to a string.
  std::string FunctionBody;
  {
    raw_string_ostream OS(FunctionBody);
    writeUleb128(OS, 0, "num locals");
    for (const WasmInitEntry &F : InitFunctions) {
      writeU8(OS, OPCODE_CALL, "CALL");
      writeUleb128(OS, F.Sym->getOutputIndex(), "function index");
    }
    writeU8(OS, OPCODE_END, "END");
  }

  // Once we know the size of the body we can create the final function body
  raw_string_ostream OS(CtorFunctionBody);
  writeUleb128(OS, FunctionBody.size(), "function size");
  OS.flush();
  CtorFunctionBody += FunctionBody;

  const WasmSignature *Sig = WasmSym::CallCtors->getFunctionType();
  SyntheticFunction *F = make<SyntheticFunction>(
      *Sig, toArrayRef(CtorFunctionBody), WasmSym::CallCtors->getName());

  F->setOutputIndex(FunctionIndex);
  F->Live = true;
  WasmSym::CallCtors->Function = F;
  InputFunctions.emplace_back(F);
}

// Populate InitFunctions vector with init functions from all input objects.
// This is then used either when creating the output linking section or to
// synthesize the "__wasm_call_ctors" function.
void Writer::calculateInitFunctions() {
  for (ObjFile *File : Symtab->ObjectFiles) {
    const WasmLinkingData &L = File->getWasmObj()->linkingData();
    for (const WasmInitFunc &F : L.InitFunctions)
      InitFunctions.emplace_back(
          WasmInitEntry{File->getFunctionSymbol(F.Symbol), F.Priority});
  }

  // Sort in order of priority (lowest first) so that they are called
  // in the correct order.
  std::stable_sort(InitFunctions.begin(), InitFunctions.end(),
                   [](const WasmInitEntry &L, const WasmInitEntry &R) {
                     return L.Priority < R.Priority;
                   });
}

void Writer::run() {
  if (Config->Relocatable)
    Config->GlobalBase = 0;

  log("-- calculateImports");
  calculateImports();
  log("-- assignIndexes");
  assignIndexes();
  log("-- calculateInitFunctions");
  calculateInitFunctions();
  if (!Config->Relocatable)
    createCtorFunction();
  log("-- calculateTypes");
  calculateTypes();
  log("-- layoutMemory");
  layoutMemory();
  log("-- calculateExports");
  calculateExports();
  log("-- assignSymtab");
  assignSymtab();

  if (errorHandler().Verbose) {
    log("Defined Functions: " + Twine(InputFunctions.size()));
    log("Defined Globals  : " + Twine(InputGlobals.size()));
    log("Function Imports : " + Twine(NumImportedFunctions));
    log("Global Imports   : " + Twine(NumImportedGlobals));
    for (ObjFile *File : Symtab->ObjectFiles)
      File->dumpInfo();
  }

  createHeader();
  log("-- createSections");
  createSections();

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
