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
#include "OutputSections.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Threads.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LEB128.h"

#include <cstdarg>

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;
using namespace lld;
using namespace lld::wasm;

static constexpr int kStackAlignment = 16;

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
    uintptr_t Value = 0;
    Value += DenseMapInfo<int32_t>::getHashValue(Sig.ReturnType);
    for (int32_t Param : Sig.ParamTypes)
      Value += DenseMapInfo<int32_t>::getHashValue(Param);
    return Value;
  }
  static bool isEqual(const WasmSignature &LHS, const WasmSignature &RHS) {
    return LHS == RHS;
  }
};

// The writer writes a SymbolTable result to a file.
class Writer {
public:
  void run();

private:
  void openFile();

  uint32_t getTypeIndex(const WasmSignature &Sig);
  void assignSymbolIndexes();
  void calculateImports();
  void calculateOffsets();
  void calculateTypes();
  void createOutputSegments();
  void layoutMemory();
  void createHeader();
  void createSections();
  SyntheticSection *createSyntheticSection(uint32_t Type,
                                           std::string Name = "");

  // Builtin sections
  void createTypeSection();
  void createFunctionSection();
  void createTableSection();
  void createGlobalSection();
  void createExportSection();
  void createImportSection();
  void createMemorySection();
  void createElemSection();
  void createStartSection();
  void createCodeSection();
  void createDataSection();

  // Custom sections
  void createRelocSections();
  void createLinkingSection();
  void createNameSection();

  void writeHeader();
  void writeSections();

  uint64_t FileSize = 0;
  uint32_t DataSize = 0;
  uint32_t NumFunctions = 0;
  uint32_t NumGlobals = 0;
  uint32_t NumMemoryPages = 0;
  uint32_t NumTableElems = 0;
  uint32_t NumElements = 0;
  uint32_t InitialTableOffset = 0;

  std::vector<const WasmSignature *> Types;
  DenseMap<WasmSignature, int32_t, WasmSignatureDenseMapInfo> TypeIndices;
  std::vector<Symbol *> FunctionImports;
  std::vector<Symbol *> GlobalImports;

  // Elements that are used to construct the final output
  std::string Header;
  std::vector<OutputSection *> OutputSections;

  std::unique_ptr<FileOutputBuffer> Buffer;

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
  uint32_t NumImports = FunctionImports.size() + GlobalImports.size();
  if (Config->ImportMemory)
    ++NumImports;

  if (NumImports == 0)
    return;

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_IMPORT);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, NumImports, "import count");

  for (Symbol *Sym : FunctionImports) {
    WasmImport Import;
    Import.Module = "env";
    Import.Field = Sym->getName();
    Import.Kind = WASM_EXTERNAL_FUNCTION;
    assert(TypeIndices.count(Sym->getFunctionType()) > 0);
    Import.SigIndex = TypeIndices.lookup(Sym->getFunctionType());
    writeImport(OS, Import);
  }

  if (Config->ImportMemory) {
    WasmImport Import;
    Import.Module = "env";
    Import.Field = "memory";
    Import.Kind = WASM_EXTERNAL_MEMORY;
    Import.Memory.Flags = 0;
    Import.Memory.Initial = NumMemoryPages;
    writeImport(OS, Import);
  }

  for (Symbol *Sym : GlobalImports) {
    WasmImport Import;
    Import.Module = "env";
    Import.Field = Sym->getName();
    Import.Kind = WASM_EXTERNAL_GLOBAL;
    Import.Global.Mutable = false;
    Import.Global.Type = WASM_TYPE_I32; // Sym->getGlobalType();
    writeImport(OS, Import);
  }
}

void Writer::createTypeSection() {
  SyntheticSection *Section = createSyntheticSection(WASM_SEC_TYPE);
  raw_ostream &OS = Section->getStream();
  writeUleb128(OS, Types.size(), "type count");
  for (const WasmSignature *Sig : Types) {
    writeSig(OS, *Sig);
  }
}

void Writer::createFunctionSection() {
  if (!NumFunctions)
    return;

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_FUNCTION);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, NumFunctions, "function count");
  for (ObjFile *File : Symtab->ObjectFiles) {
    for (uint32_t Sig : File->getWasmObj()->functionTypes()) {
      writeUleb128(OS, File->relocateTypeIndex(Sig), "sig index");
    }
  }
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
  SyntheticSection *Section = createSyntheticSection(WASM_SEC_GLOBAL);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, NumGlobals, "global count");
  for (auto &Pair : Config->SyntheticGlobals) {
    WasmGlobal &Global = Pair.second;
    writeGlobal(OS, Global);
  }

  if (Config->Relocatable || Config->EmitRelocs) {
    for (ObjFile *File : Symtab->ObjectFiles) {
      uint32_t GlobalIndex = File->NumGlobalImports();
      for (const WasmGlobal &Global : File->getWasmObj()->globals()) {
        WasmGlobal RelocatedGlobal(Global);
        if (Global.Type != WASM_TYPE_I32)
          fatal("unsupported global type: " + Twine(Global.Type));
        if (Global.InitExpr.Opcode != WASM_OPCODE_I32_CONST)
          fatal("unsupported global init opcode: " +
                Twine(Global.InitExpr.Opcode));
        RelocatedGlobal.InitExpr.Value.Int32 =
            File->getRelocatedAddress(GlobalIndex);
        writeGlobal(OS, RelocatedGlobal);
        ++GlobalIndex;
      }
    }
  }
}

void Writer::createTableSection() {
  SyntheticSection *Section = createSyntheticSection(WASM_SEC_TABLE);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, 1, "table count");
  writeSleb128(OS, WASM_TYPE_ANYFUNC, "table type");
  writeUleb128(OS, WASM_LIMITS_FLAG_HAS_MAX, "table flags");
  writeUleb128(OS, NumTableElems, "table initial size");
  writeUleb128(OS, NumTableElems, "table max size");
}

void Writer::createExportSection() {
  // Memory is and main function are exported for executables.
  bool ExportMemory = !Config->Relocatable && !Config->ImportMemory;
  bool ExportMain = !Config->Relocatable;
  bool ExportOther = true; // Config->Relocatable;

  uint32_t NumExports = 0;

  if (ExportMemory)
    ++NumExports;

  if (ExportMain && !ExportOther)
    ++NumExports;

  if (ExportOther) {
    for (ObjFile *File : Symtab->ObjectFiles) {
      for (Symbol *Sym : File->getSymbols()) {
        if (!Sym->isFunction() || Sym->isLocal() || Sym->isUndefined() ||
            Sym->WrittenToSymtab)
          continue;
        Sym->WrittenToSymtab = true;
        ++NumExports;
      }
    }
  }

  if (!NumExports)
    return;

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_EXPORT);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, NumExports, "export count");

  if (ExportMemory) {
    WasmExport MemoryExport;
    MemoryExport.Name = "memory";
    MemoryExport.Kind = WASM_EXTERNAL_MEMORY;
    MemoryExport.Index = 0;
    writeExport(OS, MemoryExport);
  }

  if (ExportMain) {
    Symbol *Sym = Symtab->find(Config->Entry);
    if (Sym->isDefined()) {
      if (!Sym->isFunction())
        fatal("entry point is not a function: " + Sym->getName());

      if (!ExportOther) {
        WasmExport MainExport;
        MainExport.Name = Config->Entry;
        MainExport.Kind = WASM_EXTERNAL_FUNCTION;
        MainExport.Index = Sym->getOutputIndex();
        writeExport(OS, MainExport);
      }
    }
  }

  if (ExportOther) {
    for (ObjFile *File : Symtab->ObjectFiles) {
      for (Symbol *Sym : File->getSymbols()) {
        if (!Sym->isFunction() || Sym->isLocal() | Sym->isUndefined() ||
            !Sym->WrittenToSymtab)
          continue;
        Sym->WrittenToSymtab = false;
        log("Export: " + Sym->getName());
        WasmExport Export;
        Export.Name = Sym->getName();
        Export.Index = Sym->getOutputIndex();
        if (Sym->isFunction())
          Export.Kind = WASM_EXTERNAL_FUNCTION;
        else
          Export.Kind = WASM_EXTERNAL_GLOBAL;
        writeExport(OS, Export);
      }
    }

    // TODO(sbc): Export local symbols too, Even though they are not part
    // of the symbol table?
  }
}

void Writer::createStartSection() {}

void Writer::createElemSection() {
  if (!NumElements)
    return;

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_ELEM);
  raw_ostream &OS = Section->getStream();

  writeUleb128(OS, 1, "segment count");
  writeUleb128(OS, 0, "table index");
  WasmInitExpr InitExpr;
  InitExpr.Opcode = WASM_OPCODE_I32_CONST;
  InitExpr.Value.Int32 = InitialTableOffset;
  writeInitExpr(OS, InitExpr);
  writeUleb128(OS, NumElements, "elem count");

  for (ObjFile *File : Symtab->ObjectFiles)
    for (const WasmElemSegment &Segment : File->getWasmObj()->elements())
      for (uint64_t FunctionIndex : Segment.Functions)
        writeUleb128(OS, File->relocateFunctionIndex(FunctionIndex),
                     "function index");
}

void Writer::createCodeSection() {
  if (!NumFunctions)
    return;

  log("createCodeSection");

  auto Section = make<CodeSection>(NumFunctions, Symtab->ObjectFiles);
  OutputSections.push_back(Section);
}

void Writer::createDataSection() {
  if (!Segments.size())
    return;

  log("createDataSection");
  auto Section = make<DataSection>(Segments);
  OutputSections.push_back(Section);
}

// Create reloctions sections in the final output.
// These are only created when relocatable output is requested.
void Writer::createRelocSections() {
  log("createRelocSections");
  // Don't use iterator here since we are adding to OutputSection
  size_t OrigSize = OutputSections.size();
  for (size_t i = 0; i < OrigSize; i++) {
    OutputSection *S = OutputSections[i];
    const char *name;
    uint32_t Count = S->numRelocations();
    if (!Count)
      continue;

    if (S->Type == WASM_SEC_DATA)
      name = "reloc.DATA";
    else if (S->Type == WASM_SEC_CODE)
      name = "reloc.CODE";
    else
      llvm_unreachable("relocations only support for code and data");

    SyntheticSection *Section = createSyntheticSection(WASM_SEC_CUSTOM, name);
    raw_ostream &OS = Section->getStream();
    writeUleb128(OS, S->Type, "reloc section");
    writeUleb128(OS, Count, "reloc count");
    S->writeRelocations(OS);
  }
}

// Create the custome "linking" section containing linker metadata.
// This is only created when relocatable output is requested.
void Writer::createLinkingSection() {
  SyntheticSection *Section =
      createSyntheticSection(WASM_SEC_CUSTOM, "linking");
  raw_ostream &OS = Section->getStream();

  SubSection DataSizeSubSection(WASM_DATA_SIZE);
  writeUleb128(DataSizeSubSection.getStream(), DataSize, "data size");
  DataSizeSubSection.finalizeContents();
  DataSizeSubSection.writeToStream(OS);

  if (Segments.size() && Config->Relocatable) {
    SubSection SubSection(WASM_SEGMENT_INFO);
    writeUleb128(SubSection.getStream(), Segments.size(), "num data segments");
    for (const OutputSegment *S : Segments) {
      writeStr(SubSection.getStream(), S->Name, "segment name");
      writeUleb128(SubSection.getStream(), S->Alignment, "alignment");
      writeUleb128(SubSection.getStream(), 0, "flags");
    }
    SubSection.finalizeContents();
    SubSection.writeToStream(OS);
  }
}

// Create the custom "name" section containing debug symbol names.
void Writer::createNameSection() {
  // Create an array of all function sorted by function index space
  std::vector<const Symbol *> Names;

  for (ObjFile *File : Symtab->ObjectFiles) {
    Names.reserve(Names.size() + File->getSymbols().size());
    for (Symbol *S : File->getSymbols()) {
      if (!S->isFunction() || S->isWeak() || S->WrittenToNameSec)
        continue;
      S->WrittenToNameSec = true;
      Names.emplace_back(S);
    }
  }

  SyntheticSection *Section = createSyntheticSection(WASM_SEC_CUSTOM, "name");

  std::sort(Names.begin(), Names.end(), [](const Symbol *A, const Symbol *B) {
    return A->getOutputIndex() < B->getOutputIndex();
  });

  SubSection FunctionSubsection(WASM_NAMES_FUNCTION);
  raw_ostream &OS = FunctionSubsection.getStream();
  writeUleb128(OS, Names.size(), "name count");

  // We have to iterate through the inputs twice so that all the imports
  // appear first before any of the local function names.
  for (const Symbol *S : Names) {
    writeUleb128(OS, S->getOutputIndex(), "func index");
    writeStr(OS, S->getName(), "symbol name");
  }

  FunctionSubsection.finalizeContents();
  FunctionSubsection.writeToStream(Section->getStream());
}

void Writer::writeHeader() {
  memcpy(Buffer->getBufferStart(), Header.data(), Header.size());
}

void Writer::writeSections() {
  uint8_t *Buf = Buffer->getBufferStart();
  parallelForEach(OutputSections, [Buf](OutputSection *S) { S->writeTo(Buf); });
}

// Fix the memory layout of the output binary.  This assigns memory offsets
// to each of the intput data sections as well as the explicit stack region.
void Writer::layoutMemory() {
  uint32_t MemoryPtr = 0;
  if (!Config->Relocatable) {
    MemoryPtr = Config->GlobalBase;
    debugPrint("mem: global base = %d\n", Config->GlobalBase);
  }

  createOutputSegments();

  // Static data comes first
  for (OutputSegment *Seg : Segments) {
    MemoryPtr = alignTo(MemoryPtr, Seg->Alignment);
    Seg->StartVA = MemoryPtr;
    debugPrint("mem: %-10s offset=%-8d size=%-4d align=%d\n",
               Seg->Name.str().c_str(), MemoryPtr, Seg->Size, Seg->Alignment);
    MemoryPtr += Seg->Size;
  }

  DataSize = MemoryPtr;
  if (!Config->Relocatable)
    DataSize -= Config->GlobalBase;
  debugPrint("mem: static data = %d\n", DataSize);

  // Stack comes after static data
  if (!Config->Relocatable) {
    MemoryPtr = alignTo(MemoryPtr, kStackAlignment);
    if (Config->ZStackSize != alignTo(Config->ZStackSize, kStackAlignment))
      error("stack size must be " + Twine(kStackAlignment) + "-byte aligned");
    debugPrint("mem: stack size  = %d\n", Config->ZStackSize);
    debugPrint("mem: stack base  = %d\n", MemoryPtr);
    MemoryPtr += Config->ZStackSize;
    Config->SyntheticGlobals[0].second.InitExpr.Value.Int32 = MemoryPtr;
    debugPrint("mem: stack top   = %d\n", MemoryPtr);
  }

  uint32_t MemSize = alignTo(MemoryPtr, WasmPageSize);
  NumMemoryPages = MemSize / WasmPageSize;
  debugPrint("mem: total pages = %d\n", NumMemoryPages);
}

SyntheticSection *Writer::createSyntheticSection(uint32_t Type,
                                                 std::string Name) {
  auto Sec = make<SyntheticSection>(Type, Name);
  log("createSection: " + toString(Sec));
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
  createStartSection();
  createElemSection();
  createCodeSection();
  createDataSection();

  // Custom sections
  if (Config->EmitRelocs || Config->Relocatable)
    createRelocSections();
  createLinkingSection();
  if (!Config->StripDebug && !Config->StripAll)
    createNameSection();

  for (OutputSection *S : OutputSections) {
    S->setOffset(FileSize);
    S->finalizeContents();
    FileSize += S->getSize();
  }
}

void Writer::calculateOffsets() {
  NumGlobals = Config->SyntheticGlobals.size();
  NumTableElems = InitialTableOffset;

  for (ObjFile *File : Symtab->ObjectFiles) {
    const WasmObjectFile *WasmFile = File->getWasmObj();

    // Function Index
    File->FunctionIndexOffset =
        FunctionImports.size() - File->NumFunctionImports() + NumFunctions;
    NumFunctions += WasmFile->functions().size();

    // Global Index
    if (Config->Relocatable || Config->EmitRelocs) {
      File->GlobalIndexOffset =
          GlobalImports.size() - File->NumGlobalImports() + NumGlobals;
      NumGlobals += WasmFile->globals().size();
    }

    // Memory
    if (WasmFile->memories().size()) {
      if (WasmFile->memories().size() > 1) {
        fatal(File->getName() + ": contains more than one memory");
      }
    }

    // Table
    uint32_t TableCount = WasmFile->tables().size();
    if (TableCount) {
      if (TableCount > 1)
        fatal(File->getName() + ": contains more than one table");
      File->TableIndexOffset = NumTableElems;
      NumTableElems += WasmFile->tables()[0].Limits.Initial;
    }

    // Elem
    uint32_t SegmentCount = WasmFile->elements().size();
    if (SegmentCount) {
      if (SegmentCount > 1)
        fatal(File->getName() + ": contains more than element segment");

      const WasmElemSegment &Segment = WasmFile->elements()[0];
      if (Segment.TableIndex != 0)
        fatal(File->getName() + ": unsupported table index");
      if (Segment.Offset.Value.Int32 != 0)
        fatal(File->getName() + ": unsupported segment offset");
      NumElements += Segment.Functions.size();
    }
  }
}

void Writer::calculateImports() {
  for (ObjFile *File : Symtab->ObjectFiles) {
    for (Symbol *Sym : File->getSymbols()) {
      if (Sym->hasOutputIndex() || Sym->isDefined() || Sym->isWeak())
        continue;

      if (Sym->isFunction()) {
        Sym->setOutputIndex(FunctionImports.size());
        FunctionImports.push_back(Sym);
      } else {
        Sym->setOutputIndex(GlobalImports.size());
        GlobalImports.push_back(Sym);
      }
    }
  }
}

uint32_t Writer::getTypeIndex(const WasmSignature &Sig) {
  auto Pair = TypeIndices.insert(std::make_pair(Sig, Types.size()));
  if (Pair.second)
    Types.push_back(&Sig);
  return Pair.first->second;
}

void Writer::calculateTypes() {
  for (ObjFile *File : Symtab->ObjectFiles) {
    File->TypeMap.reserve(File->getWasmObj()->types().size());
    for (const WasmSignature &Sig : File->getWasmObj()->types())
      File->TypeMap.push_back(getTypeIndex(Sig));
  }
}

void Writer::assignSymbolIndexes() {
  for (ObjFile *File : Symtab->ObjectFiles) {
    DEBUG(dbgs() << "assignSymbolIndexes: " << File->getName() << "\n");
    for (Symbol *Sym : File->getSymbols()) {
      if (Sym->hasOutputIndex() || !Sym->isDefined())
        continue;

      if (Sym->getFile() && isa<ObjFile>(Sym->getFile())) {
        auto *Obj = cast<ObjFile>(Sym->getFile());
        if (Sym->isFunction())
          Sym->setOutputIndex(Obj->FunctionIndexOffset +
                              Sym->getFunctionIndex());
        else
          Sym->setOutputIndex(Obj->GlobalIndexOffset + Sym->getGlobalIndex());
      }
    }
  }
}

static StringRef getOutputDataSegmentName(StringRef Name) {
  if (Config->Relocatable)
    return Name;

  for (StringRef V :
       {".text.", ".rodata.", ".data.rel.ro.", ".data.", ".bss.rel.ro.",
        ".bss.", ".init_array.", ".fini_array.", ".ctors.", ".dtors.", ".tbss.",
        ".gcc_except_table.", ".tdata.", ".ARM.exidx.", ".ARM.extab."}) {
    StringRef Prefix = V.drop_back();
    if (Name.startswith(V) || Name == Prefix)
      return Prefix;
  }

  return Name;
}

void Writer::createOutputSegments() {
  for (ObjFile *File : Symtab->ObjectFiles) {
    for (InputSegment *Segment : File->Segments) {
      StringRef Name = getOutputDataSegmentName(Segment->getName());
      OutputSegment *&S = SegmentMap[Name];
      if (S == nullptr) {
        DEBUG(dbgs() << "new segment: " << Name << "\n");
        S = make<OutputSegment>(Name);
        Segments.push_back(S);
      }
      S->addInputSegment(Segment);
      DEBUG(dbgs() << "added data: " << Name << ": " << S->Size << "\n");
      for (const WasmRelocation &R : File->DataSection->Relocations) {
        if (R.Offset >= Segment->getInputSectionOffset() &&
            R.Offset < Segment->getInputSectionOffset() + Segment->getSize()) {
          Segment->Relocations.push_back(R);
        }
      }
    }
  }
}

void Writer::run() {
  if (!Config->Relocatable)
    InitialTableOffset = 1;

  log("-- calculateTypes");
  calculateTypes();
  log("-- calculateImports");
  calculateImports();
  log("-- calculateOffsets");
  calculateOffsets();

  if (errorHandler().Verbose) {
    log("NumFunctions    : " + Twine(NumFunctions));
    log("NumGlobals      : " + Twine(NumGlobals));
    log("NumImports      : " +
        Twine(FunctionImports.size() + GlobalImports.size()));
    log("FunctionImports : " + Twine(FunctionImports.size()));
    log("GlobalImports   : " + Twine(GlobalImports.size()));
    for (ObjFile *File : Symtab->ObjectFiles)
      File->dumpInfo();
  }

  log("-- assignSymbolIndexes");
  assignSymbolIndexes();
  log("-- layoutMemory");
  layoutMemory();

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
  ::remove(Config->OutputFile.str().c_str());

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
