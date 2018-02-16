//===- InputFiles.cpp -----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"

#include "Config.h"
#include "InputChunks.h"
#include "SymbolTable.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "lld"

using namespace lld;
using namespace lld::wasm;

using namespace llvm;
using namespace llvm::object;
using namespace llvm::wasm;

Optional<MemoryBufferRef> lld::wasm::readFile(StringRef Path) {
  log("Loading: " + Path);

  auto MBOrErr = MemoryBuffer::getFile(Path);
  if (auto EC = MBOrErr.getError()) {
    error("cannot open " + Path + ": " + EC.message());
    return None;
  }
  std::unique_ptr<MemoryBuffer> &MB = *MBOrErr;
  MemoryBufferRef MBRef = MB->getMemBufferRef();
  make<std::unique_ptr<MemoryBuffer>>(std::move(MB)); // take MB ownership

  return MBRef;
}

void ObjFile::dumpInfo() const {
  log("info for: " + getName() + "\n" +
      "      Total Functions : " + Twine(FunctionSymbols.size()) + "\n" +
      "        Total Globals : " + Twine(GlobalSymbols.size()) + "\n" +
      "     Function Imports : " + Twine(NumFunctionImports) + "\n" +
      "       Global Imports : " + Twine(NumGlobalImports) + "\n");
}

uint32_t ObjFile::relocateVirtualAddress(uint32_t GlobalIndex) const {
  if (auto *DG = dyn_cast<DefinedGlobal>(getGlobalSymbol(GlobalIndex)))
    return DG->getVirtualAddress();
  else
    return 0;
}

uint32_t ObjFile::relocateFunctionIndex(uint32_t Original) const {
  const FunctionSymbol *Sym = getFunctionSymbol(Original);
  uint32_t Index = Sym->getOutputIndex();
  DEBUG(dbgs() << "relocateFunctionIndex: " << toString(*Sym) << ": "
               << Original << " -> " << Index << "\n");
  return Index;
}

uint32_t ObjFile::relocateTypeIndex(uint32_t Original) const {
  assert(TypeIsUsed[Original]);
  return TypeMap[Original];
}

uint32_t ObjFile::relocateTableIndex(uint32_t Original) const {
  const FunctionSymbol *Sym = getFunctionSymbol(Original);
  uint32_t Index = Sym->hasTableIndex() ? Sym->getTableIndex() : 0;
  DEBUG(dbgs() << "relocateTableIndex: " << toString(*Sym) << ": " << Original
               << " -> " << Index << "\n");
  return Index;
}

uint32_t ObjFile::relocateGlobalIndex(uint32_t Original) const {
  const Symbol *Sym = getGlobalSymbol(Original);
  uint32_t Index = Sym->getOutputIndex();
  DEBUG(dbgs() << "relocateGlobalIndex: " << toString(*Sym) << ": " << Original
               << " -> " << Index << "\n");
  return Index;
}

// Relocations contain an index into the function, global or table index
// space of the input file.  This function takes a relocation and returns the
// relocated index (i.e. translates from the input index space to the output
// index space).
uint32_t ObjFile::calcNewIndex(const WasmRelocation &Reloc) const {
  switch (Reloc.Type) {
  case R_WEBASSEMBLY_TYPE_INDEX_LEB:
    return relocateTypeIndex(Reloc.Index);
  case R_WEBASSEMBLY_FUNCTION_INDEX_LEB:
  case R_WEBASSEMBLY_TABLE_INDEX_I32:
  case R_WEBASSEMBLY_TABLE_INDEX_SLEB:
    return relocateFunctionIndex(Reloc.Index);
  case R_WEBASSEMBLY_GLOBAL_INDEX_LEB:
  case R_WEBASSEMBLY_MEMORY_ADDR_LEB:
  case R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
  case R_WEBASSEMBLY_MEMORY_ADDR_I32:
    return relocateGlobalIndex(Reloc.Index);
  default:
    llvm_unreachable("unknown relocation type");
  }
}

// Translate from the relocation's index into the final linked output value.
uint32_t ObjFile::calcNewValue(const WasmRelocation &Reloc) const {
  switch (Reloc.Type) {
  case R_WEBASSEMBLY_TABLE_INDEX_I32:
  case R_WEBASSEMBLY_TABLE_INDEX_SLEB:
    return relocateTableIndex(Reloc.Index);
  case R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
  case R_WEBASSEMBLY_MEMORY_ADDR_I32:
  case R_WEBASSEMBLY_MEMORY_ADDR_LEB:
    return relocateVirtualAddress(Reloc.Index) + Reloc.Addend;
  case R_WEBASSEMBLY_TYPE_INDEX_LEB:
    return relocateTypeIndex(Reloc.Index);
  case R_WEBASSEMBLY_FUNCTION_INDEX_LEB:
    return relocateFunctionIndex(Reloc.Index);
  case R_WEBASSEMBLY_GLOBAL_INDEX_LEB:
    return relocateGlobalIndex(Reloc.Index);
  default:
    llvm_unreachable("unknown relocation type");
  }
}

void ObjFile::parse() {
  // Parse a memory buffer as a wasm file.
  DEBUG(dbgs() << "Parsing object: " << toString(this) << "\n");
  std::unique_ptr<Binary> Bin = CHECK(createBinary(MB), toString(this));

  auto *Obj = dyn_cast<WasmObjectFile>(Bin.get());
  if (!Obj)
    fatal(toString(this) + ": not a wasm file");
  if (!Obj->isRelocatableObject())
    fatal(toString(this) + ": not a relocatable wasm file");

  Bin.release();
  WasmObj.reset(Obj);

  // Find the code and data sections.  Wasm objects can have at most one code
  // and one data section.
  for (const SectionRef &Sec : WasmObj->sections()) {
    const WasmSection &Section = WasmObj->getWasmSection(Sec);
    if (Section.Type == WASM_SEC_CODE)
      CodeSection = &Section;
    else if (Section.Type == WASM_SEC_DATA)
      DataSection = &Section;
  }

  TypeMap.resize(getWasmObj()->types().size());
  TypeIsUsed.resize(getWasmObj()->types().size(), false);

  initializeSymbols();
}

// Return the InputSegment in which a given symbol is defined.
InputSegment *ObjFile::getSegment(const WasmSymbol &WasmSym) const {
  uint32_t Address = WasmObj->getWasmSymbolValue(WasmSym);
  for (InputSegment *Segment : Segments) {
    if (Address >= Segment->startVA() && Address < Segment->endVA()) {
      DEBUG(dbgs() << "Found symbol in segment: " << WasmSym.Name << " -> "
                   << Segment->getName() << "\n");

      return Segment;
    }
  }
  error("symbol not found in any segment: " + WasmSym.Name);
  return nullptr;
}

// Get the value stored in the wasm global represented by this symbol.
// This represents the virtual address of the symbol in the input file.
uint32_t ObjFile::getGlobalValue(const WasmSymbol &Sym) const {
  const WasmGlobal &Global =
      getWasmObj()->globals()[Sym.ElementIndex - NumGlobalImports];
  assert(Global.Type.Type == llvm::wasm::WASM_TYPE_I32);
  return Global.InitExpr.Value.Int32;
}

// Get the signature for a given function symbol, either by looking
// it up in function sections (for defined functions), of the imports section
// (for imported functions).
const WasmSignature *ObjFile::getFunctionSig(const WasmSymbol &Sym) const {
  DEBUG(dbgs() << "getFunctionSig: " << Sym.Name << "\n");
  return &WasmObj->types()[Sym.FunctionType];
}

InputFunction *ObjFile::getFunction(const WasmSymbol &Sym) const {
  uint32_t FunctionIndex = Sym.ElementIndex - NumFunctionImports;
  return Functions[FunctionIndex];
}

bool ObjFile::isExcludedByComdat(InputChunk *Chunk) const {
  StringRef Comdat = Chunk->getComdat();
  return !Comdat.empty() && Symtab->findComdat(Comdat) != this;
}

void ObjFile::initializeSymbols() {
  Symbols.reserve(WasmObj->getNumberOfSymbols());

  for (const WasmImport &Import : WasmObj->imports()) {
    switch (Import.Kind) {
    case WASM_EXTERNAL_FUNCTION:
      ++NumFunctionImports;
      break;
    case WASM_EXTERNAL_GLOBAL:
      ++NumGlobalImports;
      break;
    }
  }

  FunctionSymbols.resize(NumFunctionImports + WasmObj->functions().size());
  GlobalSymbols.resize(NumGlobalImports + WasmObj->globals().size());

  ArrayRef<WasmFunction> Funcs = WasmObj->functions();
  ArrayRef<uint32_t> FuncTypes = WasmObj->functionTypes();
  ArrayRef<WasmSignature> Types = WasmObj->types();
  ArrayRef<WasmGlobal> Globals = WasmObj->globals();

  for (const auto &C : WasmObj->comdats())
    Symtab->addComdat(C, this);

  FunctionSymbols.resize(NumFunctionImports + Funcs.size());
  GlobalSymbols.resize(NumGlobalImports + Globals.size());

  for (const WasmSegment &S : WasmObj->dataSegments()) {
    InputSegment *Seg = make<InputSegment>(S, this);
    Seg->copyRelocations(*DataSection);
    Segments.emplace_back(Seg);
  }

  for (size_t I = 0; I < Funcs.size(); ++I) {
    const WasmFunction &Func = Funcs[I];
    const WasmSignature &Sig = Types[FuncTypes[I]];
    InputFunction *F = make<InputFunction>(Sig, &Func, this);
    F->copyRelocations(*CodeSection);
    Functions.emplace_back(F);
  }

  // Populate `FunctionSymbols` and `GlobalSymbols` based on the WasmSymbols
  // in the object
  for (const SymbolRef &Sym : WasmObj->symbols()) {
    const WasmSymbol &WasmSym = WasmObj->getWasmSymbol(Sym.getRawDataRefImpl());
    Symbol *S;
    switch (WasmSym.Type) {
    case WasmSymbol::SymbolType::FUNCTION_EXPORT: {
      InputFunction *Function = getFunction(WasmSym);
      if (!isExcludedByComdat(Function)) {
        S = createDefinedFunction(WasmSym, Function);
        break;
      }
      Function->Live = false;
      LLVM_FALLTHROUGH; // Exclude function, and add the symbol as undefined
    }
    case WasmSymbol::SymbolType::FUNCTION_IMPORT:
      S = createUndefined(WasmSym, Symbol::Kind::UndefinedFunctionKind,
                          getFunctionSig(WasmSym));
      break;
    case WasmSymbol::SymbolType::GLOBAL_EXPORT: {
      InputSegment *Segment = getSegment(WasmSym);
      if (!isExcludedByComdat(Segment)) {
        S = createDefinedGlobal(WasmSym, Segment, getGlobalValue(WasmSym));
        break;
      }
      Segment->Live = false;
      LLVM_FALLTHROUGH; // Exclude global, and add the symbol as undefined
    }
    case WasmSymbol::SymbolType::GLOBAL_IMPORT:
      S = createUndefined(WasmSym, Symbol::Kind::UndefinedGlobalKind);
      break;
    }

    Symbols.push_back(S);
    if (WasmSym.isTypeFunction()) {
      FunctionSymbols[WasmSym.ElementIndex] = S;
      if (WasmSym.HasAltIndex)
        FunctionSymbols[WasmSym.AltIndex] = S;
    } else {
      GlobalSymbols[WasmSym.ElementIndex] = S;
      if (WasmSym.HasAltIndex)
        GlobalSymbols[WasmSym.AltIndex] = S;
    }
  }

  DEBUG(for (size_t I = 0; I < FunctionSymbols.size(); ++I)
            assert(FunctionSymbols[I] != nullptr);
        for (size_t I = 0; I < GlobalSymbols.size(); ++I)
            assert(GlobalSymbols[I] != nullptr););

  DEBUG(dbgs() << "Functions   : " << FunctionSymbols.size() << "\n");
  DEBUG(dbgs() << "Globals     : " << GlobalSymbols.size() << "\n");
}

Symbol *ObjFile::createUndefined(const WasmSymbol &Sym, Symbol::Kind Kind,
                                 const WasmSignature *Signature) {
  return Symtab->addUndefined(Sym.Name, Kind, Sym.Flags, this, Signature);
}

Symbol *ObjFile::createDefinedFunction(const WasmSymbol &Sym,
                                       InputChunk *Chunk) {
  if (Sym.isBindingLocal())
    return make<DefinedFunction>(Sym.Name, Sym.Flags, this, Chunk);
  return Symtab->addDefined(true, Sym.Name, Sym.Flags, this, Chunk);
}

Symbol *ObjFile::createDefinedGlobal(const WasmSymbol &Sym, InputChunk *Chunk,
                                     uint32_t Address) {
  if (Sym.isBindingLocal())
    return make<DefinedGlobal>(Sym.Name, Sym.Flags, this, Chunk, Address);
  return Symtab->addDefined(false, Sym.Name, Sym.Flags, this, Chunk, Address);
}

void ArchiveFile::parse() {
  // Parse a MemoryBufferRef as an archive file.
  DEBUG(dbgs() << "Parsing library: " << toString(this) << "\n");
  File = CHECK(Archive::create(MB), toString(this));

  // Read the symbol table to construct Lazy symbols.
  int Count = 0;
  for (const Archive::Symbol &Sym : File->symbols()) {
    Symtab->addLazy(this, &Sym);
    ++Count;
  }
  DEBUG(dbgs() << "Read " << Count << " symbols\n");
}

void ArchiveFile::addMember(const Archive::Symbol *Sym) {
  const Archive::Child &C =
      CHECK(Sym->getMember(),
            "could not get the member for symbol " + Sym->getName());

  // Don't try to load the same member twice (this can happen when members
  // mutually reference each other).
  if (!Seen.insert(C.getChildOffset()).second)
    return;

  DEBUG(dbgs() << "loading lazy: " << Sym->getName() << "\n");
  DEBUG(dbgs() << "from archive: " << toString(this) << "\n");

  MemoryBufferRef MB =
      CHECK(C.getMemoryBufferRef(),
            "could not get the buffer for the member defining symbol " +
                Sym->getName());

  if (identify_magic(MB.getBuffer()) != file_magic::wasm_object) {
    error("unknown file type: " + MB.getBufferIdentifier());
    return;
  }

  InputFile *Obj = make<ObjFile>(MB);
  Obj->ParentName = ParentName;
  Symtab->addFile(Obj);
}

// Returns a string in the format of "foo.o" or "foo.a(bar.o)".
std::string lld::toString(const wasm::InputFile *File) {
  if (!File)
    return "<internal>";

  if (File->ParentName.empty())
    return File->getName();

  return (File->ParentName + "(" + File->getName() + ")").str();
}
