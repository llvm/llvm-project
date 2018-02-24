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
#include "InputGlobal.h"
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
      "              Symbols : " + Twine(Symbols.size()) + "\n" +
      "     Function Imports : " + Twine(NumFunctionImports) + "\n" +
      "       Global Imports : " + Twine(NumGlobalImports) + "\n");
}

uint32_t ObjFile::relocateVirtualAddress(uint32_t GlobalIndex) const {
  if (auto *DG = dyn_cast<DefinedData>(getDataSymbol(GlobalIndex)))
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
  // The null case is possible, if you take the address of a weak function
  // that's simply not supplied.
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

uint32_t ObjFile::relocateSymbolIndex(uint32_t Original) const {
  Symbol *Sym = getSymbol(Original);
  uint32_t Index = Sym->getOutputSymbolIndex();
  DEBUG(dbgs() << "relocateSymbolIndex: " << toString(*Sym) << ": " << Original
               << " -> " << Index << "\n");
  return Index;
}

// Relocations contain an index into the function, global or table index
// space of the input file.  This function takes a relocation and returns the
// relocated index (i.e. translates from the input index space to the output
// index space).
uint32_t ObjFile::calcNewIndex(const WasmRelocation &Reloc) const {
  if (Reloc.Type == R_WEBASSEMBLY_TYPE_INDEX_LEB)
    return relocateTypeIndex(Reloc.Index);

  return relocateSymbolIndex(Reloc.Index);
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
  return Segments[WasmSym.Info.DataRef.Segment];
}

InputFunction *ObjFile::getFunction(const WasmSymbol &Sym) const {
  assert(Sym.Info.ElementIndex >= NumFunctionImports);
  uint32_t FunctionIndex = Sym.Info.ElementIndex - NumFunctionImports;
  return Functions[FunctionIndex];
}

InputGlobal *ObjFile::getGlobal(const WasmSymbol &Sym) const {
  assert(Sym.Info.ElementIndex >= NumGlobalImports);
  uint32_t GlobalIndex = Sym.Info.ElementIndex - NumGlobalImports;
  return Globals[GlobalIndex];
}

bool ObjFile::isExcludedByComdat(InputChunk *Chunk) const {
  StringRef Comdat = Chunk->getComdat();
  return !Comdat.empty() && Symtab->findComdat(Comdat) != this;
}

FunctionSymbol *ObjFile::getFunctionSymbol(uint32_t Index) const {
  return cast<FunctionSymbol>(Symbols[Index]);
}

GlobalSymbol *ObjFile::getGlobalSymbol(uint32_t Index) const {
  return cast<GlobalSymbol>(Symbols[Index]);
}

DataSymbol *ObjFile::getDataSymbol(uint32_t Index) const {
  return cast<DataSymbol>(Symbols[Index]);
}

void ObjFile::initializeSymbols() {
  Symbols.reserve(WasmObj->getNumberOfSymbols());

  NumFunctionImports = WasmObj->getNumImportedFunctions();
  NumGlobalImports = WasmObj->getNumImportedGlobals();

  ArrayRef<WasmFunction> Funcs = WasmObj->functions();
  ArrayRef<uint32_t> FuncTypes = WasmObj->functionTypes();
  ArrayRef<WasmSignature> Types = WasmObj->types();

  for (StringRef Name : WasmObj->comdats())
    Symtab->addComdat(Name, this);

  for (const WasmSegment &S : WasmObj->dataSegments()) {
    InputSegment *Seg = make<InputSegment>(S, this);
    Seg->copyRelocations(*DataSection);
    Segments.emplace_back(Seg);
  }

  for (const WasmGlobal &G : WasmObj->globals())
    Globals.emplace_back(make<InputGlobal>(G));

  for (size_t I = 0; I < Funcs.size(); ++I) {
    const WasmFunction &Func = Funcs[I];
    const WasmSignature &Sig = Types[FuncTypes[I]];
    InputFunction *F = make<InputFunction>(Sig, &Func, this);
    F->copyRelocations(*CodeSection);
    Functions.emplace_back(F);
  }

  // Populate `Symbols` based on the WasmSymbols in the object
  for (const SymbolRef &Sym : WasmObj->symbols()) {
    const WasmSymbol &WasmSym = WasmObj->getWasmSymbol(Sym.getRawDataRefImpl());
    bool IsDefined = WasmSym.isDefined();

    if (IsDefined) {
      switch (WasmSym.Info.Kind) {
      case WASM_SYMBOL_TYPE_FUNCTION: {
        InputFunction *Function = getFunction(WasmSym);
        if (isExcludedByComdat(Function)) {
          Function->Live = false;
          IsDefined = false;
          break;
        }
        Symbols.push_back(createDefinedFunction(WasmSym, Function));
        break;
      }
      case WASM_SYMBOL_TYPE_DATA: {
        InputSegment *Segment = getSegment(WasmSym);
        if (isExcludedByComdat(Segment)) {
          Segment->Live = false;
          IsDefined = false;
          break;
        }
        Symbols.push_back(createDefinedData(WasmSym, Segment,
                                            WasmSym.Info.DataRef.Offset,
                                            WasmSym.Info.DataRef.Size));
        break;
      }
      case WASM_SYMBOL_TYPE_GLOBAL:
        Symbols.push_back(createDefinedGlobal(WasmSym, getGlobal(WasmSym)));
        break;
      default:
        llvm_unreachable("unkown symbol kind");
        break;
      }
    }

    // Either the the symbol itself was undefined, or was excluded via comdat
    // in which case this simply insertes the existing symbol into the correct
    // slot in the Symbols array.
    if (!IsDefined)
      Symbols.push_back(createUndefined(WasmSym));
  }
}

Symbol *ObjFile::createUndefined(const WasmSymbol &Sym) {
  return Symtab->addUndefined(
      Sym.Info.Name, static_cast<WasmSymbolType>(Sym.Info.Kind), Sym.Info.Flags,
      this, Sym.FunctionType, Sym.GlobalType);
}

Symbol *ObjFile::createDefinedFunction(const WasmSymbol &Sym,
                                       InputFunction *Function) {
  if (Sym.isBindingLocal())
    return make<DefinedFunction>(Sym.Info.Name, Sym.Info.Flags, this, Function);
  return Symtab->addDefinedFunction(Sym.Info.Name, Sym.Info.Flags, this,
                                    Function);
}

Symbol *ObjFile::createDefinedData(const WasmSymbol &Sym, InputSegment *Segment,
                                   uint32_t Offset, uint32_t Size) {
  if (Sym.isBindingLocal())
    return make<DefinedData>(Sym.Info.Name, Sym.Info.Flags, this, Segment,
                             Offset, Size);
  return Symtab->addDefinedData(Sym.Info.Name, Sym.Info.Flags, this, Segment,
                                Offset, Size);
}

Symbol *ObjFile::createDefinedGlobal(const WasmSymbol &Sym,
                                     InputGlobal *Global) {
  if (Sym.isBindingLocal())
    return make<DefinedGlobal>(Sym.Info.Name, Sym.Info.Flags, this, Global);
  return Symtab->addDefinedGlobal(Sym.Info.Name, Sym.Info.Flags, this, Global);
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
