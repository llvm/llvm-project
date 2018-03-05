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
  log("info for: " + getName() +
      "\n              Symbols : " + Twine(Symbols.size()) +
      "\n     Function Imports : " + Twine(WasmObj->getNumImportedFunctions()) +
      "\n       Global Imports : " + Twine(WasmObj->getNumImportedGlobals()));
}

// Relocations contain an index into the function, global or table index
// space of the input file.  This function takes a relocation and returns the
// relocated index (i.e. translates from the input index space to the output
// index space).
uint32_t ObjFile::calcNewIndex(const WasmRelocation &Reloc) const {
  if (Reloc.Type == R_WEBASSEMBLY_TYPE_INDEX_LEB) {
    assert(TypeIsUsed[Reloc.Index]);
    return TypeMap[Reloc.Index];
  }
  return Symbols[Reloc.Index]->getOutputSymbolIndex();
}

// Translate from the relocation's index into the final linked output value.
uint32_t ObjFile::calcNewValue(const WasmRelocation &Reloc) const {
  switch (Reloc.Type) {
  case R_WEBASSEMBLY_TABLE_INDEX_I32:
  case R_WEBASSEMBLY_TABLE_INDEX_SLEB: {
    // The null case is possible, if you take the address of a weak function
    // that's simply not supplied.
    FunctionSymbol *Sym = getFunctionSymbol(Reloc.Index);
    if (Sym->hasTableIndex())
      return Sym->getTableIndex();
    return 0;
  }
  case R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
  case R_WEBASSEMBLY_MEMORY_ADDR_I32:
  case R_WEBASSEMBLY_MEMORY_ADDR_LEB:
    if (auto *Sym = dyn_cast<DefinedData>(getDataSymbol(Reloc.Index)))
      return Sym->getVirtualAddress() + Reloc.Addend;
    return Reloc.Addend;
  case R_WEBASSEMBLY_TYPE_INDEX_LEB:
    return TypeMap[Reloc.Index];
  case R_WEBASSEMBLY_FUNCTION_INDEX_LEB:
    return getFunctionSymbol(Reloc.Index)->getOutputIndex();
  case R_WEBASSEMBLY_GLOBAL_INDEX_LEB:
    return getGlobalSymbol(Reloc.Index)->getOutputIndex();
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

  // Populate `Segments`.
  for (const WasmSegment &S : WasmObj->dataSegments()) {
    InputSegment *Seg = make<InputSegment>(S, this);
    Seg->copyRelocations(*DataSection);
    Segments.emplace_back(Seg);
  }

  // Populate `Functions`.
  ArrayRef<WasmFunction> Funcs = WasmObj->functions();
  ArrayRef<uint32_t> FuncTypes = WasmObj->functionTypes();
  ArrayRef<WasmSignature> Types = WasmObj->types();
  Functions.reserve(Funcs.size());

  for (size_t I = 0, E = Funcs.size(); I != E; ++I) {
    InputFunction *F =
        make<InputFunction>(Types[FuncTypes[I]], &Funcs[I], this);
    F->copyRelocations(*CodeSection);
    Functions.emplace_back(F);
  }

  // Populate `Globals`.
  for (const WasmGlobal &G : WasmObj->globals())
    Globals.emplace_back(make<InputGlobal>(G));

  // Populate `Symbols` based on the WasmSymbols in the object.
  Symbols.reserve(WasmObj->getNumberOfSymbols());
  for (const SymbolRef &Sym : WasmObj->symbols()) {
    const WasmSymbol &WasmSym = WasmObj->getWasmSymbol(Sym.getRawDataRefImpl());
    if (Symbol *Sym = createDefined(WasmSym))
      Symbols.push_back(Sym);
    else
      Symbols.push_back(createUndefined(WasmSym));
  }
}

bool ObjFile::isExcludedByComdat(InputChunk *Chunk) const {
  StringRef S = Chunk->getComdat();
  if (S.empty())
    return false;
  return !Symtab->addComdat(S, this);
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

Symbol *ObjFile::createDefined(const WasmSymbol &Sym) {
  if (!Sym.isDefined())
    return nullptr;

  StringRef Name = Sym.Info.Name;
  uint32_t Flags = Sym.Info.Flags;

  switch (Sym.Info.Kind) {
  case WASM_SYMBOL_TYPE_FUNCTION: {
    InputFunction *Func =
        Functions[Sym.Info.ElementIndex - WasmObj->getNumImportedFunctions()];
    if (isExcludedByComdat(Func)) {
      Func->Live = false;
      return nullptr;
    }

    if (Sym.isBindingLocal())
      return make<DefinedFunction>(Name, Flags, this, Func);
    return Symtab->addDefinedFunction(Name, Flags, this, Func);
  }
  case WASM_SYMBOL_TYPE_DATA: {
    InputSegment *Seg = Segments[Sym.Info.DataRef.Segment];
    if (isExcludedByComdat(Seg)) {
      Seg->Live = false;
      return nullptr;
    }

    uint32_t Offset = Sym.Info.DataRef.Offset;
    uint32_t Size = Sym.Info.DataRef.Size;

    if (Sym.isBindingLocal())
      return make<DefinedData>(Name, Flags, this, Seg, Offset, Size);
    return Symtab->addDefinedData(Name, Flags, this, Seg, Offset, Size);
  }
  case WASM_SYMBOL_TYPE_GLOBAL:
    InputGlobal *Global =
        Globals[Sym.Info.ElementIndex - WasmObj->getNumImportedGlobals()];
    if (Sym.isBindingLocal())
      return make<DefinedGlobal>(Name, Flags, this, Global);
    return Symtab->addDefinedGlobal(Name, Flags, this, Global);
  }
  llvm_unreachable("unkown symbol kind");
}

Symbol *ObjFile::createUndefined(const WasmSymbol &Sym) {
  StringRef Name = Sym.Info.Name;
  uint32_t Flags = Sym.Info.Flags;

  switch (Sym.Info.Kind) {
  case WASM_SYMBOL_TYPE_FUNCTION:
    return Symtab->addUndefinedFunction(Name, Flags, this, Sym.FunctionType);
  case WASM_SYMBOL_TYPE_DATA:
    return Symtab->addUndefinedData(Name, Flags, this);
  case WASM_SYMBOL_TYPE_GLOBAL:
    return Symtab->addUndefinedGlobal(Name, Flags, this, Sym.GlobalType);
  }
  llvm_unreachable("unkown symbol kind");
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
