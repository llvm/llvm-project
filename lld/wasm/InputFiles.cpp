//===- InputFiles.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Config.h"
#include "InputChunks.h"
#include "InputEvent.h"
#include "InputGlobal.h"
#include "SymbolTable.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Reproduce.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/TarWriter.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "lld"

using namespace lld;
using namespace lld::wasm;

using namespace llvm;
using namespace llvm::object;
using namespace llvm::wasm;

std::unique_ptr<llvm::TarWriter> lld::wasm::Tar;

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

  if (Tar)
    Tar->append(relativeToRoot(Path), MBRef.getBuffer());
  return MBRef;
}

InputFile *lld::wasm::createObjectFile(MemoryBufferRef MB,
                                       StringRef ArchiveName) {
  file_magic Magic = identify_magic(MB.getBuffer());
  if (Magic == file_magic::wasm_object) {
    std::unique_ptr<Binary> Bin = check(createBinary(MB));
    auto *Obj = cast<WasmObjectFile>(Bin.get());
    if (Obj->isSharedObject())
      return make<SharedFile>(MB);
    return make<ObjFile>(MB, ArchiveName);
  }

  if (Magic == file_magic::bitcode)
    return make<BitcodeFile>(MB, ArchiveName);

  fatal("unknown file type: " + MB.getBufferIdentifier());
}

void ObjFile::dumpInfo() const {
  log("info for: " + toString(this) +
      "\n              Symbols : " + Twine(Symbols.size()) +
      "\n     Function Imports : " + Twine(WasmObj->getNumImportedFunctions()) +
      "\n       Global Imports : " + Twine(WasmObj->getNumImportedGlobals()) +
      "\n        Event Imports : " + Twine(WasmObj->getNumImportedEvents()));
}

// Relocations contain either symbol or type indices.  This function takes a
// relocation and returns relocated index (i.e. translates from the input
// symbol/type space to the output symbol/type space).
uint32_t ObjFile::calcNewIndex(const WasmRelocation &Reloc) const {
  if (Reloc.Type == R_WASM_TYPE_INDEX_LEB) {
    assert(TypeIsUsed[Reloc.Index]);
    return TypeMap[Reloc.Index];
  }
  const Symbol *Sym = Symbols[Reloc.Index];
  if (auto *SS = dyn_cast<SectionSymbol>(Sym))
    Sym = SS->getOutputSectionSymbol();
  return Sym->getOutputSymbolIndex();
}

// Relocations can contain addend for combined sections. This function takes a
// relocation and returns updated addend by offset in the output section.
uint32_t ObjFile::calcNewAddend(const WasmRelocation &Reloc) const {
  switch (Reloc.Type) {
  case R_WASM_MEMORY_ADDR_LEB:
  case R_WASM_MEMORY_ADDR_SLEB:
  case R_WASM_MEMORY_ADDR_I32:
  case R_WASM_FUNCTION_OFFSET_I32:
    return Reloc.Addend;
  case R_WASM_SECTION_OFFSET_I32:
    return getSectionSymbol(Reloc.Index)->Section->OutputOffset + Reloc.Addend;
  default:
    llvm_unreachable("unexpected relocation type");
  }
}

// Calculate the value we expect to find at the relocation location.
// This is used as a sanity check before applying a relocation to a given
// location.  It is useful for catching bugs in the compiler and linker.
uint32_t ObjFile::calcExpectedValue(const WasmRelocation &Reloc) const {
  switch (Reloc.Type) {
  case R_WASM_TABLE_INDEX_I32:
  case R_WASM_TABLE_INDEX_SLEB:
  case R_WASM_TABLE_INDEX_REL_SLEB: {
    const WasmSymbol &Sym = WasmObj->syms()[Reloc.Index];
    return TableEntries[Sym.Info.ElementIndex];
  }
  case R_WASM_MEMORY_ADDR_SLEB:
  case R_WASM_MEMORY_ADDR_I32:
  case R_WASM_MEMORY_ADDR_LEB:
  case R_WASM_MEMORY_ADDR_REL_SLEB: {
    const WasmSymbol &Sym = WasmObj->syms()[Reloc.Index];
    if (Sym.isUndefined())
      return 0;
    const WasmSegment &Segment =
        WasmObj->dataSegments()[Sym.Info.DataRef.Segment];
    return Segment.Data.Offset.Value.Int32 + Sym.Info.DataRef.Offset +
           Reloc.Addend;
  }
  case R_WASM_FUNCTION_OFFSET_I32: {
    const WasmSymbol &Sym = WasmObj->syms()[Reloc.Index];
    InputFunction *F =
        Functions[Sym.Info.ElementIndex - WasmObj->getNumImportedFunctions()];
    return F->getFunctionInputOffset() + F->getFunctionCodeOffset() +
           Reloc.Addend;
  }
  case R_WASM_SECTION_OFFSET_I32:
    return Reloc.Addend;
  case R_WASM_TYPE_INDEX_LEB:
    return Reloc.Index;
  case R_WASM_FUNCTION_INDEX_LEB:
  case R_WASM_GLOBAL_INDEX_LEB:
  case R_WASM_EVENT_INDEX_LEB: {
    const WasmSymbol &Sym = WasmObj->syms()[Reloc.Index];
    return Sym.Info.ElementIndex;
  }
  default:
    llvm_unreachable("unknown relocation type");
  }
}

// Translate from the relocation's index into the final linked output value.
uint32_t ObjFile::calcNewValue(const WasmRelocation &Reloc) const {
  const Symbol* Sym = nullptr;
  if (Reloc.Type != R_WASM_TYPE_INDEX_LEB) {
    Sym = Symbols[Reloc.Index];

    // We can end up with relocations against non-live symbols.  For example
    // in debug sections.
    if ((isa<FunctionSymbol>(Sym) || isa<DataSymbol>(Sym)) && !Sym->isLive())
      return 0;

    // Special handling for undefined data symbols.  Most relocations against
    // such symbols cannot be resolved.
    if (isa<DataSymbol>(Sym) && Sym->isUndefined()) {
      if (Sym->isWeak() || Config->Relocatable)
        return 0;
      // R_WASM_MEMORY_ADDR_I32 relocations in PIC code are turned into runtime
      // fixups in __wasm_apply_relocs
      if (Config->Pic && Reloc.Type == R_WASM_MEMORY_ADDR_I32)
        return 0;
      if (Reloc.Type != R_WASM_GLOBAL_INDEX_LEB) {
        llvm_unreachable(
          ("invalid relocation against undefined data symbol: " + toString(*Sym))
              .c_str());
      }
    }
  }

  switch (Reloc.Type) {
  case R_WASM_TABLE_INDEX_I32:
  case R_WASM_TABLE_INDEX_SLEB:
  case R_WASM_TABLE_INDEX_REL_SLEB:
    if (Config->Pic && !getFunctionSymbol(Reloc.Index)->hasTableIndex())
      return 0;
    return getFunctionSymbol(Reloc.Index)->getTableIndex();
  case R_WASM_MEMORY_ADDR_SLEB:
  case R_WASM_MEMORY_ADDR_I32:
  case R_WASM_MEMORY_ADDR_LEB:
  case R_WASM_MEMORY_ADDR_REL_SLEB:
    return cast<DefinedData>(Sym)->getVirtualAddress() + Reloc.Addend;
  case R_WASM_TYPE_INDEX_LEB:
    return TypeMap[Reloc.Index];
  case R_WASM_FUNCTION_INDEX_LEB:
    return getFunctionSymbol(Reloc.Index)->getFunctionIndex();
  case R_WASM_GLOBAL_INDEX_LEB:
    if (auto GS = dyn_cast<GlobalSymbol>(Sym))
      return GS->getGlobalIndex();
    return Sym->getGOTIndex();
  case R_WASM_EVENT_INDEX_LEB:
    return getEventSymbol(Reloc.Index)->getEventIndex();
  case R_WASM_FUNCTION_OFFSET_I32: {
    auto *F = cast<DefinedFunction>(Sym);
    return F->Function->OutputOffset + F->Function->getFunctionCodeOffset() +
           Reloc.Addend;
  }
  case R_WASM_SECTION_OFFSET_I32:
    return getSectionSymbol(Reloc.Index)->Section->OutputOffset + Reloc.Addend;
  default:
    llvm_unreachable("unknown relocation type");
  }
}

template <class T>
static void setRelocs(const std::vector<T *> &Chunks,
                      const WasmSection *Section) {
  if (!Section)
    return;

  ArrayRef<WasmRelocation> Relocs = Section->Relocations;
  assert(std::is_sorted(Relocs.begin(), Relocs.end(),
                        [](const WasmRelocation &R1, const WasmRelocation &R2) {
                          return R1.Offset < R2.Offset;
                        }));
  assert(std::is_sorted(
      Chunks.begin(), Chunks.end(), [](InputChunk *C1, InputChunk *C2) {
        return C1->getInputSectionOffset() < C2->getInputSectionOffset();
      }));

  auto RelocsNext = Relocs.begin();
  auto RelocsEnd = Relocs.end();
  auto RelocLess = [](const WasmRelocation &R, uint32_t Val) {
    return R.Offset < Val;
  };
  for (InputChunk *C : Chunks) {
    auto RelocsStart = std::lower_bound(RelocsNext, RelocsEnd,
                                        C->getInputSectionOffset(), RelocLess);
    RelocsNext = std::lower_bound(
        RelocsStart, RelocsEnd, C->getInputSectionOffset() + C->getInputSize(),
        RelocLess);
    C->setRelocations(ArrayRef<WasmRelocation>(RelocsStart, RelocsNext));
  }
}

void ObjFile::parse(bool IgnoreComdats) {
  // Parse a memory buffer as a wasm file.
  LLVM_DEBUG(dbgs() << "Parsing object: " << toString(this) << "\n");
  std::unique_ptr<Binary> Bin = CHECK(createBinary(MB), toString(this));

  auto *Obj = dyn_cast<WasmObjectFile>(Bin.get());
  if (!Obj)
    fatal(toString(this) + ": not a wasm file");
  if (!Obj->isRelocatableObject())
    fatal(toString(this) + ": not a relocatable wasm file");

  Bin.release();
  WasmObj.reset(Obj);

  // Build up a map of function indices to table indices for use when
  // verifying the existing table index relocations
  uint32_t TotalFunctions =
      WasmObj->getNumImportedFunctions() + WasmObj->functions().size();
  TableEntries.resize(TotalFunctions);
  for (const WasmElemSegment &Seg : WasmObj->elements()) {
    if (Seg.Offset.Opcode != WASM_OPCODE_I32_CONST)
      fatal(toString(this) + ": invalid table elements");
    uint32_t Offset = Seg.Offset.Value.Int32;
    for (uint32_t Index = 0; Index < Seg.Functions.size(); Index++) {

      uint32_t FunctionIndex = Seg.Functions[Index];
      TableEntries[FunctionIndex] = Offset + Index;
    }
  }

  uint32_t SectionIndex = 0;

  // Bool for each symbol, true if called directly.  This allows us to implement
  // a weaker form of signature checking where undefined functions that are not
  // called directly (i.e. only address taken) don't have to match the defined
  // function's signature.  We cannot do this for directly called functions
  // because those signatures are checked at validation times.
  // See https://bugs.llvm.org/show_bug.cgi?id=40412
  std::vector<bool> IsCalledDirectly(WasmObj->getNumberOfSymbols(), false);
  for (const SectionRef &Sec : WasmObj->sections()) {
    const WasmSection &Section = WasmObj->getWasmSection(Sec);
    // Wasm objects can have at most one code and one data section.
    if (Section.Type == WASM_SEC_CODE) {
      assert(!CodeSection);
      CodeSection = &Section;
    } else if (Section.Type == WASM_SEC_DATA) {
      assert(!DataSection);
      DataSection = &Section;
    } else if (Section.Type == WASM_SEC_CUSTOM) {
      CustomSections.emplace_back(make<InputSection>(Section, this));
      CustomSections.back()->setRelocations(Section.Relocations);
      CustomSectionsByIndex[SectionIndex] = CustomSections.back();
    }
    SectionIndex++;
    // Scans relocations to dermine determine if a function symbol is called
    // directly
    for (const WasmRelocation &Reloc : Section.Relocations)
      if (Reloc.Type == R_WASM_FUNCTION_INDEX_LEB)
        IsCalledDirectly[Reloc.Index] = true;
  }

  TypeMap.resize(getWasmObj()->types().size());
  TypeIsUsed.resize(getWasmObj()->types().size(), false);

  ArrayRef<StringRef> Comdats = WasmObj->linkingData().Comdats;
  for (StringRef Comdat : Comdats) {
    bool IsNew = IgnoreComdats || Symtab->addComdat(Comdat);
    KeptComdats.push_back(IsNew);
  }

  // Populate `Segments`.
  for (const WasmSegment &S : WasmObj->dataSegments()) {
    auto* Seg = make<InputSegment>(S, this);
    Seg->Discarded = isExcludedByComdat(Seg);
    Segments.emplace_back(Seg);
  }
  setRelocs(Segments, DataSection);

  // Populate `Functions`.
  ArrayRef<WasmFunction> Funcs = WasmObj->functions();
  ArrayRef<uint32_t> FuncTypes = WasmObj->functionTypes();
  ArrayRef<WasmSignature> Types = WasmObj->types();
  Functions.reserve(Funcs.size());

  for (size_t I = 0, E = Funcs.size(); I != E; ++I) {
    auto* Func = make<InputFunction>(Types[FuncTypes[I]], &Funcs[I], this);
    Func->Discarded = isExcludedByComdat(Func);
    Functions.emplace_back(Func);
  }
  setRelocs(Functions, CodeSection);

  // Populate `Globals`.
  for (const WasmGlobal &G : WasmObj->globals())
    Globals.emplace_back(make<InputGlobal>(G, this));

  // Populate `Events`.
  for (const WasmEvent &E : WasmObj->events())
    Events.emplace_back(make<InputEvent>(Types[E.Type.SigIndex], E, this));

  // Populate `Symbols` based on the WasmSymbols in the object.
  Symbols.reserve(WasmObj->getNumberOfSymbols());
  for (const SymbolRef &Sym : WasmObj->symbols()) {
    const WasmSymbol &WasmSym = WasmObj->getWasmSymbol(Sym.getRawDataRefImpl());
    if (WasmSym.isDefined()) {
      // createDefined may fail if the symbol is comdat excluded in which case
      // we fall back to creating an undefined symbol
      if (Symbol *D = createDefined(WasmSym)) {
        Symbols.push_back(D);
        continue;
      }
    }
    size_t Idx = Symbols.size();
    Symbols.push_back(createUndefined(WasmSym, IsCalledDirectly[Idx]));
  }
}

bool ObjFile::isExcludedByComdat(InputChunk *Chunk) const {
  uint32_t C = Chunk->getComdat();
  if (C == UINT32_MAX)
    return false;
  return !KeptComdats[C];
}

FunctionSymbol *ObjFile::getFunctionSymbol(uint32_t Index) const {
  return cast<FunctionSymbol>(Symbols[Index]);
}

GlobalSymbol *ObjFile::getGlobalSymbol(uint32_t Index) const {
  return cast<GlobalSymbol>(Symbols[Index]);
}

EventSymbol *ObjFile::getEventSymbol(uint32_t Index) const {
  return cast<EventSymbol>(Symbols[Index]);
}

SectionSymbol *ObjFile::getSectionSymbol(uint32_t Index) const {
  return cast<SectionSymbol>(Symbols[Index]);
}

DataSymbol *ObjFile::getDataSymbol(uint32_t Index) const {
  return cast<DataSymbol>(Symbols[Index]);
}

Symbol *ObjFile::createDefined(const WasmSymbol &Sym) {
  StringRef Name = Sym.Info.Name;
  uint32_t Flags = Sym.Info.Flags;

  switch (Sym.Info.Kind) {
  case WASM_SYMBOL_TYPE_FUNCTION: {
    InputFunction *Func =
        Functions[Sym.Info.ElementIndex - WasmObj->getNumImportedFunctions()];
    if (Func->Discarded)
      return nullptr;
    if (Sym.isBindingLocal())
      return make<DefinedFunction>(Name, Flags, this, Func);
    return Symtab->addDefinedFunction(Name, Flags, this, Func);
  }
  case WASM_SYMBOL_TYPE_DATA: {
    InputSegment *Seg = Segments[Sym.Info.DataRef.Segment];
    if (Seg->Discarded)
      return nullptr;

    uint32_t Offset = Sym.Info.DataRef.Offset;
    uint32_t Size = Sym.Info.DataRef.Size;

    if (Sym.isBindingLocal())
      return make<DefinedData>(Name, Flags, this, Seg, Offset, Size);
    return Symtab->addDefinedData(Name, Flags, this, Seg, Offset, Size);
  }
  case WASM_SYMBOL_TYPE_GLOBAL: {
    InputGlobal *Global =
        Globals[Sym.Info.ElementIndex - WasmObj->getNumImportedGlobals()];
    if (Sym.isBindingLocal())
      return make<DefinedGlobal>(Name, Flags, this, Global);
    return Symtab->addDefinedGlobal(Name, Flags, this, Global);
  }
  case WASM_SYMBOL_TYPE_SECTION: {
    InputSection *Section = CustomSectionsByIndex[Sym.Info.ElementIndex];
    assert(Sym.isBindingLocal());
    return make<SectionSymbol>(Flags, Section, this);
  }
  case WASM_SYMBOL_TYPE_EVENT: {
    InputEvent *Event =
        Events[Sym.Info.ElementIndex - WasmObj->getNumImportedEvents()];
    if (Sym.isBindingLocal())
      return make<DefinedEvent>(Name, Flags, this, Event);
    return Symtab->addDefinedEvent(Name, Flags, this, Event);
  }
  }
  llvm_unreachable("unknown symbol kind");
}

Symbol *ObjFile::createUndefined(const WasmSymbol &Sym, bool IsCalledDirectly) {
  StringRef Name = Sym.Info.Name;
  uint32_t Flags = Sym.Info.Flags;

  switch (Sym.Info.Kind) {
  case WASM_SYMBOL_TYPE_FUNCTION:
    if (Sym.isBindingLocal())
      return make<UndefinedFunction>(Name, Sym.Info.ImportName,
                                     Sym.Info.ImportModule, Flags, this,
                                     Sym.Signature, IsCalledDirectly);
    return Symtab->addUndefinedFunction(Name, Sym.Info.ImportName,
                                        Sym.Info.ImportModule, Flags, this,
                                        Sym.Signature, IsCalledDirectly);
  case WASM_SYMBOL_TYPE_DATA:
    if (Sym.isBindingLocal())
      return make<UndefinedData>(Name, Flags, this);
    return Symtab->addUndefinedData(Name, Flags, this);
  case WASM_SYMBOL_TYPE_GLOBAL:
    if (Sym.isBindingLocal())
      return make<UndefinedGlobal>(Name, Sym.Info.ImportName,
                                   Sym.Info.ImportModule, Flags, this,
                                   Sym.GlobalType);
    return Symtab->addUndefinedGlobal(Name, Sym.Info.ImportName,
                                      Sym.Info.ImportModule, Flags, this,
                                      Sym.GlobalType);
  case WASM_SYMBOL_TYPE_SECTION:
    llvm_unreachable("section symbols cannot be undefined");
  }
  llvm_unreachable("unknown symbol kind");
}

void ArchiveFile::parse() {
  // Parse a MemoryBufferRef as an archive file.
  LLVM_DEBUG(dbgs() << "Parsing library: " << toString(this) << "\n");
  File = CHECK(Archive::create(MB), toString(this));

  // Read the symbol table to construct Lazy symbols.
  int Count = 0;
  for (const Archive::Symbol &Sym : File->symbols()) {
    Symtab->addLazy(this, &Sym);
    ++Count;
  }
  LLVM_DEBUG(dbgs() << "Read " << Count << " symbols\n");
}

void ArchiveFile::addMember(const Archive::Symbol *Sym) {
  const Archive::Child &C =
      CHECK(Sym->getMember(),
            "could not get the member for symbol " + Sym->getName());

  // Don't try to load the same member twice (this can happen when members
  // mutually reference each other).
  if (!Seen.insert(C.getChildOffset()).second)
    return;

  LLVM_DEBUG(dbgs() << "loading lazy: " << Sym->getName() << "\n");
  LLVM_DEBUG(dbgs() << "from archive: " << toString(this) << "\n");

  MemoryBufferRef MB =
      CHECK(C.getMemoryBufferRef(),
            "could not get the buffer for the member defining symbol " +
                Sym->getName());

  InputFile *Obj = createObjectFile(MB, getName());
  Symtab->addFile(Obj);
}

static uint8_t mapVisibility(GlobalValue::VisibilityTypes GvVisibility) {
  switch (GvVisibility) {
  case GlobalValue::DefaultVisibility:
    return WASM_SYMBOL_VISIBILITY_DEFAULT;
  case GlobalValue::HiddenVisibility:
  case GlobalValue::ProtectedVisibility:
    return WASM_SYMBOL_VISIBILITY_HIDDEN;
  }
  llvm_unreachable("unknown visibility");
}

static Symbol *createBitcodeSymbol(const std::vector<bool> &KeptComdats,
                                   const lto::InputFile::Symbol &ObjSym,
                                   BitcodeFile &F) {
  StringRef Name = Saver.save(ObjSym.getName());

  uint32_t Flags = ObjSym.isWeak() ? WASM_SYMBOL_BINDING_WEAK : 0;
  Flags |= mapVisibility(ObjSym.getVisibility());

  int C = ObjSym.getComdatIndex();
  bool ExcludedByComdat = C != -1 && !KeptComdats[C];

  if (ObjSym.isUndefined() || ExcludedByComdat) {
    if (ObjSym.isExecutable())
      return Symtab->addUndefinedFunction(Name, Name, DefaultModule, Flags, &F,
                                          nullptr, true);
    return Symtab->addUndefinedData(Name, Flags, &F);
  }

  if (ObjSym.isExecutable())
    return Symtab->addDefinedFunction(Name, Flags, &F, nullptr);
  return Symtab->addDefinedData(Name, Flags, &F, nullptr, 0, 0);
}

void BitcodeFile::parse() {
  Obj = check(lto::InputFile::create(MemoryBufferRef(
      MB.getBuffer(), Saver.save(ArchiveName + MB.getBufferIdentifier()))));
  Triple T(Obj->getTargetTriple());
  if (T.getArch() != Triple::wasm32) {
    error(toString(MB.getBufferIdentifier()) + ": machine type must be wasm32");
    return;
  }
  std::vector<bool> KeptComdats;
  for (StringRef S : Obj->getComdatTable())
    KeptComdats.push_back(Symtab->addComdat(S));

  for (const lto::InputFile::Symbol &ObjSym : Obj->symbols())
    Symbols.push_back(createBitcodeSymbol(KeptComdats, ObjSym, *this));
}

// Returns a string in the format of "foo.o" or "foo.a(bar.o)".
std::string lld::toString(const wasm::InputFile *File) {
  if (!File)
    return "<internal>";

  if (File->ArchiveName.empty())
    return File->getName();

  return (File->ArchiveName + "(" + File->getName() + ")").str();
}
