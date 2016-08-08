//===- InputFiles.cpp -----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
#include "Config.h"
#include "Driver.h"
#include "Error.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/LTO/legacy/LTOModule.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm-c/lto.h"
#include <cstring>
#include <system_error>
#include <utility>

using namespace llvm::COFF;
using namespace llvm::object;
using namespace llvm::support::endian;

using llvm::Triple;
using llvm::support::ulittle32_t;

namespace lld {
namespace coff {

int InputFile::NextIndex = 0;
llvm::LLVMContext BitcodeFile::Context;

// Returns the last element of a path, which is supposed to be a filename.
static StringRef getBasename(StringRef Path) {
  size_t Pos = Path.find_last_of("\\/");
  if (Pos == StringRef::npos)
    return Path;
  return Path.substr(Pos + 1);
}

// Returns a string in the format of "foo.obj" or "foo.obj(bar.lib)".
std::string InputFile::getShortName() {
  if (ParentName == "")
    return getName().lower();
  std::string Res = (getBasename(ParentName) + "(" +
                     getBasename(getName()) + ")").str();
  return StringRef(Res).lower();
}

void ArchiveFile::parse() {
  // Parse a MemoryBufferRef as an archive file.
  File = check(Archive::create(MB), getShortName());

  // Allocate a buffer for Lazy objects.
  size_t NumSyms = File->getNumberOfSymbols();
  LazySymbols.reserve(NumSyms);

  // Read the symbol table to construct Lazy objects.
  for (const Archive::Symbol &Sym : File->symbols())
    LazySymbols.emplace_back(this, Sym);

  // Seen is a map from member files to boolean values. Initially
  // all members are mapped to false, which indicates all these files
  // are not read yet.
  Error Err;
  for (auto &Child : File->children(Err))
    Seen[Child.getChildOffset()].clear();
  if (Err)
    fatal(Err, getShortName());
}

// Returns a buffer pointing to a member file containing a given symbol.
// This function is thread-safe.
MemoryBufferRef ArchiveFile::getMember(const Archive::Symbol *Sym) {
  const Archive::Child &C =
      check(Sym->getMember(),
            "could not get the member for symbol " + Sym->getName());

  // Return an empty buffer if we have already returned the same buffer.
  if (Seen[C.getChildOffset()].test_and_set())
    return MemoryBufferRef();

  MemoryBufferRef MB =
      check(C.getMemoryBufferRef(),
            "could not get the buffer for the member defining symbol " +
                Sym->getName());
  if (C.getParent()->isThin() && Driver->Cpio)
    Driver->Cpio->append(relativeToRoot(check(C.getFullName())),
                         MB.getBuffer());

  return MB;
}

void ObjectFile::parse() {
  // Parse a memory buffer as a COFF file.
  std::unique_ptr<Binary> Bin =
      check(createBinary(MB), getShortName());

  if (auto *Obj = dyn_cast<COFFObjectFile>(Bin.get())) {
    Bin.release();
    COFFObj.reset(Obj);
  } else {
    fatal(getShortName() + " is not a COFF file");
  }

  // Read section and symbol tables.
  initializeChunks();
  initializeSymbols();
  initializeSEH();
}

void ObjectFile::initializeChunks() {
  uint32_t NumSections = COFFObj->getNumberOfSections();
  Chunks.reserve(NumSections);
  SparseChunks.resize(NumSections + 1);
  for (uint32_t I = 1; I < NumSections + 1; ++I) {
    const coff_section *Sec;
    StringRef Name;
    if (auto EC = COFFObj->getSection(I, Sec))
      fatal(EC, "getSection failed: #" + Twine(I));
    if (auto EC = COFFObj->getSectionName(Sec, Name))
      fatal(EC, "getSectionName failed: #" + Twine(I));
    if (Name == ".sxdata") {
      SXData = Sec;
      continue;
    }
    if (Name == ".drectve") {
      ArrayRef<uint8_t> Data;
      COFFObj->getSectionContents(Sec, Data);
      Directives = std::string((const char *)Data.data(), Data.size());
      continue;
    }
    // Skip non-DWARF debug info. MSVC linker converts the sections into
    // a PDB file, but we don't support that.
    if (Name == ".debug" || Name.startswith(".debug$"))
      continue;
    // We want to preserve DWARF debug sections only when /debug is on.
    if (!Config->Debug && Name.startswith(".debug"))
      continue;
    if (Sec->Characteristics & llvm::COFF::IMAGE_SCN_LNK_REMOVE)
      continue;
    auto *C = new (Alloc) SectionChunk(this, Sec);
    Chunks.push_back(C);
    SparseChunks[I] = C;
  }
}

void ObjectFile::initializeSymbols() {
  uint32_t NumSymbols = COFFObj->getNumberOfSymbols();
  SymbolBodies.reserve(NumSymbols);
  SparseSymbolBodies.resize(NumSymbols);
  llvm::SmallVector<std::pair<Undefined *, uint32_t>, 8> WeakAliases;
  int32_t LastSectionNumber = 0;
  for (uint32_t I = 0; I < NumSymbols; ++I) {
    // Get a COFFSymbolRef object.
    COFFSymbolRef Sym =
        check(COFFObj->getSymbol(I), "broken object file: " + getShortName());

    const void *AuxP = nullptr;
    if (Sym.getNumberOfAuxSymbols())
      AuxP = COFFObj->getSymbol(I + 1)->getRawPtr();
    bool IsFirst = (LastSectionNumber != Sym.getSectionNumber());

    SymbolBody *Body = nullptr;
    if (Sym.isUndefined()) {
      Body = createUndefined(Sym);
    } else if (Sym.isWeakExternal()) {
      Body = createUndefined(Sym);
      uint32_t TagIndex =
          static_cast<const coff_aux_weak_external *>(AuxP)->TagIndex;
      WeakAliases.emplace_back((Undefined *)Body, TagIndex);
    } else {
      Body = createDefined(Sym, AuxP, IsFirst);
    }
    if (Body) {
      SymbolBodies.push_back(Body);
      SparseSymbolBodies[I] = Body;
    }
    I += Sym.getNumberOfAuxSymbols();
    LastSectionNumber = Sym.getSectionNumber();
  }
  for (auto WeakAlias : WeakAliases)
    WeakAlias.first->WeakAlias = SparseSymbolBodies[WeakAlias.second];
}

Undefined *ObjectFile::createUndefined(COFFSymbolRef Sym) {
  StringRef Name;
  COFFObj->getSymbolName(Sym, Name);
  return new (Alloc) Undefined(Name);
}

Defined *ObjectFile::createDefined(COFFSymbolRef Sym, const void *AuxP,
                                   bool IsFirst) {
  StringRef Name;
  if (Sym.isCommon()) {
    auto *C = new (Alloc) CommonChunk(Sym);
    Chunks.push_back(C);
    return new (Alloc) DefinedCommon(this, Sym, C);
  }
  if (Sym.isAbsolute()) {
    COFFObj->getSymbolName(Sym, Name);
    // Skip special symbols.
    if (Name == "@comp.id")
      return nullptr;
    // COFF spec 5.10.1. The .sxdata section.
    if (Name == "@feat.00") {
      if (Sym.getValue() & 1)
        SEHCompat = true;
      return nullptr;
    }
    return new (Alloc) DefinedAbsolute(Name, Sym);
  }
  int32_t SectionNumber = Sym.getSectionNumber();
  if (SectionNumber == llvm::COFF::IMAGE_SYM_DEBUG)
    return nullptr;

  // Reserved sections numbers don't have contents.
  if (llvm::COFF::isReservedSectionNumber(SectionNumber))
    fatal("broken object file: " + getShortName());

  // This symbol references a section which is not present in the section
  // header.
  if ((uint32_t)SectionNumber >= SparseChunks.size())
    fatal("broken object file: " + getShortName());

  // Nothing else to do without a section chunk.
  auto *SC = cast_or_null<SectionChunk>(SparseChunks[SectionNumber]);
  if (!SC)
    return nullptr;

  // Handle section definitions
  if (IsFirst && AuxP) {
    auto *Aux = reinterpret_cast<const coff_aux_section_definition *>(AuxP);
    if (Aux->Selection == IMAGE_COMDAT_SELECT_ASSOCIATIVE)
      if (auto *ParentSC = cast_or_null<SectionChunk>(
              SparseChunks[Aux->getNumber(Sym.isBigObj())]))
        ParentSC->addAssociative(SC);
    SC->Checksum = Aux->CheckSum;
  }

  auto *B = new (Alloc) DefinedRegular(this, Sym, SC);
  if (SC->isCOMDAT() && Sym.getValue() == 0 && !AuxP)
    SC->setSymbol(B);

  return B;
}

void ObjectFile::initializeSEH() {
  if (!SEHCompat || !SXData)
    return;
  ArrayRef<uint8_t> A;
  COFFObj->getSectionContents(SXData, A);
  if (A.size() % 4 != 0)
    fatal(".sxdata must be an array of symbol table indices");
  auto *I = reinterpret_cast<const ulittle32_t *>(A.data());
  auto *E = reinterpret_cast<const ulittle32_t *>(A.data() + A.size());
  for (; I != E; ++I)
    SEHandlers.insert(SparseSymbolBodies[*I]);
}

MachineTypes ObjectFile::getMachineType() {
  if (COFFObj)
    return static_cast<MachineTypes>(COFFObj->getMachine());
  return IMAGE_FILE_MACHINE_UNKNOWN;
}

StringRef ltrim1(StringRef S, const char *Chars) {
  if (!S.empty() && strchr(Chars, S[0]))
    return S.substr(1);
  return S;
}

void ImportFile::parse() {
  const char *Buf = MB.getBufferStart();
  const char *End = MB.getBufferEnd();
  const auto *Hdr = reinterpret_cast<const coff_import_header *>(Buf);

  // Check if the total size is valid.
  if ((size_t)(End - Buf) != (sizeof(*Hdr) + Hdr->SizeOfData))
    fatal("broken import library");

  // Read names and create an __imp_ symbol.
  StringRef Name = StringAlloc.save(StringRef(Buf + sizeof(*Hdr)));
  StringRef ImpName = StringAlloc.save("__imp_" + Name);
  const char *NameStart = Buf + sizeof(coff_import_header) + Name.size() + 1;
  DLLName = StringRef(NameStart);
  StringRef ExtName;
  switch (Hdr->getNameType()) {
  case IMPORT_ORDINAL:
    ExtName = "";
    break;
  case IMPORT_NAME:
    ExtName = Name;
    break;
  case IMPORT_NAME_NOPREFIX:
    ExtName = ltrim1(Name, "?@_");
    break;
  case IMPORT_NAME_UNDECORATE:
    ExtName = ltrim1(Name, "?@_");
    ExtName = ExtName.substr(0, ExtName.find('@'));
    break;
  }
  ImpSym = new (Alloc) DefinedImportData(DLLName, ImpName, ExtName, Hdr);
  SymbolBodies.push_back(ImpSym);

  // If type is function, we need to create a thunk which jump to an
  // address pointed by the __imp_ symbol. (This allows you to call
  // DLL functions just like regular non-DLL functions.)
  if (Hdr->getType() != llvm::COFF::IMPORT_CODE)
    return;
  ThunkSym = new (Alloc) DefinedImportThunk(Name, ImpSym, Hdr->Machine);
  SymbolBodies.push_back(ThunkSym);
}

void BitcodeFile::parse() {
  // Usually parse() is thread-safe, but bitcode file is an exception.
  std::lock_guard<std::mutex> Lock(Mu);

  Context.enableDebugTypeODRUniquing();
  ErrorOr<std::unique_ptr<LTOModule>> ModOrErr = LTOModule::createFromBuffer(
      Context, MB.getBufferStart(), MB.getBufferSize(), llvm::TargetOptions());
  M = check(std::move(ModOrErr), "could not create LTO module");

  llvm::StringSaver Saver(Alloc);
  for (unsigned I = 0, E = M->getSymbolCount(); I != E; ++I) {
    lto_symbol_attributes Attrs = M->getSymbolAttributes(I);
    if ((Attrs & LTO_SYMBOL_SCOPE_MASK) == LTO_SYMBOL_SCOPE_INTERNAL)
      continue;

    StringRef SymName = Saver.save(M->getSymbolName(I));
    int SymbolDef = Attrs & LTO_SYMBOL_DEFINITION_MASK;
    if (SymbolDef == LTO_SYMBOL_DEFINITION_UNDEFINED) {
      SymbolBodies.push_back(new (Alloc) Undefined(SymName));
    } else {
      bool Replaceable =
          (SymbolDef == LTO_SYMBOL_DEFINITION_TENTATIVE || // common
           (Attrs & LTO_SYMBOL_COMDAT) ||                  // comdat
           (SymbolDef == LTO_SYMBOL_DEFINITION_WEAK &&     // weak external
            (Attrs & LTO_SYMBOL_ALIAS)));
      SymbolBodies.push_back(new (Alloc) DefinedBitcode(this, SymName,
                                                        Replaceable));
    }
  }

  Directives = M->getLinkerOpts();
}

MachineTypes BitcodeFile::getMachineType() {
  if (!M)
    return IMAGE_FILE_MACHINE_UNKNOWN;
  switch (Triple(M->getTargetTriple()).getArch()) {
  case Triple::x86_64:
    return AMD64;
  case Triple::x86:
    return I386;
  case Triple::arm:
    return ARMNT;
  default:
    return IMAGE_FILE_MACHINE_UNKNOWN;
  }
}

std::mutex BitcodeFile::Mu;

} // namespace coff
} // namespace lld
