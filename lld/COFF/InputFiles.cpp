//===- InputFiles.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Chunks.h"
#include "Config.h"
#include "Driver.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm-c/lto.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetOptions.h"
#include <cstring>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace llvm::COFF;
using namespace llvm::object;
using namespace llvm::support::endian;

using llvm::Triple;
using llvm::support::ulittle32_t;

namespace lld {
namespace coff {

std::vector<ObjFile *> ObjFile::Instances;
std::vector<ImportFile *> ImportFile::Instances;
std::vector<BitcodeFile *> BitcodeFile::Instances;

/// Checks that Source is compatible with being a weak alias to Target.
/// If Source is Undefined and has no weak alias set, makes it a weak
/// alias to Target.
static void checkAndSetWeakAlias(SymbolTable *Symtab, InputFile *F,
                                 Symbol *Source, Symbol *Target) {
  if (auto *U = dyn_cast<Undefined>(Source)) {
    if (U->WeakAlias && U->WeakAlias != Target) {
      // Weak aliases as produced by GCC are named in the form
      // .weak.<weaksymbol>.<othersymbol>, where <othersymbol> is the name
      // of another symbol emitted near the weak symbol.
      // Just use the definition from the first object file that defined
      // this weak symbol.
      if (Config->MinGW)
        return;
      Symtab->reportDuplicate(Source, F);
    }
    U->WeakAlias = Target;
  }
}

ArchiveFile::ArchiveFile(MemoryBufferRef M) : InputFile(ArchiveKind, M) {}

void ArchiveFile::parse() {
  // Parse a MemoryBufferRef as an archive file.
  File = CHECK(Archive::create(MB), this);

  // Read the symbol table to construct Lazy objects.
  for (const Archive::Symbol &Sym : File->symbols())
    Symtab->addLazy(this, Sym);
}

// Returns a buffer pointing to a member file containing a given symbol.
void ArchiveFile::addMember(const Archive::Symbol *Sym) {
  const Archive::Child &C =
      CHECK(Sym->getMember(),
            "could not get the member for symbol " + Sym->getName());

  // Return an empty buffer if we have already returned the same buffer.
  if (!Seen.insert(C.getChildOffset()).second)
    return;

  Driver->enqueueArchiveMember(C, Sym->getName(), getName());
}

std::vector<MemoryBufferRef> getArchiveMembers(Archive *File) {
  std::vector<MemoryBufferRef> V;
  Error Err = Error::success();
  for (const ErrorOr<Archive::Child> &COrErr : File->children(Err)) {
    Archive::Child C =
        CHECK(COrErr,
              File->getFileName() + ": could not get the child of the archive");
    MemoryBufferRef MBRef =
        CHECK(C.getMemoryBufferRef(),
              File->getFileName() +
                  ": could not get the buffer for a child of the archive");
    V.push_back(MBRef);
  }
  if (Err)
    fatal(File->getFileName() +
          ": Archive::children failed: " + toString(std::move(Err)));
  return V;
}

void ObjFile::parse() {
  // Parse a memory buffer as a COFF file.
  std::unique_ptr<Binary> Bin = CHECK(createBinary(MB), this);

  if (auto *Obj = dyn_cast<COFFObjectFile>(Bin.get())) {
    Bin.release();
    COFFObj.reset(Obj);
  } else {
    fatal(toString(this) + " is not a COFF file");
  }

  // Read section and symbol tables.
  initializeChunks();
  initializeSymbols();
}

const coff_section* ObjFile::getSection(uint32_t I) {
  const coff_section *Sec;
  if (auto EC = COFFObj->getSection(I, Sec))
    fatal("getSection failed: #" + Twine(I) + ": " + EC.message());
  return Sec;
}

// We set SectionChunk pointers in the SparseChunks vector to this value
// temporarily to mark comdat sections as having an unknown resolution. As we
// walk the object file's symbol table, once we visit either a leader symbol or
// an associative section definition together with the parent comdat's leader,
// we set the pointer to either nullptr (to mark the section as discarded) or a
// valid SectionChunk for that section.
static SectionChunk *const PendingComdat = reinterpret_cast<SectionChunk *>(1);

void ObjFile::initializeChunks() {
  uint32_t NumSections = COFFObj->getNumberOfSections();
  Chunks.reserve(NumSections);
  SparseChunks.resize(NumSections + 1);
  for (uint32_t I = 1; I < NumSections + 1; ++I) {
    const coff_section *Sec = getSection(I);
    if (Sec->Characteristics & IMAGE_SCN_LNK_COMDAT)
      SparseChunks[I] = PendingComdat;
    else
      SparseChunks[I] = readSection(I, nullptr, "");
  }
}

SectionChunk *ObjFile::readSection(uint32_t SectionNumber,
                                   const coff_aux_section_definition *Def,
                                   StringRef LeaderName) {
  const coff_section *Sec = getSection(SectionNumber);

  StringRef Name;
  if (auto EC = COFFObj->getSectionName(Sec, Name))
    fatal("getSectionName failed: #" + Twine(SectionNumber) + ": " +
          EC.message());

  if (Name == ".drectve") {
    ArrayRef<uint8_t> Data;
    COFFObj->getSectionContents(Sec, Data);
    Directives = std::string((const char *)Data.data(), Data.size());
    return nullptr;
  }

  if (Name == ".llvm_addrsig") {
    AddrsigSec = Sec;
    return nullptr;
  }

  // Object files may have DWARF debug info or MS CodeView debug info
  // (or both).
  //
  // DWARF sections don't need any special handling from the perspective
  // of the linker; they are just a data section containing relocations.
  // We can just link them to complete debug info.
  //
  // CodeView needs linker support. We need to interpret debug info,
  // and then write it to a separate .pdb file.

  // Ignore DWARF debug info unless /debug is given.
  if (!Config->Debug && Name.startswith(".debug_"))
    return nullptr;

  if (Sec->Characteristics & llvm::COFF::IMAGE_SCN_LNK_REMOVE)
    return nullptr;
  auto *C = make<SectionChunk>(this, Sec);
  if (Def)
    C->Checksum = Def->CheckSum;

  // CodeView sections are stored to a different vector because they are not
  // linked in the regular manner.
  if (C->isCodeView())
    DebugChunks.push_back(C);
  else if (Config->GuardCF != GuardCFLevel::Off && Name == ".gfids$y")
    GuardFidChunks.push_back(C);
  else if (Config->GuardCF != GuardCFLevel::Off && Name == ".gljmp$y")
    GuardLJmpChunks.push_back(C);
  else if (Name == ".sxdata")
    SXDataChunks.push_back(C);
  else if (Config->TailMerge && Sec->NumberOfRelocations == 0 &&
           Name == ".rdata" && LeaderName.startswith("??_C@"))
    // COFF sections that look like string literal sections (i.e. no
    // relocations, in .rdata, leader symbol name matches the MSVC name mangling
    // for string literals) are subject to string tail merging.
    MergeChunk::addSection(C);
  else
    Chunks.push_back(C);

  return C;
}

void ObjFile::readAssociativeDefinition(
    COFFSymbolRef Sym, const coff_aux_section_definition *Def) {
  readAssociativeDefinition(Sym, Def, Def->getNumber(Sym.isBigObj()));
}

void ObjFile::readAssociativeDefinition(COFFSymbolRef Sym,
                                        const coff_aux_section_definition *Def,
                                        uint32_t ParentIndex) {
  SectionChunk *Parent = SparseChunks[ParentIndex];
  int32_t SectionNumber = Sym.getSectionNumber();

  auto Diag = [&]() {
    StringRef Name, ParentName;
    COFFObj->getSymbolName(Sym, Name);

    const coff_section *ParentSec = getSection(ParentIndex);
    COFFObj->getSectionName(ParentSec, ParentName);
    error(toString(this) + ": associative comdat " + Name + " (sec " +
          Twine(SectionNumber) + ") has invalid reference to section " +
          ParentName + " (sec " + Twine(ParentIndex) + ")");
  };

  if (Parent == PendingComdat) {
    // This can happen if an associative comdat refers to another associative
    // comdat that appears after it (invalid per COFF spec) or to a section
    // without any symbols.
    Diag();
    return;
  }

  // Check whether the parent is prevailing. If it is, so are we, and we read
  // the section; otherwise mark it as discarded.
  if (Parent) {
    SectionChunk *C = readSection(SectionNumber, Def, "");
    SparseChunks[SectionNumber] = C;
    if (C) {
      C->Selection = IMAGE_COMDAT_SELECT_ASSOCIATIVE;
      Parent->addAssociative(C);
    }
  } else {
    SparseChunks[SectionNumber] = nullptr;
  }
}

void ObjFile::recordPrevailingSymbolForMingw(
    COFFSymbolRef Sym, DenseMap<StringRef, uint32_t> &PrevailingSectionMap) {
  // For comdat symbols in executable sections, where this is the copy
  // of the section chunk we actually include instead of discarding it,
  // add the symbol to a map to allow using it for implicitly
  // associating .[px]data$<func> sections to it.
  int32_t SectionNumber = Sym.getSectionNumber();
  SectionChunk *SC = SparseChunks[SectionNumber];
  if (SC && SC->getOutputCharacteristics() & IMAGE_SCN_MEM_EXECUTE) {
    StringRef Name;
    COFFObj->getSymbolName(Sym, Name);
    PrevailingSectionMap[Name] = SectionNumber;
  }
}

void ObjFile::maybeAssociateSEHForMingw(
    COFFSymbolRef Sym, const coff_aux_section_definition *Def,
    const DenseMap<StringRef, uint32_t> &PrevailingSectionMap) {
  StringRef Name;
  COFFObj->getSymbolName(Sym, Name);
  if (Name.consume_front(".pdata$") || Name.consume_front(".xdata$")) {
    // For MinGW, treat .[px]data$<func> as implicitly associative to
    // the symbol <func>.
    auto ParentSym = PrevailingSectionMap.find(Name);
    if (ParentSym != PrevailingSectionMap.end())
      readAssociativeDefinition(Sym, Def, ParentSym->second);
  }
}

Symbol *ObjFile::createRegular(COFFSymbolRef Sym) {
  SectionChunk *SC = SparseChunks[Sym.getSectionNumber()];
  if (Sym.isExternal()) {
    StringRef Name;
    COFFObj->getSymbolName(Sym, Name);
    if (SC)
      return Symtab->addRegular(this, Name, Sym.getGeneric(), SC);
    // For MinGW symbols named .weak.* that point to a discarded section,
    // don't create an Undefined symbol. If nothing ever refers to the symbol,
    // everything should be fine. If something actually refers to the symbol
    // (e.g. the undefined weak alias), linking will fail due to undefined
    // references at the end.
    if (Config->MinGW && Name.startswith(".weak."))
      return nullptr;
    return Symtab->addUndefined(Name, this, false);
  }
  if (SC)
    return make<DefinedRegular>(this, /*Name*/ "", /*IsCOMDAT*/ false,
                                /*IsExternal*/ false, Sym.getGeneric(), SC);
  return nullptr;
}

void ObjFile::initializeSymbols() {
  uint32_t NumSymbols = COFFObj->getNumberOfSymbols();
  Symbols.resize(NumSymbols);

  SmallVector<std::pair<Symbol *, uint32_t>, 8> WeakAliases;
  std::vector<uint32_t> PendingIndexes;
  PendingIndexes.reserve(NumSymbols);

  DenseMap<StringRef, uint32_t> PrevailingSectionMap;
  std::vector<const coff_aux_section_definition *> ComdatDefs(
      COFFObj->getNumberOfSections() + 1);

  for (uint32_t I = 0; I < NumSymbols; ++I) {
    COFFSymbolRef COFFSym = check(COFFObj->getSymbol(I));
    bool PrevailingComdat;
    if (COFFSym.isUndefined()) {
      Symbols[I] = createUndefined(COFFSym);
    } else if (COFFSym.isWeakExternal()) {
      Symbols[I] = createUndefined(COFFSym);
      uint32_t TagIndex = COFFSym.getAux<coff_aux_weak_external>()->TagIndex;
      WeakAliases.emplace_back(Symbols[I], TagIndex);
    } else if (Optional<Symbol *> OptSym =
                   createDefined(COFFSym, ComdatDefs, PrevailingComdat)) {
      Symbols[I] = *OptSym;
      if (Config->MinGW && PrevailingComdat)
        recordPrevailingSymbolForMingw(COFFSym, PrevailingSectionMap);
    } else {
      // createDefined() returns None if a symbol belongs to a section that
      // was pending at the point when the symbol was read. This can happen in
      // two cases:
      // 1) section definition symbol for a comdat leader;
      // 2) symbol belongs to a comdat section associated with another section.
      // In both of these cases, we can expect the section to be resolved by
      // the time we finish visiting the remaining symbols in the symbol
      // table. So we postpone the handling of this symbol until that time.
      PendingIndexes.push_back(I);
    }
    I += COFFSym.getNumberOfAuxSymbols();
  }

  for (uint32_t I : PendingIndexes) {
    COFFSymbolRef Sym = check(COFFObj->getSymbol(I));
    if (const coff_aux_section_definition *Def = Sym.getSectionDefinition()) {
      if (Def->Selection == IMAGE_COMDAT_SELECT_ASSOCIATIVE)
        readAssociativeDefinition(Sym, Def);
      else if (Config->MinGW)
        maybeAssociateSEHForMingw(Sym, Def, PrevailingSectionMap);
    }
    if (SparseChunks[Sym.getSectionNumber()] == PendingComdat) {
      StringRef Name;
      COFFObj->getSymbolName(Sym, Name);
      log("comdat section " + Name +
          " without leader and unassociated, discarding");
      continue;
    }
    Symbols[I] = createRegular(Sym);
  }

  for (auto &KV : WeakAliases) {
    Symbol *Sym = KV.first;
    uint32_t Idx = KV.second;
    checkAndSetWeakAlias(Symtab, this, Sym, Symbols[Idx]);
  }
}

Symbol *ObjFile::createUndefined(COFFSymbolRef Sym) {
  StringRef Name;
  COFFObj->getSymbolName(Sym, Name);
  return Symtab->addUndefined(Name, this, Sym.isWeakExternal());
}

void ObjFile::handleComdatSelection(COFFSymbolRef Sym, COMDATType &Selection,
                                    bool &Prevailing, DefinedRegular *Leader) {
  if (Prevailing)
    return;
  // There's already an existing comdat for this symbol: `Leader`.
  // Use the comdats's selection field to determine if the new
  // symbol in `Sym` should be discarded, produce a duplicate symbol
  // error, etc.

  SectionChunk *LeaderChunk = nullptr;
  COMDATType LeaderSelection = IMAGE_COMDAT_SELECT_ANY;

  if (Leader->Data) {
    LeaderChunk = Leader->getChunk();
    LeaderSelection = LeaderChunk->Selection;
  } else {
    // FIXME: comdats from LTO files don't know their selection; treat them
    // as "any".
    Selection = LeaderSelection;
  }

  if ((Selection == IMAGE_COMDAT_SELECT_ANY &&
       LeaderSelection == IMAGE_COMDAT_SELECT_LARGEST) ||
      (Selection == IMAGE_COMDAT_SELECT_LARGEST &&
       LeaderSelection == IMAGE_COMDAT_SELECT_ANY)) {
    // cl.exe picks "any" for vftables when building with /GR- and
    // "largest" when building with /GR. To be able to link object files
    // compiled with each flag, "any" and "largest" are merged as "largest".
    LeaderSelection = Selection = IMAGE_COMDAT_SELECT_LARGEST;
  }

  // Other than that, comdat selections must match.  This is a bit more
  // strict than link.exe which allows merging "any" and "largest" if "any"
  // is the first symbol the linker sees, and it allows merging "largest"
  // with everything (!) if "largest" is the first symbol the linker sees.
  // Making this symmetric independent of which selection is seen first
  // seems better though.
  // (This behavior matches ModuleLinker::getComdatResult().)
  if (Selection != LeaderSelection) {
    log(("conflicting comdat type for " + toString(*Leader) + ": " +
         Twine((int)LeaderSelection) + " in " + toString(Leader->getFile()) +
         " and " + Twine((int)Selection) + " in " + toString(this))
            .str());
    Symtab->reportDuplicate(Leader, this);
    return;
  }

  switch (Selection) {
  case IMAGE_COMDAT_SELECT_NODUPLICATES:
    Symtab->reportDuplicate(Leader, this);
    break;

  case IMAGE_COMDAT_SELECT_ANY:
    // Nothing to do.
    break;

  case IMAGE_COMDAT_SELECT_SAME_SIZE:
    if (LeaderChunk->getSize() != getSection(Sym)->SizeOfRawData)
      Symtab->reportDuplicate(Leader, this);
    break;

  case IMAGE_COMDAT_SELECT_EXACT_MATCH: {
    SectionChunk NewChunk(this, getSection(Sym));
    // link.exe only compares section contents here and doesn't complain
    // if the two comdat sections have e.g. different alignment.
    // Match that.
    if (LeaderChunk->getContents() != NewChunk.getContents())
      Symtab->reportDuplicate(Leader, this);
    break;
  }

  case IMAGE_COMDAT_SELECT_ASSOCIATIVE:
    // createDefined() is never called for IMAGE_COMDAT_SELECT_ASSOCIATIVE.
    // (This means lld-link doesn't produce duplicate symbol errors for
    // associative comdats while link.exe does, but associate comdats
    // are never extern in practice.)
    llvm_unreachable("createDefined not called for associative comdats");

  case IMAGE_COMDAT_SELECT_LARGEST:
    if (LeaderChunk->getSize() < getSection(Sym)->SizeOfRawData) {
      // Replace the existing comdat symbol with the new one.
      StringRef Name;
      COFFObj->getSymbolName(Sym, Name);
      // FIXME: This is incorrect: With /opt:noref, the previous sections
      // make it into the final executable as well. Correct handling would
      // be to undo reading of the whole old section that's being replaced,
      // or doing one pass that determines what the final largest comdat
      // is for all IMAGE_COMDAT_SELECT_LARGEST comdats and then reading
      // only the largest one.
      replaceSymbol<DefinedRegular>(Leader, this, Name, /*IsCOMDAT*/ true,
                                    /*IsExternal*/ true, Sym.getGeneric(),
                                    nullptr);
      Prevailing = true;
    }
    break;

  case IMAGE_COMDAT_SELECT_NEWEST:
    llvm_unreachable("should have been rejected earlier");
  }
}

Optional<Symbol *> ObjFile::createDefined(
    COFFSymbolRef Sym,
    std::vector<const coff_aux_section_definition *> &ComdatDefs,
    bool &Prevailing) {
  Prevailing = false;
  auto GetName = [&]() {
    StringRef S;
    COFFObj->getSymbolName(Sym, S);
    return S;
  };

  if (Sym.isCommon()) {
    auto *C = make<CommonChunk>(Sym);
    Chunks.push_back(C);
    return Symtab->addCommon(this, GetName(), Sym.getValue(), Sym.getGeneric(),
                             C);
  }

  if (Sym.isAbsolute()) {
    StringRef Name = GetName();

    // Skip special symbols.
    if (Name == "@comp.id")
      return nullptr;
    if (Name == "@feat.00") {
      Feat00Flags = Sym.getValue();
      return nullptr;
    }

    if (Sym.isExternal())
      return Symtab->addAbsolute(Name, Sym);
    return make<DefinedAbsolute>(Name, Sym);
  }

  int32_t SectionNumber = Sym.getSectionNumber();
  if (SectionNumber == llvm::COFF::IMAGE_SYM_DEBUG)
    return nullptr;

  if (llvm::COFF::isReservedSectionNumber(SectionNumber))
    fatal(toString(this) + ": " + GetName() +
          " should not refer to special section " + Twine(SectionNumber));

  if ((uint32_t)SectionNumber >= SparseChunks.size())
    fatal(toString(this) + ": " + GetName() +
          " should not refer to non-existent section " + Twine(SectionNumber));

  // Comdat handling.
  // A comdat symbol consists of two symbol table entries.
  // The first symbol entry has the name of the section (e.g. .text), fixed
  // values for the other fields, and one auxilliary record.
  // The second symbol entry has the name of the comdat symbol, called the
  // "comdat leader".
  // When this function is called for the first symbol entry of a comdat,
  // it sets ComdatDefs and returns None, and when it's called for the second
  // symbol entry it reads ComdatDefs and then sets it back to nullptr.

  // Handle comdat leader.
  if (const coff_aux_section_definition *Def = ComdatDefs[SectionNumber]) {
    ComdatDefs[SectionNumber] = nullptr;
    DefinedRegular *Leader;

    if (Sym.isExternal()) {
      std::tie(Leader, Prevailing) =
          Symtab->addComdat(this, GetName(), Sym.getGeneric());
    } else {
      Leader = make<DefinedRegular>(this, /*Name*/ "", /*IsCOMDAT*/ false,
                                    /*IsExternal*/ false, Sym.getGeneric());
      Prevailing = true;
    }

    if (Def->Selection < (int)IMAGE_COMDAT_SELECT_NODUPLICATES ||
        // Intentionally ends at IMAGE_COMDAT_SELECT_LARGEST: link.exe
        // doesn't understand IMAGE_COMDAT_SELECT_NEWEST either.
        Def->Selection > (int)IMAGE_COMDAT_SELECT_LARGEST) {
      fatal("unknown comdat type " + std::to_string((int)Def->Selection) +
            " for " + GetName() + " in " + toString(this));
    }
    COMDATType Selection = (COMDATType)Def->Selection;

    if (Leader->isCOMDAT())
      handleComdatSelection(Sym, Selection, Prevailing, Leader);

    if (Prevailing) {
      SectionChunk *C = readSection(SectionNumber, Def, GetName());
      SparseChunks[SectionNumber] = C;
      C->Sym = cast<DefinedRegular>(Leader);
      C->Selection = Selection;
      cast<DefinedRegular>(Leader)->Data = &C->Repl;
    } else {
      SparseChunks[SectionNumber] = nullptr;
    }
    return Leader;
  }

  // Prepare to handle the comdat leader symbol by setting the section's
  // ComdatDefs pointer if we encounter a non-associative comdat.
  if (SparseChunks[SectionNumber] == PendingComdat) {
    if (const coff_aux_section_definition *Def = Sym.getSectionDefinition()) {
      if (Def->Selection != IMAGE_COMDAT_SELECT_ASSOCIATIVE)
        ComdatDefs[SectionNumber] = Def;
    }
    return None;
  }

  return createRegular(Sym);
}

MachineTypes ObjFile::getMachineType() {
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
  StringRef Name = Saver.save(StringRef(Buf + sizeof(*Hdr)));
  StringRef ImpName = Saver.save("__imp_" + Name);
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

  this->Hdr = Hdr;
  ExternalName = ExtName;

  ImpSym = Symtab->addImportData(ImpName, this);
  // If this was a duplicate, we logged an error but may continue;
  // in this case, ImpSym is nullptr.
  if (!ImpSym)
    return;

  if (Hdr->getType() == llvm::COFF::IMPORT_CONST)
    static_cast<void>(Symtab->addImportData(Name, this));

  // If type is function, we need to create a thunk which jump to an
  // address pointed by the __imp_ symbol. (This allows you to call
  // DLL functions just like regular non-DLL functions.)
  if (Hdr->getType() == llvm::COFF::IMPORT_CODE)
    ThunkSym = Symtab->addImportThunk(
        Name, cast_or_null<DefinedImportData>(ImpSym), Hdr->Machine);
}

void BitcodeFile::parse() {
  Obj = check(lto::InputFile::create(MemoryBufferRef(
      MB.getBuffer(), Saver.save(ParentName + MB.getBufferIdentifier()))));
  std::vector<std::pair<Symbol *, bool>> Comdat(Obj->getComdatTable().size());
  for (size_t I = 0; I != Obj->getComdatTable().size(); ++I)
    // FIXME: lto::InputFile doesn't keep enough data to do correct comdat
    // selection handling.
    Comdat[I] = Symtab->addComdat(this, Saver.save(Obj->getComdatTable()[I]));
  for (const lto::InputFile::Symbol &ObjSym : Obj->symbols()) {
    StringRef SymName = Saver.save(ObjSym.getName());
    int ComdatIndex = ObjSym.getComdatIndex();
    Symbol *Sym;
    if (ObjSym.isUndefined()) {
      Sym = Symtab->addUndefined(SymName, this, false);
    } else if (ObjSym.isCommon()) {
      Sym = Symtab->addCommon(this, SymName, ObjSym.getCommonSize());
    } else if (ObjSym.isWeak() && ObjSym.isIndirect()) {
      // Weak external.
      Sym = Symtab->addUndefined(SymName, this, true);
      std::string Fallback = ObjSym.getCOFFWeakExternalFallback();
      Symbol *Alias = Symtab->addUndefined(Saver.save(Fallback));
      checkAndSetWeakAlias(Symtab, this, Sym, Alias);
    } else if (ComdatIndex != -1) {
      if (SymName == Obj->getComdatTable()[ComdatIndex])
        Sym = Comdat[ComdatIndex].first;
      else if (Comdat[ComdatIndex].second)
        Sym = Symtab->addRegular(this, SymName);
      else
        Sym = Symtab->addUndefined(SymName, this, false);
    } else {
      Sym = Symtab->addRegular(this, SymName);
    }
    Symbols.push_back(Sym);
    if (ObjSym.isUsed())
      Config->GCRoot.push_back(Sym);
  }
  Directives = Obj->getCOFFLinkerOpts();
}

MachineTypes BitcodeFile::getMachineType() {
  switch (Triple(Obj->getTargetTriple()).getArch()) {
  case Triple::x86_64:
    return AMD64;
  case Triple::x86:
    return I386;
  case Triple::arm:
    return ARMNT;
  case Triple::aarch64:
    return ARM64;
  default:
    return IMAGE_FILE_MACHINE_UNKNOWN;
  }
}
} // namespace coff
} // namespace lld

// Returns the last element of a path, which is supposed to be a filename.
static StringRef getBasename(StringRef Path) {
  return sys::path::filename(Path, sys::path::Style::windows);
}

// Returns a string in the format of "foo.obj" or "foo.obj(bar.lib)".
std::string lld::toString(const coff::InputFile *File) {
  if (!File)
    return "<internal>";
  if (File->ParentName.empty())
    return File->getName();

  return (getBasename(File->ParentName) + "(" + getBasename(File->getName()) +
          ")")
      .str();
}
