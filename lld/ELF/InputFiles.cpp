//===- InputFiles.cpp -----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "InputSection.h"
#include "Error.h"
#include "Symbols.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::sys::fs;

using namespace lld;
using namespace lld::elf2;

namespace {
class ECRAII {
  std::error_code EC;

public:
  std::error_code &getEC() { return EC; }
  ~ECRAII() { fatal(EC); }
};
}

template <class ELFT>
ELFFileBase<ELFT>::ELFFileBase(Kind K, MemoryBufferRef M)
    : InputFile(K, M), ELFObj(MB.getBuffer(), ECRAII().getEC()) {}

template <class ELFT>
ELFKind ELFFileBase<ELFT>::getELFKind() {
  if (ELFT::TargetEndianness == support::little)
    return ELFT::Is64Bits ? ELF64LEKind : ELF32LEKind;
  return ELFT::Is64Bits ? ELF64BEKind : ELF32BEKind;
}

template <class ELFT>
typename ELFFileBase<ELFT>::Elf_Sym_Range
ELFFileBase<ELFT>::getSymbolsHelper(bool Local) {
  if (!Symtab)
    return Elf_Sym_Range(nullptr, nullptr);
  Elf_Sym_Range Syms = ELFObj.symbols(Symtab);
  uint32_t NumSymbols = std::distance(Syms.begin(), Syms.end());
  uint32_t FirstNonLocal = Symtab->sh_info;
  if (FirstNonLocal > NumSymbols)
    fatal("Invalid sh_info in symbol table");
  if (!Local)
    return make_range(Syms.begin() + FirstNonLocal, Syms.end());
  // +1 to skip over dummy symbol.
  return make_range(Syms.begin() + 1, Syms.begin() + FirstNonLocal);
}

template <class ELFT>
uint32_t ELFFileBase<ELFT>::getSectionIndex(const Elf_Sym &Sym) const {
  uint32_t I = Sym.st_shndx;
  if (I == ELF::SHN_XINDEX)
    return ELFObj.getExtendedSymbolTableIndex(&Sym, Symtab, SymtabSHNDX);
  if (I >= ELF::SHN_LORESERVE || I == ELF::SHN_ABS)
    return 0;
  return I;
}

template <class ELFT> void ELFFileBase<ELFT>::initStringTable() {
  if (!Symtab)
    return;
  ErrorOr<StringRef> StringTableOrErr = ELFObj.getStringTableForSymtab(*Symtab);
  fatal(StringTableOrErr);
  StringTable = *StringTableOrErr;
}

template <class ELFT>
typename ELFFileBase<ELFT>::Elf_Sym_Range
ELFFileBase<ELFT>::getNonLocalSymbols() {
  return getSymbolsHelper(false);
}

template <class ELFT>
ObjectFile<ELFT>::ObjectFile(MemoryBufferRef M)
    : ELFFileBase<ELFT>(Base::ObjectKind, M) {}

template <class ELFT>
typename ObjectFile<ELFT>::Elf_Sym_Range ObjectFile<ELFT>::getLocalSymbols() {
  return this->getSymbolsHelper(true);
}

template <class ELFT> uint32_t ObjectFile<ELFT>::getMipsGp0() const {
  if (MipsReginfo)
    return MipsReginfo->Reginfo->ri_gp_value;
  return 0;
}

template <class ELFT>
const typename ObjectFile<ELFT>::Elf_Sym *
ObjectFile<ELFT>::getLocalSymbol(uintX_t SymIndex) {
  uint32_t FirstNonLocal = this->Symtab->sh_info;
  if (SymIndex >= FirstNonLocal)
    return nullptr;
  Elf_Sym_Range Syms = this->ELFObj.symbols(this->Symtab);
  return Syms.begin() + SymIndex;
}

template <class ELFT>
void ObjectFile<ELFT>::parse(DenseSet<StringRef> &ComdatGroups) {
  // Read section and symbol tables.
  initializeSections(ComdatGroups);
  initializeSymbols();
}

// Sections with SHT_GROUP and comdat bits define comdat section groups.
// They are identified and deduplicated by group name. This function
// returns a group name.
template <class ELFT>
StringRef ObjectFile<ELFT>::getShtGroupSignature(const Elf_Shdr &Sec) {
  const ELFFile<ELFT> &Obj = this->ELFObj;
  uint32_t SymtabdSectionIndex = Sec.sh_link;
  ErrorOr<const Elf_Shdr *> SecOrErr = Obj.getSection(SymtabdSectionIndex);
  fatal(SecOrErr);
  const Elf_Shdr *SymtabSec = *SecOrErr;
  uint32_t SymIndex = Sec.sh_info;
  const Elf_Sym *Sym = Obj.getSymbol(SymtabSec, SymIndex);
  ErrorOr<StringRef> StringTableOrErr = Obj.getStringTableForSymtab(*SymtabSec);
  fatal(StringTableOrErr);
  ErrorOr<StringRef> SignatureOrErr = Sym->getName(*StringTableOrErr);
  fatal(SignatureOrErr);
  return *SignatureOrErr;
}

template <class ELFT>
ArrayRef<typename ObjectFile<ELFT>::uint32_X>
ObjectFile<ELFT>::getShtGroupEntries(const Elf_Shdr &Sec) {
  const ELFFile<ELFT> &Obj = this->ELFObj;
  ErrorOr<ArrayRef<uint32_X>> EntriesOrErr =
      Obj.template getSectionContentsAsArray<uint32_X>(&Sec);
  fatal(EntriesOrErr);
  ArrayRef<uint32_X> Entries = *EntriesOrErr;
  if (Entries.empty() || Entries[0] != GRP_COMDAT)
    fatal("Unsupported SHT_GROUP format");
  return Entries.slice(1);
}

template <class ELFT>
static bool shouldMerge(const typename ELFFile<ELFT>::Elf_Shdr &Sec) {
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;
  uintX_t Flags = Sec.sh_flags;
  if (!(Flags & SHF_MERGE))
    return false;
  if (Flags & SHF_WRITE)
    fatal("Writable SHF_MERGE sections are not supported");
  uintX_t EntSize = Sec.sh_entsize;
  if (!EntSize || Sec.sh_size % EntSize)
    fatal("SHF_MERGE section size must be a multiple of sh_entsize");

  // Don't try to merge if the aligment is larger than the sh_entsize.
  //
  // If this is not a SHF_STRINGS, we would need to pad after every entity. It
  // would be equivalent for the producer of the .o to just set a larger
  // sh_entsize.
  //
  // If this is a SHF_STRINGS, the larger alignment makes sense. Unfortunately
  // it would complicate tail merging. This doesn't seem that common to
  // justify the effort.
  if (Sec.sh_addralign > EntSize)
    return false;

  return true;
}

template <class ELFT>
void ObjectFile<ELFT>::initializeSections(DenseSet<StringRef> &ComdatGroups) {
  uint64_t Size = this->ELFObj.getNumSections();
  Sections.resize(Size);
  unsigned I = -1;
  const ELFFile<ELFT> &Obj = this->ELFObj;
  for (const Elf_Shdr &Sec : Obj.sections()) {
    ++I;
    if (Sections[I] == &InputSection<ELFT>::Discarded)
      continue;

    switch (Sec.sh_type) {
    case SHT_GROUP:
      Sections[I] = &InputSection<ELFT>::Discarded;
      if (ComdatGroups.insert(getShtGroupSignature(Sec)).second)
        continue;
      for (uint32_t SecIndex : getShtGroupEntries(Sec)) {
        if (SecIndex >= Size)
          fatal("Invalid section index in group");
        Sections[SecIndex] = &InputSection<ELFT>::Discarded;
      }
      break;
    case SHT_SYMTAB:
      this->Symtab = &Sec;
      break;
    case SHT_SYMTAB_SHNDX: {
      ErrorOr<ArrayRef<Elf_Word>> ErrorOrTable = Obj.getSHNDXTable(Sec);
      fatal(ErrorOrTable);
      this->SymtabSHNDX = *ErrorOrTable;
      break;
    }
    case SHT_STRTAB:
    case SHT_NULL:
      break;
    case SHT_RELA:
    case SHT_REL: {
      uint32_t RelocatedSectionIndex = Sec.sh_info;
      if (RelocatedSectionIndex >= Size)
        fatal("Invalid relocated section index");
      InputSectionBase<ELFT> *RelocatedSection =
          Sections[RelocatedSectionIndex];
      if (!RelocatedSection)
        fatal("Unsupported relocation reference");
      if (auto *S = dyn_cast<InputSection<ELFT>>(RelocatedSection)) {
        S->RelocSections.push_back(&Sec);
      } else if (auto *S = dyn_cast<EHInputSection<ELFT>>(RelocatedSection)) {
        if (S->RelocSection)
          fatal("Multiple relocation sections to .eh_frame are not supported");
        S->RelocSection = &Sec;
      } else {
        fatal("Relocations pointing to SHF_MERGE are not supported");
      }
      break;
    }
    default:
      Sections[I] = createInputSection(Sec);
    }
  }
}

template <class ELFT> InputSectionBase<ELFT> *
ObjectFile<ELFT>::createInputSection(const Elf_Shdr &Sec) {
  ErrorOr<StringRef> NameOrErr = this->ELFObj.getSectionName(&Sec);
  fatal(NameOrErr);
  StringRef Name = *NameOrErr;

  // .note.GNU-stack is a marker section to control the presence of
  // PT_GNU_STACK segment in outputs. Since the presence of the segment
  // is controlled only by the command line option (-z execstack) in LLD,
  // .note.GNU-stack is ignored.
  if (Name == ".note.GNU-stack")
    return &InputSection<ELFT>::Discarded;

  // A MIPS object file has a special section that contains register
  // usage info, which needs to be handled by the linker specially.
  if (Config->EMachine == EM_MIPS && Name == ".reginfo") {
    MipsReginfo = new (Alloc) MipsReginfoInputSection<ELFT>(this, &Sec);
    return MipsReginfo;
  }

  if (Name == ".eh_frame")
    return new (EHAlloc.Allocate()) EHInputSection<ELFT>(this, &Sec);
  if (shouldMerge<ELFT>(Sec))
    return new (MAlloc.Allocate()) MergeInputSection<ELFT>(this, &Sec);
  return new (Alloc) InputSection<ELFT>(this, &Sec);
}

template <class ELFT> void ObjectFile<ELFT>::initializeSymbols() {
  this->initStringTable();
  Elf_Sym_Range Syms = this->getNonLocalSymbols();
  uint32_t NumSymbols = std::distance(Syms.begin(), Syms.end());
  SymbolBodies.reserve(NumSymbols);
  for (const Elf_Sym &Sym : Syms)
    SymbolBodies.push_back(createSymbolBody(&Sym));
}

template <class ELFT>
InputSectionBase<ELFT> *
ObjectFile<ELFT>::getSection(const Elf_Sym &Sym) const {
  uint32_t Index = this->getSectionIndex(Sym);
  if (Index == 0)
    return nullptr;
  if (Index >= Sections.size() || !Sections[Index])
    fatal("Invalid section index");
  return Sections[Index];
}

template <class ELFT>
SymbolBody *ObjectFile<ELFT>::createSymbolBody(const Elf_Sym *Sym) {
  ErrorOr<StringRef> NameOrErr = Sym->getName(this->StringTable);
  fatal(NameOrErr);
  StringRef Name = *NameOrErr;

  switch (Sym->st_shndx) {
  case SHN_UNDEF:
    return new (Alloc) UndefinedElf<ELFT>(Name, *Sym);
  case SHN_COMMON:
    return new (Alloc) DefinedCommon(Name, Sym->st_size, Sym->st_value,
                                     Sym->getBinding() == llvm::ELF::STB_WEAK,
                                     Sym->getVisibility());
  }

  switch (Sym->getBinding()) {
  default:
    fatal("unexpected binding");
  case STB_GLOBAL:
  case STB_WEAK:
  case STB_GNU_UNIQUE: {
    InputSectionBase<ELFT> *Sec = getSection(*Sym);
    if (Sec == &InputSection<ELFT>::Discarded)
      return new (Alloc) UndefinedElf<ELFT>(Name, *Sym);
    return new (Alloc) DefinedRegular<ELFT>(Name, *Sym, Sec);
  }
  }
}

void ArchiveFile::parse() {
  ErrorOr<std::unique_ptr<Archive>> FileOrErr = Archive::create(MB);
  fatal(FileOrErr, "Failed to parse archive");
  File = std::move(*FileOrErr);

  // Allocate a buffer for Lazy objects.
  size_t NumSyms = File->getNumberOfSymbols();
  LazySymbols.reserve(NumSyms);

  // Read the symbol table to construct Lazy objects.
  for (const Archive::Symbol &Sym : File->symbols())
    LazySymbols.emplace_back(this, Sym);
}

// Returns a buffer pointing to a member file containing a given symbol.
MemoryBufferRef ArchiveFile::getMember(const Archive::Symbol *Sym) {
  ErrorOr<Archive::Child> COrErr = Sym->getMember();
  fatal(COrErr, "Could not get the member for symbol " + Sym->getName());
  const Archive::Child &C = *COrErr;

  if (!Seen.insert(C.getChildOffset()).second)
    return MemoryBufferRef();

  ErrorOr<MemoryBufferRef> RefOrErr = C.getMemoryBufferRef();
  if (!RefOrErr)
    fatal(RefOrErr, "Could not get the buffer for the member defining symbol " +
                        Sym->getName());
  return *RefOrErr;
}

template <class ELFT>
SharedFile<ELFT>::SharedFile(MemoryBufferRef M)
    : ELFFileBase<ELFT>(Base::SharedKind, M), AsNeeded(Config->AsNeeded) {}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Shdr *
SharedFile<ELFT>::getSection(const Elf_Sym &Sym) const {
  uint32_t Index = this->getSectionIndex(Sym);
  if (Index == 0)
    return nullptr;
  ErrorOr<const Elf_Shdr *> Ret = this->ELFObj.getSection(Index);
  fatal(Ret);
  return *Ret;
}

// Partially parse the shared object file so that we can call
// getSoName on this object.
template <class ELFT> void SharedFile<ELFT>::parseSoName() {
  typedef typename ELFFile<ELFT>::Elf_Dyn Elf_Dyn;
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;
  const Elf_Shdr *DynamicSec = nullptr;

  const ELFFile<ELFT> Obj = this->ELFObj;
  for (const Elf_Shdr &Sec : Obj.sections()) {
    switch (Sec.sh_type) {
    default:
      continue;
    case SHT_DYNSYM:
      this->Symtab = &Sec;
      break;
    case SHT_DYNAMIC:
      DynamicSec = &Sec;
      break;
    case SHT_SYMTAB_SHNDX: {
      ErrorOr<ArrayRef<Elf_Word>> ErrorOrTable = Obj.getSHNDXTable(Sec);
      fatal(ErrorOrTable);
      this->SymtabSHNDX = *ErrorOrTable;
      break;
    }
    }
  }

  this->initStringTable();
  SoName = this->getName();

  if (!DynamicSec)
    return;
  auto *Begin =
      reinterpret_cast<const Elf_Dyn *>(Obj.base() + DynamicSec->sh_offset);
  const Elf_Dyn *End = Begin + DynamicSec->sh_size / sizeof(Elf_Dyn);

  for (const Elf_Dyn &Dyn : make_range(Begin, End)) {
    if (Dyn.d_tag == DT_SONAME) {
      uintX_t Val = Dyn.getVal();
      if (Val >= this->StringTable.size())
        fatal("Invalid DT_SONAME entry");
      SoName = StringRef(this->StringTable.data() + Val);
      return;
    }
  }
}

// Fully parse the shared object file. This must be called after parseSoName().
template <class ELFT> void SharedFile<ELFT>::parseRest() {
  Elf_Sym_Range Syms = this->getNonLocalSymbols();
  uint32_t NumSymbols = std::distance(Syms.begin(), Syms.end());
  SymbolBodies.reserve(NumSymbols);
  for (const Elf_Sym &Sym : Syms) {
    ErrorOr<StringRef> NameOrErr = Sym.getName(this->StringTable);
    fatal(NameOrErr.getError());
    StringRef Name = *NameOrErr;

    if (Sym.isUndefined())
      Undefs.push_back(Name);
    else
      SymbolBodies.emplace_back(this, Name, Sym);
  }
}

template <typename T>
static std::unique_ptr<InputFile> createELFFileAux(MemoryBufferRef MB) {
  std::unique_ptr<T> Ret = llvm::make_unique<T>(MB);

  if (!Config->FirstElf)
    Config->FirstElf = Ret.get();

  if (Config->EKind == ELFNoneKind) {
    Config->EKind = Ret->getELFKind();
    Config->EMachine = Ret->getEMachine();
  }

  return std::move(Ret);
}

template <template <class> class T>
static std::unique_ptr<InputFile> createELFFile(MemoryBufferRef MB) {
  std::pair<unsigned char, unsigned char> Type = getElfArchType(MB.getBuffer());
  if (Type.second != ELF::ELFDATA2LSB && Type.second != ELF::ELFDATA2MSB)
    fatal("Invalid data encoding: " + MB.getBufferIdentifier());

  if (Type.first == ELF::ELFCLASS32) {
    if (Type.second == ELF::ELFDATA2LSB)
      return createELFFileAux<T<ELF32LE>>(MB);
    return createELFFileAux<T<ELF32BE>>(MB);
  }
  if (Type.first == ELF::ELFCLASS64) {
    if (Type.second == ELF::ELFDATA2LSB)
      return createELFFileAux<T<ELF64LE>>(MB);
    return createELFFileAux<T<ELF64BE>>(MB);
  }
  fatal("Invalid file class: " + MB.getBufferIdentifier());
}

std::unique_ptr<InputFile> elf2::createObjectFile(MemoryBufferRef MB,
                                                  StringRef ArchiveName) {
  std::unique_ptr<InputFile> F = createELFFile<ObjectFile>(MB);
  F->ArchiveName = ArchiveName;
  return F;
}

std::unique_ptr<InputFile> elf2::createSharedFile(MemoryBufferRef MB) {
  return createELFFile<SharedFile>(MB);
}

template class elf2::ELFFileBase<ELF32LE>;
template class elf2::ELFFileBase<ELF32BE>;
template class elf2::ELFFileBase<ELF64LE>;
template class elf2::ELFFileBase<ELF64BE>;

template class elf2::ObjectFile<ELF32LE>;
template class elf2::ObjectFile<ELF32BE>;
template class elf2::ObjectFile<ELF64LE>;
template class elf2::ObjectFile<ELF64BE>;

template class elf2::SharedFile<ELF32LE>;
template class elf2::SharedFile<ELF32BE>;
template class elf2::SharedFile<ELF64LE>;
template class elf2::SharedFile<ELF64BE>;
