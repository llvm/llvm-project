//===- InputFiles.cpp -----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Driver.h"
#include "Error.h"
#include "InputSection.h"
#include "LinkerScript.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::sys::fs;

using namespace lld;
using namespace lld::elf;

// Returns "(internal)", "foo.a(bar.o)" or "baz.o".
std::string elf::getFilename(const InputFile *F) {
  if (!F)
    return "(internal)";
  if (!F->ArchiveName.empty())
    return (F->ArchiveName + "(" + F->getName() + ")").str();
  return F->getName();
}

template <class ELFT>
static ELFFile<ELFT> createELFObj(MemoryBufferRef MB) {
  std::error_code EC;
  ELFFile<ELFT> F(MB.getBuffer(), EC);
  if (EC)
    error(EC, "failed to read " + MB.getBufferIdentifier());
  return F;
}

template <class ELFT> static ELFKind getELFKind() {
  if (ELFT::TargetEndianness == support::little)
    return ELFT::Is64Bits ? ELF64LEKind : ELF32LEKind;
  return ELFT::Is64Bits ? ELF64BEKind : ELF32BEKind;
}

template <class ELFT>
ELFFileBase<ELFT>::ELFFileBase(Kind K, MemoryBufferRef MB)
    : InputFile(K, MB), ELFObj(createELFObj<ELFT>(MB)) {
  EKind = getELFKind<ELFT>();
  EMachine = ELFObj.getHeader()->e_machine;
}

template <class ELFT>
typename ELFT::SymRange ELFFileBase<ELFT>::getElfSymbols(bool OnlyGlobals) {
  if (!Symtab)
    return Elf_Sym_Range(nullptr, nullptr);
  Elf_Sym_Range Syms = ELFObj.symbols(Symtab);
  uint32_t NumSymbols = std::distance(Syms.begin(), Syms.end());
  uint32_t FirstNonLocal = Symtab->sh_info;
  if (FirstNonLocal > NumSymbols)
    fatal(getFilename(this) + ": invalid sh_info in symbol table");

  if (OnlyGlobals)
    return makeArrayRef(Syms.begin() + FirstNonLocal, Syms.end());
  return makeArrayRef(Syms.begin(), Syms.end());
}

template <class ELFT>
uint32_t ELFFileBase<ELFT>::getSectionIndex(const Elf_Sym &Sym) const {
  uint32_t I = Sym.st_shndx;
  if (I == ELF::SHN_XINDEX)
    return ELFObj.getExtendedSymbolTableIndex(&Sym, Symtab, SymtabSHNDX);
  if (I >= ELF::SHN_LORESERVE)
    return 0;
  return I;
}

template <class ELFT> void ELFFileBase<ELFT>::initStringTable() {
  if (!Symtab)
    return;
  StringTable = check(ELFObj.getStringTableForSymtab(*Symtab));
}

template <class ELFT>
elf::ObjectFile<ELFT>::ObjectFile(MemoryBufferRef M)
    : ELFFileBase<ELFT>(Base::ObjectKind, M) {}

template <class ELFT>
ArrayRef<SymbolBody *> elf::ObjectFile<ELFT>::getNonLocalSymbols() {
  if (!this->Symtab)
    return this->SymbolBodies;
  uint32_t FirstNonLocal = this->Symtab->sh_info;
  return makeArrayRef(this->SymbolBodies).slice(FirstNonLocal);
}

template <class ELFT>
ArrayRef<SymbolBody *> elf::ObjectFile<ELFT>::getLocalSymbols() {
  if (!this->Symtab)
    return this->SymbolBodies;
  uint32_t FirstNonLocal = this->Symtab->sh_info;
  return makeArrayRef(this->SymbolBodies).slice(1, FirstNonLocal - 1);
}

template <class ELFT>
ArrayRef<SymbolBody *> elf::ObjectFile<ELFT>::getSymbols() {
  if (!this->Symtab)
    return this->SymbolBodies;
  return makeArrayRef(this->SymbolBodies).slice(1);
}

template <class ELFT> uint32_t elf::ObjectFile<ELFT>::getMipsGp0() const {
  if (ELFT::Is64Bits && MipsOptions && MipsOptions->Reginfo)
    return MipsOptions->Reginfo->ri_gp_value;
  if (!ELFT::Is64Bits && MipsReginfo && MipsReginfo->Reginfo)
    return MipsReginfo->Reginfo->ri_gp_value;
  return 0;
}

template <class ELFT>
void elf::ObjectFile<ELFT>::parse(DenseSet<StringRef> &ComdatGroups) {
  // Read section and symbol tables.
  initializeSections(ComdatGroups);
  initializeSymbols();
}

// Sections with SHT_GROUP and comdat bits define comdat section groups.
// They are identified and deduplicated by group name. This function
// returns a group name.
template <class ELFT>
StringRef elf::ObjectFile<ELFT>::getShtGroupSignature(const Elf_Shdr &Sec) {
  const ELFFile<ELFT> &Obj = this->ELFObj;
  const Elf_Shdr *Symtab = check(Obj.getSection(Sec.sh_link));
  const Elf_Sym *Sym = Obj.getSymbol(Symtab, Sec.sh_info);
  StringRef Strtab = check(Obj.getStringTableForSymtab(*Symtab));
  return check(Sym->getName(Strtab));
}

template <class ELFT>
ArrayRef<typename elf::ObjectFile<ELFT>::Elf_Word>
elf::ObjectFile<ELFT>::getShtGroupEntries(const Elf_Shdr &Sec) {
  const ELFFile<ELFT> &Obj = this->ELFObj;
  ArrayRef<Elf_Word> Entries =
      check(Obj.template getSectionContentsAsArray<Elf_Word>(&Sec));
  if (Entries.empty() || Entries[0] != GRP_COMDAT)
    fatal(getFilename(this) + ": unsupported SHT_GROUP format");
  return Entries.slice(1);
}

template <class ELFT>
bool elf::ObjectFile<ELFT>::shouldMerge(const Elf_Shdr &Sec) {
  // We don't merge sections if -O0 (default is -O1). This makes sometimes
  // the linker significantly faster, although the output will be bigger.
  if (Config->Optimize == 0)
    return false;

  // We don't merge if linker script has SECTIONS command. When script
  // do layout it can merge several sections with different attributes
  // into single output sections. We currently do not support adding
  // mergeable input sections to regular output ones as well as adding
  // regular input sections to mergeable output.
  if (ScriptConfig->HasContents)
    return false;

  // A mergeable section with size 0 is useless because they don't have
  // any data to merge. A mergeable string section with size 0 can be
  // argued as invalid because it doesn't end with a null character.
  // We'll avoid a mess by handling them as if they were non-mergeable.
  if (Sec.sh_size == 0)
    return false;

  uintX_t Flags = Sec.sh_flags;
  if (!(Flags & SHF_MERGE))
    return false;
  if (Flags & SHF_WRITE)
    fatal(getFilename(this) + ": writable SHF_MERGE section is not supported");
  uintX_t EntSize = Sec.sh_entsize;
  if (!EntSize || Sec.sh_size % EntSize)
    fatal(getFilename(this) +
          ": SHF_MERGE section size must be a multiple of sh_entsize");

  // Don't try to merge if the alignment is larger than the sh_entsize and this
  // is not SHF_STRINGS.
  //
  // Since this is not a SHF_STRINGS, we would need to pad after every entity.
  // It would be equivalent for the producer of the .o to just set a larger
  // sh_entsize.
  if (Flags & SHF_STRINGS)
    return true;

  return Sec.sh_addralign <= EntSize;
}

template <class ELFT>
void elf::ObjectFile<ELFT>::initializeSections(
    DenseSet<StringRef> &ComdatGroups) {
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
          fatal(getFilename(this) + ": invalid section index in group: " +
                Twine(SecIndex));
        Sections[SecIndex] = &InputSection<ELFT>::Discarded;
      }
      break;
    case SHT_SYMTAB:
      this->Symtab = &Sec;
      break;
    case SHT_SYMTAB_SHNDX:
      this->SymtabSHNDX = check(Obj.getSHNDXTable(Sec));
      break;
    case SHT_STRTAB:
    case SHT_NULL:
      break;
    default:
      Sections[I] = createInputSection(Sec);
    }
  }
}

template <class ELFT>
InputSectionBase<ELFT> *
elf::ObjectFile<ELFT>::getRelocTarget(const Elf_Shdr &Sec) {
  uint32_t Idx = Sec.sh_info;
  if (Idx >= Sections.size())
    fatal(getFilename(this) + ": invalid relocated section index: " +
          Twine(Idx));
  InputSectionBase<ELFT> *Target = Sections[Idx];

  // Strictly speaking, a relocation section must be included in the
  // group of the section it relocates. However, LLVM 3.3 and earlier
  // would fail to do so, so we gracefully handle that case.
  if (Target == &InputSection<ELFT>::Discarded)
    return nullptr;

  if (!Target)
    fatal(getFilename(this) + ": unsupported relocation reference");
  return Target;
}

template <class ELFT>
InputSectionBase<ELFT> *
elf::ObjectFile<ELFT>::createInputSection(const Elf_Shdr &Sec) {
  StringRef Name = check(this->ELFObj.getSectionName(&Sec));

  switch (Sec.sh_type) {
  case SHT_ARM_ATTRIBUTES:
    // FIXME: ARM meta-data section. At present attributes are ignored,
    // they can be used to reason about object compatibility.
    return &InputSection<ELFT>::Discarded;
  case SHT_MIPS_REGINFO:
    MipsReginfo.reset(new MipsReginfoInputSection<ELFT>(this, &Sec, Name));
    return MipsReginfo.get();
  case SHT_MIPS_OPTIONS:
    MipsOptions.reset(new MipsOptionsInputSection<ELFT>(this, &Sec, Name));
    return MipsOptions.get();
  case SHT_MIPS_ABIFLAGS:
    MipsAbiFlags.reset(new MipsAbiFlagsInputSection<ELFT>(this, &Sec, Name));
    return MipsAbiFlags.get();
  case SHT_RELA:
  case SHT_REL: {
    // This section contains relocation information.
    // If -r is given, we do not interpret or apply relocation
    // but just copy relocation sections to output.
    if (Config->Relocatable)
      return new (IAlloc.Allocate()) InputSection<ELFT>(this, &Sec, Name);

    // Find the relocation target section and associate this
    // section with it.
    InputSectionBase<ELFT> *Target = getRelocTarget(Sec);
    if (!Target)
      return nullptr;
    if (auto *S = dyn_cast<InputSection<ELFT>>(Target)) {
      S->RelocSections.push_back(&Sec);
      return nullptr;
    }
    if (auto *S = dyn_cast<EhInputSection<ELFT>>(Target)) {
      if (S->RelocSection)
        fatal(getFilename(this) +
              ": multiple relocation sections to .eh_frame are not supported");
      S->RelocSection = &Sec;
      return nullptr;
    }
    fatal(getFilename(this) +
          ": relocations pointing to SHF_MERGE are not supported");
  }
  }

  // .note.GNU-stack is a marker section to control the presence of
  // PT_GNU_STACK segment in outputs. Since the presence of the segment
  // is controlled only by the command line option (-z execstack) in LLD,
  // .note.GNU-stack is ignored.
  if (Name == ".note.GNU-stack")
    return &InputSection<ELFT>::Discarded;

  if (Name == ".note.GNU-split-stack") {
    error("objects using splitstacks are not supported");
    return &InputSection<ELFT>::Discarded;
  }

  if (Config->Strip != StripPolicy::None && Name.startswith(".debug"))
    return &InputSection<ELFT>::Discarded;

  // The linker merges EH (exception handling) frames and creates a
  // .eh_frame_hdr section for runtime. So we handle them with a special
  // class. For relocatable outputs, they are just passed through.
  if (Name == ".eh_frame" && !Config->Relocatable)
    return new (EHAlloc.Allocate()) EhInputSection<ELFT>(this, &Sec, Name);

  if (shouldMerge(Sec))
    return new (MAlloc.Allocate()) MergeInputSection<ELFT>(this, &Sec, Name);
  return new (IAlloc.Allocate()) InputSection<ELFT>(this, &Sec, Name);
}

template <class ELFT> void elf::ObjectFile<ELFT>::initializeSymbols() {
  this->initStringTable();
  Elf_Sym_Range Syms = this->getElfSymbols(false);
  uint32_t NumSymbols = std::distance(Syms.begin(), Syms.end());
  SymbolBodies.reserve(NumSymbols);
  for (const Elf_Sym &Sym : Syms)
    SymbolBodies.push_back(createSymbolBody(&Sym));
}

template <class ELFT>
InputSectionBase<ELFT> *
elf::ObjectFile<ELFT>::getSection(const Elf_Sym &Sym) const {
  uint32_t Index = this->getSectionIndex(Sym);
  if (Index == 0)
    return nullptr;
  if (Index >= Sections.size())
    fatal(getFilename(this) + ": invalid section index: " + Twine(Index));
  InputSectionBase<ELFT> *S = Sections[Index];
  // We found that GNU assembler 2.17.50 [FreeBSD] 2007-07-03
  // could generate broken objects. STT_SECTION symbols can be
  // associated with SHT_REL[A]/SHT_SYMTAB/SHT_STRTAB sections.
  // In this case it is fine for section to be null here as we
  // do not allocate sections of these types.
  if (!S || S == &InputSectionBase<ELFT>::Discarded)
    return S;
  return S->Repl;
}

template <class ELFT>
SymbolBody *elf::ObjectFile<ELFT>::createSymbolBody(const Elf_Sym *Sym) {
  int Binding = Sym->getBinding();
  InputSectionBase<ELFT> *Sec = getSection(*Sym);
  if (Binding == STB_LOCAL) {
    if (Sym->st_shndx == SHN_UNDEF)
      return new (this->Alloc)
          Undefined(Sym->st_name, Sym->st_other, Sym->getType(), this);
    return new (this->Alloc) DefinedRegular<ELFT>(*Sym, Sec);
  }

  StringRef Name = check(Sym->getName(this->StringTable));

  switch (Sym->st_shndx) {
  case SHN_UNDEF:
    return elf::Symtab<ELFT>::X
        ->addUndefined(Name, Binding, Sym->st_other, Sym->getType(),
                       /*CanOmitFromDynSym*/ false, /*HasUnnamedAddr*/ false,
                       this)
        ->body();
  case SHN_COMMON:
    return elf::Symtab<ELFT>::X
        ->addCommon(Name, Sym->st_size, Sym->st_value, Binding, Sym->st_other,
                    Sym->getType(), /*HasUnnamedAddr*/ false, this)
        ->body();
  }

  switch (Binding) {
  default:
    fatal(getFilename(this) + ": unexpected binding: " + Twine(Binding));
  case STB_GLOBAL:
  case STB_WEAK:
  case STB_GNU_UNIQUE:
    if (Sec == &InputSection<ELFT>::Discarded)
      return elf::Symtab<ELFT>::X
          ->addUndefined(Name, Binding, Sym->st_other, Sym->getType(),
                         /*CanOmitFromDynSym*/ false,
                         /*HasUnnamedAddr*/ false, this)
          ->body();
    return elf::Symtab<ELFT>::X->addRegular(Name, *Sym, Sec)->body();
  }
}

template <class ELFT> void ArchiveFile::parse() {
  File = check(Archive::create(MB), "failed to parse archive");

  // Read the symbol table to construct Lazy objects.
  for (const Archive::Symbol &Sym : File->symbols())
    Symtab<ELFT>::X->addLazyArchive(this, Sym);
}

// Returns a buffer pointing to a member file containing a given symbol.
MemoryBufferRef ArchiveFile::getMember(const Archive::Symbol *Sym) {
  Archive::Child C =
      check(Sym->getMember(),
            "could not get the member for symbol " + Sym->getName());

  if (!Seen.insert(C.getChildOffset()).second)
    return MemoryBufferRef();

  MemoryBufferRef Ret =
      check(C.getMemoryBufferRef(),
            "could not get the buffer for the member defining symbol " +
                Sym->getName());

  if (C.getParent()->isThin() && Driver->Cpio)
    Driver->Cpio->append(relativeToRoot(check(C.getFullName())),
                         Ret.getBuffer());

  return Ret;
}

template <class ELFT>
SharedFile<ELFT>::SharedFile(MemoryBufferRef M)
    : ELFFileBase<ELFT>(Base::SharedKind, M), AsNeeded(Config->AsNeeded) {}

template <class ELFT>
const typename ELFT::Shdr *
SharedFile<ELFT>::getSection(const Elf_Sym &Sym) const {
  uint32_t Index = this->getSectionIndex(Sym);
  if (Index == 0)
    return nullptr;
  return check(this->ELFObj.getSection(Index));
}

// Partially parse the shared object file so that we can call
// getSoName on this object.
template <class ELFT> void SharedFile<ELFT>::parseSoName() {
  typedef typename ELFT::Dyn Elf_Dyn;
  typedef typename ELFT::uint uintX_t;
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
    case SHT_SYMTAB_SHNDX:
      this->SymtabSHNDX = check(Obj.getSHNDXTable(Sec));
      break;
    case SHT_GNU_versym:
      this->VersymSec = &Sec;
      break;
    case SHT_GNU_verdef:
      this->VerdefSec = &Sec;
      break;
    }
  }

  this->initStringTable();

  // DSOs are identified by soname, and they usually contain
  // DT_SONAME tag in their header. But if they are missing,
  // filenames are used as default sonames.
  SoName = sys::path::filename(this->getName());

  if (!DynamicSec)
    return;
  auto *Begin =
      reinterpret_cast<const Elf_Dyn *>(Obj.base() + DynamicSec->sh_offset);
  const Elf_Dyn *End = Begin + DynamicSec->sh_size / sizeof(Elf_Dyn);

  for (const Elf_Dyn &Dyn : make_range(Begin, End)) {
    if (Dyn.d_tag == DT_SONAME) {
      uintX_t Val = Dyn.getVal();
      if (Val >= this->StringTable.size())
        fatal(getFilename(this) + ": invalid DT_SONAME entry");
      SoName = StringRef(this->StringTable.data() + Val);
      return;
    }
  }
}

// Parse the version definitions in the object file if present. Returns a vector
// whose nth element contains a pointer to the Elf_Verdef for version identifier
// n. Version identifiers that are not definitions map to nullptr. The array
// always has at least length 1.
template <class ELFT>
std::vector<const typename ELFT::Verdef *>
SharedFile<ELFT>::parseVerdefs(const Elf_Versym *&Versym) {
  std::vector<const Elf_Verdef *> Verdefs(1);
  // We only need to process symbol versions for this DSO if it has both a
  // versym and a verdef section, which indicates that the DSO contains symbol
  // version definitions.
  if (!VersymSec || !VerdefSec)
    return Verdefs;

  // The location of the first global versym entry.
  Versym = reinterpret_cast<const Elf_Versym *>(this->ELFObj.base() +
                                                VersymSec->sh_offset) +
           this->Symtab->sh_info;

  // We cannot determine the largest verdef identifier without inspecting
  // every Elf_Verdef, but both bfd and gold assign verdef identifiers
  // sequentially starting from 1, so we predict that the largest identifier
  // will be VerdefCount.
  unsigned VerdefCount = VerdefSec->sh_info;
  Verdefs.resize(VerdefCount + 1);

  // Build the Verdefs array by following the chain of Elf_Verdef objects
  // from the start of the .gnu.version_d section.
  const uint8_t *Verdef = this->ELFObj.base() + VerdefSec->sh_offset;
  for (unsigned I = 0; I != VerdefCount; ++I) {
    auto *CurVerdef = reinterpret_cast<const Elf_Verdef *>(Verdef);
    Verdef += CurVerdef->vd_next;
    unsigned VerdefIndex = CurVerdef->vd_ndx;
    if (Verdefs.size() <= VerdefIndex)
      Verdefs.resize(VerdefIndex + 1);
    Verdefs[VerdefIndex] = CurVerdef;
  }

  return Verdefs;
}

// Fully parse the shared object file. This must be called after parseSoName().
template <class ELFT> void SharedFile<ELFT>::parseRest() {
  // Create mapping from version identifiers to Elf_Verdef entries.
  const Elf_Versym *Versym = nullptr;
  std::vector<const Elf_Verdef *> Verdefs = parseVerdefs(Versym);

  Elf_Sym_Range Syms = this->getElfSymbols(true);
  for (const Elf_Sym &Sym : Syms) {
    unsigned VersymIndex = 0;
    if (Versym) {
      VersymIndex = Versym->vs_index;
      ++Versym;
    }

    StringRef Name = check(Sym.getName(this->StringTable));
    if (Sym.isUndefined()) {
      Undefs.push_back(Name);
      continue;
    }

    if (Versym) {
      // Ignore local symbols and non-default versions.
      if (VersymIndex == VER_NDX_LOCAL || (VersymIndex & VERSYM_HIDDEN))
        continue;
    }

    const Elf_Verdef *V =
        VersymIndex == VER_NDX_GLOBAL ? nullptr : Verdefs[VersymIndex];
    elf::Symtab<ELFT>::X->addShared(this, Name, Sym, V);
  }
}

static ELFKind getBitcodeELFKind(MemoryBufferRef MB) {
  Triple T(getBitcodeTargetTriple(MB, Driver->Context));
  if (T.isLittleEndian())
    return T.isArch64Bit() ? ELF64LEKind : ELF32LEKind;
  return T.isArch64Bit() ? ELF64BEKind : ELF32BEKind;
}

static uint8_t getBitcodeMachineKind(MemoryBufferRef MB) {
  Triple T(getBitcodeTargetTriple(MB, Driver->Context));
  switch (T.getArch()) {
  case Triple::aarch64:
    return EM_AARCH64;
  case Triple::arm:
    return EM_ARM;
  case Triple::mips:
  case Triple::mipsel:
  case Triple::mips64:
  case Triple::mips64el:
    return EM_MIPS;
  case Triple::ppc:
    return EM_PPC;
  case Triple::ppc64:
    return EM_PPC64;
  case Triple::x86:
    return T.isOSIAMCU() ? EM_IAMCU : EM_386;
  case Triple::x86_64:
    return EM_X86_64;
  default:
    fatal(MB.getBufferIdentifier() +
          ": could not infer e_machine from bitcode target triple " + T.str());
  }
}

BitcodeFile::BitcodeFile(MemoryBufferRef MB) : InputFile(BitcodeKind, MB) {
  EKind = getBitcodeELFKind(MB);
  EMachine = getBitcodeMachineKind(MB);
}

static uint8_t getGvVisibility(const GlobalValue *GV) {
  switch (GV->getVisibility()) {
  case GlobalValue::DefaultVisibility:
    return STV_DEFAULT;
  case GlobalValue::HiddenVisibility:
    return STV_HIDDEN;
  case GlobalValue::ProtectedVisibility:
    return STV_PROTECTED;
  }
  llvm_unreachable("unknown visibility");
}

template <class ELFT>
Symbol *BitcodeFile::createSymbol(const DenseSet<const Comdat *> &KeptComdats,
                                  const IRObjectFile &Obj,
                                  const BasicSymbolRef &Sym) {
  const GlobalValue *GV = Obj.getSymbolGV(Sym.getRawDataRefImpl());

  SmallString<64> Name;
  raw_svector_ostream OS(Name);
  Sym.printName(OS);
  StringRef NameRef = Saver.save(StringRef(Name));

  uint32_t Flags = Sym.getFlags();
  uint32_t Binding = (Flags & BasicSymbolRef::SF_Weak) ? STB_WEAK : STB_GLOBAL;

  uint8_t Type = STT_NOTYPE;
  uint8_t Visibility;
  bool CanOmitFromDynSym = false;
  bool HasUnnamedAddr = false;

  // FIXME: Expose a thread-local flag for module asm symbols.
  if (GV) {
    if (GV->isThreadLocal())
      Type = STT_TLS;
    CanOmitFromDynSym = canBeOmittedFromSymbolTable(GV);
    Visibility = getGvVisibility(GV);
    HasUnnamedAddr =
        GV->getUnnamedAddr() == llvm::GlobalValue::UnnamedAddr::Global;
  } else {
    // FIXME: Set SF_Hidden flag correctly for module asm symbols, and expose
    // protected visibility.
    Visibility = STV_DEFAULT;
  }

  if (GV)
    if (const Comdat *C = GV->getComdat())
      if (!KeptComdats.count(C))
        return Symtab<ELFT>::X->addUndefined(NameRef, Binding, Visibility, Type,
                                             CanOmitFromDynSym, HasUnnamedAddr,
                                             this);

  const Module &M = Obj.getModule();
  if (Flags & BasicSymbolRef::SF_Undefined)
    return Symtab<ELFT>::X->addUndefined(NameRef, Binding, Visibility, Type,
                                         CanOmitFromDynSym, HasUnnamedAddr,
                                         this);
  if (Flags & BasicSymbolRef::SF_Common) {
    // FIXME: Set SF_Common flag correctly for module asm symbols, and expose
    // size and alignment.
    assert(GV);
    const DataLayout &DL = M.getDataLayout();
    uint64_t Size = DL.getTypeAllocSize(GV->getValueType());
    return Symtab<ELFT>::X->addCommon(NameRef, Size, GV->getAlignment(),
                                      Binding, Visibility, STT_OBJECT,
                                      HasUnnamedAddr, this);
  }
  return Symtab<ELFT>::X->addBitcode(NameRef, Binding, Visibility, Type,
                                     CanOmitFromDynSym, HasUnnamedAddr, this);
}

bool BitcodeFile::shouldSkip(uint32_t Flags) {
  return !(Flags & BasicSymbolRef::SF_Global) ||
         (Flags & BasicSymbolRef::SF_FormatSpecific);
}

template <class ELFT>
void BitcodeFile::parse(DenseSet<StringRef> &ComdatGroups) {
  Obj = check(IRObjectFile::create(MB, Driver->Context));
  const Module &M = Obj->getModule();

  DenseSet<const Comdat *> KeptComdats;
  for (const auto &P : M.getComdatSymbolTable()) {
    StringRef N = Saver.save(P.first());
    if (ComdatGroups.insert(N).second)
      KeptComdats.insert(&P.second);
  }

  for (const BasicSymbolRef &Sym : Obj->symbols())
    if (!shouldSkip(Sym.getFlags()))
      Symbols.push_back(createSymbol<ELFT>(KeptComdats, *Obj, Sym));
}

template <template <class> class T>
static std::unique_ptr<InputFile> createELFFile(MemoryBufferRef MB) {
  unsigned char Size;
  unsigned char Endian;
  std::tie(Size, Endian) = getElfArchType(MB.getBuffer());
  if (Endian != ELFDATA2LSB && Endian != ELFDATA2MSB)
    fatal("invalid data encoding: " + MB.getBufferIdentifier());

  std::unique_ptr<InputFile> Obj;
  if (Size == ELFCLASS32 && Endian == ELFDATA2LSB)
    Obj.reset(new T<ELF32LE>(MB));
  else if (Size == ELFCLASS32 && Endian == ELFDATA2MSB)
    Obj.reset(new T<ELF32BE>(MB));
  else if (Size == ELFCLASS64 && Endian == ELFDATA2LSB)
    Obj.reset(new T<ELF64LE>(MB));
  else if (Size == ELFCLASS64 && Endian == ELFDATA2MSB)
    Obj.reset(new T<ELF64BE>(MB));
  else
    fatal("invalid file class: " + MB.getBufferIdentifier());

  if (!Config->FirstElf)
    Config->FirstElf = Obj.get();
  return Obj;
}

static bool isBitcode(MemoryBufferRef MB) {
  using namespace sys::fs;
  return identify_magic(MB.getBuffer()) == file_magic::bitcode;
}

std::unique_ptr<InputFile> elf::createObjectFile(MemoryBufferRef MB,
                                                 StringRef ArchiveName) {
  std::unique_ptr<InputFile> F;
  if (isBitcode(MB))
    F.reset(new BitcodeFile(MB));
  else
    F = createELFFile<ObjectFile>(MB);
  F->ArchiveName = ArchiveName;
  return F;
}

std::unique_ptr<InputFile> elf::createSharedFile(MemoryBufferRef MB) {
  return createELFFile<SharedFile>(MB);
}

MemoryBufferRef LazyObjectFile::getBuffer() {
  if (Seen)
    return MemoryBufferRef();
  Seen = true;
  return MB;
}

template <class ELFT>
void LazyObjectFile::parse() {
  for (StringRef Sym : getSymbols())
    Symtab<ELFT>::X->addLazyObject(Sym, *this);
}

template <class ELFT> std::vector<StringRef> LazyObjectFile::getElfSymbols() {
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::SymRange Elf_Sym_Range;

  const ELFFile<ELFT> Obj = createELFObj<ELFT>(this->MB);
  for (const Elf_Shdr &Sec : Obj.sections()) {
    if (Sec.sh_type != SHT_SYMTAB)
      continue;
    Elf_Sym_Range Syms = Obj.symbols(&Sec);
    uint32_t FirstNonLocal = Sec.sh_info;
    StringRef StringTable = check(Obj.getStringTableForSymtab(Sec));
    std::vector<StringRef> V;
    for (const Elf_Sym &Sym : Syms.slice(FirstNonLocal))
      if (Sym.st_shndx != SHN_UNDEF)
        V.push_back(check(Sym.getName(StringTable)));
    return V;
  }
  return {};
}

std::vector<StringRef> LazyObjectFile::getBitcodeSymbols() {
  LLVMContext Context;
  std::unique_ptr<IRObjectFile> Obj =
      check(IRObjectFile::create(this->MB, Context));
  std::vector<StringRef> V;
  for (const BasicSymbolRef &Sym : Obj->symbols()) {
    uint32_t Flags = Sym.getFlags();
    if (BitcodeFile::shouldSkip(Flags))
      continue;
    if (Flags & BasicSymbolRef::SF_Undefined)
      continue;
    SmallString<64> Name;
    raw_svector_ostream OS(Name);
    Sym.printName(OS);
    V.push_back(Saver.save(StringRef(Name)));
  }
  return V;
}

// Returns a vector of globally-visible defined symbol names.
std::vector<StringRef> LazyObjectFile::getSymbols() {
  if (isBitcode(this->MB))
    return getBitcodeSymbols();

  unsigned char Size;
  unsigned char Endian;
  std::tie(Size, Endian) = getElfArchType(this->MB.getBuffer());
  if (Size == ELFCLASS32) {
    if (Endian == ELFDATA2LSB)
      return getElfSymbols<ELF32LE>();
    return getElfSymbols<ELF32BE>();
  }
  if (Endian == ELFDATA2LSB)
    return getElfSymbols<ELF64LE>();
  return getElfSymbols<ELF64BE>();
}

template void ArchiveFile::parse<ELF32LE>();
template void ArchiveFile::parse<ELF32BE>();
template void ArchiveFile::parse<ELF64LE>();
template void ArchiveFile::parse<ELF64BE>();

template void BitcodeFile::parse<ELF32LE>(DenseSet<StringRef> &);
template void BitcodeFile::parse<ELF32BE>(DenseSet<StringRef> &);
template void BitcodeFile::parse<ELF64LE>(DenseSet<StringRef> &);
template void BitcodeFile::parse<ELF64BE>(DenseSet<StringRef> &);

template void LazyObjectFile::parse<ELF32LE>();
template void LazyObjectFile::parse<ELF32BE>();
template void LazyObjectFile::parse<ELF64LE>();
template void LazyObjectFile::parse<ELF64BE>();

template class elf::ELFFileBase<ELF32LE>;
template class elf::ELFFileBase<ELF32BE>;
template class elf::ELFFileBase<ELF64LE>;
template class elf::ELFFileBase<ELF64BE>;

template class elf::ObjectFile<ELF32LE>;
template class elf::ObjectFile<ELF32BE>;
template class elf::ObjectFile<ELF64LE>;
template class elf::ObjectFile<ELF64BE>;

template class elf::SharedFile<ELF32LE>;
template class elf::SharedFile<ELF32BE>;
template class elf::SharedFile<ELF64LE>;
template class elf::SharedFile<ELF64BE>;
