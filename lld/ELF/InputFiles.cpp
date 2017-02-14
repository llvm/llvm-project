//===- InputFiles.cpp -----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Error.h"
#include "InputSection.h"
#include "LinkerScript.h"
#include "Memory.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/LTO/LTO.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TarWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::sys::fs;

using namespace lld;
using namespace lld::elf;

TarWriter *elf::Tar;

namespace {
// In ELF object file all section addresses are zero. If we have multiple
// .text sections (when using -ffunction-section or comdat group) then
// LLVM DWARF parser will not be able to parse .debug_line correctly, unless
// we assign each section some unique address. This callback method assigns
// each section an address equal to its offset in ELF object file.
class ObjectInfo : public LoadedObjectInfo {
public:
  uint64_t getSectionLoadAddress(const object::SectionRef &Sec) const override {
    return static_cast<const ELFSectionRef &>(Sec).getOffset();
  }
  std::unique_ptr<LoadedObjectInfo> clone() const override {
    return std::unique_ptr<LoadedObjectInfo>();
  }
};
}

Optional<MemoryBufferRef> elf::readFile(StringRef Path) {
  if (Config->Verbose)
    outs() << Path << "\n";

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

template <class ELFT> void elf::ObjectFile<ELFT>::initializeDwarfLine() {
  std::unique_ptr<object::ObjectFile> Obj =
      check(object::ObjectFile::createObjectFile(this->MB),
            "createObjectFile failed");

  ObjectInfo ObjInfo;
  DWARFContextInMemory Dwarf(*Obj, &ObjInfo);
  DwarfLine.reset(new DWARFDebugLine(&Dwarf.getLineSection().Relocs));
  DataExtractor LineData(Dwarf.getLineSection().Data,
                         ELFT::TargetEndianness == support::little,
                         ELFT::Is64Bits ? 8 : 4);

  // The second parameter is offset in .debug_line section
  // for compilation unit (CU) of interest. We have only one
  // CU (object file), so offset is always 0.
  DwarfLine->getOrParseLineTable(LineData, 0);
}

// Returns source line information for a given offset
// using DWARF debug info.
template <class ELFT>
std::string elf::ObjectFile<ELFT>::getLineInfo(InputSectionBase<ELFT> *S,
                                               uintX_t Offset) {
  if (!DwarfLine)
    initializeDwarfLine();

  // The offset to CU is 0.
  const DWARFDebugLine::LineTable *Tbl = DwarfLine->getLineTable(0);
  if (!Tbl)
    return "";

  // Use fake address calcuated by adding section file offset and offset in
  // section. See comments for ObjectInfo class.
  DILineInfo Info;
  Tbl->getFileLineInfoForAddress(
      S->Offset + Offset, nullptr,
      DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath, Info);
  if (Info.Line == 0)
    return "";
  return Info.FileName + ":" + std::to_string(Info.Line);
}

// Returns "(internal)", "foo.a(bar.o)" or "baz.o".
std::string lld::toString(const InputFile *F) {
  if (!F)
    return "(internal)";
  if (!F->ArchiveName.empty())
    return (F->ArchiveName + "(" + F->getName() + ")").str();
  return F->getName();
}

template <class ELFT> static ELFKind getELFKind() {
  if (ELFT::TargetEndianness == support::little)
    return ELFT::Is64Bits ? ELF64LEKind : ELF32LEKind;
  return ELFT::Is64Bits ? ELF64BEKind : ELF32BEKind;
}

template <class ELFT>
ELFFileBase<ELFT>::ELFFileBase(Kind K, MemoryBufferRef MB) : InputFile(K, MB) {
  EKind = getELFKind<ELFT>();
  EMachine = getObj().getHeader()->e_machine;
  OSABI = getObj().getHeader()->e_ident[llvm::ELF::EI_OSABI];
}

template <class ELFT>
typename ELFT::SymRange ELFFileBase<ELFT>::getGlobalSymbols() {
  return makeArrayRef(Symbols.begin() + FirstNonLocal, Symbols.end());
}

template <class ELFT>
uint32_t ELFFileBase<ELFT>::getSectionIndex(const Elf_Sym &Sym) const {
  return check(getObj().getSectionIndex(&Sym, Symbols, SymtabSHNDX));
}

template <class ELFT>
void ELFFileBase<ELFT>::initSymtab(ArrayRef<Elf_Shdr> Sections,
                                   const Elf_Shdr *Symtab) {
  FirstNonLocal = Symtab->sh_info;
  Symbols = check(getObj().symbols(Symtab));
  if (FirstNonLocal == 0 || FirstNonLocal > Symbols.size())
    fatal(toString(this) + ": invalid sh_info in symbol table");

  StringTable = check(getObj().getStringTableForSymtab(*Symtab, Sections));
}

template <class ELFT>
elf::ObjectFile<ELFT>::ObjectFile(MemoryBufferRef M)
    : ELFFileBase<ELFT>(Base::ObjectKind, M) {}

template <class ELFT>
ArrayRef<SymbolBody *> elf::ObjectFile<ELFT>::getNonLocalSymbols() {
  return makeArrayRef(this->SymbolBodies).slice(this->FirstNonLocal);
}

template <class ELFT>
ArrayRef<SymbolBody *> elf::ObjectFile<ELFT>::getLocalSymbols() {
  if (this->SymbolBodies.empty())
    return this->SymbolBodies;
  return makeArrayRef(this->SymbolBodies).slice(1, this->FirstNonLocal - 1);
}

template <class ELFT>
ArrayRef<SymbolBody *> elf::ObjectFile<ELFT>::getSymbols() {
  if (this->SymbolBodies.empty())
    return this->SymbolBodies;
  return makeArrayRef(this->SymbolBodies).slice(1);
}

template <class ELFT>
void elf::ObjectFile<ELFT>::parse(DenseSet<CachedHashStringRef> &ComdatGroups) {
  // Read section and symbol tables.
  initializeSections(ComdatGroups);
  initializeSymbols();
}

// Sections with SHT_GROUP and comdat bits define comdat section groups.
// They are identified and deduplicated by group name. This function
// returns a group name.
template <class ELFT>
StringRef
elf::ObjectFile<ELFT>::getShtGroupSignature(ArrayRef<Elf_Shdr> Sections,
                                            const Elf_Shdr &Sec) {
  if (this->Symbols.empty())
    this->initSymtab(Sections,
                     check(object::getSection<ELFT>(Sections, Sec.sh_link)));
  const Elf_Sym *Sym =
      check(object::getSymbol<ELFT>(this->Symbols, Sec.sh_info));
  return check(Sym->getName(this->StringTable));
}

template <class ELFT>
ArrayRef<typename elf::ObjectFile<ELFT>::Elf_Word>
elf::ObjectFile<ELFT>::getShtGroupEntries(const Elf_Shdr &Sec) {
  const ELFFile<ELFT> &Obj = this->getObj();
  ArrayRef<Elf_Word> Entries =
      check(Obj.template getSectionContentsAsArray<Elf_Word>(&Sec));
  if (Entries.empty() || Entries[0] != GRP_COMDAT)
    fatal(toString(this) + ": unsupported SHT_GROUP format");
  return Entries.slice(1);
}

template <class ELFT>
bool elf::ObjectFile<ELFT>::shouldMerge(const Elf_Shdr &Sec) {
  // We don't merge sections if -O0 (default is -O1). This makes sometimes
  // the linker significantly faster, although the output will be bigger.
  if (Config->Optimize == 0)
    return false;

  // Do not merge sections if generating a relocatable object. It makes
  // the code simpler because we do not need to update relocation addends
  // to reflect changes introduced by merging. Instead of that we write
  // such "merge" sections into separate OutputSections and keep SHF_MERGE
  // / SHF_STRINGS flags and sh_entsize value to be able to perform merging
  // later during a final linking.
  if (Config->Relocatable)
    return false;

  // A mergeable section with size 0 is useless because they don't have
  // any data to merge. A mergeable string section with size 0 can be
  // argued as invalid because it doesn't end with a null character.
  // We'll avoid a mess by handling them as if they were non-mergeable.
  if (Sec.sh_size == 0)
    return false;

  // Check for sh_entsize. The ELF spec is not clear about the zero
  // sh_entsize. It says that "the member [sh_entsize] contains 0 if
  // the section does not hold a table of fixed-size entries". We know
  // that Rust 1.13 produces a string mergeable section with a zero
  // sh_entsize. Here we just accept it rather than being picky about it.
  uintX_t EntSize = Sec.sh_entsize;
  if (EntSize == 0)
    return false;
  if (Sec.sh_size % EntSize)
    fatal(toString(this) +
          ": SHF_MERGE section size must be a multiple of sh_entsize");

  uintX_t Flags = Sec.sh_flags;
  if (!(Flags & SHF_MERGE))
    return false;
  if (Flags & SHF_WRITE)
    fatal(toString(this) + ": writable SHF_MERGE section is not supported");

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
    DenseSet<CachedHashStringRef> &ComdatGroups) {
  ArrayRef<Elf_Shdr> ObjSections = check(this->getObj().sections());
  const ELFFile<ELFT> &Obj = this->getObj();
  uint64_t Size = ObjSections.size();
  Sections.resize(Size);
  unsigned I = -1;
  StringRef SectionStringTable = check(Obj.getSectionStringTable(ObjSections));
  for (const Elf_Shdr &Sec : ObjSections) {
    ++I;
    if (Sections[I] == &InputSection<ELFT>::Discarded)
      continue;

    // SHF_EXCLUDE'ed sections are discarded by the linker. However,
    // if -r is given, we'll let the final link discard such sections.
    // This is compatible with GNU.
    if ((Sec.sh_flags & SHF_EXCLUDE) && !Config->Relocatable) {
      Sections[I] = &InputSection<ELFT>::Discarded;
      continue;
    }

    switch (Sec.sh_type) {
    case SHT_GROUP:
      Sections[I] = &InputSection<ELFT>::Discarded;
      if (ComdatGroups.insert(CachedHashStringRef(
                                  getShtGroupSignature(ObjSections, Sec)))
              .second)
        continue;
      for (uint32_t SecIndex : getShtGroupEntries(Sec)) {
        if (SecIndex >= Size)
          fatal(toString(this) + ": invalid section index in group: " +
                Twine(SecIndex));
        Sections[SecIndex] = &InputSection<ELFT>::Discarded;
      }
      break;
    case SHT_SYMTAB:
      this->initSymtab(ObjSections, &Sec);
      break;
    case SHT_SYMTAB_SHNDX:
      this->SymtabSHNDX = check(Obj.getSHNDXTable(Sec, ObjSections));
      break;
    case SHT_STRTAB:
    case SHT_NULL:
      break;
    default:
      Sections[I] = createInputSection(Sec, SectionStringTable);
    }

    // .ARM.exidx sections have a reverse dependency on the InputSection they
    // have a SHF_LINK_ORDER dependency, this is identified by the sh_link.
    if (Sec.sh_flags & SHF_LINK_ORDER) {
      if (Sec.sh_link >= Sections.size())
        fatal(toString(this) + ": invalid sh_link index: " +
              Twine(Sec.sh_link));
      auto *IS = cast<InputSection<ELFT>>(Sections[Sec.sh_link]);
      IS->DependentSection = Sections[I];
    }
  }
}

template <class ELFT>
InputSectionBase<ELFT> *
elf::ObjectFile<ELFT>::getRelocTarget(const Elf_Shdr &Sec) {
  uint32_t Idx = Sec.sh_info;
  if (Idx >= Sections.size())
    fatal(toString(this) + ": invalid relocated section index: " + Twine(Idx));
  InputSectionBase<ELFT> *Target = Sections[Idx];

  // Strictly speaking, a relocation section must be included in the
  // group of the section it relocates. However, LLVM 3.3 and earlier
  // would fail to do so, so we gracefully handle that case.
  if (Target == &InputSection<ELFT>::Discarded)
    return nullptr;

  if (!Target)
    fatal(toString(this) + ": unsupported relocation reference");
  return Target;
}

template <class ELFT>
InputSectionBase<ELFT> *
elf::ObjectFile<ELFT>::createInputSection(const Elf_Shdr &Sec,
                                          StringRef SectionStringTable) {
  StringRef Name =
      check(this->getObj().getSectionName(&Sec, SectionStringTable));

  switch (Sec.sh_type) {
  case SHT_ARM_ATTRIBUTES:
    // FIXME: ARM meta-data section. Retain the first attribute section
    // we see. The eglibc ARM dynamic loaders require the presence of an
    // attribute section for dlopen to work.
    // In a full implementation we would merge all attribute sections.
    if (In<ELFT>::ARMAttributes == nullptr) {
      In<ELFT>::ARMAttributes = make<InputSection<ELFT>>(this, &Sec, Name);
      return In<ELFT>::ARMAttributes;
    }
    return &InputSection<ELFT>::Discarded;
  case SHT_RELA:
  case SHT_REL: {
    // Find the relocation target section and associate this
    // section with it. Target can be discarded, for example
    // if it is a duplicated member of SHT_GROUP section, we
    // do not create or proccess relocatable sections then.
    InputSectionBase<ELFT> *Target = getRelocTarget(Sec);
    if (!Target)
      return nullptr;

    // This section contains relocation information.
    // If -r is given, we do not interpret or apply relocation
    // but just copy relocation sections to output.
    if (Config->Relocatable)
      return make<InputSection<ELFT>>(this, &Sec, Name);

    if (Target->FirstRelocation)
      fatal(toString(this) +
            ": multiple relocation sections to one section are not supported");
    if (!isa<InputSection<ELFT>>(Target) && !isa<EhInputSection<ELFT>>(Target))
      fatal(toString(this) +
            ": relocations pointing to SHF_MERGE are not supported");

    size_t NumRelocations;
    if (Sec.sh_type == SHT_RELA) {
      ArrayRef<Elf_Rela> Rels = check(this->getObj().relas(&Sec));
      Target->FirstRelocation = Rels.begin();
      NumRelocations = Rels.size();
      Target->AreRelocsRela = true;
    } else {
      ArrayRef<Elf_Rel> Rels = check(this->getObj().rels(&Sec));
      Target->FirstRelocation = Rels.begin();
      NumRelocations = Rels.size();
      Target->AreRelocsRela = false;
    }
    assert(isUInt<31>(NumRelocations));
    Target->NumRelocations = NumRelocations;

    // Relocation sections processed by the linker are usually removed
    // from the output, so returning `nullptr` for the normal case.
    // However, if -emit-relocs is given, we need to leave them in the output.
    // (Some post link analysis tools need this information.)
    if (Config->EmitRelocs)
      return make<InputSection<ELFT>>(this, &Sec, Name);
    return nullptr;
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

  // The linkonce feature is a sort of proto-comdat. Some glibc i386 object
  // files contain definitions of symbol "__x86.get_pc_thunk.bx" in linkonce
  // sections. Drop those sections to avoid duplicate symbol errors.
  // FIXME: This is glibc PR20543, we should remove this hack once that has been
  // fixed for a while.
  if (Name.startswith(".gnu.linkonce."))
    return &InputSection<ELFT>::Discarded;

  // The linker merges EH (exception handling) frames and creates a
  // .eh_frame_hdr section for runtime. So we handle them with a special
  // class. For relocatable outputs, they are just passed through.
  if (Name == ".eh_frame" && !Config->Relocatable)
    return make<EhInputSection<ELFT>>(this, &Sec, Name);

  if (shouldMerge(Sec))
    return make<MergeInputSection<ELFT>>(this, &Sec, Name);
  return make<InputSection<ELFT>>(this, &Sec, Name);
}

template <class ELFT> void elf::ObjectFile<ELFT>::initializeSymbols() {
  SymbolBodies.reserve(this->Symbols.size());
  for (const Elf_Sym &Sym : this->Symbols)
    SymbolBodies.push_back(createSymbolBody(&Sym));
}

template <class ELFT>
InputSectionBase<ELFT> *
elf::ObjectFile<ELFT>::getSection(const Elf_Sym &Sym) const {
  uint32_t Index = this->getSectionIndex(Sym);
  if (Index >= Sections.size())
    fatal(toString(this) + ": invalid section index: " + Twine(Index));
  InputSectionBase<ELFT> *S = Sections[Index];

  // We found that GNU assembler 2.17.50 [FreeBSD] 2007-07-03 could
  // generate broken objects. STT_SECTION/STT_NOTYPE symbols can be
  // associated with SHT_REL[A]/SHT_SYMTAB/SHT_STRTAB sections.
  // In this case it is fine for section to be null here as we do not
  // allocate sections of these types.
  if (!S) {
    if (Index == 0 || Sym.getType() == STT_SECTION ||
        Sym.getType() == STT_NOTYPE)
      return nullptr;
    fatal(toString(this) + ": invalid section index: " + Twine(Index));
  }

  if (S == &InputSection<ELFT>::Discarded)
    return S;
  return S->Repl;
}

template <class ELFT>
SymbolBody *elf::ObjectFile<ELFT>::createSymbolBody(const Elf_Sym *Sym) {
  int Binding = Sym->getBinding();
  InputSectionBase<ELFT> *Sec = getSection(*Sym);

  uint8_t StOther = Sym->st_other;
  uint8_t Type = Sym->getType();
  uintX_t Value = Sym->st_value;
  uintX_t Size = Sym->st_size;

  if (Binding == STB_LOCAL) {
    if (Sym->getType() == STT_FILE)
      SourceFile = check(Sym->getName(this->StringTable));

    if (this->StringTable.size() <= Sym->st_name)
      fatal(toString(this) + ": invalid symbol name offset");

    StringRefZ Name = this->StringTable.data() + Sym->st_name;
    if (Sym->st_shndx == SHN_UNDEF)
      return new (BAlloc)
          Undefined(Name, /*IsLocal=*/true, StOther, Type, this);

    return new (BAlloc) DefinedRegular<ELFT>(Name, /*IsLocal=*/true, StOther,
                                             Type, Value, Size, Sec, this);
  }

  StringRef Name = check(Sym->getName(this->StringTable));

  switch (Sym->st_shndx) {
  case SHN_UNDEF:
    return elf::Symtab<ELFT>::X
        ->addUndefined(Name, /*IsLocal=*/false, Binding, StOther, Type,
                       /*CanOmitFromDynSym=*/false, this)
        ->body();
  case SHN_COMMON:
    if (Value == 0 || Value >= UINT32_MAX)
      fatal(toString(this) + ": common symbol '" + Name +
            "' has invalid alignment: " + Twine(Value));
    return elf::Symtab<ELFT>::X
        ->addCommon(Name, Size, Value, Binding, StOther, Type, this)
        ->body();
  }

  switch (Binding) {
  default:
    fatal(toString(this) + ": unexpected binding: " + Twine(Binding));
  case STB_GLOBAL:
  case STB_WEAK:
  case STB_GNU_UNIQUE:
    if (Sec == &InputSection<ELFT>::Discarded)
      return elf::Symtab<ELFT>::X
          ->addUndefined(Name, /*IsLocal=*/false, Binding, StOther, Type,
                         /*CanOmitFromDynSym=*/false, this)
          ->body();
    return elf::Symtab<ELFT>::X
        ->addRegular(Name, StOther, Type, Value, Size, Binding, Sec, this)
        ->body();
  }
}

template <class ELFT> void ArchiveFile::parse() {
  File = check(Archive::create(MB),
               MB.getBufferIdentifier() + ": failed to parse archive");

  // Read the symbol table to construct Lazy objects.
  for (const Archive::Symbol &Sym : File->symbols())
    Symtab<ELFT>::X->addLazyArchive(this, Sym);
}

// Returns a buffer pointing to a member file containing a given symbol.
std::pair<MemoryBufferRef, uint64_t>
ArchiveFile::getMember(const Archive::Symbol *Sym) {
  Archive::Child C =
      check(Sym->getMember(),
            "could not get the member for symbol " + Sym->getName());

  if (!Seen.insert(C.getChildOffset()).second)
    return {MemoryBufferRef(), 0};

  MemoryBufferRef Ret =
      check(C.getMemoryBufferRef(),
            "could not get the buffer for the member defining symbol " +
                Sym->getName());

  if (C.getParent()->isThin() && Tar)
    Tar->append(relativeToRoot(check(C.getFullName())), Ret.getBuffer());
  if (C.getParent()->isThin())
    return {Ret, 0};
  return {Ret, C.getChildOffset()};
}

template <class ELFT>
SharedFile<ELFT>::SharedFile(MemoryBufferRef M)
    : ELFFileBase<ELFT>(Base::SharedKind, M), AsNeeded(Config->AsNeeded) {}

template <class ELFT>
const typename ELFT::Shdr *
SharedFile<ELFT>::getSection(const Elf_Sym &Sym) const {
  return check(
      this->getObj().getSection(&Sym, this->Symbols, this->SymtabSHNDX));
}

// Partially parse the shared object file so that we can call
// getSoName on this object.
template <class ELFT> void SharedFile<ELFT>::parseSoName() {
  const Elf_Shdr *DynamicSec = nullptr;

  const ELFFile<ELFT> Obj = this->getObj();
  ArrayRef<Elf_Shdr> Sections = check(Obj.sections());
  for (const Elf_Shdr &Sec : Sections) {
    switch (Sec.sh_type) {
    default:
      continue;
    case SHT_DYNSYM:
      this->initSymtab(Sections, &Sec);
      break;
    case SHT_DYNAMIC:
      DynamicSec = &Sec;
      break;
    case SHT_SYMTAB_SHNDX:
      this->SymtabSHNDX = check(Obj.getSHNDXTable(Sec, Sections));
      break;
    case SHT_GNU_versym:
      this->VersymSec = &Sec;
      break;
    case SHT_GNU_verdef:
      this->VerdefSec = &Sec;
      break;
    }
  }

  if (this->VersymSec && this->Symbols.empty())
    error("SHT_GNU_versym should be associated with symbol table");

  // DSOs are identified by soname, and they usually contain
  // DT_SONAME tag in their header. But if they are missing,
  // filenames are used as default sonames.
  SoName = sys::path::filename(this->getName());

  if (!DynamicSec)
    return;

  ArrayRef<Elf_Dyn> Arr =
      check(Obj.template getSectionContentsAsArray<Elf_Dyn>(DynamicSec),
            toString(this) + ": getSectionContentsAsArray failed");
  for (const Elf_Dyn &Dyn : Arr) {
    if (Dyn.d_tag == DT_SONAME) {
      uintX_t Val = Dyn.getVal();
      if (Val >= this->StringTable.size())
        fatal(toString(this) + ": invalid DT_SONAME entry");
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
  const char *Base = this->MB.getBuffer().data();
  Versym = reinterpret_cast<const Elf_Versym *>(Base + VersymSec->sh_offset) +
           this->FirstNonLocal;

  // We cannot determine the largest verdef identifier without inspecting
  // every Elf_Verdef, but both bfd and gold assign verdef identifiers
  // sequentially starting from 1, so we predict that the largest identifier
  // will be VerdefCount.
  unsigned VerdefCount = VerdefSec->sh_info;
  Verdefs.resize(VerdefCount + 1);

  // Build the Verdefs array by following the chain of Elf_Verdef objects
  // from the start of the .gnu.version_d section.
  const char *Verdef = Base + VerdefSec->sh_offset;
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

  Elf_Sym_Range Syms = this->getGlobalSymbols();
  for (const Elf_Sym &Sym : Syms) {
    unsigned VersymIndex = 0;
    if (Versym) {
      VersymIndex = Versym->vs_index;
      ++Versym;
    }
    bool Hidden = VersymIndex & VERSYM_HIDDEN;
    VersymIndex = VersymIndex & ~VERSYM_HIDDEN;

    StringRef Name = check(Sym.getName(this->StringTable));
    if (Sym.isUndefined()) {
      Undefs.push_back(Name);
      continue;
    }

    // Ignore local symbols.
    if (Versym && VersymIndex == VER_NDX_LOCAL)
      continue;

    const Elf_Verdef *V =
        VersymIndex == VER_NDX_GLOBAL ? nullptr : Verdefs[VersymIndex];

    if (!Hidden)
      elf::Symtab<ELFT>::X->addShared(this, Name, Sym, V);

    // Also add the symbol with the versioned name to handle undefined symbols
    // with explicit versions.
    if (V) {
      StringRef VerName = this->StringTable.data() + V->getAux()->vda_name;
      Name = Saver.save(Twine(Name) + "@" + VerName);
      elf::Symtab<ELFT>::X->addShared(this, Name, Sym, V);
    }
  }
}

static ELFKind getBitcodeELFKind(MemoryBufferRef MB) {
  Triple T(check(getBitcodeTargetTriple(MB)));
  if (T.isLittleEndian())
    return T.isArch64Bit() ? ELF64LEKind : ELF32LEKind;
  return T.isArch64Bit() ? ELF64BEKind : ELF32BEKind;
}

static uint8_t getBitcodeMachineKind(MemoryBufferRef MB) {
  Triple T(check(getBitcodeTargetTriple(MB)));
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

static uint8_t mapVisibility(GlobalValue::VisibilityTypes GvVisibility) {
  switch (GvVisibility) {
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
static Symbol *createBitcodeSymbol(const std::vector<bool> &KeptComdats,
                                   const lto::InputFile::Symbol &ObjSym,
                                   BitcodeFile *F) {
  StringRef NameRef = Saver.save(ObjSym.getName());
  uint32_t Flags = ObjSym.getFlags();
  uint32_t Binding = (Flags & BasicSymbolRef::SF_Weak) ? STB_WEAK : STB_GLOBAL;

  uint8_t Type = ObjSym.isTLS() ? STT_TLS : STT_NOTYPE;
  uint8_t Visibility = mapVisibility(ObjSym.getVisibility());
  bool CanOmitFromDynSym = ObjSym.canBeOmittedFromSymbolTable();

  int C = check(ObjSym.getComdatIndex());
  if (C != -1 && !KeptComdats[C])
    return Symtab<ELFT>::X->addUndefined(NameRef, /*IsLocal=*/false, Binding,
                                         Visibility, Type, CanOmitFromDynSym,
                                         F);

  if (Flags & BasicSymbolRef::SF_Undefined)
    return Symtab<ELFT>::X->addUndefined(NameRef, /*IsLocal=*/false, Binding,
                                         Visibility, Type, CanOmitFromDynSym,
                                         F);

  if (Flags & BasicSymbolRef::SF_Common)
    return Symtab<ELFT>::X->addCommon(NameRef, ObjSym.getCommonSize(),
                                      ObjSym.getCommonAlignment(), Binding,
                                      Visibility, STT_OBJECT, F);

  return Symtab<ELFT>::X->addBitcode(NameRef, Binding, Visibility, Type,
                                     CanOmitFromDynSym, F);
}

template <class ELFT>
void BitcodeFile::parse(DenseSet<CachedHashStringRef> &ComdatGroups) {

  // Here we pass a new MemoryBufferRef which is identified by ArchiveName
  // (the fully resolved path of the archive) + member name + offset of the
  // member in the archive.
  // ThinLTO uses the MemoryBufferRef identifier to access its internal
  // data structures and if two archives define two members with the same name,
  // this causes a collision which result in only one of the objects being
  // taken into consideration at LTO time (which very likely causes undefined
  // symbols later in the link stage).
  Obj = check(lto::InputFile::create(MemoryBufferRef(
      MB.getBuffer(), Saver.save(ArchiveName + MB.getBufferIdentifier() +
                                 utostr(OffsetInArchive)))));

  std::vector<bool> KeptComdats;
  for (StringRef S : Obj->getComdatTable()) {
    StringRef N = Saver.save(S);
    KeptComdats.push_back(ComdatGroups.insert(CachedHashStringRef(N)).second);
  }

  for (const lto::InputFile::Symbol &ObjSym : Obj->symbols())
    Symbols.push_back(createBitcodeSymbol<ELFT>(KeptComdats, ObjSym, this));
}

template <template <class> class T>
static InputFile *createELFFile(MemoryBufferRef MB) {
  unsigned char Size;
  unsigned char Endian;
  std::tie(Size, Endian) = getElfArchType(MB.getBuffer());
  if (Endian != ELFDATA2LSB && Endian != ELFDATA2MSB)
    fatal(MB.getBufferIdentifier() + ": invalid data encoding");

  size_t BufSize = MB.getBuffer().size();
  if ((Size == ELFCLASS32 && BufSize < sizeof(Elf32_Ehdr)) ||
      (Size == ELFCLASS64 && BufSize < sizeof(Elf64_Ehdr)))
    fatal(MB.getBufferIdentifier() + ": file is too short");

  InputFile *Obj;
  if (Size == ELFCLASS32 && Endian == ELFDATA2LSB)
    Obj = make<T<ELF32LE>>(MB);
  else if (Size == ELFCLASS32 && Endian == ELFDATA2MSB)
    Obj = make<T<ELF32BE>>(MB);
  else if (Size == ELFCLASS64 && Endian == ELFDATA2LSB)
    Obj = make<T<ELF64LE>>(MB);
  else if (Size == ELFCLASS64 && Endian == ELFDATA2MSB)
    Obj = make<T<ELF64BE>>(MB);
  else
    fatal(MB.getBufferIdentifier() + ": invalid file class");

  if (!Config->FirstElf)
    Config->FirstElf = Obj;
  return Obj;
}

template <class ELFT> void BinaryFile::parse() {
  StringRef Buf = MB.getBuffer();
  ArrayRef<uint8_t> Data =
      makeArrayRef<uint8_t>((const uint8_t *)Buf.data(), Buf.size());

  std::string Filename = MB.getBufferIdentifier();
  std::transform(Filename.begin(), Filename.end(), Filename.begin(),
                 [](char C) { return isalnum(C) ? C : '_'; });
  Filename = "_binary_" + Filename;
  StringRef StartName = Saver.save(Twine(Filename) + "_start");
  StringRef EndName = Saver.save(Twine(Filename) + "_end");
  StringRef SizeName = Saver.save(Twine(Filename) + "_size");

  auto *Section = make<InputSection<ELFT>>(SHF_ALLOC | SHF_WRITE, SHT_PROGBITS,
                                           8, Data, ".data");
  Sections.push_back(Section);

  elf::Symtab<ELFT>::X->addRegular(StartName, STV_DEFAULT, STT_OBJECT, 0, 0,
                                   STB_GLOBAL, Section, nullptr);
  elf::Symtab<ELFT>::X->addRegular(EndName, STV_DEFAULT, STT_OBJECT,
                                   Data.size(), 0, STB_GLOBAL, Section,
                                   nullptr);
  elf::Symtab<ELFT>::X->addRegular(SizeName, STV_DEFAULT, STT_OBJECT,
                                   Data.size(), 0, STB_GLOBAL, nullptr,
                                   nullptr);
}

static bool isBitcode(MemoryBufferRef MB) {
  using namespace sys::fs;
  return identify_magic(MB.getBuffer()) == file_magic::bitcode;
}

InputFile *elf::createObjectFile(MemoryBufferRef MB, StringRef ArchiveName,
                                 uint64_t OffsetInArchive) {
  InputFile *F =
      isBitcode(MB) ? make<BitcodeFile>(MB) : createELFFile<ObjectFile>(MB);
  F->ArchiveName = ArchiveName;
  F->OffsetInArchive = OffsetInArchive;
  return F;
}

InputFile *elf::createSharedFile(MemoryBufferRef MB) {
  return createELFFile<SharedFile>(MB);
}

MemoryBufferRef LazyObjectFile::getBuffer() {
  if (Seen)
    return MemoryBufferRef();
  Seen = true;
  return MB;
}

template <class ELFT> void LazyObjectFile::parse() {
  for (StringRef Sym : getSymbols())
    Symtab<ELFT>::X->addLazyObject(Sym, *this);
}

template <class ELFT> std::vector<StringRef> LazyObjectFile::getElfSymbols() {
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::SymRange Elf_Sym_Range;

  const ELFFile<ELFT> Obj(this->MB.getBuffer());
  ArrayRef<Elf_Shdr> Sections = check(Obj.sections());
  for (const Elf_Shdr &Sec : Sections) {
    if (Sec.sh_type != SHT_SYMTAB)
      continue;
    Elf_Sym_Range Syms = check(Obj.symbols(&Sec));
    uint32_t FirstNonLocal = Sec.sh_info;
    StringRef StringTable = check(Obj.getStringTableForSymtab(Sec, Sections));
    std::vector<StringRef> V;
    for (const Elf_Sym &Sym : Syms.slice(FirstNonLocal))
      if (Sym.st_shndx != SHN_UNDEF)
        V.push_back(check(Sym.getName(StringTable)));
    return V;
  }
  return {};
}

std::vector<StringRef> LazyObjectFile::getBitcodeSymbols() {
  std::unique_ptr<lto::InputFile> Obj = check(lto::InputFile::create(this->MB));
  std::vector<StringRef> V;
  for (const lto::InputFile::Symbol &Sym : Obj->symbols())
    if (!(Sym.getFlags() & BasicSymbolRef::SF_Undefined))
      V.push_back(Saver.save(Sym.getName()));
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

template void BitcodeFile::parse<ELF32LE>(DenseSet<CachedHashStringRef> &);
template void BitcodeFile::parse<ELF32BE>(DenseSet<CachedHashStringRef> &);
template void BitcodeFile::parse<ELF64LE>(DenseSet<CachedHashStringRef> &);
template void BitcodeFile::parse<ELF64BE>(DenseSet<CachedHashStringRef> &);

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

template void BinaryFile::parse<ELF32LE>();
template void BinaryFile::parse<ELF32BE>();
template void BinaryFile::parse<ELF64LE>();
template void BinaryFile::parse<ELF64BE>();
