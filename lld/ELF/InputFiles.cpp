//===- InputFiles.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Driver.h"
#include "InputSection.h"
#include "LinkerScript.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/LTO/LTO.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/ARMAttributeParser.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TarWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace llvm::support::endian;

using namespace lld;
using namespace lld::elf;

bool InputFile::IsInGroup;
uint32_t InputFile::NextGroupId;
std::vector<BinaryFile *> elf::BinaryFiles;
std::vector<BitcodeFile *> elf::BitcodeFiles;
std::vector<LazyObjFile *> elf::LazyObjFiles;
std::vector<InputFile *> elf::ObjectFiles;
std::vector<SharedFile *> elf::SharedFiles;

std::unique_ptr<TarWriter> elf::Tar;

static ELFKind getELFKind(MemoryBufferRef MB, StringRef ArchiveName) {
  unsigned char Size;
  unsigned char Endian;
  std::tie(Size, Endian) = getElfArchType(MB.getBuffer());

  auto Fatal = [&](StringRef Msg) {
    StringRef Filename = MB.getBufferIdentifier();
    if (ArchiveName.empty())
      fatal(Filename + ": " + Msg);
    else
      fatal(ArchiveName + "(" + Filename + "): " + Msg);
  };

  if (!MB.getBuffer().startswith(ElfMagic))
    Fatal("not an ELF file");
  if (Endian != ELFDATA2LSB && Endian != ELFDATA2MSB)
    Fatal("corrupted ELF file: invalid data encoding");
  if (Size != ELFCLASS32 && Size != ELFCLASS64)
    Fatal("corrupted ELF file: invalid file class");

  size_t BufSize = MB.getBuffer().size();
  if ((Size == ELFCLASS32 && BufSize < sizeof(Elf32_Ehdr)) ||
      (Size == ELFCLASS64 && BufSize < sizeof(Elf64_Ehdr)))
    Fatal("corrupted ELF file: file is too short");

  if (Size == ELFCLASS32)
    return (Endian == ELFDATA2LSB) ? ELF32LEKind : ELF32BEKind;
  return (Endian == ELFDATA2LSB) ? ELF64LEKind : ELF64BEKind;
}

InputFile::InputFile(Kind K, MemoryBufferRef M)
    : MB(M), GroupId(NextGroupId), FileKind(K) {
  // All files within the same --{start,end}-group get the same group ID.
  // Otherwise, a new file will get a new group ID.
  if (!IsInGroup)
    ++NextGroupId;
}

Optional<MemoryBufferRef> elf::readFile(StringRef Path) {
  // The --chroot option changes our virtual root directory.
  // This is useful when you are dealing with files created by --reproduce.
  if (!Config->Chroot.empty() && Path.startswith("/"))
    Path = Saver.save(Config->Chroot + Path);

  log(Path);

  auto MBOrErr = MemoryBuffer::getFile(Path, -1, false);
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

// All input object files must be for the same architecture
// (e.g. it does not make sense to link x86 object files with
// MIPS object files.) This function checks for that error.
static bool isCompatible(InputFile *File) {
  if (!File->isElf() && !isa<BitcodeFile>(File))
    return true;

  if (File->EKind == Config->EKind && File->EMachine == Config->EMachine) {
    if (Config->EMachine != EM_MIPS)
      return true;
    if (isMipsN32Abi(File) == Config->MipsN32Abi)
      return true;
  }

  if (!Config->Emulation.empty()) {
    error(toString(File) + " is incompatible with " + Config->Emulation);
  } else {
    InputFile *Existing;
    if (!ObjectFiles.empty())
      Existing = ObjectFiles[0];
    else if (!SharedFiles.empty())
      Existing = SharedFiles[0];
    else
      Existing = BitcodeFiles[0];

    error(toString(File) + " is incompatible with " + toString(Existing));
  }

  return false;
}

template <class ELFT> static void doParseFile(InputFile *File) {
  if (!isCompatible(File))
    return;

  // Binary file
  if (auto *F = dyn_cast<BinaryFile>(File)) {
    BinaryFiles.push_back(F);
    F->parse();
    return;
  }

  // .a file
  if (auto *F = dyn_cast<ArchiveFile>(File)) {
    F->parse();
    return;
  }

  // Lazy object file
  if (auto *F = dyn_cast<LazyObjFile>(File)) {
    LazyObjFiles.push_back(F);
    F->parse<ELFT>();
    return;
  }

  if (Config->Trace)
    message(toString(File));

  // .so file
  if (auto *F = dyn_cast<SharedFile>(File)) {
    F->parse<ELFT>();
    return;
  }

  // LLVM bitcode file
  if (auto *F = dyn_cast<BitcodeFile>(File)) {
    BitcodeFiles.push_back(F);
    F->parse<ELFT>();
    return;
  }

  // Regular object file
  ObjectFiles.push_back(File);
  cast<ObjFile<ELFT>>(File)->parse();
}

// Add symbols in File to the symbol table.
void elf::parseFile(InputFile *File) {
  switch (Config->EKind) {
  case ELF32LEKind:
    doParseFile<ELF32LE>(File);
    return;
  case ELF32BEKind:
    doParseFile<ELF32BE>(File);
    return;
  case ELF64LEKind:
    doParseFile<ELF64LE>(File);
    return;
  case ELF64BEKind:
    doParseFile<ELF64BE>(File);
    return;
  default:
    llvm_unreachable("unknown ELFT");
  }
}

// Concatenates arguments to construct a string representing an error location.
static std::string createFileLineMsg(StringRef Path, unsigned Line) {
  std::string Filename = path::filename(Path);
  std::string Lineno = ":" + std::to_string(Line);
  if (Filename == Path)
    return Filename + Lineno;
  return Filename + Lineno + " (" + Path.str() + Lineno + ")";
}

template <class ELFT>
static std::string getSrcMsgAux(ObjFile<ELFT> &File, const Symbol &Sym,
                                InputSectionBase &Sec, uint64_t Offset) {
  // In DWARF, functions and variables are stored to different places.
  // First, lookup a function for a given offset.
  if (Optional<DILineInfo> Info = File.getDILineInfo(&Sec, Offset))
    return createFileLineMsg(Info->FileName, Info->Line);

  // If it failed, lookup again as a variable.
  if (Optional<std::pair<std::string, unsigned>> FileLine =
          File.getVariableLoc(Sym.getName()))
    return createFileLineMsg(FileLine->first, FileLine->second);

  // File.SourceFile contains STT_FILE symbol, and that is a last resort.
  return File.SourceFile;
}

std::string InputFile::getSrcMsg(const Symbol &Sym, InputSectionBase &Sec,
                                 uint64_t Offset) {
  if (kind() != ObjKind)
    return "";
  switch (Config->EKind) {
  default:
    llvm_unreachable("Invalid kind");
  case ELF32LEKind:
    return getSrcMsgAux(cast<ObjFile<ELF32LE>>(*this), Sym, Sec, Offset);
  case ELF32BEKind:
    return getSrcMsgAux(cast<ObjFile<ELF32BE>>(*this), Sym, Sec, Offset);
  case ELF64LEKind:
    return getSrcMsgAux(cast<ObjFile<ELF64LE>>(*this), Sym, Sec, Offset);
  case ELF64BEKind:
    return getSrcMsgAux(cast<ObjFile<ELF64BE>>(*this), Sym, Sec, Offset);
  }
}

template <class ELFT> void ObjFile<ELFT>::initializeDwarf() {
  Dwarf = llvm::make_unique<DWARFContext>(make_unique<LLDDwarfObj<ELFT>>(this));
  for (std::unique_ptr<DWARFUnit> &CU : Dwarf->compile_units()) {
    auto Report = [](Error Err) {
      handleAllErrors(std::move(Err),
                      [](ErrorInfoBase &Info) { warn(Info.message()); });
    };
    Expected<const DWARFDebugLine::LineTable *> ExpectedLT =
        Dwarf->getLineTableForUnit(CU.get(), Report);
    const DWARFDebugLine::LineTable *LT = nullptr;
    if (ExpectedLT)
      LT = *ExpectedLT;
    else
      Report(ExpectedLT.takeError());
    if (!LT)
      continue;
    LineTables.push_back(LT);

    // Loop over variable records and insert them to VariableLoc.
    for (const auto &Entry : CU->dies()) {
      DWARFDie Die(CU.get(), &Entry);
      // Skip all tags that are not variables.
      if (Die.getTag() != dwarf::DW_TAG_variable)
        continue;

      // Skip if a local variable because we don't need them for generating
      // error messages. In general, only non-local symbols can fail to be
      // linked.
      if (!dwarf::toUnsigned(Die.find(dwarf::DW_AT_external), 0))
        continue;

      // Get the source filename index for the variable.
      unsigned File = dwarf::toUnsigned(Die.find(dwarf::DW_AT_decl_file), 0);
      if (!LT->hasFileAtIndex(File))
        continue;

      // Get the line number on which the variable is declared.
      unsigned Line = dwarf::toUnsigned(Die.find(dwarf::DW_AT_decl_line), 0);

      // Here we want to take the variable name to add it into VariableLoc.
      // Variable can have regular and linkage name associated. At first, we try
      // to get linkage name as it can be different, for example when we have
      // two variables in different namespaces of the same object. Use common
      // name otherwise, but handle the case when it also absent in case if the
      // input object file lacks some debug info.
      StringRef Name =
          dwarf::toString(Die.find(dwarf::DW_AT_linkage_name),
                          dwarf::toString(Die.find(dwarf::DW_AT_name), ""));
      if (!Name.empty())
        VariableLoc.insert({Name, {LT, File, Line}});
    }
  }
}

// Returns the pair of file name and line number describing location of data
// object (variable, array, etc) definition.
template <class ELFT>
Optional<std::pair<std::string, unsigned>>
ObjFile<ELFT>::getVariableLoc(StringRef Name) {
  llvm::call_once(InitDwarfLine, [this]() { initializeDwarf(); });

  // Return if we have no debug information about data object.
  auto It = VariableLoc.find(Name);
  if (It == VariableLoc.end())
    return None;

  // Take file name string from line table.
  std::string FileName;
  if (!It->second.LT->getFileNameByIndex(
          It->second.File, nullptr,
          DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath, FileName))
    return None;

  return std::make_pair(FileName, It->second.Line);
}

// Returns source line information for a given offset
// using DWARF debug info.
template <class ELFT>
Optional<DILineInfo> ObjFile<ELFT>::getDILineInfo(InputSectionBase *S,
                                                  uint64_t Offset) {
  llvm::call_once(InitDwarfLine, [this]() { initializeDwarf(); });

  // Detect SectionIndex for specified section.
  uint64_t SectionIndex = object::SectionedAddress::UndefSection;
  ArrayRef<InputSectionBase *> Sections = S->File->getSections();
  for (uint64_t CurIndex = 0; CurIndex < Sections.size(); ++CurIndex) {
    if (S == Sections[CurIndex]) {
      SectionIndex = CurIndex;
      break;
    }
  }

  // Use fake address calcuated by adding section file offset and offset in
  // section. See comments for ObjectInfo class.
  DILineInfo Info;
  for (const llvm::DWARFDebugLine::LineTable *LT : LineTables) {
    if (LT->getFileLineInfoForAddress(
            {S->getOffsetInFile() + Offset, SectionIndex}, nullptr,
            DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath, Info))
      return Info;
  }
  return None;
}

// Returns "<internal>", "foo.a(bar.o)" or "baz.o".
std::string lld::toString(const InputFile *F) {
  if (!F)
    return "<internal>";

  if (F->ToStringCache.empty()) {
    if (F->ArchiveName.empty())
      F->ToStringCache = F->getName();
    else
      F->ToStringCache = (F->ArchiveName + "(" + F->getName() + ")").str();
  }
  return F->ToStringCache;
}

ELFFileBase::ELFFileBase(Kind K, MemoryBufferRef MB) : InputFile(K, MB) {
  EKind = getELFKind(MB, "");

  switch (EKind) {
  case ELF32LEKind:
    init<ELF32LE>();
    break;
  case ELF32BEKind:
    init<ELF32BE>();
    break;
  case ELF64LEKind:
    init<ELF64LE>();
    break;
  case ELF64BEKind:
    init<ELF64BE>();
    break;
  default:
    llvm_unreachable("getELFKind");
  }
}

template <typename Elf_Shdr>
static const Elf_Shdr *findSection(ArrayRef<Elf_Shdr> Sections, uint32_t Type) {
  for (const Elf_Shdr &Sec : Sections)
    if (Sec.sh_type == Type)
      return &Sec;
  return nullptr;
}

template <class ELFT> void ELFFileBase::init() {
  using Elf_Shdr = typename ELFT::Shdr;
  using Elf_Sym = typename ELFT::Sym;

  // Initialize trivial attributes.
  const ELFFile<ELFT> &Obj = getObj<ELFT>();
  EMachine = Obj.getHeader()->e_machine;
  OSABI = Obj.getHeader()->e_ident[llvm::ELF::EI_OSABI];
  ABIVersion = Obj.getHeader()->e_ident[llvm::ELF::EI_ABIVERSION];

  ArrayRef<Elf_Shdr> Sections = CHECK(Obj.sections(), this);

  // Find a symbol table.
  bool IsDSO =
      (identify_magic(MB.getBuffer()) == file_magic::elf_shared_object);
  const Elf_Shdr *SymtabSec =
      findSection(Sections, IsDSO ? SHT_DYNSYM : SHT_SYMTAB);

  if (!SymtabSec)
    return;

  // Initialize members corresponding to a symbol table.
  FirstGlobal = SymtabSec->sh_info;

  ArrayRef<Elf_Sym> ESyms = CHECK(Obj.symbols(SymtabSec), this);
  if (FirstGlobal == 0 || FirstGlobal > ESyms.size())
    fatal(toString(this) + ": invalid sh_info in symbol table");

  ELFSyms = reinterpret_cast<const void *>(ESyms.data());
  NumELFSyms = ESyms.size();
  StringTable = CHECK(Obj.getStringTableForSymtab(*SymtabSec, Sections), this);
}

template <class ELFT>
uint32_t ObjFile<ELFT>::getSectionIndex(const Elf_Sym &Sym) const {
  return CHECK(
      this->getObj().getSectionIndex(&Sym, getELFSyms<ELFT>(), ShndxTable),
      this);
}

template <class ELFT> ArrayRef<Symbol *> ObjFile<ELFT>::getLocalSymbols() {
  if (this->Symbols.empty())
    return {};
  return makeArrayRef(this->Symbols).slice(1, this->FirstGlobal - 1);
}

template <class ELFT> ArrayRef<Symbol *> ObjFile<ELFT>::getGlobalSymbols() {
  return makeArrayRef(this->Symbols).slice(this->FirstGlobal);
}

template <class ELFT> void ObjFile<ELFT>::parse(bool IgnoreComdats) {
  // Read a section table. JustSymbols is usually false.
  if (this->JustSymbols)
    initializeJustSymbols();
  else
    initializeSections(IgnoreComdats);

  // Read a symbol table.
  initializeSymbols();
}

// Sections with SHT_GROUP and comdat bits define comdat section groups.
// They are identified and deduplicated by group name. This function
// returns a group name.
template <class ELFT>
StringRef ObjFile<ELFT>::getShtGroupSignature(ArrayRef<Elf_Shdr> Sections,
                                              const Elf_Shdr &Sec) {
  const Elf_Sym *Sym =
      CHECK(object::getSymbol<ELFT>(this->getELFSyms<ELFT>(), Sec.sh_info), this);
  StringRef Signature = CHECK(Sym->getName(this->StringTable), this);

  // As a special case, if a symbol is a section symbol and has no name,
  // we use a section name as a signature.
  //
  // Such SHT_GROUP sections are invalid from the perspective of the ELF
  // standard, but GNU gold 1.14 (the newest version as of July 2017) or
  // older produce such sections as outputs for the -r option, so we need
  // a bug-compatibility.
  if (Signature.empty() && Sym->getType() == STT_SECTION)
    return getSectionName(Sec);
  return Signature;
}

template <class ELFT> bool ObjFile<ELFT>::shouldMerge(const Elf_Shdr &Sec) {
  // On a regular link we don't merge sections if -O0 (default is -O1). This
  // sometimes makes the linker significantly faster, although the output will
  // be bigger.
  //
  // Doing the same for -r would create a problem as it would combine sections
  // with different sh_entsize. One option would be to just copy every SHF_MERGE
  // section as is to the output. While this would produce a valid ELF file with
  // usable SHF_MERGE sections, tools like (llvm-)?dwarfdump get confused when
  // they see two .debug_str. We could have separate logic for combining
  // SHF_MERGE sections based both on their name and sh_entsize, but that seems
  // to be more trouble than it is worth. Instead, we just use the regular (-O1)
  // logic for -r.
  if (Config->Optimize == 0 && !Config->Relocatable)
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
  uint64_t EntSize = Sec.sh_entsize;
  if (EntSize == 0)
    return false;
  if (Sec.sh_size % EntSize)
    fatal(toString(this) +
          ": SHF_MERGE section size must be a multiple of sh_entsize");

  uint64_t Flags = Sec.sh_flags;
  if (!(Flags & SHF_MERGE))
    return false;
  if (Flags & SHF_WRITE)
    fatal(toString(this) + ": writable SHF_MERGE section is not supported");

  return true;
}

// This is for --just-symbols.
//
// --just-symbols is a very minor feature that allows you to link your
// output against other existing program, so that if you load both your
// program and the other program into memory, your output can refer the
// other program's symbols.
//
// When the option is given, we link "just symbols". The section table is
// initialized with null pointers.
template <class ELFT> void ObjFile<ELFT>::initializeJustSymbols() {
  ArrayRef<Elf_Shdr> Sections = CHECK(this->getObj().sections(), this);
  this->Sections.resize(Sections.size());
}

// An ELF object file may contain a `.deplibs` section. If it exists, the
// section contains a list of library specifiers such as `m` for libm. This
// function resolves a given name by finding the first matching library checking
// the various ways that a library can be specified to LLD. This ELF extension
// is a form of autolinking and is called `dependent libraries`. It is currently
// unique to LLVM and lld.
static void addDependentLibrary(StringRef Specifier, const InputFile *F) {
  if (!Config->DependentLibraries)
    return;
  if (fs::exists(Specifier))
    Driver->addFile(Specifier, /*WithLOption=*/false);
  else if (Optional<std::string> S = findFromSearchPaths(Specifier))
    Driver->addFile(*S, /*WithLOption=*/true);
  else if (Optional<std::string> S = searchLibraryBaseName(Specifier))
    Driver->addFile(*S, /*WithLOption=*/true);
  else
    error(toString(F) +
          ": unable to find library from dependent library specifier: " +
          Specifier);
}

template <class ELFT>
void ObjFile<ELFT>::initializeSections(bool IgnoreComdats) {
  const ELFFile<ELFT> &Obj = this->getObj();

  ArrayRef<Elf_Shdr> ObjSections = CHECK(Obj.sections(), this);
  uint64_t Size = ObjSections.size();
  this->Sections.resize(Size);
  this->SectionStringTable =
      CHECK(Obj.getSectionStringTable(ObjSections), this);

  for (size_t I = 0, E = ObjSections.size(); I < E; I++) {
    if (this->Sections[I] == &InputSection::Discarded)
      continue;
    const Elf_Shdr &Sec = ObjSections[I];

    if (Sec.sh_type == ELF::SHT_LLVM_CALL_GRAPH_PROFILE)
      CGProfile =
          check(Obj.template getSectionContentsAsArray<Elf_CGProfile>(&Sec));

    // SHF_EXCLUDE'ed sections are discarded by the linker. However,
    // if -r is given, we'll let the final link discard such sections.
    // This is compatible with GNU.
    if ((Sec.sh_flags & SHF_EXCLUDE) && !Config->Relocatable) {
      if (Sec.sh_type == SHT_LLVM_ADDRSIG) {
        // We ignore the address-significance table if we know that the object
        // file was created by objcopy or ld -r. This is because these tools
        // will reorder the symbols in the symbol table, invalidating the data
        // in the address-significance table, which refers to symbols by index.
        if (Sec.sh_link != 0)
          this->AddrsigSec = &Sec;
        else if (Config->ICF == ICFLevel::Safe)
          warn(toString(this) + ": --icf=safe is incompatible with object "
                                "files created using objcopy or ld -r");
      }
      this->Sections[I] = &InputSection::Discarded;
      continue;
    }

    switch (Sec.sh_type) {
    case SHT_GROUP: {
      // De-duplicate section groups by their signatures.
      StringRef Signature = getShtGroupSignature(ObjSections, Sec);
      this->Sections[I] = &InputSection::Discarded;


      ArrayRef<Elf_Word> Entries =
          CHECK(Obj.template getSectionContentsAsArray<Elf_Word>(&Sec), this);
      if (Entries.empty())
        fatal(toString(this) + ": empty SHT_GROUP");

      // The first word of a SHT_GROUP section contains flags. Currently,
      // the standard defines only "GRP_COMDAT" flag for the COMDAT group.
      // An group with the empty flag doesn't define anything; such sections
      // are just skipped.
      if (Entries[0] == 0)
        continue;

      if (Entries[0] != GRP_COMDAT)
        fatal(toString(this) + ": unsupported SHT_GROUP format");

      bool IsNew =
          IgnoreComdats ||
          Symtab->ComdatGroups.try_emplace(CachedHashStringRef(Signature), this)
              .second;
      if (IsNew) {
        if (Config->Relocatable)
          this->Sections[I] = createInputSection(Sec);
        continue;
      }

      // Otherwise, discard group members.
      for (uint32_t SecIndex : Entries.slice(1)) {
        if (SecIndex >= Size)
          fatal(toString(this) +
                ": invalid section index in group: " + Twine(SecIndex));
        this->Sections[SecIndex] = &InputSection::Discarded;
      }
      break;
    }
    case SHT_SYMTAB_SHNDX:
      ShndxTable = CHECK(Obj.getSHNDXTable(Sec, ObjSections), this);
      break;
    case SHT_SYMTAB:
    case SHT_STRTAB:
    case SHT_NULL:
      break;
    default:
      this->Sections[I] = createInputSection(Sec);
    }

    // .ARM.exidx sections have a reverse dependency on the InputSection they
    // have a SHF_LINK_ORDER dependency, this is identified by the sh_link.
    if (Sec.sh_flags & SHF_LINK_ORDER) {
      InputSectionBase *LinkSec = nullptr;
      if (Sec.sh_link < this->Sections.size())
        LinkSec = this->Sections[Sec.sh_link];
      if (!LinkSec)
        fatal(toString(this) +
              ": invalid sh_link index: " + Twine(Sec.sh_link));

      InputSection *IS = cast<InputSection>(this->Sections[I]);
      LinkSec->DependentSections.push_back(IS);
      if (!isa<InputSection>(LinkSec))
        error("a section " + IS->Name +
              " with SHF_LINK_ORDER should not refer a non-regular "
              "section: " +
              toString(LinkSec));
    }
  }
}

// For ARM only, to set the EF_ARM_ABI_FLOAT_SOFT or EF_ARM_ABI_FLOAT_HARD
// flag in the ELF Header we need to look at Tag_ABI_VFP_args to find out how
// the input objects have been compiled.
static void updateARMVFPArgs(const ARMAttributeParser &Attributes,
                             const InputFile *F) {
  if (!Attributes.hasAttribute(ARMBuildAttrs::ABI_VFP_args))
    // If an ABI tag isn't present then it is implicitly given the value of 0
    // which maps to ARMBuildAttrs::BaseAAPCS. However many assembler files,
    // including some in glibc that don't use FP args (and should have value 3)
    // don't have the attribute so we do not consider an implicit value of 0
    // as a clash.
    return;

  unsigned VFPArgs = Attributes.getAttributeValue(ARMBuildAttrs::ABI_VFP_args);
  ARMVFPArgKind Arg;
  switch (VFPArgs) {
  case ARMBuildAttrs::BaseAAPCS:
    Arg = ARMVFPArgKind::Base;
    break;
  case ARMBuildAttrs::HardFPAAPCS:
    Arg = ARMVFPArgKind::VFP;
    break;
  case ARMBuildAttrs::ToolChainFPPCS:
    // Tool chain specific convention that conforms to neither AAPCS variant.
    Arg = ARMVFPArgKind::ToolChain;
    break;
  case ARMBuildAttrs::CompatibleFPAAPCS:
    // Object compatible with all conventions.
    return;
  default:
    error(toString(F) + ": unknown Tag_ABI_VFP_args value: " + Twine(VFPArgs));
    return;
  }
  // Follow ld.bfd and error if there is a mix of calling conventions.
  if (Config->ARMVFPArgs != Arg && Config->ARMVFPArgs != ARMVFPArgKind::Default)
    error(toString(F) + ": incompatible Tag_ABI_VFP_args");
  else
    Config->ARMVFPArgs = Arg;
}

// The ARM support in lld makes some use of instructions that are not available
// on all ARM architectures. Namely:
// - Use of BLX instruction for interworking between ARM and Thumb state.
// - Use of the extended Thumb branch encoding in relocation.
// - Use of the MOVT/MOVW instructions in Thumb Thunks.
// The ARM Attributes section contains information about the architecture chosen
// at compile time. We follow the convention that if at least one input object
// is compiled with an architecture that supports these features then lld is
// permitted to use them.
static void updateSupportedARMFeatures(const ARMAttributeParser &Attributes) {
  if (!Attributes.hasAttribute(ARMBuildAttrs::CPU_arch))
    return;
  auto Arch = Attributes.getAttributeValue(ARMBuildAttrs::CPU_arch);
  switch (Arch) {
  case ARMBuildAttrs::Pre_v4:
  case ARMBuildAttrs::v4:
  case ARMBuildAttrs::v4T:
    // Architectures prior to v5 do not support BLX instruction
    break;
  case ARMBuildAttrs::v5T:
  case ARMBuildAttrs::v5TE:
  case ARMBuildAttrs::v5TEJ:
  case ARMBuildAttrs::v6:
  case ARMBuildAttrs::v6KZ:
  case ARMBuildAttrs::v6K:
    Config->ARMHasBlx = true;
    // Architectures used in pre-Cortex processors do not support
    // The J1 = 1 J2 = 1 Thumb branch range extension, with the exception
    // of Architecture v6T2 (arm1156t2-s and arm1156t2f-s) that do.
    break;
  default:
    // All other Architectures have BLX and extended branch encoding
    Config->ARMHasBlx = true;
    Config->ARMJ1J2BranchEncoding = true;
    if (Arch != ARMBuildAttrs::v6_M && Arch != ARMBuildAttrs::v6S_M)
      // All Architectures used in Cortex processors with the exception
      // of v6-M and v6S-M have the MOVT and MOVW instructions.
      Config->ARMHasMovtMovw = true;
    break;
  }
}

// If a source file is compiled with x86 hardware-assisted call flow control
// enabled, the generated object file contains feature flags indicating that
// fact. This function reads the feature flags and returns it.
//
// Essentially we want to read a single 32-bit value in this function, but this
// function is rather complicated because the value is buried deep inside a
// .note.gnu.property section.
//
// The section consists of one or more NOTE records. Each NOTE record consists
// of zero or more type-length-value fields. We want to find a field of a
// certain type. It seems a bit too much to just store a 32-bit value, perhaps
// the ABI is unnecessarily complicated.
template <class ELFT>
static uint32_t readAndFeatures(ObjFile<ELFT> *Obj, ArrayRef<uint8_t> Data) {
  using Elf_Nhdr = typename ELFT::Nhdr;
  using Elf_Note = typename ELFT::Note;

  uint32_t FeaturesSet = 0;
  while (!Data.empty()) {
    // Read one NOTE record.
    if (Data.size() < sizeof(Elf_Nhdr))
      fatal(toString(Obj) + ": .note.gnu.property: section too short");

    auto *Nhdr = reinterpret_cast<const Elf_Nhdr *>(Data.data());
    if (Data.size() < Nhdr->getSize())
      fatal(toString(Obj) + ": .note.gnu.property: section too short");

    Elf_Note Note(*Nhdr);
    if (Nhdr->n_type != NT_GNU_PROPERTY_TYPE_0 || Note.getName() != "GNU") {
      Data = Data.slice(Nhdr->getSize());
      continue;
    }

    uint32_t FeatureAndType = Config->EMachine == EM_AARCH64
                                  ? GNU_PROPERTY_AARCH64_FEATURE_1_AND
                                  : GNU_PROPERTY_X86_FEATURE_1_AND;

    // Read a body of a NOTE record, which consists of type-length-value fields.
    ArrayRef<uint8_t> Desc = Note.getDesc();
    while (!Desc.empty()) {
      if (Desc.size() < 8)
        fatal(toString(Obj) + ": .note.gnu.property: section too short");

      uint32_t Type = read32le(Desc.data());
      uint32_t Size = read32le(Desc.data() + 4);

      if (Type == FeatureAndType) {
        // We found a FEATURE_1_AND field. There may be more than one of these
        // in a .note.gnu.propery section, for a relocatable object we
        // accumulate the bits set.
        FeaturesSet |= read32le(Desc.data() + 8);
      }

      // On 64-bit, a payload may be followed by a 4-byte padding to make its
      // size a multiple of 8.
      if (ELFT::Is64Bits)
        Size = alignTo(Size, 8);

      Desc = Desc.slice(Size + 8); // +8 for Type and Size
    }

    // Go to next NOTE record to look for more FEATURE_1_AND descriptions.
    Data = Data.slice(Nhdr->getSize());
  }

  return FeaturesSet;
}

template <class ELFT>
InputSectionBase *ObjFile<ELFT>::getRelocTarget(const Elf_Shdr &Sec) {
  uint32_t Idx = Sec.sh_info;
  if (Idx >= this->Sections.size())
    fatal(toString(this) + ": invalid relocated section index: " + Twine(Idx));
  InputSectionBase *Target = this->Sections[Idx];

  // Strictly speaking, a relocation section must be included in the
  // group of the section it relocates. However, LLVM 3.3 and earlier
  // would fail to do so, so we gracefully handle that case.
  if (Target == &InputSection::Discarded)
    return nullptr;

  if (!Target)
    fatal(toString(this) + ": unsupported relocation reference");
  return Target;
}

// Create a regular InputSection class that has the same contents
// as a given section.
static InputSection *toRegularSection(MergeInputSection *Sec) {
  return make<InputSection>(Sec->File, Sec->Flags, Sec->Type, Sec->Alignment,
                            Sec->data(), Sec->Name);
}

template <class ELFT>
InputSectionBase *ObjFile<ELFT>::createInputSection(const Elf_Shdr &Sec) {
  StringRef Name = getSectionName(Sec);

  switch (Sec.sh_type) {
  case SHT_ARM_ATTRIBUTES: {
    if (Config->EMachine != EM_ARM)
      break;
    ARMAttributeParser Attributes;
    ArrayRef<uint8_t> Contents = check(this->getObj().getSectionContents(&Sec));
    Attributes.Parse(Contents, /*isLittle*/ Config->EKind == ELF32LEKind);
    updateSupportedARMFeatures(Attributes);
    updateARMVFPArgs(Attributes, this);

    // FIXME: Retain the first attribute section we see. The eglibc ARM
    // dynamic loaders require the presence of an attribute section for dlopen
    // to work. In a full implementation we would merge all attribute sections.
    if (In.ARMAttributes == nullptr) {
      In.ARMAttributes = make<InputSection>(*this, Sec, Name);
      return In.ARMAttributes;
    }
    return &InputSection::Discarded;
  }
  case SHT_LLVM_DEPENDENT_LIBRARIES: {
    if (Config->Relocatable)
      break;
    ArrayRef<char> Data =
        CHECK(this->getObj().template getSectionContentsAsArray<char>(&Sec), this);
    if (!Data.empty() && Data.back() != '\0') {
      error(toString(this) +
            ": corrupted dependent libraries section (unterminated string): " +
            Name);
      return &InputSection::Discarded;
    }
    for (const char *D = Data.begin(), *E = Data.end(); D < E;) {
      StringRef S(D);
      addDependentLibrary(S, this);
      D += S.size() + 1;
    }
    return &InputSection::Discarded;
  }
  case SHT_RELA:
  case SHT_REL: {
    // Find a relocation target section and associate this section with that.
    // Target may have been discarded if it is in a different section group
    // and the group is discarded, even though it's a violation of the
    // spec. We handle that situation gracefully by discarding dangling
    // relocation sections.
    InputSectionBase *Target = getRelocTarget(Sec);
    if (!Target)
      return nullptr;

    // This section contains relocation information.
    // If -r is given, we do not interpret or apply relocation
    // but just copy relocation sections to output.
    if (Config->Relocatable) {
      InputSection *RelocSec = make<InputSection>(*this, Sec, Name);
      // We want to add a dependency to target, similar like we do for
      // -emit-relocs below. This is useful for the case when linker script
      // contains the "/DISCARD/". It is perhaps uncommon to use a script with
      // -r, but we faced it in the Linux kernel and have to handle such case
      // and not to crash.
      Target->DependentSections.push_back(RelocSec);
      return RelocSec;
    }

    if (Target->FirstRelocation)
      fatal(toString(this) +
            ": multiple relocation sections to one section are not supported");

    // ELF spec allows mergeable sections with relocations, but they are
    // rare, and it is in practice hard to merge such sections by contents,
    // because applying relocations at end of linking changes section
    // contents. So, we simply handle such sections as non-mergeable ones.
    // Degrading like this is acceptable because section merging is optional.
    if (auto *MS = dyn_cast<MergeInputSection>(Target)) {
      Target = toRegularSection(MS);
      this->Sections[Sec.sh_info] = Target;
    }

    if (Sec.sh_type == SHT_RELA) {
      ArrayRef<Elf_Rela> Rels = CHECK(getObj().relas(&Sec), this);
      Target->FirstRelocation = Rels.begin();
      Target->NumRelocations = Rels.size();
      Target->AreRelocsRela = true;
    } else {
      ArrayRef<Elf_Rel> Rels = CHECK(getObj().rels(&Sec), this);
      Target->FirstRelocation = Rels.begin();
      Target->NumRelocations = Rels.size();
      Target->AreRelocsRela = false;
    }
    assert(isUInt<31>(Target->NumRelocations));

    // Relocation sections processed by the linker are usually removed
    // from the output, so returning `nullptr` for the normal case.
    // However, if -emit-relocs is given, we need to leave them in the output.
    // (Some post link analysis tools need this information.)
    if (Config->EmitRelocs) {
      InputSection *RelocSec = make<InputSection>(*this, Sec, Name);
      // We will not emit relocation section if target was discarded.
      Target->DependentSections.push_back(RelocSec);
      return RelocSec;
    }
    return nullptr;
  }
  }

  // The GNU linker uses .note.GNU-stack section as a marker indicating
  // that the code in the object file does not expect that the stack is
  // executable (in terms of NX bit). If all input files have the marker,
  // the GNU linker adds a PT_GNU_STACK segment to tells the loader to
  // make the stack non-executable. Most object files have this section as
  // of 2017.
  //
  // But making the stack non-executable is a norm today for security
  // reasons. Failure to do so may result in a serious security issue.
  // Therefore, we make LLD always add PT_GNU_STACK unless it is
  // explicitly told to do otherwise (by -z execstack). Because the stack
  // executable-ness is controlled solely by command line options,
  // .note.GNU-stack sections are simply ignored.
  if (Name == ".note.GNU-stack")
    return &InputSection::Discarded;

  // Object files that use processor features such as Intel Control-Flow
  // Enforcement (CET) or AArch64 Branch Target Identification BTI, use a
  // .note.gnu.property section containing a bitfield of feature bits like the
  // GNU_PROPERTY_X86_FEATURE_1_IBT flag. Read a bitmap containing the flag.
  //
  // Since we merge bitmaps from multiple object files to create a new
  // .note.gnu.property containing a single AND'ed bitmap, we discard an input
  // file's .note.gnu.property section.
  if (Name == ".note.gnu.property") {
    ArrayRef<uint8_t> Contents = check(this->getObj().getSectionContents(&Sec));
    this->AndFeatures = readAndFeatures(this, Contents);
    return &InputSection::Discarded;
  }

  // Split stacks is a feature to support a discontiguous stack,
  // commonly used in the programming language Go. For the details,
  // see https://gcc.gnu.org/wiki/SplitStacks. An object file compiled
  // for split stack will include a .note.GNU-split-stack section.
  if (Name == ".note.GNU-split-stack") {
    if (Config->Relocatable) {
      error("cannot mix split-stack and non-split-stack in a relocatable link");
      return &InputSection::Discarded;
    }
    this->SplitStack = true;
    return &InputSection::Discarded;
  }

  // An object file cmpiled for split stack, but where some of the
  // functions were compiled with the no_split_stack_attribute will
  // include a .note.GNU-no-split-stack section.
  if (Name == ".note.GNU-no-split-stack") {
    this->SomeNoSplitStack = true;
    return &InputSection::Discarded;
  }

  // The linkonce feature is a sort of proto-comdat. Some glibc i386 object
  // files contain definitions of symbol "__x86.get_pc_thunk.bx" in linkonce
  // sections. Drop those sections to avoid duplicate symbol errors.
  // FIXME: This is glibc PR20543, we should remove this hack once that has been
  // fixed for a while.
  if (Name == ".gnu.linkonce.t.__x86.get_pc_thunk.bx" ||
      Name == ".gnu.linkonce.t.__i686.get_pc_thunk.bx")
    return &InputSection::Discarded;

  // If we are creating a new .build-id section, strip existing .build-id
  // sections so that the output won't have more than one .build-id.
  // This is not usually a problem because input object files normally don't
  // have .build-id sections, but you can create such files by
  // "ld.{bfd,gold,lld} -r --build-id", and we want to guard against it.
  if (Name == ".note.gnu.build-id" && Config->BuildId != BuildIdKind::None)
    return &InputSection::Discarded;

  // The linker merges EH (exception handling) frames and creates a
  // .eh_frame_hdr section for runtime. So we handle them with a special
  // class. For relocatable outputs, they are just passed through.
  if (Name == ".eh_frame" && !Config->Relocatable)
    return make<EhInputSection>(*this, Sec, Name);

  if (shouldMerge(Sec))
    return make<MergeInputSection>(*this, Sec, Name);
  return make<InputSection>(*this, Sec, Name);
}

template <class ELFT>
StringRef ObjFile<ELFT>::getSectionName(const Elf_Shdr &Sec) {
  return CHECK(getObj().getSectionName(&Sec, SectionStringTable), this);
}

// Initialize this->Symbols. this->Symbols is a parallel array as
// its corresponding ELF symbol table.
template <class ELFT> void ObjFile<ELFT>::initializeSymbols() {
  ArrayRef<Elf_Sym> ESyms = this->getELFSyms<ELFT>();
  this->Symbols.resize(ESyms.size());

  // Our symbol table may have already been partially initialized
  // because of LazyObjFile.
  for (size_t I = 0, End = ESyms.size(); I != End; ++I)
    if (!this->Symbols[I] && ESyms[I].getBinding() != STB_LOCAL)
      this->Symbols[I] =
          Symtab->insert(CHECK(ESyms[I].getName(this->StringTable), this));

  // Fill this->Symbols. A symbol is either local or global.
  for (size_t I = 0, End = ESyms.size(); I != End; ++I) {
    const Elf_Sym &ESym = ESyms[I];

    // Read symbol attributes.
    uint32_t SecIdx = getSectionIndex(ESym);
    if (SecIdx >= this->Sections.size())
      fatal(toString(this) + ": invalid section index: " + Twine(SecIdx));

    InputSectionBase *Sec = this->Sections[SecIdx];
    uint8_t Binding = ESym.getBinding();
    uint8_t StOther = ESym.st_other;
    uint8_t Type = ESym.getType();
    uint64_t Value = ESym.st_value;
    uint64_t Size = ESym.st_size;
    StringRefZ Name = this->StringTable.data() + ESym.st_name;

    // Handle local symbols. Local symbols are not added to the symbol
    // table because they are not visible from other object files. We
    // allocate symbol instances and add their pointers to Symbols.
    if (Binding == STB_LOCAL) {
      if (ESym.getType() == STT_FILE)
        SourceFile = CHECK(ESym.getName(this->StringTable), this);

      if (this->StringTable.size() <= ESym.st_name)
        fatal(toString(this) + ": invalid symbol name offset");

      if (ESym.st_shndx == SHN_UNDEF)
        this->Symbols[I] = make<Undefined>(this, Name, Binding, StOther, Type);
      else if (Sec == &InputSection::Discarded)
        this->Symbols[I] = make<Undefined>(this, Name, Binding, StOther, Type,
                                           /*DiscardedSecIdx=*/SecIdx);
      else
        this->Symbols[I] =
            make<Defined>(this, Name, Binding, StOther, Type, Value, Size, Sec);
      continue;
    }

    // Handle global undefined symbols.
    if (ESym.st_shndx == SHN_UNDEF) {
      this->Symbols[I]->resolve(Undefined{this, Name, Binding, StOther, Type});
      continue;
    }

    // Handle global common symbols.
    if (ESym.st_shndx == SHN_COMMON) {
      if (Value == 0 || Value >= UINT32_MAX)
        fatal(toString(this) + ": common symbol '" + StringRef(Name.Data) +
              "' has invalid alignment: " + Twine(Value));
      this->Symbols[I]->resolve(
          CommonSymbol{this, Name, Binding, StOther, Type, Value, Size});
      continue;
    }

    // If a defined symbol is in a discarded section, handle it as if it
    // were an undefined symbol. Such symbol doesn't comply with the
    // standard, but in practice, a .eh_frame often directly refer
    // COMDAT member sections, and if a comdat group is discarded, some
    // defined symbol in a .eh_frame becomes dangling symbols.
    if (Sec == &InputSection::Discarded) {
      this->Symbols[I]->resolve(
          Undefined{this, Name, Binding, StOther, Type, SecIdx});
      continue;
    }

    // Handle global defined symbols.
    if (Binding == STB_GLOBAL || Binding == STB_WEAK ||
        Binding == STB_GNU_UNIQUE) {
      this->Symbols[I]->resolve(
          Defined{this, Name, Binding, StOther, Type, Value, Size, Sec});
      continue;
    }

    fatal(toString(this) + ": unexpected binding: " + Twine((int)Binding));
  }
}

ArchiveFile::ArchiveFile(std::unique_ptr<Archive> &&File)
    : InputFile(ArchiveKind, File->getMemoryBufferRef()),
      File(std::move(File)) {}

void ArchiveFile::parse() {
  for (const Archive::Symbol &Sym : File->symbols())
    Symtab->addSymbol(LazyArchive{*this, Sym});
}

// Returns a buffer pointing to a member file containing a given symbol.
void ArchiveFile::fetch(const Archive::Symbol &Sym) {
  Archive::Child C =
      CHECK(Sym.getMember(), toString(this) +
                                 ": could not get the member for symbol " +
                                 Sym.getName());

  if (!Seen.insert(C.getChildOffset()).second)
    return;

  MemoryBufferRef MB =
      CHECK(C.getMemoryBufferRef(),
            toString(this) +
                ": could not get the buffer for the member defining symbol " +
                Sym.getName());

  if (Tar && C.getParent()->isThin())
    Tar->append(relativeToRoot(CHECK(C.getFullName(), this)), MB.getBuffer());

  InputFile *File = createObjectFile(
      MB, getName(), C.getParent()->isThin() ? 0 : C.getChildOffset());
  File->GroupId = GroupId;
  parseFile(File);
}

unsigned SharedFile::VernauxNum;

// Parse the version definitions in the object file if present, and return a
// vector whose nth element contains a pointer to the Elf_Verdef for version
// identifier n. Version identifiers that are not definitions map to nullptr.
template <typename ELFT>
static std::vector<const void *> parseVerdefs(const uint8_t *Base,
                                              const typename ELFT::Shdr *Sec) {
  if (!Sec)
    return {};

  // We cannot determine the largest verdef identifier without inspecting
  // every Elf_Verdef, but both bfd and gold assign verdef identifiers
  // sequentially starting from 1, so we predict that the largest identifier
  // will be VerdefCount.
  unsigned VerdefCount = Sec->sh_info;
  std::vector<const void *> Verdefs(VerdefCount + 1);

  // Build the Verdefs array by following the chain of Elf_Verdef objects
  // from the start of the .gnu.version_d section.
  const uint8_t *Verdef = Base + Sec->sh_offset;
  for (unsigned I = 0; I != VerdefCount; ++I) {
    auto *CurVerdef = reinterpret_cast<const typename ELFT::Verdef *>(Verdef);
    Verdef += CurVerdef->vd_next;
    unsigned VerdefIndex = CurVerdef->vd_ndx;
    Verdefs.resize(VerdefIndex + 1);
    Verdefs[VerdefIndex] = CurVerdef;
  }
  return Verdefs;
}

// We do not usually care about alignments of data in shared object
// files because the loader takes care of it. However, if we promote a
// DSO symbol to point to .bss due to copy relocation, we need to keep
// the original alignment requirements. We infer it in this function.
template <typename ELFT>
static uint64_t getAlignment(ArrayRef<typename ELFT::Shdr> Sections,
                             const typename ELFT::Sym &Sym) {
  uint64_t Ret = UINT64_MAX;
  if (Sym.st_value)
    Ret = 1ULL << countTrailingZeros((uint64_t)Sym.st_value);
  if (0 < Sym.st_shndx && Sym.st_shndx < Sections.size())
    Ret = std::min<uint64_t>(Ret, Sections[Sym.st_shndx].sh_addralign);
  return (Ret > UINT32_MAX) ? 0 : Ret;
}

// Fully parse the shared object file.
//
// This function parses symbol versions. If a DSO has version information,
// the file has a ".gnu.version_d" section which contains symbol version
// definitions. Each symbol is associated to one version through a table in
// ".gnu.version" section. That table is a parallel array for the symbol
// table, and each table entry contains an index in ".gnu.version_d".
//
// The special index 0 is reserved for VERF_NDX_LOCAL and 1 is for
// VER_NDX_GLOBAL. There's no table entry for these special versions in
// ".gnu.version_d".
//
// The file format for symbol versioning is perhaps a bit more complicated
// than necessary, but you can easily understand the code if you wrap your
// head around the data structure described above.
template <class ELFT> void SharedFile::parse() {
  using Elf_Dyn = typename ELFT::Dyn;
  using Elf_Shdr = typename ELFT::Shdr;
  using Elf_Sym = typename ELFT::Sym;
  using Elf_Verdef = typename ELFT::Verdef;
  using Elf_Versym = typename ELFT::Versym;

  ArrayRef<Elf_Dyn> DynamicTags;
  const ELFFile<ELFT> Obj = this->getObj<ELFT>();
  ArrayRef<Elf_Shdr> Sections = CHECK(Obj.sections(), this);

  const Elf_Shdr *VersymSec = nullptr;
  const Elf_Shdr *VerdefSec = nullptr;

  // Search for .dynsym, .dynamic, .symtab, .gnu.version and .gnu.version_d.
  for (const Elf_Shdr &Sec : Sections) {
    switch (Sec.sh_type) {
    default:
      continue;
    case SHT_DYNAMIC:
      DynamicTags =
          CHECK(Obj.template getSectionContentsAsArray<Elf_Dyn>(&Sec), this);
      break;
    case SHT_GNU_versym:
      VersymSec = &Sec;
      break;
    case SHT_GNU_verdef:
      VerdefSec = &Sec;
      break;
    }
  }

  if (VersymSec && NumELFSyms == 0) {
    error("SHT_GNU_versym should be associated with symbol table");
    return;
  }

  // Search for a DT_SONAME tag to initialize this->SoName.
  for (const Elf_Dyn &Dyn : DynamicTags) {
    if (Dyn.d_tag == DT_NEEDED) {
      uint64_t Val = Dyn.getVal();
      if (Val >= this->StringTable.size())
        fatal(toString(this) + ": invalid DT_NEEDED entry");
      DtNeeded.push_back(this->StringTable.data() + Val);
    } else if (Dyn.d_tag == DT_SONAME) {
      uint64_t Val = Dyn.getVal();
      if (Val >= this->StringTable.size())
        fatal(toString(this) + ": invalid DT_SONAME entry");
      SoName = this->StringTable.data() + Val;
    }
  }

  // DSOs are uniquified not by filename but by soname.
  DenseMap<StringRef, SharedFile *>::iterator It;
  bool WasInserted;
  std::tie(It, WasInserted) = Symtab->SoNames.try_emplace(SoName, this);

  // If a DSO appears more than once on the command line with and without
  // --as-needed, --no-as-needed takes precedence over --as-needed because a
  // user can add an extra DSO with --no-as-needed to force it to be added to
  // the dependency list.
  It->second->IsNeeded |= IsNeeded;
  if (!WasInserted)
    return;

  SharedFiles.push_back(this);

  Verdefs = parseVerdefs<ELFT>(Obj.base(), VerdefSec);

  // Parse ".gnu.version" section which is a parallel array for the symbol
  // table. If a given file doesn't have a ".gnu.version" section, we use
  // VER_NDX_GLOBAL.
  size_t Size = NumELFSyms - FirstGlobal;
  std::vector<uint32_t> Versyms(Size, VER_NDX_GLOBAL);
  if (VersymSec) {
    ArrayRef<Elf_Versym> Versym =
        CHECK(Obj.template getSectionContentsAsArray<Elf_Versym>(VersymSec),
              this)
            .slice(FirstGlobal);
    for (size_t I = 0; I < Size; ++I)
      Versyms[I] = Versym[I].vs_index;
  }

  // System libraries can have a lot of symbols with versions. Using a
  // fixed buffer for computing the versions name (foo@ver) can save a
  // lot of allocations.
  SmallString<0> VersionedNameBuffer;

  // Add symbols to the symbol table.
  ArrayRef<Elf_Sym> Syms = this->getGlobalELFSyms<ELFT>();
  for (size_t I = 0; I < Syms.size(); ++I) {
    const Elf_Sym &Sym = Syms[I];

    // ELF spec requires that all local symbols precede weak or global
    // symbols in each symbol table, and the index of first non-local symbol
    // is stored to sh_info. If a local symbol appears after some non-local
    // symbol, that's a violation of the spec.
    StringRef Name = CHECK(Sym.getName(this->StringTable), this);
    if (Sym.getBinding() == STB_LOCAL) {
      warn("found local symbol '" + Name +
           "' in global part of symbol table in file " + toString(this));
      continue;
    }

    if (Sym.isUndefined()) {
      Symbol *S = Symtab->addSymbol(
          Undefined{this, Name, Sym.getBinding(), Sym.st_other, Sym.getType()});
      S->ExportDynamic = true;
      continue;
    }

    // MIPS BFD linker puts _gp_disp symbol into DSO files and incorrectly
    // assigns VER_NDX_LOCAL to this section global symbol. Here is a
    // workaround for this bug.
    uint32_t Idx = Versyms[I] & ~VERSYM_HIDDEN;
    if (Config->EMachine == EM_MIPS && Idx == VER_NDX_LOCAL &&
        Name == "_gp_disp")
      continue;

    uint32_t Alignment = getAlignment<ELFT>(Sections, Sym);
    if (!(Versyms[I] & VERSYM_HIDDEN)) {
      Symtab->addSymbol(SharedSymbol{*this, Name, Sym.getBinding(),
                                     Sym.st_other, Sym.getType(), Sym.st_value,
                                     Sym.st_size, Alignment, Idx});
    }

    // Also add the symbol with the versioned name to handle undefined symbols
    // with explicit versions.
    if (Idx == VER_NDX_GLOBAL)
      continue;

    if (Idx >= Verdefs.size() || Idx == VER_NDX_LOCAL) {
      error("corrupt input file: version definition index " + Twine(Idx) +
            " for symbol " + Name + " is out of bounds\n>>> defined in " +
            toString(this));
      continue;
    }

    StringRef VerName =
        this->StringTable.data() +
        reinterpret_cast<const Elf_Verdef *>(Verdefs[Idx])->getAux()->vda_name;
    VersionedNameBuffer.clear();
    Name = (Name + "@" + VerName).toStringRef(VersionedNameBuffer);
    Symtab->addSymbol(SharedSymbol{*this, Saver.save(Name), Sym.getBinding(),
                                   Sym.st_other, Sym.getType(), Sym.st_value,
                                   Sym.st_size, Alignment, Idx});
  }
}

static ELFKind getBitcodeELFKind(const Triple &T) {
  if (T.isLittleEndian())
    return T.isArch64Bit() ? ELF64LEKind : ELF32LEKind;
  return T.isArch64Bit() ? ELF64BEKind : ELF32BEKind;
}

static uint8_t getBitcodeMachineKind(StringRef Path, const Triple &T) {
  switch (T.getArch()) {
  case Triple::aarch64:
    return EM_AARCH64;
  case Triple::amdgcn:
  case Triple::r600:
    return EM_AMDGPU;
  case Triple::arm:
  case Triple::thumb:
    return EM_ARM;
  case Triple::avr:
    return EM_AVR;
  case Triple::mips:
  case Triple::mipsel:
  case Triple::mips64:
  case Triple::mips64el:
    return EM_MIPS;
  case Triple::msp430:
    return EM_MSP430;
  case Triple::ppc:
    return EM_PPC;
  case Triple::ppc64:
  case Triple::ppc64le:
    return EM_PPC64;
  case Triple::x86:
    return T.isOSIAMCU() ? EM_IAMCU : EM_386;
  case Triple::x86_64:
    return EM_X86_64;
  default:
    error(Path + ": could not infer e_machine from bitcode target triple " +
          T.str());
    return EM_NONE;
  }
}

BitcodeFile::BitcodeFile(MemoryBufferRef MB, StringRef ArchiveName,
                         uint64_t OffsetInArchive)
    : InputFile(BitcodeKind, MB) {
  this->ArchiveName = ArchiveName;

  std::string Path = MB.getBufferIdentifier().str();
  if (Config->ThinLTOIndexOnly)
    Path = replaceThinLTOSuffix(MB.getBufferIdentifier());

  // ThinLTO assumes that all MemoryBufferRefs given to it have a unique
  // name. If two archives define two members with the same name, this
  // causes a collision which result in only one of the objects being taken
  // into consideration at LTO time (which very likely causes undefined
  // symbols later in the link stage). So we append file offset to make
  // filename unique.
  StringRef Name = ArchiveName.empty()
                       ? Saver.save(Path)
                       : Saver.save(ArchiveName + "(" + Path + " at " +
                                    utostr(OffsetInArchive) + ")");
  MemoryBufferRef MBRef(MB.getBuffer(), Name);

  Obj = CHECK(lto::InputFile::create(MBRef), this);

  Triple T(Obj->getTargetTriple());
  EKind = getBitcodeELFKind(T);
  EMachine = getBitcodeMachineKind(MB.getBufferIdentifier(), T);
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
                                   BitcodeFile &F) {
  StringRef Name = Saver.save(ObjSym.getName());
  uint8_t Binding = ObjSym.isWeak() ? STB_WEAK : STB_GLOBAL;
  uint8_t Type = ObjSym.isTLS() ? STT_TLS : STT_NOTYPE;
  uint8_t Visibility = mapVisibility(ObjSym.getVisibility());
  bool CanOmitFromDynSym = ObjSym.canBeOmittedFromSymbolTable();

  int C = ObjSym.getComdatIndex();
  if (ObjSym.isUndefined() || (C != -1 && !KeptComdats[C])) {
    Undefined New(&F, Name, Binding, Visibility, Type);
    if (CanOmitFromDynSym)
      New.ExportDynamic = false;
    return Symtab->addSymbol(New);
  }

  if (ObjSym.isCommon())
    return Symtab->addSymbol(
        CommonSymbol{&F, Name, Binding, Visibility, STT_OBJECT,
                     ObjSym.getCommonAlignment(), ObjSym.getCommonSize()});

  Defined New(&F, Name, Binding, Visibility, Type, 0, 0, nullptr);
  if (CanOmitFromDynSym)
    New.ExportDynamic = false;
  return Symtab->addSymbol(New);
}

template <class ELFT> void BitcodeFile::parse() {
  std::vector<bool> KeptComdats;
  for (StringRef S : Obj->getComdatTable())
    KeptComdats.push_back(
        Symtab->ComdatGroups.try_emplace(CachedHashStringRef(S), this).second);

  for (const lto::InputFile::Symbol &ObjSym : Obj->symbols())
    Symbols.push_back(createBitcodeSymbol<ELFT>(KeptComdats, ObjSym, *this));

  for (auto L : Obj->getDependentLibraries())
    addDependentLibrary(L, this);
}

void BinaryFile::parse() {
  ArrayRef<uint8_t> Data = arrayRefFromStringRef(MB.getBuffer());
  auto *Section = make<InputSection>(this, SHF_ALLOC | SHF_WRITE, SHT_PROGBITS,
                                     8, Data, ".data");
  Sections.push_back(Section);

  // For each input file foo that is embedded to a result as a binary
  // blob, we define _binary_foo_{start,end,size} symbols, so that
  // user programs can access blobs by name. Non-alphanumeric
  // characters in a filename are replaced with underscore.
  std::string S = "_binary_" + MB.getBufferIdentifier().str();
  for (size_t I = 0; I < S.size(); ++I)
    if (!isAlnum(S[I]))
      S[I] = '_';

  Symtab->addSymbol(Defined{nullptr, Saver.save(S + "_start"), STB_GLOBAL,
                            STV_DEFAULT, STT_OBJECT, 0, 0, Section});
  Symtab->addSymbol(Defined{nullptr, Saver.save(S + "_end"), STB_GLOBAL,
                            STV_DEFAULT, STT_OBJECT, Data.size(), 0, Section});
  Symtab->addSymbol(Defined{nullptr, Saver.save(S + "_size"), STB_GLOBAL,
                            STV_DEFAULT, STT_OBJECT, Data.size(), 0, nullptr});
}

InputFile *elf::createObjectFile(MemoryBufferRef MB, StringRef ArchiveName,
                                 uint64_t OffsetInArchive) {
  if (isBitcode(MB))
    return make<BitcodeFile>(MB, ArchiveName, OffsetInArchive);

  switch (getELFKind(MB, ArchiveName)) {
  case ELF32LEKind:
    return make<ObjFile<ELF32LE>>(MB, ArchiveName);
  case ELF32BEKind:
    return make<ObjFile<ELF32BE>>(MB, ArchiveName);
  case ELF64LEKind:
    return make<ObjFile<ELF64LE>>(MB, ArchiveName);
  case ELF64BEKind:
    return make<ObjFile<ELF64BE>>(MB, ArchiveName);
  default:
    llvm_unreachable("getELFKind");
  }
}

void LazyObjFile::fetch() {
  if (MB.getBuffer().empty())
    return;

  InputFile *File = createObjectFile(MB, ArchiveName, OffsetInArchive);
  File->GroupId = GroupId;

  MB = {};

  // Copy symbol vector so that the new InputFile doesn't have to
  // insert the same defined symbols to the symbol table again.
  File->Symbols = std::move(Symbols);

  parseFile(File);
}

template <class ELFT> void LazyObjFile::parse() {
  using Elf_Sym = typename ELFT::Sym;

  // A lazy object file wraps either a bitcode file or an ELF file.
  if (isBitcode(this->MB)) {
    std::unique_ptr<lto::InputFile> Obj =
        CHECK(lto::InputFile::create(this->MB), this);
    for (const lto::InputFile::Symbol &Sym : Obj->symbols()) {
      if (Sym.isUndefined())
        continue;
      Symtab->addSymbol(LazyObject{*this, Saver.save(Sym.getName())});
    }
    return;
  }

  if (getELFKind(this->MB, ArchiveName) != Config->EKind) {
    error("incompatible file: " + this->MB.getBufferIdentifier());
    return;
  }

  // Find a symbol table.
  ELFFile<ELFT> Obj = check(ELFFile<ELFT>::create(MB.getBuffer()));
  ArrayRef<typename ELFT::Shdr> Sections = CHECK(Obj.sections(), this);

  for (const typename ELFT::Shdr &Sec : Sections) {
    if (Sec.sh_type != SHT_SYMTAB)
      continue;

    // A symbol table is found.
    ArrayRef<Elf_Sym> ESyms = CHECK(Obj.symbols(&Sec), this);
    uint32_t FirstGlobal = Sec.sh_info;
    StringRef Strtab = CHECK(Obj.getStringTableForSymtab(Sec, Sections), this);
    this->Symbols.resize(ESyms.size());

    // Get existing symbols or insert placeholder symbols.
    for (size_t I = FirstGlobal, End = ESyms.size(); I != End; ++I)
      if (ESyms[I].st_shndx != SHN_UNDEF)
        this->Symbols[I] = Symtab->insert(CHECK(ESyms[I].getName(Strtab), this));

    // Replace existing symbols with LazyObject symbols.
    //
    // resolve() may trigger this->fetch() if an existing symbol is an
    // undefined symbol. If that happens, this LazyObjFile has served
    // its purpose, and we can exit from the loop early.
    for (Symbol *Sym : this->Symbols) {
      if (!Sym)
        continue;
      Sym->resolve(LazyObject{*this, Sym->getName()});

      // MemoryBuffer is emptied if this file is instantiated as ObjFile.
      if (MB.getBuffer().empty())
        return;
    }
    return;
  }
}

std::string elf::replaceThinLTOSuffix(StringRef Path) {
  StringRef Suffix = Config->ThinLTOObjectSuffixReplace.first;
  StringRef Repl = Config->ThinLTOObjectSuffixReplace.second;

  if (Path.consume_back(Suffix))
    return (Path + Repl).str();
  return Path;
}

template void BitcodeFile::parse<ELF32LE>();
template void BitcodeFile::parse<ELF32BE>();
template void BitcodeFile::parse<ELF64LE>();
template void BitcodeFile::parse<ELF64BE>();

template void LazyObjFile::parse<ELF32LE>();
template void LazyObjFile::parse<ELF32BE>();
template void LazyObjFile::parse<ELF64LE>();
template void LazyObjFile::parse<ELF64BE>();

template class elf::ObjFile<ELF32LE>;
template class elf::ObjFile<ELF32BE>;
template class elf::ObjFile<ELF64LE>;
template class elf::ObjFile<ELF64BE>;

template void SharedFile::parse<ELF32LE>();
template void SharedFile::parse<ELF32BE>();
template void SharedFile::parse<ELF64LE>();
template void SharedFile::parse<ELF64BE>();
