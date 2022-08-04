//===- InputFiles.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Config.h"
#include "DWARF.h"
#include "Driver.h"
#include "InputSection.h"
#include "LinkerScript.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/DWARF.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Support/ARMAttributeParser.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/RISCVAttributeParser.h"
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

bool InputFile::isInGroup;
uint32_t InputFile::nextGroupId;

std::unique_ptr<TarWriter> elf::tar;

// Returns "<internal>", "foo.a(bar.o)" or "baz.o".
std::string lld::toString(const InputFile *f) {
  if (!f)
    return "<internal>";

  if (f->toStringCache.empty()) {
    if (f->archiveName.empty())
      f->toStringCache = f->getName();
    else
      (f->archiveName + "(" + f->getName() + ")").toVector(f->toStringCache);
  }
  return std::string(f->toStringCache);
}

static ELFKind getELFKind(MemoryBufferRef mb, StringRef archiveName) {
  unsigned char size;
  unsigned char endian;
  std::tie(size, endian) = getElfArchType(mb.getBuffer());

  auto report = [&](StringRef msg) {
    StringRef filename = mb.getBufferIdentifier();
    if (archiveName.empty())
      fatal(filename + ": " + msg);
    else
      fatal(archiveName + "(" + filename + "): " + msg);
  };

  if (!mb.getBuffer().startswith(ElfMagic))
    report("not an ELF file");
  if (endian != ELFDATA2LSB && endian != ELFDATA2MSB)
    report("corrupted ELF file: invalid data encoding");
  if (size != ELFCLASS32 && size != ELFCLASS64)
    report("corrupted ELF file: invalid file class");

  size_t bufSize = mb.getBuffer().size();
  if ((size == ELFCLASS32 && bufSize < sizeof(Elf32_Ehdr)) ||
      (size == ELFCLASS64 && bufSize < sizeof(Elf64_Ehdr)))
    report("corrupted ELF file: file is too short");

  if (size == ELFCLASS32)
    return (endian == ELFDATA2LSB) ? ELF32LEKind : ELF32BEKind;
  return (endian == ELFDATA2LSB) ? ELF64LEKind : ELF64BEKind;
}

// For ARM only, to set the EF_ARM_ABI_FLOAT_SOFT or EF_ARM_ABI_FLOAT_HARD
// flag in the ELF Header we need to look at Tag_ABI_VFP_args to find out how
// the input objects have been compiled.
static void updateARMVFPArgs(const ARMAttributeParser &attributes,
                             const InputFile *f) {
  Optional<unsigned> attr =
      attributes.getAttributeValue(ARMBuildAttrs::ABI_VFP_args);
  if (!attr)
    // If an ABI tag isn't present then it is implicitly given the value of 0
    // which maps to ARMBuildAttrs::BaseAAPCS. However many assembler files,
    // including some in glibc that don't use FP args (and should have value 3)
    // don't have the attribute so we do not consider an implicit value of 0
    // as a clash.
    return;

  unsigned vfpArgs = *attr;
  ARMVFPArgKind arg;
  switch (vfpArgs) {
  case ARMBuildAttrs::BaseAAPCS:
    arg = ARMVFPArgKind::Base;
    break;
  case ARMBuildAttrs::HardFPAAPCS:
    arg = ARMVFPArgKind::VFP;
    break;
  case ARMBuildAttrs::ToolChainFPPCS:
    // Tool chain specific convention that conforms to neither AAPCS variant.
    arg = ARMVFPArgKind::ToolChain;
    break;
  case ARMBuildAttrs::CompatibleFPAAPCS:
    // Object compatible with all conventions.
    return;
  default:
    error(toString(f) + ": unknown Tag_ABI_VFP_args value: " + Twine(vfpArgs));
    return;
  }
  // Follow ld.bfd and error if there is a mix of calling conventions.
  if (config->armVFPArgs != arg && config->armVFPArgs != ARMVFPArgKind::Default)
    error(toString(f) + ": incompatible Tag_ABI_VFP_args");
  else
    config->armVFPArgs = arg;
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
static void updateSupportedARMFeatures(const ARMAttributeParser &attributes) {
  Optional<unsigned> attr =
      attributes.getAttributeValue(ARMBuildAttrs::CPU_arch);
  if (!attr)
    return;
  auto arch = attr.value();
  switch (arch) {
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
    config->armHasBlx = true;
    // Architectures used in pre-Cortex processors do not support
    // The J1 = 1 J2 = 1 Thumb branch range extension, with the exception
    // of Architecture v6T2 (arm1156t2-s and arm1156t2f-s) that do.
    break;
  default:
    // All other Architectures have BLX and extended branch encoding
    config->armHasBlx = true;
    config->armJ1J2BranchEncoding = true;
    if (arch != ARMBuildAttrs::v6_M && arch != ARMBuildAttrs::v6S_M)
      // All Architectures used in Cortex processors with the exception
      // of v6-M and v6S-M have the MOVT and MOVW instructions.
      config->armHasMovtMovw = true;
    break;
  }
}

InputFile::InputFile(Kind k, MemoryBufferRef m)
    : mb(m), groupId(nextGroupId), fileKind(k) {
  // All files within the same --{start,end}-group get the same group ID.
  // Otherwise, a new file will get a new group ID.
  if (!isInGroup)
    ++nextGroupId;
}

Optional<MemoryBufferRef> elf::readFile(StringRef path) {
  llvm::TimeTraceScope timeScope("Load input files", path);

  // The --chroot option changes our virtual root directory.
  // This is useful when you are dealing with files created by --reproduce.
  if (!config->chroot.empty() && path.startswith("/"))
    path = saver().save(config->chroot + path);

  log(path);
  config->dependencyFiles.insert(llvm::CachedHashString(path));

  auto mbOrErr = MemoryBuffer::getFile(path, /*IsText=*/false,
                                       /*RequiresNullTerminator=*/false);
  if (auto ec = mbOrErr.getError()) {
    error("cannot open " + path + ": " + ec.message());
    return None;
  }

  MemoryBufferRef mbref = (*mbOrErr)->getMemBufferRef();
  ctx->memoryBuffers.push_back(std::move(*mbOrErr)); // take MB ownership

  if (tar)
    tar->append(relativeToRoot(path), mbref.getBuffer());
  return mbref;
}

// All input object files must be for the same architecture
// (e.g. it does not make sense to link x86 object files with
// MIPS object files.) This function checks for that error.
static bool isCompatible(InputFile *file) {
  if (!file->isElf() && !isa<BitcodeFile>(file))
    return true;

  if (file->ekind == config->ekind && file->emachine == config->emachine) {
    if (config->emachine != EM_MIPS)
      return true;
    if (isMipsN32Abi(file) == config->mipsN32Abi)
      return true;
  }

  StringRef target =
      !config->bfdname.empty() ? config->bfdname : config->emulation;
  if (!target.empty()) {
    error(toString(file) + " is incompatible with " + target);
    return false;
  }

  InputFile *existing = nullptr;
  if (!ctx->objectFiles.empty())
    existing = ctx->objectFiles[0];
  else if (!ctx->sharedFiles.empty())
    existing = ctx->sharedFiles[0];
  else if (!ctx->bitcodeFiles.empty())
    existing = ctx->bitcodeFiles[0];
  std::string with;
  if (existing)
    with = " with " + toString(existing);
  error(toString(file) + " is incompatible" + with);
  return false;
}

template <class ELFT> static void doParseFile(InputFile *file) {
  if (!isCompatible(file))
    return;

  // Binary file
  if (auto *f = dyn_cast<BinaryFile>(file)) {
    ctx->binaryFiles.push_back(f);
    f->parse();
    return;
  }

  // Lazy object file
  if (file->lazy) {
    if (auto *f = dyn_cast<BitcodeFile>(file)) {
      ctx->lazyBitcodeFiles.push_back(f);
      f->parseLazy();
    } else {
      cast<ObjFile<ELFT>>(file)->parseLazy();
    }
    return;
  }

  if (config->trace)
    message(toString(file));

  // .so file
  if (auto *f = dyn_cast<SharedFile>(file)) {
    f->parse<ELFT>();
    return;
  }

  // LLVM bitcode file
  if (auto *f = dyn_cast<BitcodeFile>(file)) {
    ctx->bitcodeFiles.push_back(f);
    f->parse<ELFT>();
    return;
  }

  // Regular object file
  ctx->objectFiles.push_back(cast<ELFFileBase>(file));
  cast<ObjFile<ELFT>>(file)->parse();
}

// Add symbols in File to the symbol table.
void elf::parseFile(InputFile *file) { invokeELFT(doParseFile, file); }

// Concatenates arguments to construct a string representing an error location.
static std::string createFileLineMsg(StringRef path, unsigned line) {
  std::string filename = std::string(path::filename(path));
  std::string lineno = ":" + std::to_string(line);
  if (filename == path)
    return filename + lineno;
  return filename + lineno + " (" + path.str() + lineno + ")";
}

template <class ELFT>
static std::string getSrcMsgAux(ObjFile<ELFT> &file, const Symbol &sym,
                                InputSectionBase &sec, uint64_t offset) {
  // In DWARF, functions and variables are stored to different places.
  // First, look up a function for a given offset.
  if (Optional<DILineInfo> info = file.getDILineInfo(&sec, offset))
    return createFileLineMsg(info->FileName, info->Line);

  // If it failed, look up again as a variable.
  if (Optional<std::pair<std::string, unsigned>> fileLine =
          file.getVariableLoc(sym.getName()))
    return createFileLineMsg(fileLine->first, fileLine->second);

  // File.sourceFile contains STT_FILE symbol, and that is a last resort.
  return std::string(file.sourceFile);
}

std::string InputFile::getSrcMsg(const Symbol &sym, InputSectionBase &sec,
                                 uint64_t offset) {
  if (kind() != ObjKind)
    return "";
  switch (config->ekind) {
  default:
    llvm_unreachable("Invalid kind");
  case ELF32LEKind:
    return getSrcMsgAux(cast<ObjFile<ELF32LE>>(*this), sym, sec, offset);
  case ELF32BEKind:
    return getSrcMsgAux(cast<ObjFile<ELF32BE>>(*this), sym, sec, offset);
  case ELF64LEKind:
    return getSrcMsgAux(cast<ObjFile<ELF64LE>>(*this), sym, sec, offset);
  case ELF64BEKind:
    return getSrcMsgAux(cast<ObjFile<ELF64BE>>(*this), sym, sec, offset);
  }
}

StringRef InputFile::getNameForScript() const {
  if (archiveName.empty())
    return getName();

  if (nameForScriptCache.empty())
    nameForScriptCache = (archiveName + Twine(':') + getName()).str();

  return nameForScriptCache;
}

// An ELF object file may contain a `.deplibs` section. If it exists, the
// section contains a list of library specifiers such as `m` for libm. This
// function resolves a given name by finding the first matching library checking
// the various ways that a library can be specified to LLD. This ELF extension
// is a form of autolinking and is called `dependent libraries`. It is currently
// unique to LLVM and lld.
static void addDependentLibrary(StringRef specifier, const InputFile *f) {
  if (!config->dependentLibraries)
    return;
  if (Optional<std::string> s = searchLibraryBaseName(specifier))
    driver->addFile(saver().save(*s), /*withLOption=*/true);
  else if (Optional<std::string> s = findFromSearchPaths(specifier))
    driver->addFile(saver().save(*s), /*withLOption=*/true);
  else if (fs::exists(specifier))
    driver->addFile(specifier, /*withLOption=*/false);
  else
    error(toString(f) +
          ": unable to find library from dependent library specifier: " +
          specifier);
}

// Record the membership of a section group so that in the garbage collection
// pass, section group members are kept or discarded as a unit.
template <class ELFT>
static void handleSectionGroup(ArrayRef<InputSectionBase *> sections,
                               ArrayRef<typename ELFT::Word> entries) {
  bool hasAlloc = false;
  for (uint32_t index : entries.slice(1)) {
    if (index >= sections.size())
      return;
    if (InputSectionBase *s = sections[index])
      if (s != &InputSection::discarded && s->flags & SHF_ALLOC)
        hasAlloc = true;
  }

  // If any member has the SHF_ALLOC flag, the whole group is subject to garbage
  // collection. See the comment in markLive(). This rule retains .debug_types
  // and .rela.debug_types.
  if (!hasAlloc)
    return;

  // Connect the members in a circular doubly-linked list via
  // nextInSectionGroup.
  InputSectionBase *head;
  InputSectionBase *prev = nullptr;
  for (uint32_t index : entries.slice(1)) {
    InputSectionBase *s = sections[index];
    if (!s || s == &InputSection::discarded)
      continue;
    if (prev)
      prev->nextInSectionGroup = s;
    else
      head = s;
    prev = s;
  }
  if (prev)
    prev->nextInSectionGroup = head;
}

template <class ELFT> DWARFCache *ObjFile<ELFT>::getDwarf() {
  llvm::call_once(initDwarf, [this]() {
    dwarf = std::make_unique<DWARFCache>(std::make_unique<DWARFContext>(
        std::make_unique<LLDDwarfObj<ELFT>>(this), "",
        [&](Error err) { warn(getName() + ": " + toString(std::move(err))); },
        [&](Error warning) {
          warn(getName() + ": " + toString(std::move(warning)));
        }));
  });

  return dwarf.get();
}

// Returns the pair of file name and line number describing location of data
// object (variable, array, etc) definition.
template <class ELFT>
Optional<std::pair<std::string, unsigned>>
ObjFile<ELFT>::getVariableLoc(StringRef name) {
  return getDwarf()->getVariableLoc(name);
}

// Returns source line information for a given offset
// using DWARF debug info.
template <class ELFT>
Optional<DILineInfo> ObjFile<ELFT>::getDILineInfo(InputSectionBase *s,
                                                  uint64_t offset) {
  // Detect SectionIndex for specified section.
  uint64_t sectionIndex = object::SectionedAddress::UndefSection;
  ArrayRef<InputSectionBase *> sections = s->file->getSections();
  for (uint64_t curIndex = 0; curIndex < sections.size(); ++curIndex) {
    if (s == sections[curIndex]) {
      sectionIndex = curIndex;
      break;
    }
  }

  return getDwarf()->getDILineInfo(offset, sectionIndex);
}

ELFFileBase::ELFFileBase(Kind k, MemoryBufferRef mb) : InputFile(k, mb) {
  ekind = getELFKind(mb, "");

  switch (ekind) {
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
static const Elf_Shdr *findSection(ArrayRef<Elf_Shdr> sections, uint32_t type) {
  for (const Elf_Shdr &sec : sections)
    if (sec.sh_type == type)
      return &sec;
  return nullptr;
}

template <class ELFT> void ELFFileBase::init() {
  using Elf_Shdr = typename ELFT::Shdr;
  using Elf_Sym = typename ELFT::Sym;

  // Initialize trivial attributes.
  const ELFFile<ELFT> &obj = getObj<ELFT>();
  emachine = obj.getHeader().e_machine;
  osabi = obj.getHeader().e_ident[llvm::ELF::EI_OSABI];
  abiVersion = obj.getHeader().e_ident[llvm::ELF::EI_ABIVERSION];

  ArrayRef<Elf_Shdr> sections = CHECK(obj.sections(), this);
  elfShdrs = sections.data();
  numELFShdrs = sections.size();

  // Find a symbol table.
  bool isDSO =
      (identify_magic(mb.getBuffer()) == file_magic::elf_shared_object);
  const Elf_Shdr *symtabSec =
      findSection(sections, isDSO ? SHT_DYNSYM : SHT_SYMTAB);

  if (!symtabSec)
    return;

  // Initialize members corresponding to a symbol table.
  firstGlobal = symtabSec->sh_info;

  ArrayRef<Elf_Sym> eSyms = CHECK(obj.symbols(symtabSec), this);
  if (firstGlobal == 0 || firstGlobal > eSyms.size())
    fatal(toString(this) + ": invalid sh_info in symbol table");

  elfSyms = reinterpret_cast<const void *>(eSyms.data());
  numELFSyms = uint32_t(eSyms.size());
  stringTable = CHECK(obj.getStringTableForSymtab(*symtabSec, sections), this);
}

template <class ELFT>
uint32_t ObjFile<ELFT>::getSectionIndex(const Elf_Sym &sym) const {
  return CHECK(
      this->getObj().getSectionIndex(sym, getELFSyms<ELFT>(), shndxTable),
      this);
}

template <class ELFT> void ObjFile<ELFT>::parse(bool ignoreComdats) {
  object::ELFFile<ELFT> obj = this->getObj();
  // Read a section table. justSymbols is usually false.
  if (this->justSymbols)
    initializeJustSymbols();
  else
    initializeSections(ignoreComdats, obj);

  // Read a symbol table.
  initializeSymbols(obj);
}

// Sections with SHT_GROUP and comdat bits define comdat section groups.
// They are identified and deduplicated by group name. This function
// returns a group name.
template <class ELFT>
StringRef ObjFile<ELFT>::getShtGroupSignature(ArrayRef<Elf_Shdr> sections,
                                              const Elf_Shdr &sec) {
  typename ELFT::SymRange symbols = this->getELFSyms<ELFT>();
  if (sec.sh_info >= symbols.size())
    fatal(toString(this) + ": invalid symbol index");
  const typename ELFT::Sym &sym = symbols[sec.sh_info];
  return CHECK(sym.getName(this->stringTable), this);
}

template <class ELFT>
bool ObjFile<ELFT>::shouldMerge(const Elf_Shdr &sec, StringRef name) {
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
  if (config->optimize == 0 && !config->relocatable)
    return false;

  // A mergeable section with size 0 is useless because they don't have
  // any data to merge. A mergeable string section with size 0 can be
  // argued as invalid because it doesn't end with a null character.
  // We'll avoid a mess by handling them as if they were non-mergeable.
  if (sec.sh_size == 0)
    return false;

  // Check for sh_entsize. The ELF spec is not clear about the zero
  // sh_entsize. It says that "the member [sh_entsize] contains 0 if
  // the section does not hold a table of fixed-size entries". We know
  // that Rust 1.13 produces a string mergeable section with a zero
  // sh_entsize. Here we just accept it rather than being picky about it.
  uint64_t entSize = sec.sh_entsize;
  if (entSize == 0)
    return false;
  if (sec.sh_size % entSize)
    fatal(toString(this) + ":(" + name + "): SHF_MERGE section size (" +
          Twine(sec.sh_size) + ") must be a multiple of sh_entsize (" +
          Twine(entSize) + ")");

  if (sec.sh_flags & SHF_WRITE)
    fatal(toString(this) + ":(" + name +
          "): writable SHF_MERGE section is not supported");

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
  sections.resize(numELFShdrs);
}

template <class ELFT>
void ObjFile<ELFT>::initializeSections(bool ignoreComdats,
                                       const llvm::object::ELFFile<ELFT> &obj) {
  ArrayRef<Elf_Shdr> objSections = getELFShdrs<ELFT>();
  StringRef shstrtab = CHECK(obj.getSectionStringTable(objSections), this);
  uint64_t size = objSections.size();
  this->sections.resize(size);

  std::vector<ArrayRef<Elf_Word>> selectedGroups;

  for (size_t i = 0; i != size; ++i) {
    if (this->sections[i] == &InputSection::discarded)
      continue;
    const Elf_Shdr &sec = objSections[i];

    // SHF_EXCLUDE'ed sections are discarded by the linker. However,
    // if -r is given, we'll let the final link discard such sections.
    // This is compatible with GNU.
    if ((sec.sh_flags & SHF_EXCLUDE) && !config->relocatable) {
      if (sec.sh_type == SHT_LLVM_CALL_GRAPH_PROFILE)
        cgProfileSectionIndex = i;
      if (sec.sh_type == SHT_LLVM_ADDRSIG) {
        // We ignore the address-significance table if we know that the object
        // file was created by objcopy or ld -r. This is because these tools
        // will reorder the symbols in the symbol table, invalidating the data
        // in the address-significance table, which refers to symbols by index.
        if (sec.sh_link != 0)
          this->addrsigSec = &sec;
        else if (config->icf == ICFLevel::Safe)
          warn(toString(this) +
               ": --icf=safe conservatively ignores "
               "SHT_LLVM_ADDRSIG [index " +
               Twine(i) +
               "] with sh_link=0 "
               "(likely created using objcopy or ld -r)");
      }
      this->sections[i] = &InputSection::discarded;
      continue;
    }

    switch (sec.sh_type) {
    case SHT_GROUP: {
      // De-duplicate section groups by their signatures.
      StringRef signature = getShtGroupSignature(objSections, sec);
      this->sections[i] = &InputSection::discarded;

      ArrayRef<Elf_Word> entries =
          CHECK(obj.template getSectionContentsAsArray<Elf_Word>(sec), this);
      if (entries.empty())
        fatal(toString(this) + ": empty SHT_GROUP");

      Elf_Word flag = entries[0];
      if (flag && flag != GRP_COMDAT)
        fatal(toString(this) + ": unsupported SHT_GROUP format");

      bool keepGroup =
          (flag & GRP_COMDAT) == 0 || ignoreComdats ||
          symtab->comdatGroups.try_emplace(CachedHashStringRef(signature), this)
              .second;
      if (keepGroup) {
        if (config->relocatable)
          this->sections[i] = createInputSection(
              i, sec, check(obj.getSectionName(sec, shstrtab)));
        selectedGroups.push_back(entries);
        continue;
      }

      // Otherwise, discard group members.
      for (uint32_t secIndex : entries.slice(1)) {
        if (secIndex >= size)
          fatal(toString(this) +
                ": invalid section index in group: " + Twine(secIndex));
        this->sections[secIndex] = &InputSection::discarded;
      }
      break;
    }
    case SHT_SYMTAB_SHNDX:
      shndxTable = CHECK(obj.getSHNDXTable(sec, objSections), this);
      break;
    case SHT_SYMTAB:
    case SHT_STRTAB:
    case SHT_REL:
    case SHT_RELA:
    case SHT_NULL:
      break;
    case SHT_LLVM_SYMPART:
      ctx->hasSympart.store(true, std::memory_order_relaxed);
      LLVM_FALLTHROUGH;
    default:
      this->sections[i] =
          createInputSection(i, sec, check(obj.getSectionName(sec, shstrtab)));
    }
  }

  // We have a second loop. It is used to:
  // 1) handle SHF_LINK_ORDER sections.
  // 2) create SHT_REL[A] sections. In some cases the section header index of a
  //    relocation section may be smaller than that of the relocated section. In
  //    such cases, the relocation section would attempt to reference a target
  //    section that has not yet been created. For simplicity, delay creation of
  //    relocation sections until now.
  for (size_t i = 0; i != size; ++i) {
    if (this->sections[i] == &InputSection::discarded)
      continue;
    const Elf_Shdr &sec = objSections[i];

    if (sec.sh_type == SHT_REL || sec.sh_type == SHT_RELA) {
      // Find a relocation target section and associate this section with that.
      // Target may have been discarded if it is in a different section group
      // and the group is discarded, even though it's a violation of the spec.
      // We handle that situation gracefully by discarding dangling relocation
      // sections.
      const uint32_t info = sec.sh_info;
      InputSectionBase *s = getRelocTarget(i, sec, info);
      if (!s)
        continue;

      // ELF spec allows mergeable sections with relocations, but they are rare,
      // and it is in practice hard to merge such sections by contents, because
      // applying relocations at end of linking changes section contents. So, we
      // simply handle such sections as non-mergeable ones. Degrading like this
      // is acceptable because section merging is optional.
      if (auto *ms = dyn_cast<MergeInputSection>(s)) {
        s = make<InputSection>(ms->file, ms->flags, ms->type, ms->alignment,
                               ms->data(), ms->name);
        sections[info] = s;
      }

      if (s->relSecIdx != 0)
        error(
            toString(s) +
            ": multiple relocation sections to one section are not supported");
      s->relSecIdx = i;

      // Relocation sections are usually removed from the output, so return
      // `nullptr` for the normal case. However, if -r or --emit-relocs is
      // specified, we need to copy them to the output. (Some post link analysis
      // tools specify --emit-relocs to obtain the information.)
      if (config->copyRelocs) {
        auto *isec = make<InputSection>(
            *this, sec, check(obj.getSectionName(sec, shstrtab)));
        // If the relocated section is discarded (due to /DISCARD/ or
        // --gc-sections), the relocation section should be discarded as well.
        s->dependentSections.push_back(isec);
        sections[i] = isec;
      }
      continue;
    }

    // A SHF_LINK_ORDER section with sh_link=0 is handled as if it did not have
    // the flag.
    if (!sec.sh_link || !(sec.sh_flags & SHF_LINK_ORDER))
      continue;

    InputSectionBase *linkSec = nullptr;
    if (sec.sh_link < size)
      linkSec = this->sections[sec.sh_link];
    if (!linkSec)
      fatal(toString(this) + ": invalid sh_link index: " + Twine(sec.sh_link));

    // A SHF_LINK_ORDER section is discarded if its linked-to section is
    // discarded.
    InputSection *isec = cast<InputSection>(this->sections[i]);
    linkSec->dependentSections.push_back(isec);
    if (!isa<InputSection>(linkSec))
      error("a section " + isec->name +
            " with SHF_LINK_ORDER should not refer a non-regular section: " +
            toString(linkSec));
  }

  for (ArrayRef<Elf_Word> entries : selectedGroups)
    handleSectionGroup<ELFT>(this->sections, entries);
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
template <class ELFT> static uint32_t readAndFeatures(const InputSection &sec) {
  using Elf_Nhdr = typename ELFT::Nhdr;
  using Elf_Note = typename ELFT::Note;

  uint32_t featuresSet = 0;
  ArrayRef<uint8_t> data = sec.rawData;
  auto reportFatal = [&](const uint8_t *place, const char *msg) {
    fatal(toString(sec.file) + ":(" + sec.name + "+0x" +
          Twine::utohexstr(place - sec.rawData.data()) + "): " + msg);
  };
  while (!data.empty()) {
    // Read one NOTE record.
    auto *nhdr = reinterpret_cast<const Elf_Nhdr *>(data.data());
    if (data.size() < sizeof(Elf_Nhdr) || data.size() < nhdr->getSize())
      reportFatal(data.data(), "data is too short");

    Elf_Note note(*nhdr);
    if (nhdr->n_type != NT_GNU_PROPERTY_TYPE_0 || note.getName() != "GNU") {
      data = data.slice(nhdr->getSize());
      continue;
    }

    uint32_t featureAndType = config->emachine == EM_AARCH64
                                  ? GNU_PROPERTY_AARCH64_FEATURE_1_AND
                                  : GNU_PROPERTY_X86_FEATURE_1_AND;

    // Read a body of a NOTE record, which consists of type-length-value fields.
    ArrayRef<uint8_t> desc = note.getDesc();
    while (!desc.empty()) {
      const uint8_t *place = desc.data();
      if (desc.size() < 8)
        reportFatal(place, "program property is too short");
      uint32_t type = read32<ELFT::TargetEndianness>(desc.data());
      uint32_t size = read32<ELFT::TargetEndianness>(desc.data() + 4);
      desc = desc.slice(8);
      if (desc.size() < size)
        reportFatal(place, "program property is too short");

      if (type == featureAndType) {
        // We found a FEATURE_1_AND field. There may be more than one of these
        // in a .note.gnu.property section, for a relocatable object we
        // accumulate the bits set.
        if (size < 4)
          reportFatal(place, "FEATURE_1_AND entry is too short");
        featuresSet |= read32<ELFT::TargetEndianness>(desc.data());
      }

      // Padding is present in the note descriptor, if necessary.
      desc = desc.slice(alignTo<(ELFT::Is64Bits ? 8 : 4)>(size));
    }

    // Go to next NOTE record to look for more FEATURE_1_AND descriptions.
    data = data.slice(nhdr->getSize());
  }

  return featuresSet;
}

template <class ELFT>
InputSectionBase *ObjFile<ELFT>::getRelocTarget(uint32_t idx,
                                                const Elf_Shdr &sec,
                                                uint32_t info) {
  if (info < this->sections.size()) {
    InputSectionBase *target = this->sections[info];

    // Strictly speaking, a relocation section must be included in the
    // group of the section it relocates. However, LLVM 3.3 and earlier
    // would fail to do so, so we gracefully handle that case.
    if (target == &InputSection::discarded)
      return nullptr;

    if (target != nullptr)
      return target;
  }

  error(toString(this) + Twine(": relocation section (index ") + Twine(idx) +
        ") has invalid sh_info (" + Twine(info) + ")");
  return nullptr;
}

template <class ELFT>
InputSectionBase *ObjFile<ELFT>::createInputSection(uint32_t idx,
                                                    const Elf_Shdr &sec,
                                                    StringRef name) {
  if (sec.sh_type == SHT_ARM_ATTRIBUTES && config->emachine == EM_ARM) {
    ARMAttributeParser attributes;
    ArrayRef<uint8_t> contents = check(this->getObj().getSectionContents(sec));
    if (Error e = attributes.parse(contents, config->ekind == ELF32LEKind
                                                 ? support::little
                                                 : support::big)) {
      InputSection isec(*this, sec, name);
      warn(toString(&isec) + ": " + llvm::toString(std::move(e)));
    } else {
      updateSupportedARMFeatures(attributes);
      updateARMVFPArgs(attributes, this);

      // FIXME: Retain the first attribute section we see. The eglibc ARM
      // dynamic loaders require the presence of an attribute section for dlopen
      // to work. In a full implementation we would merge all attribute
      // sections.
      if (in.attributes == nullptr) {
        in.attributes = std::make_unique<InputSection>(*this, sec, name);
        return in.attributes.get();
      }
      return &InputSection::discarded;
    }
  }

  if (sec.sh_type == SHT_RISCV_ATTRIBUTES && config->emachine == EM_RISCV) {
    RISCVAttributeParser attributes;
    ArrayRef<uint8_t> contents = check(this->getObj().getSectionContents(sec));
    if (Error e = attributes.parse(contents, support::little)) {
      InputSection isec(*this, sec, name);
      warn(toString(&isec) + ": " + llvm::toString(std::move(e)));
    } else {
      // FIXME: Validate arch tag contains C if and only if EF_RISCV_RVC is
      // present.

      // FIXME: Retain the first attribute section we see. Tools such as
      // llvm-objdump make use of the attribute section to determine which
      // standard extensions to enable. In a full implementation we would merge
      // all attribute sections.
      if (in.attributes == nullptr) {
        in.attributes = std::make_unique<InputSection>(*this, sec, name);
        return in.attributes.get();
      }
      return &InputSection::discarded;
    }
  }

  if (sec.sh_type == SHT_LLVM_DEPENDENT_LIBRARIES && !config->relocatable) {
    ArrayRef<char> data =
        CHECK(this->getObj().template getSectionContentsAsArray<char>(sec), this);
    if (!data.empty() && data.back() != '\0') {
      error(toString(this) +
            ": corrupted dependent libraries section (unterminated string): " +
            name);
      return &InputSection::discarded;
    }
    for (const char *d = data.begin(), *e = data.end(); d < e;) {
      StringRef s(d);
      addDependentLibrary(s, this);
      d += s.size() + 1;
    }
    return &InputSection::discarded;
  }

  if (name.startswith(".n")) {
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
    if (name == ".note.GNU-stack")
      return &InputSection::discarded;

    // Object files that use processor features such as Intel Control-Flow
    // Enforcement (CET) or AArch64 Branch Target Identification BTI, use a
    // .note.gnu.property section containing a bitfield of feature bits like the
    // GNU_PROPERTY_X86_FEATURE_1_IBT flag. Read a bitmap containing the flag.
    //
    // Since we merge bitmaps from multiple object files to create a new
    // .note.gnu.property containing a single AND'ed bitmap, we discard an input
    // file's .note.gnu.property section.
    if (name == ".note.gnu.property") {
      this->andFeatures = readAndFeatures<ELFT>(InputSection(*this, sec, name));
      return &InputSection::discarded;
    }

    // Split stacks is a feature to support a discontiguous stack,
    // commonly used in the programming language Go. For the details,
    // see https://gcc.gnu.org/wiki/SplitStacks. An object file compiled
    // for split stack will include a .note.GNU-split-stack section.
    if (name == ".note.GNU-split-stack") {
      if (config->relocatable) {
        error(
            "cannot mix split-stack and non-split-stack in a relocatable link");
        return &InputSection::discarded;
      }
      this->splitStack = true;
      return &InputSection::discarded;
    }

    // An object file compiled for split stack, but where some of the
    // functions were compiled with the no_split_stack_attribute will
    // include a .note.GNU-no-split-stack section.
    if (name == ".note.GNU-no-split-stack") {
      this->someNoSplitStack = true;
      return &InputSection::discarded;
    }

    // Strip existing .note.gnu.build-id sections so that the output won't have
    // more than one build-id. This is not usually a problem because input
    // object files normally don't have .build-id sections, but you can create
    // such files by "ld.{bfd,gold,lld} -r --build-id", and we want to guard
    // against it.
    if (name == ".note.gnu.build-id")
      return &InputSection::discarded;
  }

  // The linker merges EH (exception handling) frames and creates a
  // .eh_frame_hdr section for runtime. So we handle them with a special
  // class. For relocatable outputs, they are just passed through.
  if (name == ".eh_frame" && !config->relocatable)
    return make<EhInputSection>(*this, sec, name);

  if ((sec.sh_flags & SHF_MERGE) && shouldMerge(sec, name))
    return make<MergeInputSection>(*this, sec, name);
  return make<InputSection>(*this, sec, name);
}

// Initialize this->Symbols. this->Symbols is a parallel array as
// its corresponding ELF symbol table.
template <class ELFT>
void ObjFile<ELFT>::initializeSymbols(const object::ELFFile<ELFT> &obj) {
  SymbolTable &symtab = *elf::symtab;

  ArrayRef<Elf_Sym> eSyms = this->getELFSyms<ELFT>();
  symbols.resize(eSyms.size());

  // Some entries have been filled by LazyObjFile.
  for (size_t i = firstGlobal, end = eSyms.size(); i != end; ++i)
    if (!symbols[i])
      symbols[i] = symtab.insert(CHECK(eSyms[i].getName(stringTable), this));

  // Perform symbol resolution on non-local symbols.
  SmallVector<unsigned, 32> undefineds;
  for (size_t i = firstGlobal, end = eSyms.size(); i != end; ++i) {
    const Elf_Sym &eSym = eSyms[i];
    uint32_t secIdx = eSym.st_shndx;
    if (secIdx == SHN_UNDEF) {
      undefineds.push_back(i);
      continue;
    }

    uint8_t binding = eSym.getBinding();
    uint8_t stOther = eSym.st_other;
    uint8_t type = eSym.getType();
    uint64_t value = eSym.st_value;
    uint64_t size = eSym.st_size;

    Symbol *sym = symbols[i];
    sym->isUsedInRegularObj = true;
    if (LLVM_UNLIKELY(eSym.st_shndx == SHN_COMMON)) {
      if (value == 0 || value >= UINT32_MAX)
        fatal(toString(this) + ": common symbol '" + sym->getName() +
              "' has invalid alignment: " + Twine(value));
      hasCommonSyms = true;
      sym->resolve(
          CommonSymbol{this, StringRef(), binding, stOther, type, value, size});
      continue;
    }

    // Handle global defined symbols. Defined::section will be set in postParse.
    sym->resolve(Defined{this, StringRef(), binding, stOther, type, value, size,
                         nullptr});
  }

  // Undefined symbols (excluding those defined relative to non-prevailing
  // sections) can trigger recursive extract. Process defined symbols first so
  // that the relative order between a defined symbol and an undefined symbol
  // does not change the symbol resolution behavior. In addition, a set of
  // interconnected symbols will all be resolved to the same file, instead of
  // being resolved to different files.
  for (unsigned i : undefineds) {
    const Elf_Sym &eSym = eSyms[i];
    Symbol *sym = symbols[i];
    sym->resolve(Undefined{this, StringRef(), eSym.getBinding(), eSym.st_other,
                           eSym.getType()});
    sym->isUsedInRegularObj = true;
    sym->referenced = true;
  }
}

template <class ELFT> void ObjFile<ELFT>::initializeLocalSymbols() {
  if (!firstGlobal)
    return;
  SymbolUnion *locals = makeThreadLocalN<SymbolUnion>(firstGlobal);

  ArrayRef<Elf_Sym> eSyms = this->getELFSyms<ELFT>();
  for (size_t i = 0, end = firstGlobal; i != end; ++i) {
    const Elf_Sym &eSym = eSyms[i];
    uint32_t secIdx = eSym.st_shndx;
    if (LLVM_UNLIKELY(secIdx == SHN_XINDEX))
      secIdx = check(getExtendedSymbolTableIndex<ELFT>(eSym, i, shndxTable));
    else if (secIdx >= SHN_LORESERVE)
      secIdx = 0;
    if (LLVM_UNLIKELY(secIdx >= sections.size()))
      fatal(toString(this) + ": invalid section index: " + Twine(secIdx));
    if (LLVM_UNLIKELY(eSym.getBinding() != STB_LOCAL))
      error(toString(this) + ": non-local symbol (" + Twine(i) +
            ") found at index < .symtab's sh_info (" + Twine(end) + ")");

    InputSectionBase *sec = sections[secIdx];
    uint8_t type = eSym.getType();
    if (type == STT_FILE)
      sourceFile = CHECK(eSym.getName(stringTable), this);
    if (LLVM_UNLIKELY(stringTable.size() <= eSym.st_name))
      fatal(toString(this) + ": invalid symbol name offset");
    StringRef name(stringTable.data() + eSym.st_name);

    symbols[i] = reinterpret_cast<Symbol *>(locals + i);
    if (eSym.st_shndx == SHN_UNDEF || sec == &InputSection::discarded)
      new (symbols[i]) Undefined(this, name, STB_LOCAL, eSym.st_other, type,
                                 /*discardedSecIdx=*/secIdx);
    else
      new (symbols[i]) Defined(this, name, STB_LOCAL, eSym.st_other, type,
                               eSym.st_value, eSym.st_size, sec);
    symbols[i]->isUsedInRegularObj = true;
  }
}

// Called after all ObjFile::parse is called for all ObjFiles. This checks
// duplicate symbols and may do symbol property merge in the future.
template <class ELFT> void ObjFile<ELFT>::postParse() {
  static std::mutex mu;
  ArrayRef<Elf_Sym> eSyms = this->getELFSyms<ELFT>();
  for (size_t i = firstGlobal, end = eSyms.size(); i != end; ++i) {
    const Elf_Sym &eSym = eSyms[i];
    Symbol &sym = *symbols[i];
    uint32_t secIdx = eSym.st_shndx;
    uint8_t binding = eSym.getBinding();
    if (LLVM_UNLIKELY(binding != STB_GLOBAL && binding != STB_WEAK &&
                      binding != STB_GNU_UNIQUE))
      errorOrWarn(toString(this) + ": symbol (" + Twine(i) +
                  ") has invalid binding: " + Twine((int)binding));

    // st_value of STT_TLS represents the assigned offset, not the actual
    // address which is used by STT_FUNC and STT_OBJECT. STT_TLS symbols can
    // only be referenced by special TLS relocations. It is usually an error if
    // a STT_TLS symbol is replaced by a non-STT_TLS symbol, vice versa.
    if (LLVM_UNLIKELY(sym.isTls()) && eSym.getType() != STT_TLS &&
        eSym.getType() != STT_NOTYPE)
      errorOrWarn("TLS attribute mismatch: " + toString(sym) + "\n>>> in " +
                  toString(sym.file) + "\n>>> in " + toString(this));

    // Handle non-COMMON defined symbol below. !sym.file allows a symbol
    // assignment to redefine a symbol without an error.
    if (!sym.file || !sym.isDefined() || secIdx == SHN_UNDEF ||
        secIdx == SHN_COMMON)
      continue;

    if (LLVM_UNLIKELY(secIdx == SHN_XINDEX))
      secIdx = check(getExtendedSymbolTableIndex<ELFT>(eSym, i, shndxTable));
    else if (secIdx >= SHN_LORESERVE)
      secIdx = 0;
    if (LLVM_UNLIKELY(secIdx >= sections.size()))
      fatal(toString(this) + ": invalid section index: " + Twine(secIdx));
    InputSectionBase *sec = sections[secIdx];
    if (sec == &InputSection::discarded) {
      if (sym.traced) {
        printTraceSymbol(Undefined{this, sym.getName(), sym.binding,
                                   sym.stOther, sym.type, secIdx},
                         sym.getName());
      }
      if (sym.file == this) {
        std::lock_guard<std::mutex> lock(mu);
        ctx->nonPrevailingSyms.emplace_back(&sym, secIdx);
      }
      continue;
    }

    if (sym.file == this) {
      cast<Defined>(sym).section = sec;
      continue;
    }

    if (binding == STB_WEAK)
      continue;
    std::lock_guard<std::mutex> lock(mu);
    ctx->duplicates.push_back({&sym, this, sec, eSym.st_value});
  }
}

// The handling of tentative definitions (COMMON symbols) in archives is murky.
// A tentative definition will be promoted to a global definition if there are
// no non-tentative definitions to dominate it. When we hold a tentative
// definition to a symbol and are inspecting archive members for inclusion
// there are 2 ways we can proceed:
//
// 1) Consider the tentative definition a 'real' definition (ie promotion from
//    tentative to real definition has already happened) and not inspect
//    archive members for Global/Weak definitions to replace the tentative
//    definition. An archive member would only be included if it satisfies some
//    other undefined symbol. This is the behavior Gold uses.
//
// 2) Consider the tentative definition as still undefined (ie the promotion to
//    a real definition happens only after all symbol resolution is done).
//    The linker searches archive members for STB_GLOBAL definitions to
//    replace the tentative definition with. This is the behavior used by
//    GNU ld.
//
//  The second behavior is inherited from SysVR4, which based it on the FORTRAN
//  COMMON BLOCK model. This behavior is needed for proper initialization in old
//  (pre F90) FORTRAN code that is packaged into an archive.
//
//  The following functions search archive members for definitions to replace
//  tentative definitions (implementing behavior 2).
static bool isBitcodeNonCommonDef(MemoryBufferRef mb, StringRef symName,
                                  StringRef archiveName) {
  IRSymtabFile symtabFile = check(readIRSymtab(mb));
  for (const irsymtab::Reader::SymbolRef &sym :
       symtabFile.TheReader.symbols()) {
    if (sym.isGlobal() && sym.getName() == symName)
      return !sym.isUndefined() && !sym.isWeak() && !sym.isCommon();
  }
  return false;
}

template <class ELFT>
static bool isNonCommonDef(MemoryBufferRef mb, StringRef symName,
                           StringRef archiveName) {
  ObjFile<ELFT> *obj = make<ObjFile<ELFT>>(mb, archiveName);
  StringRef stringtable = obj->getStringTable();

  for (auto sym : obj->template getGlobalELFSyms<ELFT>()) {
    Expected<StringRef> name = sym.getName(stringtable);
    if (name && name.get() == symName)
      return sym.isDefined() && sym.getBinding() == STB_GLOBAL &&
             !sym.isCommon();
  }
  return false;
}

static bool isNonCommonDef(MemoryBufferRef mb, StringRef symName,
                           StringRef archiveName) {
  switch (getELFKind(mb, archiveName)) {
  case ELF32LEKind:
    return isNonCommonDef<ELF32LE>(mb, symName, archiveName);
  case ELF32BEKind:
    return isNonCommonDef<ELF32BE>(mb, symName, archiveName);
  case ELF64LEKind:
    return isNonCommonDef<ELF64LE>(mb, symName, archiveName);
  case ELF64BEKind:
    return isNonCommonDef<ELF64BE>(mb, symName, archiveName);
  default:
    llvm_unreachable("getELFKind");
  }
}

unsigned SharedFile::vernauxNum;

// Parse the version definitions in the object file if present, and return a
// vector whose nth element contains a pointer to the Elf_Verdef for version
// identifier n. Version identifiers that are not definitions map to nullptr.
template <typename ELFT>
static SmallVector<const void *, 0>
parseVerdefs(const uint8_t *base, const typename ELFT::Shdr *sec) {
  if (!sec)
    return {};

  // Build the Verdefs array by following the chain of Elf_Verdef objects
  // from the start of the .gnu.version_d section.
  SmallVector<const void *, 0> verdefs;
  const uint8_t *verdef = base + sec->sh_offset;
  for (unsigned i = 0, e = sec->sh_info; i != e; ++i) {
    auto *curVerdef = reinterpret_cast<const typename ELFT::Verdef *>(verdef);
    verdef += curVerdef->vd_next;
    unsigned verdefIndex = curVerdef->vd_ndx;
    if (verdefIndex >= verdefs.size())
      verdefs.resize(verdefIndex + 1);
    verdefs[verdefIndex] = curVerdef;
  }
  return verdefs;
}

// Parse SHT_GNU_verneed to properly set the name of a versioned undefined
// symbol. We detect fatal issues which would cause vulnerabilities, but do not
// implement sophisticated error checking like in llvm-readobj because the value
// of such diagnostics is low.
template <typename ELFT>
std::vector<uint32_t> SharedFile::parseVerneed(const ELFFile<ELFT> &obj,
                                               const typename ELFT::Shdr *sec) {
  if (!sec)
    return {};
  std::vector<uint32_t> verneeds;
  ArrayRef<uint8_t> data = CHECK(obj.getSectionContents(*sec), this);
  const uint8_t *verneedBuf = data.begin();
  for (unsigned i = 0; i != sec->sh_info; ++i) {
    if (verneedBuf + sizeof(typename ELFT::Verneed) > data.end())
      fatal(toString(this) + " has an invalid Verneed");
    auto *vn = reinterpret_cast<const typename ELFT::Verneed *>(verneedBuf);
    const uint8_t *vernauxBuf = verneedBuf + vn->vn_aux;
    for (unsigned j = 0; j != vn->vn_cnt; ++j) {
      if (vernauxBuf + sizeof(typename ELFT::Vernaux) > data.end())
        fatal(toString(this) + " has an invalid Vernaux");
      auto *aux = reinterpret_cast<const typename ELFT::Vernaux *>(vernauxBuf);
      if (aux->vna_name >= this->stringTable.size())
        fatal(toString(this) + " has a Vernaux with an invalid vna_name");
      uint16_t version = aux->vna_other & VERSYM_VERSION;
      if (version >= verneeds.size())
        verneeds.resize(version + 1);
      verneeds[version] = aux->vna_name;
      vernauxBuf += aux->vna_next;
    }
    verneedBuf += vn->vn_next;
  }
  return verneeds;
}

// We do not usually care about alignments of data in shared object
// files because the loader takes care of it. However, if we promote a
// DSO symbol to point to .bss due to copy relocation, we need to keep
// the original alignment requirements. We infer it in this function.
template <typename ELFT>
static uint64_t getAlignment(ArrayRef<typename ELFT::Shdr> sections,
                             const typename ELFT::Sym &sym) {
  uint64_t ret = UINT64_MAX;
  if (sym.st_value)
    ret = 1ULL << countTrailingZeros((uint64_t)sym.st_value);
  if (0 < sym.st_shndx && sym.st_shndx < sections.size())
    ret = std::min<uint64_t>(ret, sections[sym.st_shndx].sh_addralign);
  return (ret > UINT32_MAX) ? 0 : ret;
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

  ArrayRef<Elf_Dyn> dynamicTags;
  const ELFFile<ELFT> obj = this->getObj<ELFT>();
  ArrayRef<Elf_Shdr> sections = getELFShdrs<ELFT>();

  const Elf_Shdr *versymSec = nullptr;
  const Elf_Shdr *verdefSec = nullptr;
  const Elf_Shdr *verneedSec = nullptr;

  // Search for .dynsym, .dynamic, .symtab, .gnu.version and .gnu.version_d.
  for (const Elf_Shdr &sec : sections) {
    switch (sec.sh_type) {
    default:
      continue;
    case SHT_DYNAMIC:
      dynamicTags =
          CHECK(obj.template getSectionContentsAsArray<Elf_Dyn>(sec), this);
      break;
    case SHT_GNU_versym:
      versymSec = &sec;
      break;
    case SHT_GNU_verdef:
      verdefSec = &sec;
      break;
    case SHT_GNU_verneed:
      verneedSec = &sec;
      break;
    }
  }

  if (versymSec && numELFSyms == 0) {
    error("SHT_GNU_versym should be associated with symbol table");
    return;
  }

  // Search for a DT_SONAME tag to initialize this->soName.
  for (const Elf_Dyn &dyn : dynamicTags) {
    if (dyn.d_tag == DT_NEEDED) {
      uint64_t val = dyn.getVal();
      if (val >= this->stringTable.size())
        fatal(toString(this) + ": invalid DT_NEEDED entry");
      dtNeeded.push_back(this->stringTable.data() + val);
    } else if (dyn.d_tag == DT_SONAME) {
      uint64_t val = dyn.getVal();
      if (val >= this->stringTable.size())
        fatal(toString(this) + ": invalid DT_SONAME entry");
      soName = this->stringTable.data() + val;
    }
  }

  // DSOs are uniquified not by filename but by soname.
  DenseMap<CachedHashStringRef, SharedFile *>::iterator it;
  bool wasInserted;
  std::tie(it, wasInserted) =
      symtab->soNames.try_emplace(CachedHashStringRef(soName), this);

  // If a DSO appears more than once on the command line with and without
  // --as-needed, --no-as-needed takes precedence over --as-needed because a
  // user can add an extra DSO with --no-as-needed to force it to be added to
  // the dependency list.
  it->second->isNeeded |= isNeeded;
  if (!wasInserted)
    return;

  ctx->sharedFiles.push_back(this);

  verdefs = parseVerdefs<ELFT>(obj.base(), verdefSec);
  std::vector<uint32_t> verneeds = parseVerneed<ELFT>(obj, verneedSec);

  // Parse ".gnu.version" section which is a parallel array for the symbol
  // table. If a given file doesn't have a ".gnu.version" section, we use
  // VER_NDX_GLOBAL.
  size_t size = numELFSyms - firstGlobal;
  std::vector<uint16_t> versyms(size, VER_NDX_GLOBAL);
  if (versymSec) {
    ArrayRef<Elf_Versym> versym =
        CHECK(obj.template getSectionContentsAsArray<Elf_Versym>(*versymSec),
              this)
            .slice(firstGlobal);
    for (size_t i = 0; i < size; ++i)
      versyms[i] = versym[i].vs_index;
  }

  // System libraries can have a lot of symbols with versions. Using a
  // fixed buffer for computing the versions name (foo@ver) can save a
  // lot of allocations.
  SmallString<0> versionedNameBuffer;

  // Add symbols to the symbol table.
  SymbolTable &symtab = *elf::symtab;
  ArrayRef<Elf_Sym> syms = this->getGlobalELFSyms<ELFT>();
  for (size_t i = 0, e = syms.size(); i != e; ++i) {
    const Elf_Sym &sym = syms[i];

    // ELF spec requires that all local symbols precede weak or global
    // symbols in each symbol table, and the index of first non-local symbol
    // is stored to sh_info. If a local symbol appears after some non-local
    // symbol, that's a violation of the spec.
    StringRef name = CHECK(sym.getName(stringTable), this);
    if (sym.getBinding() == STB_LOCAL) {
      warn("found local symbol '" + name +
           "' in global part of symbol table in file " + toString(this));
      continue;
    }

    uint16_t idx = versyms[i] & ~VERSYM_HIDDEN;
    if (sym.isUndefined()) {
      // For unversioned undefined symbols, VER_NDX_GLOBAL makes more sense but
      // as of binutils 2.34, GNU ld produces VER_NDX_LOCAL.
      if (idx != VER_NDX_LOCAL && idx != VER_NDX_GLOBAL) {
        if (idx >= verneeds.size()) {
          error("corrupt input file: version need index " + Twine(idx) +
                " for symbol " + name + " is out of bounds\n>>> defined in " +
                toString(this));
          continue;
        }
        StringRef verName = stringTable.data() + verneeds[idx];
        versionedNameBuffer.clear();
        name = saver().save(
            (name + "@" + verName).toStringRef(versionedNameBuffer));
      }
      Symbol *s = symtab.addSymbol(
          Undefined{this, name, sym.getBinding(), sym.st_other, sym.getType()});
      s->exportDynamic = true;
      if (s->isUndefined() && sym.getBinding() != STB_WEAK &&
          config->unresolvedSymbolsInShlib != UnresolvedPolicy::Ignore)
        requiredSymbols.push_back(s);
      continue;
    }

    // MIPS BFD linker puts _gp_disp symbol into DSO files and incorrectly
    // assigns VER_NDX_LOCAL to this section global symbol. Here is a
    // workaround for this bug.
    if (config->emachine == EM_MIPS && idx == VER_NDX_LOCAL &&
        name == "_gp_disp")
      continue;

    uint32_t alignment = getAlignment<ELFT>(sections, sym);
    if (!(versyms[i] & VERSYM_HIDDEN)) {
      auto *s = symtab.addSymbol(
          SharedSymbol{*this, name, sym.getBinding(), sym.st_other,
                       sym.getType(), sym.st_value, sym.st_size, alignment});
      if (s->file == this)
        s->verdefIndex = idx;
    }

    // Also add the symbol with the versioned name to handle undefined symbols
    // with explicit versions.
    if (idx == VER_NDX_GLOBAL)
      continue;

    if (idx >= verdefs.size() || idx == VER_NDX_LOCAL) {
      error("corrupt input file: version definition index " + Twine(idx) +
            " for symbol " + name + " is out of bounds\n>>> defined in " +
            toString(this));
      continue;
    }

    StringRef verName =
        stringTable.data() +
        reinterpret_cast<const Elf_Verdef *>(verdefs[idx])->getAux()->vda_name;
    versionedNameBuffer.clear();
    name = (name + "@" + verName).toStringRef(versionedNameBuffer);
    auto *s = symtab.addSymbol(
        SharedSymbol{*this, saver().save(name), sym.getBinding(), sym.st_other,
                     sym.getType(), sym.st_value, sym.st_size, alignment});
    if (s->file == this)
      s->verdefIndex = idx;
  }
}

static ELFKind getBitcodeELFKind(const Triple &t) {
  if (t.isLittleEndian())
    return t.isArch64Bit() ? ELF64LEKind : ELF32LEKind;
  return t.isArch64Bit() ? ELF64BEKind : ELF32BEKind;
}

static uint16_t getBitcodeMachineKind(StringRef path, const Triple &t) {
  switch (t.getArch()) {
  case Triple::aarch64:
  case Triple::aarch64_be:
    return EM_AARCH64;
  case Triple::amdgcn:
  case Triple::r600:
    return EM_AMDGPU;
  case Triple::arm:
  case Triple::thumb:
    return EM_ARM;
  case Triple::avr:
    return EM_AVR;
  case Triple::hexagon:
    return EM_HEXAGON;
  case Triple::mips:
  case Triple::mipsel:
  case Triple::mips64:
  case Triple::mips64el:
    return EM_MIPS;
  case Triple::msp430:
    return EM_MSP430;
  case Triple::ppc:
  case Triple::ppcle:
    return EM_PPC;
  case Triple::ppc64:
  case Triple::ppc64le:
    return EM_PPC64;
  case Triple::riscv32:
  case Triple::riscv64:
    return EM_RISCV;
  case Triple::x86:
    return t.isOSIAMCU() ? EM_IAMCU : EM_386;
  case Triple::x86_64:
    return EM_X86_64;
  default:
    error(path + ": could not infer e_machine from bitcode target triple " +
          t.str());
    return EM_NONE;
  }
}

static uint8_t getOsAbi(const Triple &t) {
  switch (t.getOS()) {
  case Triple::AMDHSA:
    return ELF::ELFOSABI_AMDGPU_HSA;
  case Triple::AMDPAL:
    return ELF::ELFOSABI_AMDGPU_PAL;
  case Triple::Mesa3D:
    return ELF::ELFOSABI_AMDGPU_MESA3D;
  default:
    return ELF::ELFOSABI_NONE;
  }
}

BitcodeFile::BitcodeFile(MemoryBufferRef mb, StringRef archiveName,
                         uint64_t offsetInArchive, bool lazy)
    : InputFile(BitcodeKind, mb) {
  this->archiveName = archiveName;
  this->lazy = lazy;

  std::string path = mb.getBufferIdentifier().str();
  if (config->thinLTOIndexOnly)
    path = replaceThinLTOSuffix(mb.getBufferIdentifier());

  // ThinLTO assumes that all MemoryBufferRefs given to it have a unique
  // name. If two archives define two members with the same name, this
  // causes a collision which result in only one of the objects being taken
  // into consideration at LTO time (which very likely causes undefined
  // symbols later in the link stage). So we append file offset to make
  // filename unique.
  StringRef name = archiveName.empty()
                       ? saver().save(path)
                       : saver().save(archiveName + "(" + path::filename(path) +
                                      " at " + utostr(offsetInArchive) + ")");
  MemoryBufferRef mbref(mb.getBuffer(), name);

  obj = CHECK(lto::InputFile::create(mbref), this);

  Triple t(obj->getTargetTriple());
  ekind = getBitcodeELFKind(t);
  emachine = getBitcodeMachineKind(mb.getBufferIdentifier(), t);
  osabi = getOsAbi(t);
}

static uint8_t mapVisibility(GlobalValue::VisibilityTypes gvVisibility) {
  switch (gvVisibility) {
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
static void
createBitcodeSymbol(Symbol *&sym, const std::vector<bool> &keptComdats,
                    const lto::InputFile::Symbol &objSym, BitcodeFile &f) {
  uint8_t binding = objSym.isWeak() ? STB_WEAK : STB_GLOBAL;
  uint8_t type = objSym.isTLS() ? STT_TLS : STT_NOTYPE;
  uint8_t visibility = mapVisibility(objSym.getVisibility());

  if (!sym)
    sym = symtab->insert(saver().save(objSym.getName()));

  int c = objSym.getComdatIndex();
  if (objSym.isUndefined() || (c != -1 && !keptComdats[c])) {
    Undefined newSym(&f, StringRef(), binding, visibility, type);
    sym->resolve(newSym);
    sym->referenced = true;
    return;
  }

  if (objSym.isCommon()) {
    sym->resolve(CommonSymbol{&f, StringRef(), binding, visibility, STT_OBJECT,
                              objSym.getCommonAlignment(),
                              objSym.getCommonSize()});
  } else {
    Defined newSym(&f, StringRef(), binding, visibility, type, 0, 0, nullptr);
    if (objSym.canBeOmittedFromSymbolTable())
      newSym.exportDynamic = false;
    sym->resolve(newSym);
  }
}

template <class ELFT> void BitcodeFile::parse() {
  for (std::pair<StringRef, Comdat::SelectionKind> s : obj->getComdatTable()) {
    keptComdats.push_back(
        s.second == Comdat::NoDeduplicate ||
        symtab->comdatGroups.try_emplace(CachedHashStringRef(s.first), this)
            .second);
  }

  symbols.resize(obj->symbols().size());
  // Process defined symbols first. See the comment in
  // ObjFile<ELFT>::initializeSymbols.
  for (auto it : llvm::enumerate(obj->symbols()))
    if (!it.value().isUndefined()) {
      Symbol *&sym = symbols[it.index()];
      createBitcodeSymbol<ELFT>(sym, keptComdats, it.value(), *this);
    }
  for (auto it : llvm::enumerate(obj->symbols()))
    if (it.value().isUndefined()) {
      Symbol *&sym = symbols[it.index()];
      createBitcodeSymbol<ELFT>(sym, keptComdats, it.value(), *this);
    }

  for (auto l : obj->getDependentLibraries())
    addDependentLibrary(l, this);
}

void BitcodeFile::parseLazy() {
  SymbolTable &symtab = *elf::symtab;
  symbols.resize(obj->symbols().size());
  for (auto it : llvm::enumerate(obj->symbols()))
    if (!it.value().isUndefined()) {
      auto *sym = symtab.insert(saver().save(it.value().getName()));
      sym->resolve(LazyObject{*this});
      symbols[it.index()] = sym;
    }
}

void BitcodeFile::postParse() {
  for (auto it : llvm::enumerate(obj->symbols())) {
    const Symbol &sym = *symbols[it.index()];
    const auto &objSym = it.value();
    if (sym.file == this || !sym.isDefined() || objSym.isUndefined() ||
        objSym.isCommon() || objSym.isWeak())
      continue;
    int c = objSym.getComdatIndex();
    if (c != -1 && !keptComdats[c])
      continue;
    reportDuplicate(sym, this, nullptr, 0);
  }
}

void BinaryFile::parse() {
  ArrayRef<uint8_t> data = arrayRefFromStringRef(mb.getBuffer());
  auto *section = make<InputSection>(this, SHF_ALLOC | SHF_WRITE, SHT_PROGBITS,
                                     8, data, ".data");
  sections.push_back(section);

  // For each input file foo that is embedded to a result as a binary
  // blob, we define _binary_foo_{start,end,size} symbols, so that
  // user programs can access blobs by name. Non-alphanumeric
  // characters in a filename are replaced with underscore.
  std::string s = "_binary_" + mb.getBufferIdentifier().str();
  for (size_t i = 0; i < s.size(); ++i)
    if (!isAlnum(s[i]))
      s[i] = '_';

  llvm::StringSaver &saver = lld::saver();

  symtab->addAndCheckDuplicate(Defined{nullptr, saver.save(s + "_start"),
                                       STB_GLOBAL, STV_DEFAULT, STT_OBJECT, 0,
                                       0, section});
  symtab->addAndCheckDuplicate(Defined{nullptr, saver.save(s + "_end"),
                                       STB_GLOBAL, STV_DEFAULT, STT_OBJECT,
                                       data.size(), 0, section});
  symtab->addAndCheckDuplicate(Defined{nullptr, saver.save(s + "_size"),
                                       STB_GLOBAL, STV_DEFAULT, STT_OBJECT,
                                       data.size(), 0, nullptr});
}

ELFFileBase *elf::createObjFile(MemoryBufferRef mb, StringRef archiveName,
                                bool lazy) {
  ELFFileBase *f;
  switch (getELFKind(mb, archiveName)) {
  case ELF32LEKind:
    f = make<ObjFile<ELF32LE>>(mb, archiveName);
    break;
  case ELF32BEKind:
    f = make<ObjFile<ELF32BE>>(mb, archiveName);
    break;
  case ELF64LEKind:
    f = make<ObjFile<ELF64LE>>(mb, archiveName);
    break;
  case ELF64BEKind:
    f = make<ObjFile<ELF64BE>>(mb, archiveName);
    break;
  default:
    llvm_unreachable("getELFKind");
  }
  f->lazy = lazy;
  return f;
}

template <class ELFT> void ObjFile<ELFT>::parseLazy() {
  const ArrayRef<typename ELFT::Sym> eSyms = this->getELFSyms<ELFT>();
  SymbolTable &symtab = *elf::symtab;

  symbols.resize(eSyms.size());
  for (size_t i = firstGlobal, end = eSyms.size(); i != end; ++i)
    if (eSyms[i].st_shndx != SHN_UNDEF)
      symbols[i] = symtab.insert(CHECK(eSyms[i].getName(stringTable), this));

  // Replace existing symbols with LazyObject symbols.
  //
  // resolve() may trigger this->extract() if an existing symbol is an undefined
  // symbol. If that happens, this function has served its purpose, and we can
  // exit from the loop early.
  for (Symbol *sym : makeArrayRef(symbols).slice(firstGlobal))
    if (sym) {
      sym->resolve(LazyObject{*this});
      if (!lazy)
        return;
    }
}

bool InputFile::shouldExtractForCommon(StringRef name) {
  if (isa<BitcodeFile>(this))
    return isBitcodeNonCommonDef(mb, name, archiveName);

  return isNonCommonDef(mb, name, archiveName);
}

std::string elf::replaceThinLTOSuffix(StringRef path) {
  StringRef suffix = config->thinLTOObjectSuffixReplace.first;
  StringRef repl = config->thinLTOObjectSuffixReplace.second;

  if (path.consume_back(suffix))
    return (path + repl).str();
  return std::string(path);
}

template void BitcodeFile::parse<ELF32LE>();
template void BitcodeFile::parse<ELF32BE>();
template void BitcodeFile::parse<ELF64LE>();
template void BitcodeFile::parse<ELF64BE>();

template class elf::ObjFile<ELF32LE>;
template class elf::ObjFile<ELF32BE>;
template class elf::ObjFile<ELF64LE>;
template class elf::ObjFile<ELF64BE>;

template void SharedFile::parse<ELF32LE>();
template void SharedFile::parse<ELF32BE>();
template void SharedFile::parse<ELF64LE>();
template void SharedFile::parse<ELF64BE>();
