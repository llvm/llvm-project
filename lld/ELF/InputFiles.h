//===- InputFiles.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_INPUT_FILES_H
#define LLD_ELF_INPUT_FILES_H

#include "Config.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/LLVM.h"
#include "lld/Common/Reproduce.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/IR/Comdat.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Support/Threading.h"
#include <map>

namespace llvm {
class TarWriter;
struct DILineInfo;
namespace lto {
class InputFile;
}
} // namespace llvm

namespace lld {
namespace elf {
class InputFile;
class InputSectionBase;
}

// Returns "<internal>", "foo.a(bar.o)" or "baz.o".
std::string toString(const elf::InputFile *F);

namespace elf {

using llvm::object::Archive;

class Symbol;

// If -reproduce option is given, all input files are written
// to this tar archive.
extern std::unique_ptr<llvm::TarWriter> Tar;

// Opens a given file.
llvm::Optional<MemoryBufferRef> readFile(StringRef Path);

// The root class of input files.
class InputFile {
public:
  enum Kind {
    ObjKind,
    SharedKind,
    LazyObjKind,
    ArchiveKind,
    BitcodeKind,
    BinaryKind,
  };

  Kind kind() const { return FileKind; }

  bool isElf() const {
    Kind K = kind();
    return K == ObjKind || K == SharedKind;
  }

  StringRef getName() const { return MB.getBufferIdentifier(); }
  MemoryBufferRef MB;

  // Returns sections. It is a runtime error to call this function
  // on files that don't have the notion of sections.
  ArrayRef<InputSectionBase *> getSections() const {
    assert(FileKind == ObjKind || FileKind == BinaryKind);
    return Sections;
  }

  // Returns object file symbols. It is a runtime error to call this
  // function on files of other types.
  ArrayRef<Symbol *> getSymbols() { return getMutableSymbols(); }

  std::vector<Symbol *> &getMutableSymbols() {
    assert(FileKind == BinaryKind || FileKind == ObjKind ||
           FileKind == BitcodeKind);
    return Symbols;
  }

  // Filename of .a which contained this file. If this file was
  // not in an archive file, it is the empty string. We use this
  // string for creating error messages.
  std::string ArchiveName;

  // If this is an architecture-specific file, the following members
  // have ELF type (i.e. ELF{32,64}{LE,BE}) and target machine type.
  ELFKind EKind = ELFNoneKind;
  uint16_t EMachine = llvm::ELF::EM_NONE;
  uint8_t OSABI = 0;
  uint8_t ABIVersion = 0;

  // Cache for toString(). Only toString() should use this member.
  mutable std::string ToStringCache;

  std::string getSrcMsg(const Symbol &Sym, InputSectionBase &Sec,
                        uint64_t Offset);

  // True if this is an argument for --just-symbols. Usually false.
  bool JustSymbols = false;

  // On PPC64 we need to keep track of which files contain small code model
  // relocations that access the .toc section. To minimize the chance of a
  // relocation overflow, files that do contain said relocations should have
  // their .toc sections sorted closer to the .got section than files that do
  // not contain any small code model relocations. Thats because the toc-pointer
  // is defined to point at .got + 0x8000 and the instructions used with small
  // code model relocations support immediates in the range [-0x8000, 0x7FFC],
  // making the addressable range relative to the toc pointer
  // [.got, .got + 0xFFFC].
  bool PPC64SmallCodeModelTocRelocs = false;

  // GroupId is used for --warn-backrefs which is an optional error
  // checking feature. All files within the same --{start,end}-group or
  // --{start,end}-lib get the same group ID. Otherwise, each file gets a new
  // group ID. For more info, see checkDependency() in SymbolTable.cpp.
  uint32_t GroupId;
  static bool IsInGroup;
  static uint32_t NextGroupId;

  // Index of MIPS GOT built for this file.
  llvm::Optional<size_t> MipsGotIndex;

protected:
  InputFile(Kind K, MemoryBufferRef M);
  std::vector<InputSectionBase *> Sections;
  std::vector<Symbol *> Symbols;

private:
  const Kind FileKind;
};

class ELFFileBase : public InputFile {
public:
  ELFFileBase(Kind K, MemoryBufferRef M);
  template <typename ELFT> void parseHeader();
  static bool classof(const InputFile *F) { return F->isElf(); }

  template <typename ELFT> llvm::object::ELFFile<ELFT> getObj() const {
    return check(llvm::object::ELFFile<ELFT>::create(MB.getBuffer()));
  }

  StringRef getStringTable() const { return StringTable; }

  template <typename ELFT> typename ELFT::SymRange getELFSyms() const {
    return typename ELFT::SymRange(
        reinterpret_cast<const typename ELFT::Sym *>(ELFSyms), NumELFSyms);
  }
  template <typename ELFT> typename ELFT::SymRange getGlobalELFSyms() const {
    return getELFSyms<ELFT>().slice(FirstGlobal);
  }

protected:
  const void *ELFSyms = nullptr;
  size_t NumELFSyms = 0;
  uint32_t FirstGlobal = 0;
  StringRef StringTable;
  template <typename ELFT>
  void initSymtab(ArrayRef<typename ELFT::Shdr> Sections,
                  const typename ELFT::Shdr *Symtab);
};

// .o file.
template <class ELFT> class ObjFile : public ELFFileBase {
  using Elf_Rel = typename ELFT::Rel;
  using Elf_Rela = typename ELFT::Rela;
  using Elf_Sym = typename ELFT::Sym;
  using Elf_Shdr = typename ELFT::Shdr;
  using Elf_Word = typename ELFT::Word;
  using Elf_CGProfile = typename ELFT::CGProfile;

  StringRef getShtGroupSignature(ArrayRef<Elf_Shdr> Sections,
                                 const Elf_Shdr &Sec);

public:
  static bool classof(const InputFile *F) { return F->kind() == ObjKind; }

  llvm::object::ELFFile<ELFT> getObj() const {
    return this->ELFFileBase::getObj<ELFT>();
  }

  ArrayRef<Symbol *> getLocalSymbols();
  ArrayRef<Symbol *> getGlobalSymbols();

  ObjFile(MemoryBufferRef M, StringRef ArchiveName);
  void parse(llvm::DenseSet<llvm::CachedHashStringRef> &ComdatGroups);

  Symbol &getSymbol(uint32_t SymbolIndex) const {
    if (SymbolIndex >= this->Symbols.size())
      fatal(toString(this) + ": invalid symbol index");
    return *this->Symbols[SymbolIndex];
  }

  uint32_t getSectionIndex(const Elf_Sym &Sym) const;

  template <typename RelT> Symbol &getRelocTargetSym(const RelT &Rel) const {
    uint32_t SymIndex = Rel.getSymbol(Config->IsMips64EL);
    return getSymbol(SymIndex);
  }

  llvm::Optional<llvm::DILineInfo> getDILineInfo(InputSectionBase *, uint64_t);
  llvm::Optional<std::pair<std::string, unsigned>> getVariableLoc(StringRef Name);

  // MIPS GP0 value defined by this file. This value represents the gp value
  // used to create the relocatable object and required to support
  // R_MIPS_GPREL16 / R_MIPS_GPREL32 relocations.
  uint32_t MipsGp0 = 0;

  // Name of source file obtained from STT_FILE symbol value,
  // or empty string if there is no such symbol in object file
  // symbol table.
  StringRef SourceFile;

  // True if the file defines functions compiled with
  // -fsplit-stack. Usually false.
  bool SplitStack = false;

  // True if the file defines functions compiled with -fsplit-stack,
  // but had one or more functions with the no_split_stack attribute.
  bool SomeNoSplitStack = false;

  // Pointer to this input file's .llvm_addrsig section, if it has one.
  const Elf_Shdr *AddrsigSec = nullptr;

  // SHT_LLVM_CALL_GRAPH_PROFILE table
  ArrayRef<Elf_CGProfile> CGProfile;

private:
  void
  initializeSections(llvm::DenseSet<llvm::CachedHashStringRef> &ComdatGroups);
  void initializeSymbols();
  void initializeJustSymbols();
  void initializeDwarf();
  InputSectionBase *getRelocTarget(const Elf_Shdr &Sec);
  InputSectionBase *createInputSection(const Elf_Shdr &Sec);
  StringRef getSectionName(const Elf_Shdr &Sec);

  bool shouldMerge(const Elf_Shdr &Sec);
  Symbol *createSymbol(const Elf_Sym *Sym);

  // Each ELF symbol contains a section index which the symbol belongs to.
  // However, because the number of bits dedicated for that is limited, a
  // symbol can directly point to a section only when the section index is
  // equal to or smaller than 65280.
  //
  // If an object file contains more than 65280 sections, the file must
  // contain .symtab_shndxr section. The section contains an array of
  // 32-bit integers whose size is the same as the number of symbols.
  // Nth symbol's section index is in the Nth entry of .symtab_shndxr.
  //
  // The following variable contains the contents of .symtab_shndxr.
  // If the section does not exist (which is common), the array is empty.
  ArrayRef<Elf_Word> ShndxTable;

  // .shstrtab contents.
  StringRef SectionStringTable;

  // Debugging information to retrieve source file and line for error
  // reporting. Linker may find reasonable number of errors in a
  // single object file, so we cache debugging information in order to
  // parse it only once for each object file we link.
  std::unique_ptr<llvm::DWARFContext> Dwarf;
  std::vector<const llvm::DWARFDebugLine::LineTable *> LineTables;
  struct VarLoc {
    const llvm::DWARFDebugLine::LineTable *LT;
    unsigned File;
    unsigned Line;
  };
  llvm::DenseMap<StringRef, VarLoc> VariableLoc;
  llvm::once_flag InitDwarfLine;
};

// LazyObjFile is analogous to ArchiveFile in the sense that
// the file contains lazy symbols. The difference is that
// LazyObjFile wraps a single file instead of multiple files.
//
// This class is used for --start-lib and --end-lib options which
// instruct the linker to link object files between them with the
// archive file semantics.
class LazyObjFile : public InputFile {
public:
  LazyObjFile(MemoryBufferRef M, StringRef ArchiveName,
              uint64_t OffsetInArchive)
      : InputFile(LazyObjKind, M), OffsetInArchive(OffsetInArchive) {
    this->ArchiveName = ArchiveName;
  }

  static bool classof(const InputFile *F) { return F->kind() == LazyObjKind; }

  template <class ELFT> void parse();
  MemoryBufferRef getBuffer();
  InputFile *fetch();
  bool AddedToLink = false;

private:
  uint64_t OffsetInArchive;
};

// An ArchiveFile object represents a .a file.
class ArchiveFile : public InputFile {
public:
  explicit ArchiveFile(std::unique_ptr<Archive> &&File);
  static bool classof(const InputFile *F) { return F->kind() == ArchiveKind; }
  template <class ELFT> void parse();

  // Pulls out an object file that contains a definition for Sym and
  // returns it. If the same file was instantiated before, this
  // function returns a nullptr (so we don't instantiate the same file
  // more than once.)
  InputFile *fetch(const Archive::Symbol &Sym);

private:
  std::unique_ptr<Archive> File;
  llvm::DenseSet<uint64_t> Seen;
};

class BitcodeFile : public InputFile {
public:
  BitcodeFile(MemoryBufferRef M, StringRef ArchiveName,
              uint64_t OffsetInArchive);
  static bool classof(const InputFile *F) { return F->kind() == BitcodeKind; }
  template <class ELFT>
  void parse(llvm::DenseSet<llvm::CachedHashStringRef> &ComdatGroups);
  std::unique_ptr<llvm::lto::InputFile> Obj;
};

// .so file.
class SharedFile : public ELFFileBase {
public:
  // This is actually a vector of Elf_Verdef pointers.
  std::vector<const void *> Verdefs;

  // If the output file needs Elf_Verneed data structures for this file, this is
  // a vector of Elf_Vernaux version identifiers that map onto the entries in
  // Verdefs, otherwise it is empty.
  std::vector<unsigned> Vernauxs;

  static unsigned VernauxNum;

  std::vector<StringRef> DtNeeded;
  std::string SoName;

  static bool classof(const InputFile *F) { return F->kind() == SharedKind; }

  SharedFile(MemoryBufferRef M, StringRef DefaultSoName);

  template <typename ELFT> void parse();

  // Used for --no-allow-shlib-undefined.
  bool AllNeededIsKnown;

  // Used for --as-needed
  bool IsNeeded;
};

class BinaryFile : public InputFile {
public:
  explicit BinaryFile(MemoryBufferRef M) : InputFile(BinaryKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == BinaryKind; }
  void parse();
};

InputFile *createObjectFile(MemoryBufferRef MB, StringRef ArchiveName = "",
                            uint64_t OffsetInArchive = 0);
InputFile *createSharedFile(MemoryBufferRef MB, StringRef DefaultSoName);

inline bool isBitcode(MemoryBufferRef MB) {
  return identify_magic(MB.getBuffer()) == llvm::file_magic::bitcode;
}

std::string replaceThinLTOSuffix(StringRef Path);

extern std::vector<BinaryFile *> BinaryFiles;
extern std::vector<BitcodeFile *> BitcodeFiles;
extern std::vector<LazyObjFile *> LazyObjFiles;
extern std::vector<InputFile *> ObjectFiles;
extern std::vector<SharedFile *> SharedFiles;

} // namespace elf
} // namespace lld

#endif
