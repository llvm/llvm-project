//===- InputFiles.h ---------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_INPUT_FILES_H
#define LLD_ELF_INPUT_FILES_H

#include "Config.h"
#include "InputSection.h"
#include "Error.h"
#include "Symbols.h"

#include "lld/Core/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf2 {

using llvm::object::Archive;

class InputFile;
class Lazy;
class SymbolBody;

// The root class of input files.
class InputFile {
public:
  enum Kind { ObjectKind, SharedKind, ArchiveKind };
  Kind kind() const { return FileKind; }
  virtual ~InputFile();

  StringRef getName() const { return MB.getBufferIdentifier(); }

protected:
  InputFile(Kind K, MemoryBufferRef M) : MB(M), FileKind(K) {}
  MemoryBufferRef MB;

private:
  const Kind FileKind;
};

template <typename ELFT> class ELFFileBase : public InputFile {
public:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

  ELFFileBase(Kind K, MemoryBufferRef M);
  static bool classof(const InputFile *F) {
    Kind K = F->kind();
    return K == ObjectKind || K == SharedKind;
  }

  static ELFKind getELFKind();
  const llvm::object::ELFFile<ELFT> &getObj() const { return ELFObj; }
  llvm::object::ELFFile<ELFT> &getObj() { return ELFObj; }

  uint16_t getEMachine() const { return getObj().getHeader()->e_machine; }
  uint8_t getOSABI() const {
    return getObj().getHeader()->e_ident[llvm::ELF::EI_OSABI];
  }

  StringRef getStringTable() const { return StringTable; }

  uint32_t getSectionIndex(const Elf_Sym &Sym) const;

protected:
  llvm::object::ELFFile<ELFT> ELFObj;
  const Elf_Shdr *Symtab = nullptr;
  ArrayRef<Elf_Word> SymtabSHNDX;
  StringRef StringTable;
  void initStringTable();
  Elf_Sym_Range getNonLocalSymbols();
  Elf_Sym_Range getSymbolsHelper(bool);
};

// .o file.
template <class ELFT> class ObjectFile : public ELFFileBase<ELFT> {
  typedef ELFFileBase<ELFT> Base;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

  typedef llvm::support::detail::packed_endian_specific_integral<
      uint32_t, ELFT::TargetEndianness, 2> GroupEntryType;
  StringRef getShtGroupSignature(const Elf_Shdr &Sec);
  ArrayRef<GroupEntryType> getShtGroupEntries(const Elf_Shdr &Sec);

public:
  static bool classof(const InputFile *F) {
    return F->kind() == Base::ObjectKind;
  }

  ArrayRef<SymbolBody *> getSymbols() { return this->SymbolBodies; }

  explicit ObjectFile(MemoryBufferRef M);
  void parse(llvm::DenseSet<StringRef> &Comdats);

  ArrayRef<InputSectionBase<ELFT> *> getSections() const { return Sections; }
  InputSectionBase<ELFT> *getSection(const Elf_Sym &Sym) const;

  SymbolBody *getSymbolBody(uint32_t SymbolIndex) const {
    uint32_t FirstNonLocal = this->Symtab->sh_info;
    if (SymbolIndex < FirstNonLocal)
      return nullptr;
    return this->SymbolBodies[SymbolIndex - FirstNonLocal];
  }

  Elf_Sym_Range getLocalSymbols();
  const Elf_Sym *getLocalSymbol(uintX_t SymIndex);

  const Elf_Shdr *getSymbolTable() const { return this->Symtab; };

private:
  void initializeSections(llvm::DenseSet<StringRef> &Comdats);
  void initializeSymbols();

  SymbolBody *createSymbolBody(StringRef StringTable, const Elf_Sym *Sym);

  // List of all sections defined by this file.
  std::vector<InputSectionBase<ELFT> *> Sections;

  // List of all symbols referenced or defined by this file.
  std::vector<SymbolBody *> SymbolBodies;

  llvm::BumpPtrAllocator Alloc;
  llvm::SpecificBumpPtrAllocator<MergeInputSection<ELFT>> MAlloc;
  llvm::SpecificBumpPtrAllocator<EHInputSection<ELFT>> EHAlloc;
};

class ArchiveFile : public InputFile {
public:
  explicit ArchiveFile(MemoryBufferRef M) : InputFile(ArchiveKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == ArchiveKind; }
  void parse();

  // Returns a memory buffer for a given symbol. An empty memory buffer
  // is returned if we have already returned the same memory buffer.
  // (So that we don't instantiate same members more than once.)
  MemoryBufferRef getMember(const Archive::Symbol *Sym);

  llvm::MutableArrayRef<Lazy> getLazySymbols() { return LazySymbols; }
  std::vector<MemoryBufferRef> getMembers();

private:
  std::unique_ptr<Archive> File;
  std::vector<Lazy> LazySymbols;
  llvm::DenseSet<uint64_t> Seen;
};

// .so file.
template <class ELFT> class SharedFile : public ELFFileBase<ELFT> {
  typedef ELFFileBase<ELFT> Base;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

  std::vector<SharedSymbol<ELFT>> SymbolBodies;
  std::vector<StringRef> Undefs;
  StringRef SoName;

public:
  StringRef getSoName() const { return SoName; }
  llvm::MutableArrayRef<SharedSymbol<ELFT>> getSharedSymbols() {
    return SymbolBodies;
  }
  const Elf_Shdr *getSection(const Elf_Sym &Sym) const;
  llvm::ArrayRef<StringRef> getUndefinedSymbols() { return Undefs; }

  static bool classof(const InputFile *F) {
    return F->kind() == Base::SharedKind;
  }

  explicit SharedFile(MemoryBufferRef M);

  void parseSoName();
  void parse();

  // Used for --as-needed
  bool AsNeeded = false;
  bool IsUsed = false;
  bool isNeeded() const { return !AsNeeded || IsUsed; }
};

template <template <class> class T>
std::unique_ptr<InputFile> createELFFile(MemoryBufferRef MB);

} // namespace elf2
} // namespace lld

#endif
