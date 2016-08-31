//===- SymbolTable.h --------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SYMBOL_TABLE_H
#define LLD_ELF_SYMBOL_TABLE_H

#include "InputFiles.h"
#include "LTO.h"
#include "llvm/ADT/DenseMap.h"

namespace lld {
namespace elf {
class Lazy;
template <class ELFT> class OutputSectionBase;
struct Symbol;

typedef llvm::CachedHash<StringRef> SymName;

// SymbolTable is a bucket of all known symbols, including defined,
// undefined, or lazy symbols (the last one is symbols in archive
// files whose archive members are not yet loaded).
//
// We put all symbols of all files to a SymbolTable, and the
// SymbolTable selects the "best" symbols if there are name
// conflicts. For example, obviously, a defined symbol is better than
// an undefined symbol. Or, if there's a conflict between a lazy and a
// undefined, it'll read an archive member to read a real definition
// to replace the lazy symbol. The logic is implemented in the
// add*() functions, which are called by input files as they are parsed. There
// is one add* function per symbol type.
template <class ELFT> class SymbolTable {
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::uint uintX_t;

public:
  void addFile(std::unique_ptr<InputFile> File);
  void addCombinedLtoObject();

  llvm::ArrayRef<Symbol *> getSymbols() const { return SymVector; }

  const std::vector<std::unique_ptr<ObjectFile<ELFT>>> &getObjectFiles() const {
    return ObjectFiles;
  }

  const std::vector<std::unique_ptr<SharedFile<ELFT>>> &getSharedFiles() const {
    return SharedFiles;
  }

  DefinedRegular<ELFT> *addAbsolute(StringRef Name,
                                    uint8_t Visibility = llvm::ELF::STV_HIDDEN);
  DefinedRegular<ELFT> *addIgnored(StringRef Name,
                                   uint8_t Visibility = llvm::ELF::STV_HIDDEN);

  Symbol *addUndefined(StringRef Name);
  Symbol *addUndefined(StringRef Name, uint8_t Binding, uint8_t StOther,
                       uint8_t Type, bool CanOmitFromDynSym,
                       bool HasUnnamedAddr, InputFile *File);

  Symbol *addRegular(StringRef Name, const Elf_Sym &Sym,
                     InputSectionBase<ELFT> *Section);
  Symbol *addRegular(StringRef Name, uint8_t Binding, uint8_t StOther);
  Symbol *addSynthetic(StringRef N, OutputSectionBase<ELFT> *Section,
                       uintX_t Value, uint8_t StOther);
  void addShared(SharedFile<ELFT> *F, StringRef Name, const Elf_Sym &Sym,
                 const typename ELFT::Verdef *Verdef);

  void addLazyArchive(ArchiveFile *F, const llvm::object::Archive::Symbol S);
  void addLazyObject(StringRef Name, LazyObjectFile &Obj);
  Symbol *addBitcode(StringRef Name, uint8_t Binding, uint8_t StOther,
                     uint8_t Type, bool CanOmitFromDynSym, bool HasUnnamedAddr,
                     BitcodeFile *File);

  Symbol *addCommon(StringRef N, uint64_t Size, uint64_t Alignment,
                    uint8_t Binding, uint8_t StOther, uint8_t Type,
                    bool HasUnnamedAddr, InputFile *File);

  void scanUndefinedFlags();
  void scanShlibUndefined();
  void scanDynamicList();
  void scanVersionScript();

  SymbolBody *find(StringRef Name);

  void trace(StringRef Name);
  void wrap(StringRef Name);

private:
  std::vector<SymbolBody *> findAll(StringRef Pattern);
  std::pair<Symbol *, bool> insert(StringRef &Name);
  std::pair<Symbol *, bool> insert(StringRef &Name, uint8_t Type,
                                   uint8_t Visibility, bool CanOmitFromDynSym,
                                   bool HasUnnamedAddr, InputFile *File);

  std::string conflictMsg(SymbolBody *Existing, InputFile *NewFile);
  void reportDuplicate(SymbolBody *Existing, InputFile *NewFile);

  std::map<std::string, SymbolBody *> getDemangledSyms();

  struct SymIndex {
    SymIndex(int Idx, bool Traced) : Idx(Idx), Traced(Traced) {}
    int Idx : 31;
    unsigned Traced : 1;
  };

  // The order the global symbols are in is not defined. We can use an arbitrary
  // order, but it has to be reproducible. That is true even when cross linking.
  // The default hashing of StringRef produces different results on 32 and 64
  // bit systems so we use a map to a vector. That is arbitrary, deterministic
  // but a bit inefficient.
  // FIXME: Experiment with passing in a custom hashing or sorting the symbols
  // once symbol resolution is finished.
  llvm::DenseMap<SymName, SymIndex> Symtab;
  std::vector<Symbol *> SymVector;
  llvm::BumpPtrAllocator Alloc;

  // Comdat groups define "link once" sections. If two comdat groups have the
  // same name, only one of them is linked, and the other is ignored. This set
  // is used to uniquify them.
  llvm::DenseSet<StringRef> ComdatGroups;

  // The symbol table owns all file objects.
  std::vector<std::unique_ptr<ArchiveFile>> ArchiveFiles;
  std::vector<std::unique_ptr<ObjectFile<ELFT>>> ObjectFiles;
  std::vector<std::unique_ptr<LazyObjectFile>> LazyObjectFiles;
  std::vector<std::unique_ptr<SharedFile<ELFT>>> SharedFiles;
  std::vector<std::unique_ptr<BitcodeFile>> BitcodeFiles;

  // Set of .so files to not link the same shared object file more than once.
  llvm::DenseSet<StringRef> SoNames;

  std::unique_ptr<BitcodeCompiler> Lto;
};

template <class ELFT> struct Symtab { static SymbolTable<ELFT> *X; };
template <class ELFT> SymbolTable<ELFT> *Symtab<ELFT>::X;

} // namespace elf
} // namespace lld

#endif
