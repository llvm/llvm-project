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
// to replace the lazy symbol. The logic is implemented in resolve().
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

  SymbolBody *addUndefined(StringRef Name);
  SymbolBody *addUndefinedOpt(StringRef Name);
  DefinedRegular<ELFT> *addAbsolute(StringRef Name,
                                    uint8_t Visibility = llvm::ELF::STV_HIDDEN);
  SymbolBody *addSynthetic(StringRef Name, OutputSectionBase<ELFT> &Section,
                           uintX_t Value);
  DefinedRegular<ELFT> *addIgnored(StringRef Name,
                                   uint8_t Visibility = llvm::ELF::STV_HIDDEN);

  void scanUndefinedFlags();
  void scanShlibUndefined();
  void scanDynamicList();
  void scanVersionScript();
  SymbolBody *find(StringRef Name);
  void wrap(StringRef Name);
  InputFile *findFile(SymbolBody *B);

private:
  Symbol *insert(SymbolBody *New);
  void addLazy(Lazy *New);
  void addMemberFile(SymbolBody *Undef, Lazy *L);
  void resolve(SymbolBody *Body);
  std::string conflictMsg(SymbolBody *Old, SymbolBody *New);

  // The order the global symbols are in is not defined. We can use an arbitrary
  // order, but it has to be reproducible. That is true even when cross linking.
  // The default hashing of StringRef produces different results on 32 and 64
  // bit systems so we use a map to a vector. That is arbitrary, deterministic
  // but a bit inefficient.
  // FIXME: Experiment with passing in a custom hashing or sorting the symbols
  // once symbol resolution is finished.
  llvm::DenseMap<SymName, unsigned> Symtab;
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

} // namespace elf
} // namespace lld

#endif
