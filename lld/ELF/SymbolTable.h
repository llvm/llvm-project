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
#include "llvm/ADT/MapVector.h"

namespace lld {
namespace elf2 {
class Lazy;
template <class ELFT> class OutputSectionBase;
struct Symbol;
class Undefined;

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
public:
  SymbolTable();

  void addFile(std::unique_ptr<InputFile> File);

  const llvm::MapVector<StringRef, Symbol *> &getSymbols() const {
    return Symtab;
  }

  const std::vector<std::unique_ptr<ObjectFile<ELFT>>> &getObjectFiles() const {
    return ObjectFiles;
  }

  const std::vector<std::unique_ptr<SharedFile<ELFT>>> &getSharedFiles() const {
    return SharedFiles;
  }

  SymbolBody *addUndefined(StringRef Name);
  SymbolBody *addUndefinedOpt(StringRef Name);
  void addAbsolute(StringRef Name,
                   typename llvm::object::ELFFile<ELFT>::Elf_Sym &ESym);
  void addSynthetic(StringRef Name, OutputSectionBase<ELFT> &Section,
                    typename llvm::object::ELFFile<ELFT>::uintX_t Value);
  SymbolBody *addIgnored(StringRef Name);
  bool isUndefined(StringRef Name);
  void scanShlibUndefined();
  SymbolBody *find(StringRef Name);

private:
  Symbol *insert(SymbolBody *New);
  void addLazy(Lazy *New);
  void addMemberFile(Undefined *Undef, Lazy *L);
  void resolve(SymbolBody *Body);
  std::string conflictMsg(SymbolBody *Old, SymbolBody *New);

  std::vector<std::unique_ptr<InputFile>> ArchiveFiles;

  // The order the global symbols are in is not defined. We can use an arbitrary
  // order, but it has to be reproducible. That is true even when cross linking.
  // The default hashing of StringRef produces different results on 32 and 64
  // bit systems so we use a MapVector. That is arbitrary, deterministic but
  // a bit inefficient.
  // FIXME: Experiment with passing in a custom hashing or sorting the symbols
  // once symbol resolution is finished.
  llvm::MapVector<StringRef, Symbol *> Symtab;
  llvm::BumpPtrAllocator Alloc;

  llvm::DenseSet<StringRef> Comdats;

  // The writer needs to infer the machine type from the object files.
  std::vector<std::unique_ptr<ObjectFile<ELFT>>> ObjectFiles;

  std::vector<std::unique_ptr<SharedFile<ELFT>>> SharedFiles;
  llvm::DenseSet<StringRef> IncludedSoNames;
};

template <class ELFT>
ELFFileBase<ELFT> *
findFile(ArrayRef<std::unique_ptr<ObjectFile<ELFT>>> ObjectFiles,
         const SymbolBody *B);

} // namespace elf2
} // namespace lld

#endif
