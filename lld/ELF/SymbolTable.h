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
#include "Strings.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseMap.h"

namespace lld {
namespace elf {

struct Symbol;

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
class SymbolTable {
public:
  template <class ELFT> void addFile(InputFile *File);
  template <class ELFT> void addCombinedLTOObject();
  template <class ELFT> void addSymbolAlias(StringRef Alias, StringRef Name);
  template <class ELFT> void addSymbolWrap(StringRef Name);
  void applySymbolRenames();

  ArrayRef<Symbol *> getSymbols() const { return SymVector; }

  template <class ELFT>
  DefinedRegular *addAbsolute(StringRef Name,
                              uint8_t Visibility = llvm::ELF::STV_HIDDEN,
                              uint8_t Binding = llvm::ELF::STB_GLOBAL);
  template <class ELFT>
  DefinedRegular *addIgnored(StringRef Name,
                             uint8_t Visibility = llvm::ELF::STV_HIDDEN);

  template <class ELFT> Symbol *addUndefined(StringRef Name);
  template <class ELFT>
  Symbol *addUndefined(StringRef Name, bool IsLocal, uint8_t Binding,
                       uint8_t StOther, uint8_t Type, bool CanOmitFromDynSym,
                       InputFile *File);
  template <class ELFT>
  Symbol *addRegular(StringRef Name, uint8_t StOther, uint8_t Type,
                     uint64_t Value, uint64_t Size, uint8_t Binding,
                     SectionBase *Section, InputFile *File);

  template <class ELFT>
  void addShared(SharedFile<ELFT> *F, StringRef Name,
                 const typename ELFT::Sym &Sym,
                 const typename ELFT::Verdef *Verdef);

  template <class ELFT>
  Symbol *addLazyArchive(ArchiveFile *F, const llvm::object::Archive::Symbol S);
  template <class ELFT> void addLazyObject(StringRef Name, LazyObjFile &Obj);

  Symbol *addBitcode(StringRef Name, uint8_t Binding, uint8_t StOther,
                     uint8_t Type, bool CanOmitFromDynSym, BitcodeFile *File);

  Symbol *addCommon(StringRef N, uint64_t Size, uint32_t Alignment,
                    uint8_t Binding, uint8_t StOther, uint8_t Type,
                    InputFile *File);

  std::pair<Symbol *, bool> insert(StringRef Name);
  std::pair<Symbol *, bool> insert(StringRef Name, uint8_t Type,
                                   uint8_t Visibility, bool CanOmitFromDynSym,
                                   InputFile *File);

  template <class ELFT> void scanUndefinedFlags();
  template <class ELFT> void scanShlibUndefined();
  void scanVersionScript();

  SymbolBody *find(StringRef Name);

  void trace(StringRef Name);

private:
  std::vector<SymbolBody *> findByVersion(SymbolVersion Ver);
  std::vector<SymbolBody *> findAllByVersion(SymbolVersion Ver);

  llvm::StringMap<std::vector<SymbolBody *>> &getDemangledSyms();
  void handleAnonymousVersion();
  void assignExactVersion(SymbolVersion Ver, uint16_t VersionId,
                          StringRef VersionName);
  void assignWildcardVersion(SymbolVersion Ver, uint16_t VersionId);

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
  llvm::DenseMap<llvm::CachedHashStringRef, SymIndex> Symtab;
  std::vector<Symbol *> SymVector;

  // Comdat groups define "link once" sections. If two comdat groups have the
  // same name, only one of them is linked, and the other is ignored. This set
  // is used to uniquify them.
  llvm::DenseSet<llvm::CachedHashStringRef> ComdatGroups;

  // Set of .so files to not link the same shared object file more than once.
  llvm::DenseSet<StringRef> SoNames;

  // A map from demangled symbol names to their symbol objects.
  // This mapping is 1:N because two symbols with different versions
  // can have the same name. We use this map to handle "extern C++ {}"
  // directive in version scripts.
  llvm::Optional<llvm::StringMap<std::vector<SymbolBody *>>> DemangledSyms;

  // For LTO.
  std::unique_ptr<BitcodeCompiler> LTO;
};

extern SymbolTable *Symtab;
} // namespace elf
} // namespace lld

#endif
