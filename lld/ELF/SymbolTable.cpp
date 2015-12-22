//===- SymbolTable.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Symbol table is a bag of all known symbols. We put all symbols of
// all input files to the symbol table. The symbol Table is basically
// a hash table with the logic to resolve symbol name conflicts using
// the symbol types.
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "Config.h"
#include "Error.h"
#include "Symbols.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf2;

template <class ELFT> SymbolTable<ELFT>::SymbolTable() {}

template <class ELFT>
static void checkCompatibility(InputFile *FileP) {
  auto *F = dyn_cast<ELFFileBase<ELFT>>(FileP);
  if (!F)
    return;
  if (F->getELFKind() == Config->EKind && F->getEMachine() == Config->EMachine)
    return;
  StringRef A = F->getName();
  StringRef B = Config->Emulation;
  if (B.empty())
    B = Config->FirstElf->getName();
  error(A + " is incompatible with " + B);
}

template <class ELFT>
void SymbolTable<ELFT>::addFile(std::unique_ptr<InputFile> File) {
  InputFile *FileP = File.release();
  checkCompatibility<ELFT>(FileP);

  // .a file
  if (auto *F = dyn_cast<ArchiveFile>(FileP)) {
    ArchiveFiles.emplace_back(F);
    F->parse();
    for (Lazy &Sym : F->getLazySymbols())
      addLazy(&Sym);
    return;
  }

  // .so file
  if (auto *F = dyn_cast<SharedFile<ELFT>>(FileP)) {
    // DSOs are uniquified not by filename but by soname.
    F->parseSoName();
    if (!IncludedSoNames.insert(F->getSoName()).second)
      return;

    SharedFiles.emplace_back(F);
    F->parse();
    for (SharedSymbol<ELFT> &B : F->getSharedSymbols())
      resolve(&B);
    return;
  }

  // .o file
  auto *F = cast<ObjectFile<ELFT>>(FileP);
  ObjectFiles.emplace_back(F);
  F->parse(Comdats);
  for (SymbolBody *B : F->getSymbols())
    resolve(B);
}

template <class ELFT>
SymbolBody *SymbolTable<ELFT>::addUndefined(StringRef Name) {
  auto *Sym = new (Alloc) Undefined<ELFT>(Name, Undefined<ELFT>::Required);
  resolve(Sym);
  return Sym;
}

template <class ELFT>
SymbolBody *SymbolTable<ELFT>::addUndefinedOpt(StringRef Name) {
  auto *Sym = new (Alloc) Undefined<ELFT>(Name, Undefined<ELFT>::Optional);
  resolve(Sym);
  return Sym;
}

template <class ELFT>
void SymbolTable<ELFT>::addAbsolute(StringRef Name,
                                    typename ELFFile<ELFT>::Elf_Sym &ESym) {
  resolve(new (Alloc) DefinedAbsolute<ELFT>(Name, ESym));
}

template <class ELFT>
void SymbolTable<ELFT>::addSynthetic(StringRef Name,
                                     OutputSectionBase<ELFT> &Section,
                                     typename ELFFile<ELFT>::uintX_t Value) {
  typedef typename DefinedSynthetic<ELFT>::Elf_Sym Elf_Sym;
  auto *ESym = new (Alloc) Elf_Sym;
  memset(ESym, 0, sizeof(Elf_Sym));
  ESym->st_value = Value;
  auto *Sym = new (Alloc) DefinedSynthetic<ELFT>(Name, *ESym, Section);
  resolve(Sym);
}

template <class ELFT>
SymbolBody *SymbolTable<ELFT>::addIgnored(StringRef Name) {
  auto *Sym = new (Alloc)
      DefinedAbsolute<ELFT>(Name, DefinedAbsolute<ELFT>::IgnoreUndef);
  resolve(Sym);
  return Sym;
}

template <class ELFT> bool SymbolTable<ELFT>::isUndefined(StringRef Name) {
  if (SymbolBody *Sym = find(Name))
    return Sym->isUndefined();
  return false;
}

// Returns a file from which symbol B was created.
// If B does not belong to any file in ObjectFiles, returns a nullptr.
template <class ELFT>
ELFFileBase<ELFT> *
elf2::findFile(ArrayRef<std::unique_ptr<ObjectFile<ELFT>>> ObjectFiles,
               const SymbolBody *B) {
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

  const Elf_Sym *Sym = &cast<ELFSymbolBody<ELFT>>(*B).Sym;
  for (const std::unique_ptr<ObjectFile<ELFT>> &F : ObjectFiles) {
    Elf_Sym_Range R = F->getObj().symbols(F->getSymbolTable());
    if (R.begin() <= Sym && Sym < R.end())
      return F.get();
  }
  return nullptr;
}

template <class ELFT>
std::string SymbolTable<ELFT>::conflictMsg(SymbolBody *Old, SymbolBody *New) {
  ELFFileBase<ELFT> *OldFile = findFile<ELFT>(ObjectFiles, Old);
  ELFFileBase<ELFT> *NewFile = findFile<ELFT>(ObjectFiles, New);

  StringRef Sym = Old->getName();
  StringRef F1 = OldFile ? OldFile->getName() : "(internal)";
  StringRef F2 = NewFile ? NewFile->getName() : "(internal)";
  return (Sym + " in " + F1 + " and " + F2).str();
}

// This function resolves conflicts if there's an existing symbol with
// the same name. Decisions are made based on symbol type.
template <class ELFT> void SymbolTable<ELFT>::resolve(SymbolBody *New) {
  Symbol *Sym = insert(New);
  if (Sym->Body == New)
    return;

  SymbolBody *Existing = Sym->Body;

  if (Lazy *L = dyn_cast<Lazy>(Existing)) {
    if (auto *Undef = dyn_cast<Undefined<ELFT>>(New)) {
      addMemberFile(Undef, L);
      return;
    }
    // Found a definition for something also in an archive.
    // Ignore the archive definition.
    Sym->Body = New;
    return;
  }

  if (New->isTls() != Existing->isTls())
    error("TLS attribute mismatch for symbol: " + conflictMsg(Existing, New));

  // compare() returns -1, 0, or 1 if the lhs symbol is less preferable,
  // equivalent (conflicting), or more preferable, respectively.
  int comp = Existing->compare<ELFT>(New);
  if (comp == 0) {
    std::string S = "duplicate symbol: " + conflictMsg(Existing, New);
    if (!Config->AllowMultipleDefinition)
      error(S);
    warning(S);
    return;
  }
  if (comp < 0)
    Sym->Body = New;
}

template <class ELFT> Symbol *SymbolTable<ELFT>::insert(SymbolBody *New) {
  // Find an existing Symbol or create and insert a new one.
  StringRef Name = New->getName();
  Symbol *&Sym = Symtab[Name];
  if (!Sym)
    Sym = new (Alloc) Symbol{New};
  New->setBackref(Sym);
  return Sym;
}

template <class ELFT> SymbolBody *SymbolTable<ELFT>::find(StringRef Name) {
  auto It = Symtab.find(Name);
  if (It == Symtab.end())
    return nullptr;
  return It->second->Body;
}

template <class ELFT> void SymbolTable<ELFT>::addLazy(Lazy *L) {
  Symbol *Sym = insert(L);
  if (Sym->Body == L)
    return;
  if (auto *Undef = dyn_cast<Undefined<ELFT>>(Sym->Body)) {
    Sym->Body = L;
    addMemberFile(Undef, L);
  }
}

template <class ELFT>
void SymbolTable<ELFT>::addMemberFile(Undefined<ELFT> *Undef, Lazy *L) {
  // Weak undefined symbols should not fetch members from archives.
  // If we were to keep old symbol we would not know that an archive member was
  // available if a strong undefined symbol shows up afterwards in the link.
  // If a strong undefined symbol never shows up, this lazy symbol will
  // get to the end of the link and must be treated as the weak undefined one.
  // We set UsedInRegularObj in a similar way to what is done with shared
  // symbols and mark it as weak to reduce how many special cases are needed.
  if (Undef->isWeak()) {
    L->setUsedInRegularObj();
    L->setWeak();
    return;
  }

  // Fetch a member file that has the definition for L.
  // getMember returns nullptr if the member was already read from the library.
  if (std::unique_ptr<InputFile> File = L->getMember())
    addFile(std::move(File));
}

// This function takes care of the case in which shared libraries depend on
// the user program (not the other way, which is usual). Shared libraries
// may have undefined symbols, expecting that the user program provides
// the definitions for them. An example is BSD's __progname symbol.
// We need to put such symbols to the main program's .dynsym so that
// shared libraries can find them.
// Except this, we ignore undefined symbols in DSOs.
template <class ELFT> void SymbolTable<ELFT>::scanShlibUndefined() {
  for (std::unique_ptr<SharedFile<ELFT>> &File : SharedFiles)
    for (StringRef U : File->getUndefinedSymbols())
      if (SymbolBody *Sym = find(U))
        if (Sym->isDefined())
          Sym->setUsedInDynamicReloc();
}

template class lld::elf2::SymbolTable<ELF32LE>;
template class lld::elf2::SymbolTable<ELF32BE>;
template class lld::elf2::SymbolTable<ELF64LE>;
template class lld::elf2::SymbolTable<ELF64BE>;

template ELFFileBase<ELF32LE> *
lld::elf2::findFile(ArrayRef<std::unique_ptr<ObjectFile<ELF32LE>>>,
                    const SymbolBody *);
template ELFFileBase<ELF32BE> *
lld::elf2::findFile(ArrayRef<std::unique_ptr<ObjectFile<ELF32BE>>>,
                    const SymbolBody *);
template ELFFileBase<ELF64LE> *
lld::elf2::findFile(ArrayRef<std::unique_ptr<ObjectFile<ELF64LE>>>,
                    const SymbolBody *);
template ELFFileBase<ELF64BE> *
lld::elf2::findFile(ArrayRef<std::unique_ptr<ObjectFile<ELF64BE>>>,
                    const SymbolBody *);
