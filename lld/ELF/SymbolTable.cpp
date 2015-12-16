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

template <class ELFT> bool SymbolTable<ELFT>::shouldUseRela() const {
  ELFKind K = cast<ELFFileBase<ELFT>>(Config->FirstElf)->getELFKind();
  return K == ELF64LEKind || K == ELF64BEKind;
}

template <class ELFT>
void SymbolTable<ELFT>::addFile(std::unique_ptr<InputFile> File) {
  checkCompatibility(File);

  if (auto *AF = dyn_cast<ArchiveFile>(File.get())) {
    ArchiveFiles.emplace_back(std::move(File));
    AF->parse();
    for (Lazy &Sym : AF->getLazySymbols())
      addLazy(&Sym);
    return;
  }

  if (auto *S = dyn_cast<SharedFile<ELFT>>(File.get())) {
    S->parseSoName();
    if (!IncludedSoNames.insert(S->getSoName()).second)
      return;
    S->parse();
  } else {
    cast<ObjectFile<ELFT>>(File.get())->parse(Comdats);
  }
  addELFFile(cast<ELFFileBase<ELFT>>(File.release()));
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
void SymbolTable<ELFT>::addAbsoluteSym(StringRef Name,
                                       typename ELFFile<ELFT>::Elf_Sym &ESym) {
  resolve(new (Alloc) DefinedAbsolute<ELFT>(Name, ESym));
}

template <class ELFT>
void SymbolTable<ELFT>::addSyntheticSym(StringRef Name,
                                        OutputSectionBase<ELFT> &Section,
                                        typename ELFFile<ELFT>::uintX_t Value) {
  typedef typename DefinedSynthetic<ELFT>::Elf_Sym Elf_Sym;
  auto ESym = new (Alloc) Elf_Sym;
  memset(ESym, 0, sizeof(Elf_Sym));
  ESym->st_value = Value;
  auto Sym = new (Alloc) DefinedSynthetic<ELFT>(Name, *ESym, Section);
  resolve(Sym);
}

template <class ELFT>
SymbolBody *SymbolTable<ELFT>::addIgnoredSym(StringRef Name) {
  auto Sym = new (Alloc)
      DefinedAbsolute<ELFT>(Name, DefinedAbsolute<ELFT>::IgnoreUndef);
  resolve(Sym);
  return Sym;
}

template <class ELFT> bool SymbolTable<ELFT>::isUndefined(StringRef Name) {
  if (SymbolBody *Sym = find(Name))
    return Sym->isUndefined();
  return false;
}

template <class ELFT>
void SymbolTable<ELFT>::addELFFile(ELFFileBase<ELFT> *File) {
  if (auto *O = dyn_cast<ObjectFile<ELFT>>(File))
    ObjectFiles.emplace_back(O);
  else if (auto *S = dyn_cast<SharedFile<ELFT>>(File))
    SharedFiles.emplace_back(S);

  if (auto *O = dyn_cast<ObjectFile<ELFT>>(File)) {
    for (SymbolBody *Body : O->getSymbols())
      resolve(Body);
  }

  if (auto *S = dyn_cast<SharedFile<ELFT>>(File)) {
    for (SharedSymbol<ELFT> &Body : S->getSharedSymbols())
      resolve(&Body);
  }
}

template <class ELFT>
void SymbolTable<ELFT>::reportConflict(const Twine &Message,
                                       const SymbolBody &Old,
                                       const SymbolBody &New, bool Warning) {
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

  const Elf_Sym &OldE = cast<ELFSymbolBody<ELFT>>(Old).Sym;
  const Elf_Sym &NewE = cast<ELFSymbolBody<ELFT>>(New).Sym;
  ELFFileBase<ELFT> *OldFile = nullptr;
  ELFFileBase<ELFT> *NewFile = nullptr;

  for (const std::unique_ptr<ObjectFile<ELFT>> &File : ObjectFiles) {
    Elf_Sym_Range Syms = File->getObj().symbols(File->getSymbolTable());
    if (&OldE > Syms.begin() && &OldE < Syms.end())
      OldFile = File.get();
    if (&NewE > Syms.begin() && &NewE < Syms.end())
      NewFile = File.get();
  }

  std::string Msg = (Message + ": " + Old.getName() + " in " +
                     (OldFile ? OldFile->getName() : "(internal)") + " and " +
                     (NewFile ? NewFile->getName() : "(internal)"))
                        .str();
  if (Warning)
    warning(Msg);
  else
    error(Msg);
}

// This function resolves conflicts if there's an existing symbol with
// the same name. Decisions are made based on symbol type.
template <class ELFT> void SymbolTable<ELFT>::resolve(SymbolBody *New) {
  Symbol *Sym = insert(New);
  if (Sym->Body == New)
    return;

  SymbolBody *Existing = Sym->Body;

  if (Lazy *L = dyn_cast<Lazy>(Existing)) {
    if (New->isUndefined()) {
      if (New->isWeak()) {
        // See the explanation in SymbolTable::addLazy
        L->setUsedInRegularObj();
        L->setWeak();
        return;
      }
      addMemberFile(L);
      return;
    }

    // Found a definition for something also in an archive. Ignore the archive
    // definition.
    Sym->Body = New;
    return;
  }

  if (New->isTLS() != Existing->isTLS())
    reportConflict("TLS attribute mismatch for symbol", *Existing, *New, false);

  // compare() returns -1, 0, or 1 if the lhs symbol is less preferable,
  // equivalent (conflicting), or more preferable, respectively.
  int comp = Existing->compare<ELFT>(New);
  if (comp < 0)
    Sym->Body = New;
  else if (comp == 0)
    reportConflict("duplicate symbol", *Existing, *New,
                   Config->AllowMultipleDefinition);
}

template <class ELFT> Symbol *SymbolTable<ELFT>::insert(SymbolBody *New) {
  // Find an existing Symbol or create and insert a new one.
  StringRef Name = New->getName();
  Symbol *&Sym = Symtab[Name];
  if (!Sym) {
    Sym = new (Alloc) Symbol(New);
    New->setBackref(Sym);
    return Sym;
  }
  New->setBackref(Sym);
  return Sym;
}

template <class ELFT> SymbolBody *SymbolTable<ELFT>::find(StringRef Name) {
  auto It = Symtab.find(Name);
  if (It == Symtab.end())
    return nullptr;
  return It->second->Body;
}

template <class ELFT> void SymbolTable<ELFT>::addLazy(Lazy *New) {
  Symbol *Sym = insert(New);
  if (Sym->Body == New)
    return;
  SymbolBody *Existing = Sym->Body;
  if (Existing->isDefined() || Existing->isLazy())
    return;
  Sym->Body = New;
  assert(Existing->isUndefined() && "Unexpected symbol kind.");

  // Weak undefined symbols should not fetch members from archives.
  // If we were to keep old symbol we would not know that an archive member was
  // available if a strong undefined symbol shows up afterwards in the link.
  // If a strong undefined symbol never shows up, this lazy symbol will
  // get to the end of the link and must be treated as the weak undefined one.
  // We set UsedInRegularObj in a similar way to what is done with shared
  // symbols and mark it as weak to reduce how many special cases are needed.
  if (Existing->isWeak()) {
    New->setUsedInRegularObj();
    New->setWeak();
    return;
  }
  addMemberFile(New);
}

template <class ELFT>
void SymbolTable<ELFT>::checkCompatibility(std::unique_ptr<InputFile> &File) {
  auto *E = dyn_cast<ELFFileBase<ELFT>>(File.get());
  if (!E)
    return;
  if (E->getELFKind() == Config->EKind && E->getEMachine() == Config->EMachine)
    return;
  StringRef A = E->getName();
  StringRef B = Config->Emulation;
  if (B.empty())
    B = Config->FirstElf->getName();
  error(A + " is incompatible with " + B);
}

template <class ELFT> void SymbolTable<ELFT>::addMemberFile(Lazy *Body) {
  // getMember returns nullptr if the member was already read from the library.
  if (std::unique_ptr<InputFile> File = Body->getMember())
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
