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
// all input files to the symbol table. The symbol table is basically
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

// All input object files must be for the same architecture
// (e.g. it does not make sense to link x86 object files with
// MIPS object files.) This function checks for that error.
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

// Add symbols in File to the symbol table.
template <class ELFT>
void SymbolTable<ELFT>::addFile(std::unique_ptr<InputFile> File) {
  InputFile *FileP = File.get();
  checkCompatibility<ELFT>(FileP);

  // .a file
  if (auto *F = dyn_cast<ArchiveFile>(FileP)) {
    ArchiveFiles.emplace_back(cast<ArchiveFile>(File.release()));
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

    SharedFiles.emplace_back(cast<SharedFile<ELFT>>(File.release()));
    F->parseRest();
    for (SharedSymbol<ELFT> &B : F->getSharedSymbols())
      resolve(&B);
    return;
  }

  // .o file
  auto *F = cast<ObjectFile<ELFT>>(FileP);
  ObjectFiles.emplace_back(cast<ObjectFile<ELFT>>(File.release()));
  F->parse(ComdatGroups);
  for (SymbolBody *B : F->getSymbols())
    resolve(B);
}

// Add an undefined symbol.
template <class ELFT>
SymbolBody *SymbolTable<ELFT>::addUndefined(StringRef Name) {
  auto *Sym = new (Alloc) Undefined(Name, false, STV_DEFAULT, false);
  resolve(Sym);
  return Sym;
}

// Add an undefined symbol. Unlike addUndefined, that symbol
// doesn't have to be resolved, thus "opt" (optional).
template <class ELFT>
SymbolBody *SymbolTable<ELFT>::addUndefinedOpt(StringRef Name) {
  auto *Sym = new (Alloc) Undefined(Name, false, STV_HIDDEN, true);
  resolve(Sym);
  return Sym;
}

template <class ELFT>
void SymbolTable<ELFT>::addAbsolute(StringRef Name,
                                    typename ELFFile<ELFT>::Elf_Sym &ESym) {
  resolve(new (Alloc) DefinedRegular<ELFT>(Name, ESym, nullptr));
}

template <class ELFT>
void SymbolTable<ELFT>::addSynthetic(StringRef Name,
                                     OutputSectionBase<ELFT> &Section,
                                     typename ELFFile<ELFT>::uintX_t Value) {
  auto *Sym = new (Alloc) DefinedSynthetic<ELFT>(Name, Value, Section);
  resolve(Sym);
}

// Add Name as an "ignored" symbol. An ignored symbol is a regular
// linker-synthesized defined symbol, but it is not recorded to the output
// file's symbol table. Such symbols are useful for some linker-defined symbols.
template <class ELFT>
SymbolBody *SymbolTable<ELFT>::addIgnored(StringRef Name) {
  auto *Sym = new (Alloc)
      DefinedRegular<ELFT>(Name, ElfSym<ELFT>::IgnoreUndef, nullptr);
  resolve(Sym);
  return Sym;
}

// Returns a file from which symbol B was created.
// If B does not belong to any file, returns a nullptr.
template <class ELFT>
ELFFileBase<ELFT> *SymbolTable<ELFT>::findFile(SymbolBody *B) {
  for (const std::unique_ptr<ObjectFile<ELFT>> &F : ObjectFiles) {
    ArrayRef<SymbolBody *> Syms = F->getSymbols();
    if (std::find(Syms.begin(), Syms.end(), B) != Syms.end())
      return F.get();
  }
  return nullptr;
}

template <class ELFT>
std::string SymbolTable<ELFT>::conflictMsg(SymbolBody *Old, SymbolBody *New) {
  ELFFileBase<ELFT> *OldFile = findFile(Old);
  ELFFileBase<ELFT> *NewFile = findFile(New);

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
    if (auto *Undef = dyn_cast<Undefined>(New)) {
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
  int Comp = Existing->compare<ELFT>(New);
  if (Comp == 0) {
    std::string S = "duplicate symbol: " + conflictMsg(Existing, New);
    if (!Config->AllowMultipleDefinition)
      error(S);
    warning(S);
    return;
  }
  if (Comp < 0)
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
  if (auto *Undef = dyn_cast<Undefined>(Sym->Body)) {
    Sym->Body = L;
    addMemberFile(Undef, L);
  }
}

template <class ELFT>
void SymbolTable<ELFT>::addMemberFile(Undefined *Undef, Lazy *L) {
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

template class elf2::SymbolTable<ELF32LE>;
template class elf2::SymbolTable<ELF32BE>;
template class elf2::SymbolTable<ELF64LE>;
template class elf2::SymbolTable<ELF64BE>;
