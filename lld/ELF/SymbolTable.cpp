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
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Linker/IRMover.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf;

// All input object files must be for the same architecture
// (e.g. it does not make sense to link x86 object files with
// MIPS object files.) This function checks for that error.
template <class ELFT> static bool isCompatible(InputFile *FileP) {
  auto *F = dyn_cast<ELFFileBase<ELFT>>(FileP);
  if (!F)
    return true;
  if (F->getELFKind() == Config->EKind && F->getEMachine() == Config->EMachine)
    return true;
  StringRef A = F->getName();
  StringRef B = Config->Emulation;
  if (B.empty())
    B = Config->FirstElf->getName();
  error(A + " is incompatible with " + B);
  return false;
}

// Add symbols in File to the symbol table.
template <class ELFT>
void SymbolTable<ELFT>::addFile(std::unique_ptr<InputFile> File) {
  InputFile *FileP = File.get();
  if (!isCompatible<ELFT>(FileP))
    return;

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
    if (!SoNames.insert(F->getSoName()).second)
      return;

    SharedFiles.emplace_back(cast<SharedFile<ELFT>>(File.release()));
    F->parseRest();
    for (SharedSymbol<ELFT> &B : F->getSharedSymbols())
      resolve(&B);
    return;
  }

  // LLVM bitcode file.
  if (auto *F = dyn_cast<BitcodeFile>(FileP)) {
    BitcodeFiles.emplace_back(cast<BitcodeFile>(File.release()));
    F->parse(ComdatGroups);
    for (SymbolBody *B : F->getSymbols())
      resolve(B);
    return;
  }

  // .o file
  auto *F = cast<ObjectFile<ELFT>>(FileP);
  ObjectFiles.emplace_back(cast<ObjectFile<ELFT>>(File.release()));
  F->parse(ComdatGroups);
  for (SymbolBody *B : F->getSymbols())
    resolve(B);
}

// Codegen the module M and returns the resulting InputFile.
template <class ELFT>
std::unique_ptr<InputFile> SymbolTable<ELFT>::codegen(Module &M) {
  StringRef TripleStr = M.getTargetTriple();
  Triple TheTriple(TripleStr);

  // FIXME: Should we have a default triple? The gold plugin uses
  // sys::getDefaultTargetTriple(), but that is probably wrong given that this
  // might be a cross linker.

  std::string ErrMsg;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleStr, ErrMsg);
  if (!TheTarget)
    fatal("Target not found: " + ErrMsg);

  TargetOptions Options;
  Reloc::Model R = Config->Shared ? Reloc::PIC_ : Reloc::Static;
  std::unique_ptr<TargetMachine> TM(
      TheTarget->createTargetMachine(TripleStr, "", "", Options, R));

  raw_svector_ostream OS(OwningLTOData);
  legacy::PassManager CodeGenPasses;
  if (TM->addPassesToEmitFile(CodeGenPasses, OS,
                              TargetMachine::CGFT_ObjectFile))
    fatal("Failed to setup codegen");
  CodeGenPasses.run(M);
  LtoBuffer = MemoryBuffer::getMemBuffer(OwningLTOData, "", false);
  return createObjectFile(*LtoBuffer);
}

static void addBitcodeFile(IRMover &Mover, BitcodeFile &F,
                           LLVMContext &Context) {
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer(F.MB, false);
  std::unique_ptr<Module> M =
      check(getLazyBitcodeModule(std::move(Buffer), Context,
                                 /*ShouldLazyLoadMetadata*/ true));
  std::vector<GlobalValue *> Keep;
  for (SymbolBody *B : F.getSymbols()) {
    if (B->repl() != B)
      continue;
    auto *DB = dyn_cast<DefinedBitcode>(B);
    if (!DB)
      continue;
    GlobalValue *GV = M->getNamedValue(B->getName());
    assert(GV);
    Keep.push_back(GV);
  }
  Mover.move(std::move(M), Keep, [](GlobalValue &, IRMover::ValueAdder) {});
}

// Merge all the bitcode files we have seen, codegen the result and return
// the resulting ObjectFile.
template <class ELFT>
ObjectFile<ELFT> *SymbolTable<ELFT>::createCombinedLtoObject() {
  LLVMContext Context;
  Module Combined("ld-temp.o", Context);
  IRMover Mover(Combined);
  for (const std::unique_ptr<BitcodeFile> &F : BitcodeFiles)
    addBitcodeFile(Mover, *F, Context);
  std::unique_ptr<InputFile> F = codegen(Combined);
  ObjectFiles.emplace_back(cast<ObjectFile<ELFT>>(F.release()));
  return &*ObjectFiles.back();
}

template <class ELFT> void SymbolTable<ELFT>::addCombinedLtoObject() {
  if (BitcodeFiles.empty())
    return;
  ObjectFile<ELFT> *Obj = createCombinedLtoObject();
  llvm::DenseSet<StringRef> DummyGroups;
  Obj->parse(DummyGroups);
  for (SymbolBody *Body : Obj->getSymbols()) {
    Symbol *Sym = insert(Body);
    if (!Sym->Body->isUndefined() && Body->isUndefined())
      continue;
    Sym->Body = Body;
  }
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
SymbolBody *SymbolTable<ELFT>::addAbsolute(StringRef Name, Elf_Sym &ESym) {
  // Pass nullptr because absolute symbols have no corresponding input sections.
  auto *Sym = new (Alloc) DefinedRegular<ELFT>(Name, ESym, nullptr);
  resolve(Sym);
  return Sym;
}

template <class ELFT>
SymbolBody *SymbolTable<ELFT>::addSynthetic(StringRef Name,
                                            OutputSectionBase<ELFT> &Sec,
                                            uintX_t Val, uint8_t Visibility) {
  auto *Sym = new (Alloc) DefinedSynthetic<ELFT>(Name, Val, Sec, Visibility);
  resolve(Sym);
  return Sym;
}

// Add Name as an "ignored" symbol. An ignored symbol is a regular
// linker-synthesized defined symbol, but it is not recorded to the output
// file's symbol table. Such symbols are useful for some linker-defined symbols.
template <class ELFT>
SymbolBody *SymbolTable<ELFT>::addIgnored(StringRef Name) {
  return addAbsolute(Name, ElfSym<ELFT>::Ignored);
}

// Rename SYM as __wrap_SYM. The original symbol is preserved as __real_SYM.
// Used to implement --wrap.
template <class ELFT> void SymbolTable<ELFT>::wrap(StringRef Name) {
  if (Symtab.count(Name) == 0)
    return;
  StringSaver Saver(Alloc);
  Symbol *Sym = addUndefined(Name)->getSymbol();
  Symbol *Real = addUndefined(Saver.save("__real_" + Name))->getSymbol();
  Symbol *Wrap = addUndefined(Saver.save("__wrap_" + Name))->getSymbol();
  Real->Body = Sym->Body;
  Sym->Body = Wrap->Body;
}

// Returns a file from which symbol B was created.
// If B does not belong to any file, returns a nullptr.
template <class ELFT> InputFile *SymbolTable<ELFT>::findFile(SymbolBody *B) {
  for (const std::unique_ptr<ObjectFile<ELFT>> &F : ObjectFiles) {
    ArrayRef<SymbolBody *> Syms = F->getSymbols();
    if (std::find(Syms.begin(), Syms.end(), B) != Syms.end())
      return F.get();
  }
  for (const std::unique_ptr<BitcodeFile> &F : BitcodeFiles) {
    ArrayRef<SymbolBody *> Syms = F->getSymbols();
    if (std::find(Syms.begin(), Syms.end(), B) != Syms.end())
      return F.get();
  }
  return nullptr;
}

// Returns "(internal)", "foo.a(bar.o)" or "baz.o".
static std::string getFilename(InputFile *F) {
  if (!F)
    return "(internal)";
  if (!F->ArchiveName.empty())
    return (F->ArchiveName + "(" + F->getName() + ")").str();
  return F->getName();
}

// Construct a string in the form of "Sym in File1 and File2".
// Used to construct an error message.
template <class ELFT>
std::string SymbolTable<ELFT>::conflictMsg(SymbolBody *Old, SymbolBody *New) {
  InputFile *F1 = findFile(Old);
  InputFile *F2 = findFile(New);
  StringRef Sym = Old->getName();
  return demangle(Sym) + " in " + getFilename(F1) + " and " + getFilename(F2);
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

  if (New->IsTls != Existing->IsTls) {
    error("TLS attribute mismatch for symbol: " + conflictMsg(Existing, New));
    return;
  }

  // compare() returns -1, 0, or 1 if the lhs symbol is less preferable,
  // equivalent (conflicting), or more preferable, respectively.
  int Comp = Existing->compare<ELFT>(New);
  if (Comp == 0) {
    std::string S = "duplicate symbol: " + conflictMsg(Existing, New);
    if (Config->AllowMultipleDefinition)
      warning(S);
    else
      error(S);
    return;
  }
  if (Comp < 0)
    Sym->Body = New;
}

// Find an existing symbol or create and insert a new one.
template <class ELFT> Symbol *SymbolTable<ELFT>::insert(SymbolBody *New) {
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
  // symbols and copy information to reduce how many special cases are needed.
  if (Undef->isWeak()) {
    L->setUsedInRegularObj();
    L->setWeak();

    // FIXME: Do we need to copy more?
    L->IsTls |= Undef->IsTls;
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
          Sym->MustBeInDynSym = true;
}

template class elf::SymbolTable<ELF32LE>;
template class elf::SymbolTable<ELF32BE>;
template class elf::SymbolTable<ELF64LE>;
template class elf::SymbolTable<ELF64BE>;
