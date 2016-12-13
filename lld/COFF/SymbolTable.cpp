//===- SymbolTable.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "Config.h"
#include "Driver.h"
#include "Error.h"
#include "Symbols.h"
#include "lld/Support/Memory.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/LTO/legacy/LTOCodeGenerator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;

namespace lld {
namespace coff {

SymbolTable *Symtab;

void SymbolTable::addFile(InputFile *File) {
  if (Config->Verbose)
    outs() << "Reading " << toString(File) << "\n";
  File->parse();

  MachineTypes MT = File->getMachineType();
  if (Config->Machine == IMAGE_FILE_MACHINE_UNKNOWN) {
    Config->Machine = MT;
  } else if (MT != IMAGE_FILE_MACHINE_UNKNOWN && Config->Machine != MT) {
    fatal(toString(File) + ": machine type " + machineToStr(MT) +
          " conflicts with " + machineToStr(Config->Machine));
  }

  if (auto *F = dyn_cast<ObjectFile>(File)) {
    ObjectFiles.push_back(F);
  } else if (auto *F = dyn_cast<BitcodeFile>(File)) {
    BitcodeFiles.push_back(F);
  } else if (auto *F = dyn_cast<ImportFile>(File)) {
    ImportFiles.push_back(F);
  }

  StringRef S = File->getDirectives();
  if (S.empty())
    return;

  if (Config->Verbose)
    outs() << "Directives: " << toString(File) << ": " << S << "\n";
  Driver->parseDirectives(S);
}

void SymbolTable::reportRemainingUndefines() {
  SmallPtrSet<SymbolBody *, 8> Undefs;
  for (auto &I : Symtab) {
    Symbol *Sym = I.second;
    auto *Undef = dyn_cast<Undefined>(Sym->body());
    if (!Undef)
      continue;
    if (!Sym->IsUsedInRegularObj)
      continue;
    StringRef Name = Undef->getName();
    // A weak alias may have been resolved, so check for that.
    if (Defined *D = Undef->getWeakAlias()) {
      // We resolve weak aliases by replacing the alias's SymbolBody with the
      // target's SymbolBody. This causes all SymbolBody pointers referring to
      // the old symbol to instead refer to the new symbol. However, we can't
      // just blindly copy sizeof(Symbol::Body) bytes from D to Sym->Body
      // because D may be an internal symbol, and internal symbols are stored as
      // "unparented" SymbolBodies. For that reason we need to check which type
      // of symbol we are dealing with and copy the correct number of bytes.
      if (isa<DefinedRegular>(D))
        memcpy(Sym->Body.buffer, D, sizeof(DefinedRegular));
      else if (isa<DefinedAbsolute>(D))
        memcpy(Sym->Body.buffer, D, sizeof(DefinedAbsolute));
      else
        // No other internal symbols are possible.
        Sym->Body = D->symbol()->Body;
      continue;
    }
    // If we can resolve a symbol by removing __imp_ prefix, do that.
    // This odd rule is for compatibility with MSVC linker.
    if (Name.startswith("__imp_")) {
      Symbol *Imp = find(Name.substr(strlen("__imp_")));
      if (Imp && isa<Defined>(Imp->body())) {
        auto *D = cast<Defined>(Imp->body());
        replaceBody<DefinedLocalImport>(Sym, Name, D);
        LocalImportChunks.push_back(
            cast<DefinedLocalImport>(Sym->body())->getChunk());
        continue;
      }
    }
    // Remaining undefined symbols are not fatal if /force is specified.
    // They are replaced with dummy defined symbols.
    if (Config->Force)
      replaceBody<DefinedAbsolute>(Sym, Name, 0);
    Undefs.insert(Sym->body());
  }
  if (Undefs.empty())
    return;
  for (SymbolBody *B : Config->GCRoot)
    if (Undefs.count(B))
      errs() << "<root>: undefined symbol: " << B->getName() << "\n";
  for (ObjectFile *File : ObjectFiles)
    for (SymbolBody *Sym : File->getSymbols())
      if (Undefs.count(Sym))
        errs() << toString(File) << ": undefined symbol: " << Sym->getName()
               << "\n";
  if (!Config->Force)
    fatal("link failed");
}

std::pair<Symbol *, bool> SymbolTable::insert(StringRef Name) {
  Symbol *&Sym = Symtab[CachedHashStringRef(Name)];
  if (Sym)
    return {Sym, false};
  Sym = make<Symbol>();
  Sym->IsUsedInRegularObj = false;
  return {Sym, true};
}

Symbol *SymbolTable::addUndefined(StringRef Name, InputFile *F,
                                  bool IsWeakAlias) {
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);
  if (!F || !isa<BitcodeFile>(F))
    S->IsUsedInRegularObj = true;
  if (WasInserted || (isa<Lazy>(S->body()) && IsWeakAlias)) {
    replaceBody<Undefined>(S, Name);
    return S;
  }
  if (auto *L = dyn_cast<Lazy>(S->body()))
    addMemberFile(L->File, L->Sym);
  return S;
}

void SymbolTable::addLazy(ArchiveFile *F, const Archive::Symbol Sym) {
  StringRef Name = Sym.getName();
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);
  if (WasInserted) {
    replaceBody<Lazy>(S, F, Sym);
    return;
  }
  auto *U = dyn_cast<Undefined>(S->body());
  if (!U || U->WeakAlias)
    return;
  addMemberFile(F, Sym);
}

void SymbolTable::reportDuplicate(Symbol *Existing, InputFile *NewFile) {
  fatal("duplicate symbol: " + toString(*Existing->body()) + " in " +
        toString(Existing->body()->getFile()) + " and in " +
        (NewFile ? toString(NewFile) : "(internal)"));
}

Symbol *SymbolTable::addAbsolute(StringRef N, COFFSymbolRef Sym) {
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(N);
  S->IsUsedInRegularObj = true;
  if (WasInserted || isa<Undefined>(S->body()) || isa<Lazy>(S->body()))
    replaceBody<DefinedAbsolute>(S, N, Sym);
  else if (!isa<DefinedCOFF>(S->body()))
    reportDuplicate(S, nullptr);
  return S;
}

Symbol *SymbolTable::addAbsolute(StringRef N, uint64_t VA) {
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(N);
  S->IsUsedInRegularObj = true;
  if (WasInserted || isa<Undefined>(S->body()) || isa<Lazy>(S->body()))
    replaceBody<DefinedAbsolute>(S, N, VA);
  else if (!isa<DefinedCOFF>(S->body()))
    reportDuplicate(S, nullptr);
  return S;
}

Symbol *SymbolTable::addRelative(StringRef N, uint64_t VA) {
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(N);
  S->IsUsedInRegularObj = true;
  if (WasInserted || isa<Undefined>(S->body()) || isa<Lazy>(S->body()))
    replaceBody<DefinedRelative>(S, N, VA);
  else if (!isa<DefinedCOFF>(S->body()))
    reportDuplicate(S, nullptr);
  return S;
}

Symbol *SymbolTable::addRegular(ObjectFile *F, COFFSymbolRef Sym,
                                SectionChunk *C) {
  StringRef Name;
  F->getCOFFObj()->getSymbolName(Sym, Name);
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);
  S->IsUsedInRegularObj = true;
  if (WasInserted || isa<Undefined>(S->body()) || isa<Lazy>(S->body()))
    replaceBody<DefinedRegular>(S, F, Sym, C);
  else if (auto *R = dyn_cast<DefinedRegular>(S->body())) {
    if (!C->isCOMDAT() || !R->isCOMDAT())
      reportDuplicate(S, F);
  } else if (auto *B = dyn_cast<DefinedBitcode>(S->body())) {
    if (B->IsReplaceable)
      replaceBody<DefinedRegular>(S, F, Sym, C);
    else if (!C->isCOMDAT())
      reportDuplicate(S, F);
  } else
    replaceBody<DefinedRegular>(S, F, Sym, C);
  return S;
}

Symbol *SymbolTable::addBitcode(BitcodeFile *F, StringRef N, bool IsReplaceable) {
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(N);
  if (WasInserted || isa<Undefined>(S->body()) || isa<Lazy>(S->body())) {
    replaceBody<DefinedBitcode>(S, F, N, IsReplaceable);
    return S;
  }
  if (isa<DefinedCommon>(S->body()))
    return S;
  if (IsReplaceable)
    if (isa<DefinedRegular>(S->body()) || isa<DefinedBitcode>(S->body()))
      return S;
  reportDuplicate(S, F);
  return S;
}

Symbol *SymbolTable::addCommon(ObjectFile *F, COFFSymbolRef Sym,
                               CommonChunk *C) {
  StringRef Name;
  F->getCOFFObj()->getSymbolName(Sym, Name);
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);
  S->IsUsedInRegularObj = true;
  if (WasInserted || !isa<DefinedCOFF>(S->body()))
    replaceBody<DefinedCommon>(S, F, Sym, C);
  else if (auto *DC = dyn_cast<DefinedCommon>(S->body()))
    if (Sym.getValue() > DC->getSize())
      replaceBody<DefinedCommon>(S, F, Sym, C);
  return S;
}

Symbol *SymbolTable::addImportData(StringRef N, ImportFile *F) {
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(N);
  S->IsUsedInRegularObj = true;
  if (WasInserted || isa<Undefined>(S->body()) || isa<Lazy>(S->body()))
    replaceBody<DefinedImportData>(S, N, F);
  else if (!isa<DefinedCOFF>(S->body()))
    reportDuplicate(S, nullptr);
  return S;
}

Symbol *SymbolTable::addImportThunk(StringRef Name, DefinedImportData *ID,
                                    uint16_t Machine) {
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);
  S->IsUsedInRegularObj = true;
  if (WasInserted || isa<Undefined>(S->body()) || isa<Lazy>(S->body()))
    replaceBody<DefinedImportThunk>(S, Name, ID, Machine);
  else if (!isa<DefinedCOFF>(S->body()))
    reportDuplicate(S, nullptr);
  return S;
}

// Reads an archive member file pointed by a given symbol.
void SymbolTable::addMemberFile(ArchiveFile *F, const Archive::Symbol Sym) {
  InputFile *File = F->getMember(&Sym);

  // getMember returns an empty buffer if the member was already
  // read from the library.
  if (!File)
    return;
  if (Config->Verbose)
    outs() << "Loaded " << toString(File) << " for " << Sym.getName() << "\n";
  addFile(File);
}

std::vector<Chunk *> SymbolTable::getChunks() {
  std::vector<Chunk *> Res;
  for (ObjectFile *File : ObjectFiles) {
    std::vector<Chunk *> &V = File->getChunks();
    Res.insert(Res.end(), V.begin(), V.end());
  }
  return Res;
}

Symbol *SymbolTable::find(StringRef Name) {
  auto It = Symtab.find(CachedHashStringRef(Name));
  if (It == Symtab.end())
    return nullptr;
  return It->second;
}

Symbol *SymbolTable::findUnderscore(StringRef Name) {
  if (Config->Machine == I386)
    return find(("_" + Name).str());
  return find(Name);
}

StringRef SymbolTable::findByPrefix(StringRef Prefix) {
  for (auto Pair : Symtab) {
    StringRef Name = Pair.first.val();
    if (Name.startswith(Prefix))
      return Name;
  }
  return "";
}

StringRef SymbolTable::findMangle(StringRef Name) {
  if (Symbol *Sym = find(Name))
    if (!isa<Undefined>(Sym->body()))
      return Name;
  if (Config->Machine != I386)
    return findByPrefix(("?" + Name + "@@Y").str());
  if (!Name.startswith("_"))
    return "";
  // Search for x86 C function.
  StringRef S = findByPrefix((Name + "@").str());
  if (!S.empty())
    return S;
  // Search for x86 C++ non-member function.
  return findByPrefix(("?" + Name.substr(1) + "@@Y").str());
}

void SymbolTable::mangleMaybe(SymbolBody *B) {
  auto *U = dyn_cast<Undefined>(B);
  if (!U || U->WeakAlias)
    return;
  StringRef Alias = findMangle(U->getName());
  if (!Alias.empty())
    U->WeakAlias = addUndefined(Alias);
}

SymbolBody *SymbolTable::addUndefined(StringRef Name) {
  return addUndefined(Name, nullptr, false)->body();
}

void SymbolTable::printMap(llvm::raw_ostream &OS) {
  for (ObjectFile *File : ObjectFiles) {
    OS << toString(File) << ":\n";
    for (SymbolBody *Body : File->getSymbols())
      if (auto *R = dyn_cast<DefinedRegular>(Body))
        if (R->getChunk()->isLive())
          OS << Twine::utohexstr(Config->ImageBase + R->getRVA())
             << " " << R->getName() << "\n";
  }
}

void SymbolTable::addCombinedLTOObjects() {
  if (BitcodeFiles.empty())
    return;

  // Create an object file and add it to the symbol table by replacing any
  // DefinedBitcode symbols with the definitions in the object file.
  LTOCodeGenerator CG(BitcodeFile::Context);
  CG.setOptLevel(Config->LTOOptLevel);
  std::vector<ObjectFile *> Objs = createLTOObjects(&CG);

  size_t NumBitcodeFiles = BitcodeFiles.size();
  for (ObjectFile *Obj : Objs)
    Obj->parse();
  if (BitcodeFiles.size() != NumBitcodeFiles)
    fatal("LTO: late loaded symbol created new bitcode reference");
}

// Combine and compile bitcode files and then return the result
// as a vector of regular COFF object files.
std::vector<ObjectFile *> SymbolTable::createLTOObjects(LTOCodeGenerator *CG) {
  // All symbols referenced by non-bitcode objects, including GC roots, must be
  // preserved. We must also replace bitcode symbols with undefined symbols so
  // that they may be replaced with real definitions without conflicting.
  for (BitcodeFile *File : BitcodeFiles)
    for (SymbolBody *Body : File->getSymbols()) {
      if (!isa<DefinedBitcode>(Body))
        continue;
      if (Body->symbol()->IsUsedInRegularObj)
        CG->addMustPreserveSymbol(Body->getName());
      replaceBody<Undefined>(Body->symbol(), Body->getName());
    }

  CG->setModule(BitcodeFiles[0]->takeModule());
  for (unsigned I = 1, E = BitcodeFiles.size(); I != E; ++I)
    CG->addModule(BitcodeFiles[I]->takeModule().get());

  bool DisableVerify = true;
#ifdef NDEBUG
  DisableVerify = false;
#endif
  if (!CG->optimize(DisableVerify, false, false, false))
    fatal(""); // optimize() should have emitted any error message.

  Objs.resize(Config->LTOJobs);
  // Use std::list to avoid invalidation of pointers in OSPtrs.
  std::list<raw_svector_ostream> OSs;
  std::vector<raw_pwrite_stream *> OSPtrs;
  for (SmallString<0> &Obj : Objs) {
    OSs.emplace_back(Obj);
    OSPtrs.push_back(&OSs.back());
  }

  if (!CG->compileOptimized(OSPtrs))
    fatal(""); // compileOptimized() should have emitted any error message.

  std::vector<ObjectFile *> ObjFiles;
  for (SmallString<0> &Obj : Objs) {
    auto *ObjFile = make<ObjectFile>(MemoryBufferRef(Obj, "<LTO object>"));
    ObjectFiles.push_back(ObjFile);
    ObjFiles.push_back(ObjFile);
  }

  return ObjFiles;
}

} // namespace coff
} // namespace lld
