//===- SymbolTable.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "LinkerScript.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Strings.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf;

SymbolTable *elf::Symtab;

// This function is where all the optimizations of link-time
// optimization happens. When LTO is in use, some input files are
// not in native object file format but in the LLVM bitcode format.
// This function compiles bitcode files into a few big native files
// using LLVM functions and replaces bitcode symbols with the results.
// Because all bitcode files that the program consists of are passed
// to the compiler at once, it can do whole-program optimization.
template <class ELFT> void SymbolTable::addCombinedLTOObject() {
  // Compile bitcode files and replace bitcode symbols.
  LTO.reset(new BitcodeCompiler);
  for (BitcodeFile *F : BitcodeFiles)
    LTO->add(*F);

  for (InputFile *File : LTO->compile()) {
    DenseSet<CachedHashStringRef> DummyGroups;
    auto *Obj = cast<ObjFile<ELFT>>(File);
    Obj->parse(DummyGroups);
    for (Symbol *Sym : Obj->getGlobalSymbols())
      Sym->parseSymbolVersion();
    ObjectFiles.push_back(File);
  }
}

// Set a flag for --trace-symbol so that we can print out a log message
// if a new symbol with the same name is inserted into the symbol table.
void SymbolTable::trace(StringRef Name) {
  SymMap.insert({CachedHashStringRef(Name), -1});
}

void SymbolTable::wrap(Symbol *Sym, Symbol *Real, Symbol *Wrap) {
  // Swap symbols as instructed by -wrap.
  int &Idx1 = SymMap[CachedHashStringRef(Sym->getName())];
  int &Idx2 = SymMap[CachedHashStringRef(Real->getName())];
  int &Idx3 = SymMap[CachedHashStringRef(Wrap->getName())];

  Idx2 = Idx1;
  Idx1 = Idx3;

  // Now renaming is complete. No one refers Real symbol. We could leave
  // Real as-is, but if Real is written to the symbol table, that may
  // contain irrelevant values. So, we copy all values from Sym to Real.
  StringRef S = Real->getName();
  memcpy(Real, Sym, sizeof(SymbolUnion));
  Real->setName(S);
}

static uint8_t getMinVisibility(uint8_t VA, uint8_t VB) {
  if (VA == STV_DEFAULT)
    return VB;
  if (VB == STV_DEFAULT)
    return VA;
  return std::min(VA, VB);
}

// Find an existing symbol or create and insert a new one.
Symbol *SymbolTable::insert(const Symbol &New) {
  // <name>@@<version> means the symbol is the default version. In that
  // case <name>@@<version> will be used to resolve references to <name>.
  //
  // Since this is a hot path, the following string search code is
  // optimized for speed. StringRef::find(char) is much faster than
  // StringRef::find(StringRef).
  StringRef Name = New.getName();
  size_t Pos = Name.find('@');
  if (Pos != StringRef::npos && Pos + 1 < Name.size() && Name[Pos + 1] == '@')
    Name = Name.take_front(Pos);

  auto P = SymMap.insert({CachedHashStringRef(Name), (int)SymVector.size()});
  int &SymIndex = P.first->second;
  bool IsNew = P.second;
  bool Traced = false;

  if (SymIndex == -1) {
    SymIndex = SymVector.size();
    IsNew = true;
    Traced = true;
  }

  Symbol *Old;
  if (IsNew) {
    Old = reinterpret_cast<Symbol *>(make<SymbolUnion>());
    SymVector.push_back(Old);

    Old->SymbolKind = Symbol::PlaceholderKind;
    Old->VersionId = Config->DefaultSymbolVersion;
    Old->Visibility = STV_DEFAULT;
    Old->IsUsedInRegularObj = false;
    Old->ExportDynamic = false;
    Old->CanInline = true;
    Old->Traced = Traced;
    Old->ScriptDefined = false;
  } else {
    Old = SymVector[SymIndex];
  }

  return Old;
}

// Merge symbol properties.
//
// When we have many symbols of the same name, we choose one of them,
// and that's the result of symbol resolution. However, symbols that
// were not chosen still affect some symbol properteis.
void SymbolTable::mergeProperties(Symbol *Old, const Symbol &New) {
  // Merge symbol properties.
  Old->ExportDynamic = Old->ExportDynamic || New.ExportDynamic;
  Old->IsUsedInRegularObj = Old->IsUsedInRegularObj || New.IsUsedInRegularObj;

  // DSO symbols do not affect visibility in the output.
  if (!isa<SharedSymbol>(New))
    Old->Visibility = getMinVisibility(Old->Visibility, New.Visibility);
}

Symbol *SymbolTable::addUndefined(const Undefined &New) {
  Symbol *Old = insert(New);
  mergeProperties(Old, New);

  if (Old->isPlaceholder()) {
    replaceSymbol(Old, &New);
    return Old;
  }

  // An undefined symbol with non default visibility must be satisfied
  // in the same DSO.
  if (Old->isShared() && New.Visibility != STV_DEFAULT) {
    replaceSymbol(Old, &New);
    return Old;
  }

  if (Old->isShared() || Old->isLazy() ||
      (Old->isUndefined() && New.Binding != STB_WEAK))
    Old->Binding = New.Binding;

  if (Old->isLazy()) {
    // An undefined weak will not fetch archive members. See comment on Lazy in
    // Symbols.h for the details.
    if (New.Binding == STB_WEAK) {
      Old->Type = New.Type;
      return Old;
    }

    // Do extra check for --warn-backrefs.
    //
    // --warn-backrefs is an option to prevent an undefined reference from
    // fetching an archive member written earlier in the command line. It can be
    // used to keep compatibility with GNU linkers to some degree.
    // I'll explain the feature and why you may find it useful in this comment.
    //
    // lld's symbol resolution semantics is more relaxed than traditional Unix
    // linkers. For example,
    //
    //   ld.lld foo.a bar.o
    //
    // succeeds even if bar.o contains an undefined symbol that has to be
    // resolved by some object file in foo.a. Traditional Unix linkers don't
    // allow this kind of backward reference, as they visit each file only once
    // from left to right in the command line while resolving all undefined
    // symbols at the moment of visiting.
    //
    // In the above case, since there's no undefined symbol when a linker visits
    // foo.a, no files are pulled out from foo.a, and because the linker forgets
    // about foo.a after visiting, it can't resolve undefined symbols in bar.o
    // that could have been resolved otherwise.
    //
    // That lld accepts more relaxed form means that (besides it'd make more
    // sense) you can accidentally write a command line or a build file that
    // works only with lld, even if you have a plan to distribute it to wider
    // users who may be using GNU linkers. With --warn-backrefs, you can detect
    // a library order that doesn't work with other Unix linkers.
    //
    // The option is also useful to detect cyclic dependencies between static
    // archives. Again, lld accepts
    //
    //   ld.lld foo.a bar.a
    //
    // even if foo.a and bar.a depend on each other. With --warn-backrefs, it is
    // handled as an error.
    //
    // Here is how the option works. We assign a group ID to each file. A file
    // with a smaller group ID can pull out object files from an archive file
    // with an equal or greater group ID. Otherwise, it is a reverse dependency
    // and an error.
    //
    // A file outside --{start,end}-group gets a fresh ID when instantiated. All
    // files within the same --{start,end}-group get the same group ID. E.g.
    //
    //   ld.lld A B --start-group C D --end-group E
    //
    // A forms group 0. B form group 1. C and D (including their member object
    // files) form group 2. E forms group 3. I think that you can see how this
    // group assignment rule simulates the traditional linker's semantics.
    bool Backref = Config->WarnBackrefs && New.File &&
                   Old->File->GroupId < New.File->GroupId;
    fetchLazy(Old);

    // We don't report backward references to weak symbols as they can be
    // overridden later.
    if (Backref && !Old->isWeak())
      warn("backward reference detected: " + New.getName() + " in " +
           toString(New.File) + " refers to " + toString(Old->File));
  }
  return Old;
}

// Using .symver foo,foo@@VER unfortunately creates two symbols: foo and
// foo@@VER. We want to effectively ignore foo, so give precedence to
// foo@@VER.
// FIXME: If users can transition to using
// .symver foo,foo@@@VER
// we can delete this hack.
static int compareVersion(StringRef OldName, StringRef NewName) {
  bool A = OldName.contains("@@");
  bool B = NewName.contains("@@");
  if (!A && B)
    return 1;
  if (A && !B)
    return -1;
  return 0;
}

// Compare two symbols. Return 1 if the new symbol should win, -1 if
// the new symbol should lose, or 0 if there is a conflict.
static int compare(const Symbol *Old, const Symbol *New) {
  assert(New->isDefined() || New->isCommon());

  if (!Old->isDefined() && !Old->isCommon())
    return 1;

  if (int Cmp = compareVersion(Old->getName(), New->getName()))
    return Cmp;

  if (New->isWeak())
    return -1;

  if (Old->isWeak())
    return 1;

  if (Old->isCommon() && New->isCommon()) {
    if (Config->WarnCommon)
      warn("multiple common of " + Old->getName());
    return 0;
  }

  if (Old->isCommon()) {
    if (Config->WarnCommon)
      warn("common " + Old->getName() + " is overridden");
    return 1;
  }

  if (New->isCommon()) {
    if (Config->WarnCommon)
      warn("common " + Old->getName() + " is overridden");
    return -1;
  }

  auto *OldSym = cast<Defined>(Old);
  auto *NewSym = cast<Defined>(New);

  if (New->File && isa<BitcodeFile>(New->File))
    return 0;

  if (!OldSym->Section && !NewSym->Section && OldSym->Value == NewSym->Value &&
      NewSym->Binding == STB_GLOBAL)
    return -1;

  return 0;
}

Symbol *SymbolTable::addCommon(const CommonSymbol &New) {
  Symbol *Old = insert(New);
  mergeProperties(Old, New);

  if (Old->isPlaceholder()) {
    replaceSymbol(Old, &New);
    return Old;
  }

  int Cmp = compare(Old, &New);
  if (Cmp < 0)
    return Old;

  if (Cmp > 0) {
    replaceSymbol(Old, &New);
    return Old;
  }

  CommonSymbol *OldSym = cast<CommonSymbol>(Old);

  OldSym->Alignment = std::max(OldSym->Alignment, New.Alignment);
  if (OldSym->Size < New.Size) {
    OldSym->File = New.File;
    OldSym->Size = New.Size;
  }
  return OldSym;
}

static void reportDuplicate(Symbol *Sym, InputFile *NewFile,
                            InputSectionBase *ErrSec, uint64_t ErrOffset) {
  if (Config->AllowMultipleDefinition)
    return;

  Defined *D = cast<Defined>(Sym);
  if (!D->Section || !ErrSec) {
    error("duplicate symbol: " + toString(*Sym) + "\n>>> defined in " +
          toString(Sym->File) + "\n>>> defined in " + toString(NewFile));
    return;
  }

  // Construct and print an error message in the form of:
  //
  //   ld.lld: error: duplicate symbol: foo
  //   >>> defined at bar.c:30
  //   >>>            bar.o (/home/alice/src/bar.o)
  //   >>> defined at baz.c:563
  //   >>>            baz.o in archive libbaz.a
  auto *Sec1 = cast<InputSectionBase>(D->Section);
  std::string Src1 = Sec1->getSrcMsg(*Sym, D->Value);
  std::string Obj1 = Sec1->getObjMsg(D->Value);
  std::string Src2 = ErrSec->getSrcMsg(*Sym, ErrOffset);
  std::string Obj2 = ErrSec->getObjMsg(ErrOffset);

  std::string Msg = "duplicate symbol: " + toString(*Sym) + "\n>>> defined at ";
  if (!Src1.empty())
    Msg += Src1 + "\n>>>            ";
  Msg += Obj1 + "\n>>> defined at ";
  if (!Src2.empty())
    Msg += Src2 + "\n>>>            ";
  Msg += Obj2;
  error(Msg);
}

Symbol *SymbolTable::addDefined(const Defined &New) {
  Symbol *Old = insert(New);
  mergeProperties(Old, New);

  if (Old->isPlaceholder()) {
    replaceSymbol(Old, &New);
    return Old;
  }

  int Cmp = compare(Old, &New);
  if (Cmp > 0)
    replaceSymbol(Old, &New);
  else if (Cmp == 0)
    reportDuplicate(Old, New.File,
                    dyn_cast_or_null<InputSectionBase>(New.Section), New.Value);
  return Old;
}

Symbol *SymbolTable::addShared(const SharedSymbol &New) {
  Symbol *Old = insert(New);
  mergeProperties(Old, New);

  // Make sure we preempt DSO symbols with default visibility.
  if (New.Visibility == STV_DEFAULT)
    Old->ExportDynamic = true;

  if (Old->isPlaceholder()) {
    replaceSymbol(Old, &New);
    return Old;
  }

  if (Old->Visibility == STV_DEFAULT && (Old->isUndefined() || Old->isLazy())) {
    // An undefined symbol with non default visibility must be satisfied
    // in the same DSO.
    uint8_t Binding = Old->Binding;
    replaceSymbol(Old, &New);
    Old->Binding = Binding;
  }
  return Old;
}

Symbol *SymbolTable::find(StringRef Name) {
  auto It = SymMap.find(CachedHashStringRef(Name));
  if (It == SymMap.end())
    return nullptr;
  if (It->second == -1)
    return nullptr;
  return SymVector[It->second];
}

template <class LazyT> Symbol *SymbolTable::addLazy(const LazyT &New) {
  Symbol *Old = insert(New);
  mergeProperties(Old, New);

  if (Old->isPlaceholder()) {
    replaceSymbol(Old, &New);
    return Old;
  }

  if (!Old->isUndefined())
    return Old;

  // An undefined weak will not fetch archive members. See comment on Lazy in
  // Symbols.h for the details.
  if (Old->isWeak()) {
    uint8_t Type = Old->Type;
    replaceSymbol(Old, &New);
    Old->Type = Type;
    Old->Binding = STB_WEAK;
    return Old;
  }

  if (InputFile *F = New.fetch())
    parseFile(F);
  return Old;
}

Symbol *SymbolTable::addLazyArchive(const LazyArchive &New) {
  return addLazy(New);
}

Symbol *SymbolTable::addLazyObject(const LazyObject &New) {
  return addLazy(New);
}

void SymbolTable::fetchLazy(Symbol *Sym) {
  if (auto *S = dyn_cast<LazyArchive>(Sym)) {
    if (InputFile *File = S->fetch())
      parseFile(File);
    return;
  }

  auto *S = cast<LazyObject>(Sym);
  if (InputFile *File = cast<LazyObjFile>(S->File)->fetch())
    parseFile(File);
}

// Initialize DemangledSyms with a map from demangled symbols to symbol
// objects. Used to handle "extern C++" directive in version scripts.
//
// The map will contain all demangled symbols. That can be very large,
// and in LLD we generally want to avoid do anything for each symbol.
// Then, why are we doing this? Here's why.
//
// Users can use "extern C++ {}" directive to match against demangled
// C++ symbols. For example, you can write a pattern such as
// "llvm::*::foo(int, ?)". Obviously, there's no way to handle this
// other than trying to match a pattern against all demangled symbols.
// So, if "extern C++" feature is used, we need to demangle all known
// symbols.
StringMap<std::vector<Symbol *>> &SymbolTable::getDemangledSyms() {
  if (!DemangledSyms) {
    DemangledSyms.emplace();
    for (Symbol *Sym : SymVector) {
      if (!Sym->isDefined() && !Sym->isCommon())
        continue;
      if (Optional<std::string> S = demangleItanium(Sym->getName()))
        (*DemangledSyms)[*S].push_back(Sym);
      else
        (*DemangledSyms)[Sym->getName()].push_back(Sym);
    }
  }
  return *DemangledSyms;
}

std::vector<Symbol *> SymbolTable::findByVersion(SymbolVersion Ver) {
  if (Ver.IsExternCpp)
    return getDemangledSyms().lookup(Ver.Name);
  if (Symbol *B = find(Ver.Name))
    if (B->isDefined() || B->isCommon())
      return {B};
  return {};
}

std::vector<Symbol *> SymbolTable::findAllByVersion(SymbolVersion Ver) {
  std::vector<Symbol *> Res;
  StringMatcher M(Ver.Name);

  if (Ver.IsExternCpp) {
    for (auto &P : getDemangledSyms())
      if (M.match(P.first()))
        Res.insert(Res.end(), P.second.begin(), P.second.end());
    return Res;
  }

  for (Symbol *Sym : SymVector)
    if ((Sym->isDefined() || Sym->isCommon()) && M.match(Sym->getName()))
      Res.push_back(Sym);
  return Res;
}

// If there's only one anonymous version definition in a version
// script file, the script does not actually define any symbol version,
// but just specifies symbols visibilities.
void SymbolTable::handleAnonymousVersion() {
  for (SymbolVersion &Ver : Config->VersionScriptGlobals)
    assignExactVersion(Ver, VER_NDX_GLOBAL, "global");
  for (SymbolVersion &Ver : Config->VersionScriptGlobals)
    assignWildcardVersion(Ver, VER_NDX_GLOBAL);
  for (SymbolVersion &Ver : Config->VersionScriptLocals)
    assignExactVersion(Ver, VER_NDX_LOCAL, "local");
  for (SymbolVersion &Ver : Config->VersionScriptLocals)
    assignWildcardVersion(Ver, VER_NDX_LOCAL);
}

// Handles -dynamic-list.
void SymbolTable::handleDynamicList() {
  for (SymbolVersion &Ver : Config->DynamicList) {
    std::vector<Symbol *> Syms;
    if (Ver.HasWildcard)
      Syms = findAllByVersion(Ver);
    else
      Syms = findByVersion(Ver);

    for (Symbol *B : Syms) {
      if (!Config->Shared)
        B->ExportDynamic = true;
      else if (B->includeInDynsym())
        B->IsPreemptible = true;
    }
  }
}

// Set symbol versions to symbols. This function handles patterns
// containing no wildcard characters.
void SymbolTable::assignExactVersion(SymbolVersion Ver, uint16_t VersionId,
                                     StringRef VersionName) {
  if (Ver.HasWildcard)
    return;

  // Get a list of symbols which we need to assign the version to.
  std::vector<Symbol *> Syms = findByVersion(Ver);
  if (Syms.empty()) {
    if (!Config->UndefinedVersion)
      error("version script assignment of '" + VersionName + "' to symbol '" +
            Ver.Name + "' failed: symbol not defined");
    return;
  }

  // Assign the version.
  for (Symbol *Sym : Syms) {
    // Skip symbols containing version info because symbol versions
    // specified by symbol names take precedence over version scripts.
    // See parseSymbolVersion().
    if (Sym->getName().contains('@'))
      continue;

    if (Sym->VersionId != Config->DefaultSymbolVersion &&
        Sym->VersionId != VersionId)
      error("duplicate symbol '" + Ver.Name + "' in version script");
    Sym->VersionId = VersionId;
  }
}

void SymbolTable::assignWildcardVersion(SymbolVersion Ver, uint16_t VersionId) {
  if (!Ver.HasWildcard)
    return;

  // Exact matching takes precendence over fuzzy matching,
  // so we set a version to a symbol only if no version has been assigned
  // to the symbol. This behavior is compatible with GNU.
  for (Symbol *B : findAllByVersion(Ver))
    if (B->VersionId == Config->DefaultSymbolVersion)
      B->VersionId = VersionId;
}

// This function processes version scripts by updating VersionId
// member of symbols.
void SymbolTable::scanVersionScript() {
  // Handle edge cases first.
  handleAnonymousVersion();
  handleDynamicList();

  // Now we have version definitions, so we need to set version ids to symbols.
  // Each version definition has a glob pattern, and all symbols that match
  // with the pattern get that version.

  // First, we assign versions to exact matching symbols,
  // i.e. version definitions not containing any glob meta-characters.
  for (VersionDefinition &V : Config->VersionDefinitions)
    for (SymbolVersion &Ver : V.Globals)
      assignExactVersion(Ver, V.Id, V.Name);

  // Next, we assign versions to fuzzy matching symbols,
  // i.e. version definitions containing glob meta-characters.
  // Note that because the last match takes precedence over previous matches,
  // we iterate over the definitions in the reverse order.
  for (VersionDefinition &V : llvm::reverse(Config->VersionDefinitions))
    for (SymbolVersion &Ver : V.Globals)
      assignWildcardVersion(Ver, V.Id);

  // Symbol themselves might know their versions because symbols
  // can contain versions in the form of <name>@<version>.
  // Let them parse and update their names to exclude version suffix.
  for (Symbol *Sym : SymVector)
    Sym->parseSymbolVersion();
}

template void SymbolTable::addCombinedLTOObject<ELF32LE>();
template void SymbolTable::addCombinedLTOObject<ELF32BE>();
template void SymbolTable::addCombinedLTOObject<ELF64LE>();
template void SymbolTable::addCombinedLTOObject<ELF64BE>();
