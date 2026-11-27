//===- SymbolTable.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_SYMBOL_TABLE_H
#define LLD_COFF_SYMBOL_TABLE_H

#include "InputFiles.h"
#include "LTO.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
struct LTOCodeGenerator;
}

namespace lld::coff {

class Chunk;
class CommonChunk;
class COFFLinkerContext;
class Defined;
class DefinedAbsolute;
class DefinedRegular;
class ImportThunkChunk;
class LazyArchive;
class SameAddressThunkARM64EC;
class SectionChunk;
class Symbol;

// This data structure is instantiated for each -wrap option.
struct WrappedSymbol {
  Symbol *sym;
  Symbol *real;
  Symbol *wrap;
};

struct UndefinedDiag;

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
// add*() functions, which are called by input files as they are parsed.
// There is one add* function per symbol type.
class SymbolTable {
public:
  SymbolTable(COFFLinkerContext &c,
              llvm::COFF::MachineTypes machine = IMAGE_FILE_MACHINE_UNKNOWN)
      : ctx(c), machine(machine) {}

  // Emit errors for symbols that cannot be resolved.
  void reportUnresolvable();

  // Try to resolve any undefined symbols and update the symbol table
  // accordingly, then print an error message for any remaining undefined
  // symbols and warn about imported local symbols.
  void resolveRemainingUndefines(std::vector<Undefined *> &aliases);

  // Try to resolve undefined symbols with alternate names.
  void resolveAlternateNames();

  // Load lazy objects that are needed for MinGW automatic import and for
  // doing stdcall fixups.
  void loadMinGWSymbols(bool loadStdcallFixups, bool loadAutoImports);
  bool handleMinGWAutomaticImport(Symbol *sym, StringRef name);

  // Returns a symbol for a given name. Returns a nullptr if not found.
  Symbol *find(StringRef name) const;
  Symbol *findUnderscore(StringRef name) const;

  void addUndefinedGlob(StringRef arg);

  // Occasionally we have to resolve an undefined symbol to its
  // mangled symbol. This function tries to find a mangled name
  // for U from the symbol table, and if found, set the symbol as
  // a weak alias for U.
  Symbol *findMangle(StringRef name);
  StringRef mangleMaybe(Symbol *s);

  // Symbol names are mangled by prepending "_" on x86.
  StringRef mangle(StringRef sym);

  // Windows specific -- "main" is not the only main function in Windows.
  // You can choose one from these four -- {w,}{WinMain,main}.
  // There are four different entry point functions for them,
  // {w,}{WinMain,main}CRTStartup, respectively. The linker needs to
  // choose the right one depending on which "main" function is defined.
  // This function looks up the symbol table and resolve corresponding
  // entry point name.
  StringRef findDefaultEntry();
  WindowsSubsystem inferSubsystem();

  // Build a set of COFF objects representing the combined contents of
  // BitcodeFiles and add them to the symbol table. Called after all files are
  // added and before the writer writes results to a file.
  void compileBitcodeFiles();

  void waitForLTOCleanup();

  // Creates an Undefined symbol and marks it as live.
  Symbol *addGCRoot(StringRef sym, bool aliasEC = false);

  // Creates an Undefined symbol for a given name.
  Symbol *addUndefined(StringRef name);

  Symbol *addSynthetic(StringRef n, Chunk *c);
  Symbol *addAbsolute(StringRef n, uint64_t va);

  Symbol *addUndefined(StringRef name, InputFile *f, bool overrideLazy,
                       bool markReference = true);
  // Registers a definition from a discarded section without creating a
  // reference or forcing a lazy provider.
  Symbol *addDiscardedDefinition(StringRef name);
  Symbol *canonicalizeDeferredCOMDATSymbol(Symbol *symbol) const;
  bool hasDeferredCOMDATWork() const;
  void addWeakAlias(InputFile *f, Symbol *source, Symbol *target,
                    bool isAntiDep);
  void addReference(Symbol *symbol);
  void deferSectionReferences(SectionChunk *chunk);
  void addUnrelocatedReference(Symbol *symbol);
  void resolveDeferredReferences();
  void finalizeDeferredSymbols();
  void restoreDeferredSymbolState(Undefined *source);
  void resolveDeferredSymbols();
  void addLazyArchive(ArchiveFile *f, const Archive::Symbol &sym);
  void addLazyObject(InputFile *f, StringRef n);
  void addLazyDLLSymbol(DLLFile *f, DLLFile::Symbol *sym, StringRef n);
  Symbol *addAbsolute(StringRef n, COFFSymbolRef s);
  Symbol *addRegular(InputFile *f, StringRef n,
                     const llvm::object::coff_symbol_generic *s = nullptr,
                     SectionChunk *c = nullptr, uint32_t sectionOffset = 0,
                     bool isWeak = false);
  Symbol *addDeferredComdatRegular(BitcodeFile *f, StringRef n, SectionChunk *c,
                                   bool isWeak, Symbol *comdatLeader);
  std::pair<DefinedRegular *, bool>
  addComdat(InputFile *f, StringRef n,
            const llvm::object::coff_symbol_generic *s = nullptr);
  Symbol *addCommon(InputFile *f, StringRef n, uint64_t size,
                    const llvm::object::coff_symbol_generic *s = nullptr,
                    CommonChunk *c = nullptr);
  DefinedImportData *addImportData(StringRef n, ImportFile *f,
                                   Chunk *&location);
  Defined *addImportThunk(StringRef name, DefinedImportData *s,
                          ImportThunkChunk *chunk);
  void addLibcall(StringRef name);
  void addEntryThunk(Symbol *from, Symbol *to);
  void addExitThunk(Symbol *from, Symbol *to);
  void initializeECThunks();
  void initializeSameAddressThunks();

  void reportDuplicate(Symbol *existing, InputFile *newFile,
                       SectionChunk *newSc = nullptr,
                       uint32_t newSectionOffset = 0);

  COFFLinkerContext &ctx;
  llvm::COFF::MachineTypes machine;

  bool isEC() const { return machine == ARM64EC; }

  // An entry point symbol.
  Symbol *entry = nullptr;

  // A list of chunks which to be added to .rdata.
  std::vector<Chunk *> localImportChunks;

  // A list of EC EXP+ symbols.
  std::vector<Symbol *> expSymbols;

  std::vector<SameAddressThunkARM64EC *> sameAddressThunks;

  // A list of DLL exports.
  std::vector<Export> exports;
  llvm::DenseSet<StringRef> directivesExports;
  bool hadExplicitExports;

  Chunk *edataStart = nullptr;
  Chunk *edataEnd = nullptr;

  Symbol *delayLoadHelper = nullptr;
  Chunk *tailMergeUnwindInfoChunk = nullptr;

  // A list of wrapped symbols.
  std::vector<WrappedSymbol> wrapped;

  // Used for /alternatename.
  std::map<StringRef, StringRef> alternateNames;

  // Used for /aligncomm.
  std::map<std::string, int> alignComm;

  void fixupExports();
  void assignExportOrdinals();
  void parseModuleDefs(StringRef path);
  void parseAlternateName(StringRef);
  void parseAligncomm(StringRef);

  // Iterates symbols in non-determinstic hash table order.
  template <typename T> void forEachSymbol(T callback) {
    for (auto &pair : symMap)
      callback(pair.second);
  }

  std::vector<BitcodeFile *> bitcodeFileInstances;

  DefinedRegular *loadConfigSym = nullptr;
  uint32_t loadConfigSize = 0;
  void initializeLoadConfig();

  std::string printSymbol(Symbol *sym) const;

private:
  struct DeferredWeakAlias {
    InputFile *file;
    Symbol *target;
    bool isAntiDep;
  };
  struct DeferredWeakAliases {
    Symbol *initialTarget = nullptr;
    bool initialIsAntiDep = false;
    llvm::SmallVector<DeferredWeakAlias, 2> suppressed;
  };
  struct DeferredDuplicate {
    SectionChunk *chunk;
    Symbol *existing;
    InputFile *newFile;
    uint32_t sectionOffset;
  };
  struct DeferredImportThunk {
    ImportFile *file;
    StringRef importName;
  };

  using DeferredProviderId = uint32_t;
  static constexpr DeferredProviderId noDeferredProvider = ~uint32_t{0};

  // Deferred regular definitions are represented directly by their replay
  // inputs. Other provider kinds keep their existing stable Symbol object as
  // payload. COMDAT leaders intentionally remain materialized symbols because
  // SectionChunk::sym and selection state use their identity while parsing.
  struct DeferredProvider {
    Symbol *symbol = nullptr;
    InputFile *regularFile = nullptr;
    const llvm::object::coff_symbol_generic *regularSym = nullptr;
    SectionChunk *regularChunk = nullptr;
    Symbol *parent = nullptr;
    uint32_t regularValue = 0;
    DeferredProviderId next = noDeferredProvider;
    bool isRegular = false;
    bool regularIsWeak = false;
  };

  struct DeferredSourceState {
    DeferredProviderId first = noDeferredProvider;
    DeferredProviderId last = noDeferredProvider;
  };

  void deferDefinedSymbol(Symbol *source, Symbol *provider,
                          bool isUsedInRegularObj, Symbol *parent = nullptr);
  void deferRegularSymbol(Symbol *source, InputFile *file,
                          const llvm::object::coff_symbol_generic *coffSym,
                          SectionChunk *chunk, uint32_t value, bool isWeak,
                          Symbol *parent = nullptr);
  void appendDeferredProvider(Symbol *source, DeferredProvider provider);
  Symbol *cloneDeferredSymbol(Symbol *symbol);
  void replayDeferredSymbol(Symbol *source, DeferredProviderId providerId);
  void replayDeferredSources(ArrayRef<Symbol *> sources, bool importThunks,
                             bool final);
  Symbol *
  addRegularWithDeferredCOMDAT(InputFile *file, StringRef name,
                               const llvm::object::coff_symbol_generic *coffSym,
                               SectionChunk *chunk, uint32_t sectionOffset,
                               bool isWeak, Symbol *existing, bool provisional,
                               bool existingProvisional);
  void updateDeferredCommonLiveness(DeferredProviderId firstProvider,
                                    Symbol *source);
  void updateDeferredImportLiveness();
  void markReferenced(Symbol *symbol, bool forceLazyProvider = true);

  /// Given a name without "__imp_" prefix, returns a defined symbol
  /// with the "__imp_" prefix, if it exists.
  Defined *impSymbol(StringRef name);
  /// Inserts symbol if not already present.
  std::pair<Symbol *, bool> insert(StringRef name);
  /// Same as insert(Name), but also sets isUsedInRegularObj.
  std::pair<Symbol *, bool> insert(StringRef name, InputFile *f);

  bool findUnderscoreMangle(StringRef sym);
  std::vector<Symbol *> getSymsWithPrefix(StringRef prefix);

  llvm::DenseMap<llvm::CachedHashStringRef, Symbol *> symMap;
  llvm::DenseMap<Symbol *, DeferredWeakAliases> deferredWeakAliases;
  llvm::DenseMap<Symbol *, DeferredSourceState> deferredSymbols;
  llvm::SmallVector<DeferredProvider, 0> deferredProviders;
  llvm::SmallVector<Symbol *, 0> deferredSymbolSources;
  llvm::SmallVector<Symbol *, 0> deferredSymbolsToResolve;
  llvm::SmallPtrSet<Symbol *, 8> deferredSymbolsQueued;
  llvm::SmallVector<SectionChunk *, 0> deferredReferenceChunks;
  llvm::SmallVector<DeferredDuplicate, 0> deferredDuplicates;
  llvm::DenseMap<Symbol *, DeferredImportThunk> deferredImportThunks;
  llvm::DenseMap<Symbol *, Symbol *> deferredComdatLeaderSources;
  llvm::DenseMap<Symbol *, bool> deferredComdatSelections;
  llvm::DenseMap<Symbol *, Symbol *> selectedDeferredComdats;
  llvm::SmallPtrSet<ImportFile *, 8> deferredImportFiles;
  bool replayingDeferredSymbols = false;
  std::unique_ptr<BitcodeCompiler> lto;
  std::vector<std::pair<Symbol *, Symbol *>> entryThunks;
  llvm::DenseMap<Symbol *, Symbol *> exitThunks;

  void
  reportProblemSymbols(const llvm::SmallPtrSetImpl<Symbol *> &undefs,
                       const llvm::DenseMap<Symbol *, Symbol *> *localImports,
                       bool needBitcodeFiles);
  void reportUndefinedSymbol(const UndefinedDiag &undefDiag);
};

std::vector<std::string> getSymbolLocations(ObjFile *file, uint32_t symIndex);

StringRef ltrim1(StringRef s, const char *chars);

} // namespace lld::coff

#endif
