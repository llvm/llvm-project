//===- SymbolTable.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_SYMBOL_TABLE_H
#define LLD_WASM_SYMBOL_TABLE_H

#include "InputFiles.h"
#include "LTO.h"
#include "Symbols.h"
#include "lld/Common/LLVM.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseSet.h"

namespace lld {
namespace wasm {

class InputSegment;

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
  void wrap(Symbol *Sym, Symbol *Real, Symbol *Wrap);

  void addFile(InputFile *File);

  void addCombinedLTOObject();

  void reportRemainingUndefines();

  ArrayRef<Symbol *> getSymbols() const { return SymVector; }

  Symbol *find(StringRef Name);

  void replace(StringRef Name, Symbol* Sym);

  void trace(StringRef Name);

  Symbol *addDefinedFunction(StringRef Name, uint32_t Flags, InputFile *File,
                             InputFunction *Function);
  Symbol *addDefinedData(StringRef Name, uint32_t Flags, InputFile *File,
                         InputSegment *Segment, uint32_t Address,
                         uint32_t Size);
  Symbol *addDefinedGlobal(StringRef Name, uint32_t Flags, InputFile *File,
                           InputGlobal *G);
  Symbol *addDefinedEvent(StringRef Name, uint32_t Flags, InputFile *File,
                          InputEvent *E);

  Symbol *addUndefinedFunction(StringRef Name, StringRef ImportName,
                               StringRef ImportModule, uint32_t Flags,
                               InputFile *File, const WasmSignature *Signature,
                               bool IsCalledDirectly);
  Symbol *addUndefinedData(StringRef Name, uint32_t Flags, InputFile *File);
  Symbol *addUndefinedGlobal(StringRef Name, StringRef ImportName,
                             StringRef ImportModule,  uint32_t Flags,
                             InputFile *File, const WasmGlobalType *Type);

  void addLazy(ArchiveFile *F, const llvm::object::Archive::Symbol *Sym);

  bool addComdat(StringRef Name);

  DefinedData *addSyntheticDataSymbol(StringRef Name, uint32_t Flags);
  DefinedGlobal *addSyntheticGlobal(StringRef Name, uint32_t Flags,
                                    InputGlobal *Global);
  DefinedFunction *addSyntheticFunction(StringRef Name, uint32_t Flags,
                                        InputFunction *Function);
  DefinedData *addOptionalDataSymbol(StringRef Name, uint32_t Value = 0,
                                     uint32_t Flags = 0);

  void handleSymbolVariants();
  void handleWeakUndefines();

  std::vector<ObjFile *> ObjectFiles;
  std::vector<SharedFile *> SharedFiles;
  std::vector<BitcodeFile *> BitcodeFiles;
  std::vector<InputFunction *> SyntheticFunctions;
  std::vector<InputGlobal *> SyntheticGlobals;

private:
  std::pair<Symbol *, bool> insert(StringRef Name, const InputFile *File);
  std::pair<Symbol *, bool> insertName(StringRef Name);

  bool getFunctionVariant(Symbol* Sym, const WasmSignature *Sig,
                          const InputFile *File, Symbol **Out);
  InputFunction *replaceWithUnreachable(Symbol *Sym, const WasmSignature &Sig,
                                        StringRef DebugName);

  // Maps symbol names to index into the SymVector.  -1 means that symbols
  // is to not yet in the vector but it should have tracing enabled if it is
  // ever added.
  llvm::DenseMap<llvm::CachedHashStringRef, int> SymMap;
  std::vector<Symbol *> SymVector;

  // For certain symbols types, e.g. function symbols, we allow for muliple
  // variants of the same symbol with different signatures.
  llvm::DenseMap<llvm::CachedHashStringRef, std::vector<Symbol *>> SymVariants;

  // Comdat groups define "link once" sections. If two comdat groups have the
  // same name, only one of them is linked, and the other is ignored. This set
  // is used to uniquify them.
  llvm::DenseSet<llvm::CachedHashStringRef> ComdatGroups;

  // For LTO.
  std::unique_ptr<BitcodeCompiler> LTO;
};

extern SymbolTable *Symtab;

} // namespace wasm
} // namespace lld

#endif
