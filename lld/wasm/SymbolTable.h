//===- SymbolTable.h --------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_SYMBOL_TABLE_H
#define LLD_WASM_SYMBOL_TABLE_H

#include "InputFiles.h"
#include "Symbols.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

using llvm::wasm::WasmSignature;
using llvm::wasm::WasmGlobalType;

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
  void addFile(InputFile *File);

  std::vector<ObjFile *> ObjectFiles;

  void reportRemainingUndefines();

  ArrayRef<Symbol *> getSymbols() const { return SymVector; }
  Symbol *find(StringRef Name);
  ObjFile *findComdat(StringRef Name) const;

  Symbol *addDefinedFunction(StringRef Name, uint32_t Flags, InputFile *F,
                             InputFunction *Function = nullptr);
  Symbol *addDefinedData(StringRef Name, uint32_t Flags, InputFile *F,
                         InputSegment *Segment = nullptr, uint32_t Address = 0,
                         uint32_t Size = 0);
  Symbol *addDefinedGlobal(StringRef Name, uint32_t Flags, InputFile *F,
                           InputGlobal *G);
  Symbol *addUndefinedFunction(StringRef Name, const WasmSignature *Type);
  Symbol *addUndefined(StringRef Name, WasmSymbolType Type, uint32_t Flags,
                       InputFile *F, const WasmSignature *Signature = nullptr,
                       const WasmGlobalType *GlobalType = nullptr);
  void addLazy(ArchiveFile *F, const Archive::Symbol *Sym);
  bool addComdat(StringRef Name, ObjFile *);

  DefinedData *addSyntheticDataSymbol(StringRef Name, uint32_t Flags = 0);
  DefinedGlobal *addSyntheticGlobal(StringRef Name, uint32_t Flags,
                                    InputGlobal *Global);
  DefinedFunction *addSyntheticFunction(StringRef Name,
                                        const WasmSignature *Type,
                                        uint32_t Flags = 0);

private:
  std::pair<Symbol *, bool> insert(StringRef Name);

  llvm::DenseMap<llvm::CachedHashStringRef, ObjFile *> ComdatMap;
  llvm::DenseMap<llvm::CachedHashStringRef, Symbol *> SymMap;
  std::vector<Symbol *> SymVector;
};

extern SymbolTable *Symtab;

} // namespace wasm
} // namespace lld

#endif
