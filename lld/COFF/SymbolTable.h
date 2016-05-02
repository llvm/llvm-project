//===- SymbolTable.h --------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_SYMBOL_TABLE_H
#define LLD_COFF_SYMBOL_TABLE_H

#include "InputFiles.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

#ifdef _MSC_VER
// <future> depends on <eh.h> for __uncaught_exception.
#include <eh.h>
#endif

#include <future>

namespace llvm {
struct LTOCodeGenerator;
}

namespace lld {
namespace coff {

class Chunk;
class Defined;
class Lazy;
class SymbolBody;
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
// to replace the lazy symbol. The logic is implemented in resolve().
class SymbolTable {
public:
  void addFile(std::unique_ptr<InputFile> File);
  std::vector<std::unique_ptr<InputFile>> &getFiles() { return Files; }
  void step();
  void run();
  bool queueEmpty();

  // Print an error message on undefined symbols. If Resolve is true, try to
  // resolve any undefined symbols and update the symbol table accordingly.
  void reportRemainingUndefines(bool Resolve);

  // Returns a list of chunks of selected symbols.
  std::vector<Chunk *> getChunks();

  // Returns a symbol for a given name. Returns a nullptr if not found.
  Symbol *find(StringRef Name);
  Symbol *findUnderscore(StringRef Name);

  // Occasionally we have to resolve an undefined symbol to its
  // mangled symbol. This function tries to find a mangled name
  // for U from the symbol table, and if found, set the symbol as
  // a weak alias for U.
  void mangleMaybe(Undefined *U);
  StringRef findMangle(StringRef Name);

  // Print a layout map to OS.
  void printMap(llvm::raw_ostream &OS);

  // Build a set of COFF objects representing the combined contents of
  // BitcodeFiles and add them to the symbol table. Called after all files are
  // added and before the writer writes results to a file.
  void addCombinedLTOObjects();

  // The writer needs to handle DLL import libraries specially in
  // order to create the import descriptor table.
  std::vector<ImportFile *> ImportFiles;

  // The writer needs to infer the machine type from the object files.
  std::vector<ObjectFile *> ObjectFiles;

  // Creates an Undefined symbol for a given name.
  Undefined *addUndefined(StringRef Name);
  DefinedRelative *addRelative(StringRef Name, uint64_t VA);
  DefinedAbsolute *addAbsolute(StringRef Name, uint64_t VA);

  // A list of chunks which to be added to .rdata.
  std::vector<Chunk *> LocalImportChunks;

private:
  void readArchives();
  void readObjects();

  void addSymbol(SymbolBody *New);
  void addLazy(Lazy *New, std::vector<Symbol *> *Accum);
  Symbol *insert(SymbolBody *New);
  StringRef findByPrefix(StringRef Prefix);

  void addMemberFile(Lazy *Body);
  void addCombinedLTOObject(ObjectFile *Obj);
  std::vector<ObjectFile *> createLTOObjects(llvm::LTOCodeGenerator *CG);

  llvm::DenseMap<StringRef, Symbol *> Symtab;

  std::vector<std::unique_ptr<InputFile>> Files;
  std::vector<std::future<ArchiveFile *>> ArchiveQueue;
  std::vector<std::future<InputFile *>> ObjectQueue;

  std::vector<BitcodeFile *> BitcodeFiles;
  std::vector<SmallString<0>> Objs;
  llvm::BumpPtrAllocator Alloc;
};

} // namespace coff
} // namespace lld

#endif
