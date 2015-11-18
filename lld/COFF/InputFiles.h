//===- InputFiles.h ---------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_INPUT_FILES_H
#define LLD_COFF_INPUT_FILES_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/StringSaver.h"
#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace lld {
namespace coff {

using llvm::LTOModule;
using llvm::COFF::IMAGE_FILE_MACHINE_UNKNOWN;
using llvm::COFF::MachineTypes;
using llvm::object::Archive;
using llvm::object::COFFObjectFile;
using llvm::object::COFFSymbolRef;
using llvm::object::coff_section;

class Chunk;
class Defined;
class DefinedImportData;
class DefinedImportThunk;
class Lazy;
class SymbolBody;
class Undefined;

// The root class of input files.
class InputFile {
public:
  enum Kind { ArchiveKind, ObjectKind, ImportKind, BitcodeKind };
  Kind kind() const { return FileKind; }
  virtual ~InputFile() {}

  // Returns the filename.
  StringRef getName() { return MB.getBufferIdentifier(); }

  // Returns symbols defined by this file.
  virtual std::vector<SymbolBody *> &getSymbols() = 0;

  // Reads a file (the constructor doesn't do that).
  virtual void parse() = 0;

  // Returns the CPU type this file was compiled to.
  virtual MachineTypes getMachineType() { return IMAGE_FILE_MACHINE_UNKNOWN; }

  // Returns a short, human-friendly filename. If this is a member of
  // an archive file, a returned value includes parent's filename.
  // Used for logging or debugging.
  std::string getShortName();

  // Sets a parent filename if this file is created from an archive.
  void setParentName(StringRef N) { ParentName = N; }

  // Returns .drectve section contents if exist.
  StringRef getDirectives() { return StringRef(Directives).trim(); }

  // Each file has a unique index. The index number is used to
  // resolve ties in symbol resolution.
  int Index;
  static int NextIndex;

protected:
  InputFile(Kind K, MemoryBufferRef M)
      : Index(NextIndex++), MB(M), FileKind(K) {}

  MemoryBufferRef MB;
  std::string Directives;

private:
  const Kind FileKind;
  StringRef ParentName;
};

// .lib or .a file.
class ArchiveFile : public InputFile {
public:
  explicit ArchiveFile(MemoryBufferRef M) : InputFile(ArchiveKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == ArchiveKind; }
  void parse() override;

  // Returns a memory buffer for a given symbol. An empty memory buffer
  // is returned if we have already returned the same memory buffer.
  // (So that we don't instantiate same members more than once.)
  MemoryBufferRef getMember(const Archive::Symbol *Sym);

  llvm::MutableArrayRef<Lazy> getLazySymbols() { return LazySymbols; }

  // All symbols returned by ArchiveFiles are of Lazy type.
  std::vector<SymbolBody *> &getSymbols() override {
    llvm_unreachable("internal error");
  }

private:
  std::unique_ptr<Archive> File;
  std::string Filename;
  std::vector<Lazy> LazySymbols;
  std::map<uint64_t, std::atomic_flag> Seen;
};

// .obj or .o file. This may be a member of an archive file.
class ObjectFile : public InputFile {
public:
  explicit ObjectFile(MemoryBufferRef M) : InputFile(ObjectKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == ObjectKind; }
  void parse() override;
  MachineTypes getMachineType() override;
  std::vector<Chunk *> &getChunks() { return Chunks; }
  std::vector<SymbolBody *> &getSymbols() override { return SymbolBodies; }

  // Returns a SymbolBody object for the SymbolIndex'th symbol in the
  // underlying object file.
  SymbolBody *getSymbolBody(uint32_t SymbolIndex) {
    return SparseSymbolBodies[SymbolIndex];
  }

  // Returns the underying COFF file.
  COFFObjectFile *getCOFFObj() { return COFFObj.get(); }

  // True if this object file is compatible with SEH.
  // COFF-specific and x86-only.
  bool SEHCompat = false;

  // The list of safe exception handlers listed in .sxdata section.
  // COFF-specific and x86-only.
  std::set<SymbolBody *> SEHandlers;

private:
  void initializeChunks();
  void initializeSymbols();
  void initializeSEH();

  Defined *createDefined(COFFSymbolRef Sym, const void *Aux, bool IsFirst);
  Undefined *createUndefined(COFFSymbolRef Sym);
  Undefined *createWeakExternal(COFFSymbolRef Sym, const void *Aux);

  std::unique_ptr<COFFObjectFile> COFFObj;
  llvm::BumpPtrAllocator Alloc;
  const coff_section *SXData = nullptr;

  // List of all chunks defined by this file. This includes both section
  // chunks and non-section chunks for common symbols.
  std::vector<Chunk *> Chunks;

  // This vector contains the same chunks as Chunks, but they are
  // indexed such that you can get a SectionChunk by section index.
  // Nonexistent section indices are filled with null pointers.
  // (Because section number is 1-based, the first slot is always a
  // null pointer.)
  std::vector<Chunk *> SparseChunks;

  // List of all symbols referenced or defined by this file.
  std::vector<SymbolBody *> SymbolBodies;

  // This vector contains the same symbols as SymbolBodies, but they
  // are indexed such that you can get a SymbolBody by symbol
  // index. Nonexistent indices (which are occupied by auxiliary
  // symbols in the real symbol table) are filled with null pointers.
  std::vector<SymbolBody *> SparseSymbolBodies;
};

// This type represents import library members that contain DLL names
// and symbols exported from the DLLs. See Microsoft PE/COFF spec. 7
// for details about the format.
class ImportFile : public InputFile {
public:
  explicit ImportFile(MemoryBufferRef M)
      : InputFile(ImportKind, M), StringAlloc(StringAllocAux) {}
  static bool classof(const InputFile *F) { return F->kind() == ImportKind; }
  std::vector<SymbolBody *> &getSymbols() override { return SymbolBodies; }

  DefinedImportData *ImpSym = nullptr;
  DefinedImportThunk *ThunkSym = nullptr;
  std::string DLLName;

private:
  void parse() override;

  std::vector<SymbolBody *> SymbolBodies;
  llvm::BumpPtrAllocator Alloc;
  llvm::BumpPtrAllocator StringAllocAux;
  llvm::StringSaver StringAlloc;
};

// Used for LTO.
class BitcodeFile : public InputFile {
public:
  explicit BitcodeFile(MemoryBufferRef M) : InputFile(BitcodeKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == BitcodeKind; }
  std::vector<SymbolBody *> &getSymbols() override { return SymbolBodies; }
  MachineTypes getMachineType() override;

  std::unique_ptr<LTOModule> takeModule() { return std::move(M); }

private:
  void parse() override;

  std::vector<SymbolBody *> SymbolBodies;
  llvm::BumpPtrAllocator Alloc;
  std::unique_ptr<LTOModule> M;
  static std::mutex Mu;
};

} // namespace coff
} // namespace lld

#endif
