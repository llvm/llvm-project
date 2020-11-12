//===- InputFiles.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_INPUT_FILES_H
#define LLD_COFF_INPUT_FILES_H

#include "Config.h"
#include "lld/Common/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/StringSaver.h"
#include <memory>
#include <set>
#include <vector>

namespace llvm {
struct DILineInfo;
namespace pdb {
class DbiModuleDescriptorBuilder;
class NativeSession;
}
namespace lto {
class InputFile;
}
}

namespace lld {
class DWARFCache;

namespace coff {

std::vector<MemoryBufferRef> getArchiveMembers(llvm::object::Archive *file);

using llvm::COFF::IMAGE_FILE_MACHINE_UNKNOWN;
using llvm::COFF::MachineTypes;
using llvm::object::Archive;
using llvm::object::COFFObjectFile;
using llvm::object::COFFSymbolRef;
using llvm::object::coff_import_header;
using llvm::object::coff_section;

class Chunk;
class Defined;
class DefinedImportData;
class DefinedImportThunk;
class DefinedRegular;
class SectionChunk;
class Symbol;
class Undefined;
class TpiSource;

// The root class of input files.
class InputFile {
public:
  enum Kind {
    ArchiveKind,
    ObjectKind,
    LazyObjectKind,
    PDBKind,
    ImportKind,
    BitcodeKind
  };
  Kind kind() const { return fileKind; }
  virtual ~InputFile() {}

  // Returns the filename.
  StringRef getName() const { return mb.getBufferIdentifier(); }

  // Reads a file (the constructor doesn't do that).
  virtual void parse() = 0;

  // Returns the CPU type this file was compiled to.
  virtual MachineTypes getMachineType() { return IMAGE_FILE_MACHINE_UNKNOWN; }

  MemoryBufferRef mb;

  // An archive file name if this file is created from an archive.
  StringRef parentName;

  // Returns .drectve section contents if exist.
  StringRef getDirectives() { return directives; }

protected:
  InputFile(Kind k, MemoryBufferRef m) : mb(m), fileKind(k) {}

  StringRef directives;

private:
  const Kind fileKind;
};

// .lib or .a file.
class ArchiveFile : public InputFile {
public:
  explicit ArchiveFile(MemoryBufferRef m);
  static bool classof(const InputFile *f) { return f->kind() == ArchiveKind; }
  void parse() override;

  // Enqueues an archive member load for the given symbol. If we've already
  // enqueued a load for the same archive member, this function does nothing,
  // which ensures that we don't load the same member more than once.
  void addMember(const Archive::Symbol &sym);

private:
  std::unique_ptr<Archive> file;
  llvm::DenseSet<uint64_t> seen;
};

// .obj or .o file between -start-lib and -end-lib.
class LazyObjFile : public InputFile {
public:
  explicit LazyObjFile(MemoryBufferRef m) : InputFile(LazyObjectKind, m) {}
  static bool classof(const InputFile *f) {
    return f->kind() == LazyObjectKind;
  }
  // Makes this object file part of the link.
  void fetch();
  // Adds the symbols in this file to the symbol table as LazyObject symbols.
  void parse() override;

private:
  std::vector<Symbol *> symbols;
};

// .obj or .o file. This may be a member of an archive file.
class ObjFile : public InputFile {
public:
  explicit ObjFile(MemoryBufferRef m) : InputFile(ObjectKind, m) {}
  explicit ObjFile(MemoryBufferRef m, std::vector<Symbol *> &&symbols)
      : InputFile(ObjectKind, m), symbols(std::move(symbols)) {}
  static bool classof(const InputFile *f) { return f->kind() == ObjectKind; }
  void parse() override;
  MachineTypes getMachineType() override;
  ArrayRef<Chunk *> getChunks() { return chunks; }
  ArrayRef<SectionChunk *> getDebugChunks() { return debugChunks; }
  ArrayRef<SectionChunk *> getSXDataChunks() { return sxDataChunks; }
  ArrayRef<SectionChunk *> getGuardFidChunks() { return guardFidChunks; }
  ArrayRef<SectionChunk *> getGuardLJmpChunks() { return guardLJmpChunks; }
  ArrayRef<Symbol *> getSymbols() { return symbols; }

  MutableArrayRef<Symbol *> getMutableSymbols() { return symbols; }

  ArrayRef<uint8_t> getDebugSection(StringRef secName);

  // Returns a Symbol object for the symbolIndex'th symbol in the
  // underlying object file.
  Symbol *getSymbol(uint32_t symbolIndex) {
    return symbols[symbolIndex];
  }

  // Returns the underlying COFF file.
  COFFObjectFile *getCOFFObj() { return coffObj.get(); }

  // Add a symbol for a range extension thunk. Return the new symbol table
  // index. This index can be used to modify a relocation.
  uint32_t addRangeThunkSymbol(Symbol *thunk) {
    symbols.push_back(thunk);
    return symbols.size() - 1;
  }

  void includeResourceChunks();

  bool isResourceObjFile() const { return !resourceChunks.empty(); }

  static std::vector<ObjFile *> instances;

  // Flags in the absolute @feat.00 symbol if it is present. These usually
  // indicate if an object was compiled with certain security features enabled
  // like stack guard, safeseh, /guard:cf, or other things.
  uint32_t feat00Flags = 0;

  // True if this object file is compatible with SEH.  COFF-specific and
  // x86-only. COFF spec 5.10.1. The .sxdata section.
  bool hasSafeSEH() { return feat00Flags & 0x1; }

  // True if this file was compiled with /guard:cf.
  bool hasGuardCF() { return feat00Flags & 0x800; }

  // Pointer to the PDB module descriptor builder. Various debug info records
  // will reference object files by "module index", which is here. Things like
  // source files and section contributions are also recorded here. Will be null
  // if we are not producing a PDB.
  llvm::pdb::DbiModuleDescriptorBuilder *moduleDBI = nullptr;

  const coff_section *addrsigSec = nullptr;

  const coff_section *callgraphSec = nullptr;

  // When using Microsoft precompiled headers, this is the PCH's key.
  // The same key is used by both the precompiled object, and objects using the
  // precompiled object. Any difference indicates out-of-date objects.
  llvm::Optional<uint32_t> pchSignature;

  // Whether this file was compiled with /hotpatch.
  bool hotPatchable = false;

  // Whether the object was already merged into the final PDB.
  bool mergedIntoPDB = false;

  // If the OBJ has a .debug$T stream, this tells how it will be handled.
  TpiSource *debugTypesObj = nullptr;

  // The .debug$P or .debug$T section data if present. Empty otherwise.
  ArrayRef<uint8_t> debugTypes;

  llvm::Optional<std::pair<StringRef, uint32_t>>
  getVariableLocation(StringRef var);

  llvm::Optional<llvm::DILineInfo> getDILineInfo(uint32_t offset,
                                                 uint32_t sectionIndex);

private:
  const coff_section* getSection(uint32_t i);
  const coff_section *getSection(COFFSymbolRef sym) {
    return getSection(sym.getSectionNumber());
  }

  void initializeChunks();
  void initializeSymbols();
  void initializeFlags();
  void initializeDependencies();

  SectionChunk *
  readSection(uint32_t sectionNumber,
              const llvm::object::coff_aux_section_definition *def,
              StringRef leaderName);

  void readAssociativeDefinition(
      COFFSymbolRef coffSym,
      const llvm::object::coff_aux_section_definition *def);

  void readAssociativeDefinition(
      COFFSymbolRef coffSym,
      const llvm::object::coff_aux_section_definition *def,
      uint32_t parentSection);

  void recordPrevailingSymbolForMingw(
      COFFSymbolRef coffSym,
      llvm::DenseMap<StringRef, uint32_t> &prevailingSectionMap);

  void maybeAssociateSEHForMingw(
      COFFSymbolRef sym, const llvm::object::coff_aux_section_definition *def,
      const llvm::DenseMap<StringRef, uint32_t> &prevailingSectionMap);

  // Given a new symbol Sym with comdat selection Selection, if the new
  // symbol is not (yet) Prevailing and the existing comdat leader set to
  // Leader, emits a diagnostic if the new symbol and its selection doesn't
  // match the existing symbol and its selection. If either old or new
  // symbol have selection IMAGE_COMDAT_SELECT_LARGEST, Sym might replace
  // the existing leader. In that case, Prevailing is set to true.
  void
  handleComdatSelection(COFFSymbolRef sym, llvm::COFF::COMDATType &selection,
                        bool &prevailing, DefinedRegular *leader,
                        const llvm::object::coff_aux_section_definition *def);

  llvm::Optional<Symbol *>
  createDefined(COFFSymbolRef sym,
                std::vector<const llvm::object::coff_aux_section_definition *>
                    &comdatDefs,
                bool &prevailingComdat);
  Symbol *createRegular(COFFSymbolRef sym);
  Symbol *createUndefined(COFFSymbolRef sym);

  std::unique_ptr<COFFObjectFile> coffObj;

  // List of all chunks defined by this file. This includes both section
  // chunks and non-section chunks for common symbols.
  std::vector<Chunk *> chunks;

  std::vector<SectionChunk *> resourceChunks;

  // CodeView debug info sections.
  std::vector<SectionChunk *> debugChunks;

  // Chunks containing symbol table indices of exception handlers. Only used for
  // 32-bit x86.
  std::vector<SectionChunk *> sxDataChunks;

  // Chunks containing symbol table indices of address taken symbols and longjmp
  // targets.  These are not linked into the final binary when /guard:cf is set.
  std::vector<SectionChunk *> guardFidChunks;
  std::vector<SectionChunk *> guardLJmpChunks;

  // This vector contains a list of all symbols defined or referenced by this
  // file. They are indexed such that you can get a Symbol by symbol
  // index. Nonexistent indices (which are occupied by auxiliary
  // symbols in the real symbol table) are filled with null pointers.
  std::vector<Symbol *> symbols;

  // This vector contains the same chunks as Chunks, but they are
  // indexed such that you can get a SectionChunk by section index.
  // Nonexistent section indices are filled with null pointers.
  // (Because section number is 1-based, the first slot is always a
  // null pointer.) This vector is only valid during initialization.
  std::vector<SectionChunk *> sparseChunks;

  DWARFCache *dwarf = nullptr;
};

// This is a PDB type server dependency, that is not a input file per se, but
// needs to be treated like one. Such files are discovered from the debug type
// stream.
class PDBInputFile : public InputFile {
public:
  explicit PDBInputFile(MemoryBufferRef m);
  ~PDBInputFile();
  static bool classof(const InputFile *f) { return f->kind() == PDBKind; }
  void parse() override;

  static void enqueue(StringRef path, ObjFile *fromFile);

  static PDBInputFile *findFromRecordPath(StringRef path, ObjFile *fromFile);

  static std::map<std::string, PDBInputFile *> instances;

  // Record possible errors while opening the PDB file
  llvm::Optional<Error> loadErr;

  // This is the actual interface to the PDB (if it was opened successfully)
  std::unique_ptr<llvm::pdb::NativeSession> session;

  // If the PDB has a .debug$T stream, this tells how it will be handled.
  TpiSource *debugTypesObj = nullptr;
};

// This type represents import library members that contain DLL names
// and symbols exported from the DLLs. See Microsoft PE/COFF spec. 7
// for details about the format.
class ImportFile : public InputFile {
public:
  explicit ImportFile(MemoryBufferRef m) : InputFile(ImportKind, m) {}

  static bool classof(const InputFile *f) { return f->kind() == ImportKind; }

  static std::vector<ImportFile *> instances;

  Symbol *impSym = nullptr;
  Symbol *thunkSym = nullptr;
  std::string dllName;

private:
  void parse() override;

public:
  StringRef externalName;
  const coff_import_header *hdr;
  Chunk *location = nullptr;

  // We want to eliminate dllimported symbols if no one actually refers them.
  // These "Live" bits are used to keep track of which import library members
  // are actually in use.
  //
  // If the Live bit is turned off by MarkLive, Writer will ignore dllimported
  // symbols provided by this import library member. We also track whether the
  // imported symbol is used separately from whether the thunk is used in order
  // to avoid creating unnecessary thunks.
  bool live = !config->doGC;
  bool thunkLive = !config->doGC;
};

// Used for LTO.
class BitcodeFile : public InputFile {
public:
  BitcodeFile(MemoryBufferRef mb, StringRef archiveName,
              uint64_t offsetInArchive);
  explicit BitcodeFile(MemoryBufferRef m, StringRef archiveName,
                       uint64_t offsetInArchive,
                       std::vector<Symbol *> &&symbols);
  ~BitcodeFile();
  static bool classof(const InputFile *f) { return f->kind() == BitcodeKind; }
  ArrayRef<Symbol *> getSymbols() { return symbols; }
  MachineTypes getMachineType() override;
  static std::vector<BitcodeFile *> instances;
  std::unique_ptr<llvm::lto::InputFile> obj;

private:
  void parse() override;

  std::vector<Symbol *> symbols;
};

inline bool isBitcode(MemoryBufferRef mb) {
  return identify_magic(mb.getBuffer()) == llvm::file_magic::bitcode;
}

std::string replaceThinLTOSuffix(StringRef path);
} // namespace coff

std::string toString(const coff::InputFile *file);
} // namespace lld

#endif
