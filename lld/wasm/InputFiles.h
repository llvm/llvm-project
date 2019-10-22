//===- InputFiles.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_INPUT_FILES_H
#define LLD_WASM_INPUT_FILES_H

#include "Symbols.h"
#include "lld/Common/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/MemoryBuffer.h"
#include <vector>

namespace llvm {
class TarWriter;
}

namespace lld {
namespace wasm {

class InputChunk;
class InputFunction;
class InputSegment;
class InputGlobal;
class InputEvent;
class InputSection;

// If --reproduce option is given, all input files are written
// to this tar archive.
extern std::unique_ptr<llvm::TarWriter> Tar;

class InputFile {
public:
  enum Kind {
    ObjectKind,
    SharedKind,
    ArchiveKind,
    BitcodeKind,
  };

  virtual ~InputFile() {}

  // Returns the filename.
  StringRef getName() const { return MB.getBufferIdentifier(); }

  Kind kind() const { return FileKind; }

  // An archive file name if this file is created from an archive.
  StringRef ArchiveName;

  ArrayRef<Symbol *> getSymbols() const { return Symbols; }

  MutableArrayRef<Symbol *> getMutableSymbols() { return Symbols; }

protected:
  InputFile(Kind K, MemoryBufferRef M) : MB(M), FileKind(K) {}
  MemoryBufferRef MB;

  // List of all symbols referenced or defined by this file.
  std::vector<Symbol *> Symbols;

private:
  const Kind FileKind;
};

// .a file (ar archive)
class ArchiveFile : public InputFile {
public:
  explicit ArchiveFile(MemoryBufferRef M) : InputFile(ArchiveKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == ArchiveKind; }

  void addMember(const llvm::object::Archive::Symbol *Sym);

  void parse();

private:
  std::unique_ptr<llvm::object::Archive> File;
  llvm::DenseSet<uint64_t> Seen;
};

// .o file (wasm object file)
class ObjFile : public InputFile {
public:
  explicit ObjFile(MemoryBufferRef M, StringRef ArchiveName)
      : InputFile(ObjectKind, M) {
    this->ArchiveName = ArchiveName;
  }
  static bool classof(const InputFile *F) { return F->kind() == ObjectKind; }

  void parse(bool IgnoreComdats = false);

  // Returns the underlying wasm file.
  const WasmObjectFile *getWasmObj() const { return WasmObj.get(); }

  void dumpInfo() const;

  uint32_t calcNewIndex(const WasmRelocation &Reloc) const;
  uint32_t calcNewValue(const WasmRelocation &Reloc) const;
  uint32_t calcNewAddend(const WasmRelocation &Reloc) const;
  uint32_t calcExpectedValue(const WasmRelocation &Reloc) const;
  Symbol *getSymbol(const WasmRelocation &Reloc) const {
    return Symbols[Reloc.Index];
  };

  const WasmSection *CodeSection = nullptr;
  const WasmSection *DataSection = nullptr;

  // Maps input type indices to output type indices
  std::vector<uint32_t> TypeMap;
  std::vector<bool> TypeIsUsed;
  // Maps function indices to table indices
  std::vector<uint32_t> TableEntries;
  std::vector<bool> KeptComdats;
  std::vector<InputSegment *> Segments;
  std::vector<InputFunction *> Functions;
  std::vector<InputGlobal *> Globals;
  std::vector<InputEvent *> Events;
  std::vector<InputSection *> CustomSections;
  llvm::DenseMap<uint32_t, InputSection *> CustomSectionsByIndex;

  Symbol *getSymbol(uint32_t Index) const { return Symbols[Index]; }
  FunctionSymbol *getFunctionSymbol(uint32_t Index) const;
  DataSymbol *getDataSymbol(uint32_t Index) const;
  GlobalSymbol *getGlobalSymbol(uint32_t Index) const;
  SectionSymbol *getSectionSymbol(uint32_t Index) const;
  EventSymbol *getEventSymbol(uint32_t Index) const;

private:
  Symbol *createDefined(const WasmSymbol &Sym);
  Symbol *createUndefined(const WasmSymbol &Sym, bool IsCalledDirectly);

  bool isExcludedByComdat(InputChunk *Chunk) const;

  std::unique_ptr<WasmObjectFile> WasmObj;
};

// .so file.
class SharedFile : public InputFile {
public:
  explicit SharedFile(MemoryBufferRef M) : InputFile(SharedKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == SharedKind; }
};

// .bc file
class BitcodeFile : public InputFile {
public:
  explicit BitcodeFile(MemoryBufferRef M, StringRef ArchiveName)
      : InputFile(BitcodeKind, M) {
    this->ArchiveName = ArchiveName;
  }
  static bool classof(const InputFile *F) { return F->kind() == BitcodeKind; }

  void parse();
  std::unique_ptr<llvm::lto::InputFile> Obj;
};

// Will report a fatal() error if the input buffer is not a valid bitcode
// or wasm object file.
InputFile *createObjectFile(MemoryBufferRef MB, StringRef ArchiveName = "");

// Opens a given file.
llvm::Optional<MemoryBufferRef> readFile(StringRef Path);

} // namespace wasm

std::string toString(const wasm::InputFile *File);

} // namespace lld

#endif
