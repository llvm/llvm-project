//===- InputFiles.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_INPUT_FILES_H
#define LLD_MACHO_INPUT_FILES_H

#include "MachOStructs.h"

#include "lld/Common/LLVM.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TextAPI/MachO/InterfaceFile.h"
#include "llvm/TextAPI/MachO/TextAPIReader.h"

#include <map>
#include <vector>

namespace llvm {
namespace lto {
class InputFile;
} // namespace lto
} // namespace llvm

namespace lld {
namespace macho {

class InputSection;
class Symbol;
struct Reloc;

// If .subsections_via_symbols is set, each InputSection will be split along
// symbol boundaries. The keys of a SubsectionMap represent the offsets of
// each subsection from the start of the original pre-split InputSection.
using SubsectionMap = std::map<uint32_t, InputSection *>;

class InputFile {
public:
  enum Kind {
    ObjKind,
    OpaqueKind,
    DylibKind,
    ArchiveKind,
    BitcodeKind,
  };

  virtual ~InputFile() = default;
  Kind kind() const { return fileKind; }
  StringRef getName() const { return name; }

  MemoryBufferRef mb;
  std::vector<Symbol *> symbols;
  ArrayRef<llvm::MachO::section_64> sectionHeaders;
  std::vector<SubsectionMap> subsections;

protected:
  InputFile(Kind kind, MemoryBufferRef mb)
      : mb(mb), fileKind(kind), name(mb.getBufferIdentifier()) {}

  InputFile(Kind kind, const llvm::MachO::InterfaceFile &interface)
      : fileKind(kind), name(saver.save(interface.getPath())) {}

  void parseSections(ArrayRef<llvm::MachO::section_64>);

  void parseSymbols(ArrayRef<lld::structs::nlist_64> nList, const char *strtab,
                    bool subsectionsViaSymbols);

  Symbol *parseNonSectionSymbol(const structs::nlist_64 &sym, StringRef name);

  void parseRelocations(const llvm::MachO::section_64 &, SubsectionMap &);

private:
  const Kind fileKind;
  const StringRef name;
};

// .o file
class ObjFile : public InputFile {
public:
  explicit ObjFile(MemoryBufferRef mb);
  static bool classof(const InputFile *f) { return f->kind() == ObjKind; }
};

// command-line -sectcreate file
class OpaqueFile : public InputFile {
public:
  explicit OpaqueFile(MemoryBufferRef mb, StringRef segName,
                      StringRef sectName);
  static bool classof(const InputFile *f) { return f->kind() == OpaqueKind; }
};

// .dylib file
class DylibFile : public InputFile {
public:
  // Mach-O dylibs can re-export other dylibs as sub-libraries, meaning that the
  // symbols in those sub-libraries will be available under the umbrella
  // library's namespace. Those sub-libraries can also have their own
  // re-exports. When loading a re-exported dylib, `umbrella` should be set to
  // the root dylib to ensure symbols in the child library are correctly bound
  // to the root. On the other hand, if a dylib is being directly loaded
  // (through an -lfoo flag), then `umbrella` should be a nullptr.
  explicit DylibFile(MemoryBufferRef mb, DylibFile *umbrella = nullptr);

  explicit DylibFile(const llvm::MachO::InterfaceFile &interface,
                     DylibFile *umbrella = nullptr);

  static bool classof(const InputFile *f) { return f->kind() == DylibKind; }

  StringRef dylibName;
  uint64_t ordinal = 0; // Ordinal numbering starts from 1, so 0 is a sentinel
  bool reexport = false;
  bool forceWeakImport = false;
  std::vector<DylibFile *> reexported;
};

// .a file
class ArchiveFile : public InputFile {
public:
  explicit ArchiveFile(std::unique_ptr<llvm::object::Archive> &&file);
  static bool classof(const InputFile *f) { return f->kind() == ArchiveKind; }
  void fetch(const llvm::object::Archive::Symbol &sym);

private:
  std::unique_ptr<llvm::object::Archive> file;
  // Keep track of children fetched from the archive by tracking
  // which address offsets have been fetched already.
  llvm::DenseSet<uint64_t> seen;
};

class BitcodeFile : public InputFile {
public:
  explicit BitcodeFile(MemoryBufferRef mb);
  static bool classof(const InputFile *f) { return f->kind() == BitcodeKind; }

  std::unique_ptr<llvm::lto::InputFile> obj;
};

extern std::vector<InputFile *> inputFiles;

llvm::Optional<MemoryBufferRef> readFile(StringRef path);

const llvm::MachO::load_command *
findCommand(const llvm::MachO::mach_header_64 *, uint32_t type);

} // namespace macho

std::string toString(const macho::InputFile *file);
} // namespace lld

#endif
