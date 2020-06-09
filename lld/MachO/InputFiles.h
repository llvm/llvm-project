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
#include "llvm/ADT/DenseSet.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TextAPI/MachO/InterfaceFile.h"
#include "llvm/TextAPI/MachO/TextAPIReader.h"

#include <map>
#include <vector>

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
    DylibKind,
    ArchiveKind,
  };

  virtual ~InputFile() = default;
  Kind kind() const { return fileKind; }
  StringRef getName() const { return mb.getBufferIdentifier(); }

  MemoryBufferRef mb;
  std::vector<Symbol *> symbols;
  ArrayRef<llvm::MachO::section_64> sectionHeaders;
  std::vector<SubsectionMap> subsections;

protected:
  InputFile(Kind kind, MemoryBufferRef mb) : mb(mb), fileKind(kind) {}

  void parseSections(ArrayRef<llvm::MachO::section_64>);

  void parseSymbols(ArrayRef<lld::structs::nlist_64> nList, const char *strtab,
                    bool subsectionsViaSymbols);

  void parseRelocations(const llvm::MachO::section_64 &, SubsectionMap &);

private:
  const Kind fileKind;
};

// .o file
class ObjFile : public InputFile {
public:
  explicit ObjFile(MemoryBufferRef mb);
  static bool classof(const InputFile *f) { return f->kind() == ObjKind; }
};

// .dylib file
class DylibFile : public InputFile {
public:
  explicit DylibFile(std::shared_ptr<llvm::MachO::InterfaceFile> interface,
                     DylibFile *umbrella = nullptr);

  // Mach-O dylibs can re-export other dylibs as sub-libraries, meaning that the
  // symbols in those sub-libraries will be available under the umbrella
  // library's namespace. Those sub-libraries can also have their own
  // re-exports. When loading a re-exported dylib, `umbrella` should be set to
  // the root dylib to ensure symbols in the child library are correctly bound
  // to the root. On the other hand, if a dylib is being directly loaded
  // (through an -lfoo flag), then `umbrella` should be a nullptr.
  explicit DylibFile(MemoryBufferRef mb, DylibFile *umbrella = nullptr);

  static bool classof(const InputFile *f) { return f->kind() == DylibKind; }

  // Do not use this constructor!! This is meant only for createLibSystemMock(),
  // but it cannot be made private as we call it via make().
  DylibFile();
  static DylibFile *createLibSystemMock();

  StringRef dylibName;
  uint64_t ordinal = 0; // Ordinal numbering starts from 1, so 0 is a sentinel
  bool reexport = false;
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

extern std::vector<InputFile *> inputFiles;

llvm::Optional<MemoryBufferRef> readFile(StringRef path);

} // namespace macho

std::string toString(const macho::InputFile *file);
} // namespace lld

#endif
