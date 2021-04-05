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
#include "llvm/ADT/SetVector.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TextAPI/TextAPIReader.h"

#include <map>
#include <vector>

namespace llvm {
namespace lto {
class InputFile;
} // namespace lto
namespace MachO {
class InterfaceFile;
} // namespace MachO
class TarWriter;
} // namespace llvm

namespace lld {
namespace macho {

class InputSection;
class Symbol;
struct Reloc;
enum class RefState : uint8_t;

// If --reproduce option is given, all input files are written
// to this tar archive.
extern std::unique_ptr<llvm::TarWriter> tar;

// If .subsections_via_symbols is set, each InputSection will be split along
// symbol boundaries. The field offset represents the offset of the subsection
// from the start of the original pre-split InputSection.
struct SubsectionEntry {
  uint64_t offset;
  InputSection *isec;
};
using SubsectionMapping = std::vector<SubsectionEntry>;

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
  std::vector<SubsectionMapping> subsections;
  // Provides an easy way to sort InputFiles deterministically.
  const int id;

  // If not empty, this stores the name of the archive containing this file.
  // We use this string for creating error messages.
  std::string archiveName;

protected:
  InputFile(Kind kind, MemoryBufferRef mb)
      : mb(mb), id(idCount++), fileKind(kind), name(mb.getBufferIdentifier()) {}

  InputFile(Kind, const llvm::MachO::InterfaceFile &);

private:
  const Kind fileKind;
  const StringRef name;

  static int idCount;
};

// .o file
class ObjFile : public InputFile {
public:
  ObjFile(MemoryBufferRef mb, uint32_t modTime, StringRef archiveName);
  static bool classof(const InputFile *f) { return f->kind() == ObjKind; }

  llvm::DWARFUnit *compileUnit = nullptr;
  const uint32_t modTime;
  std::vector<InputSection *> debugSections;

private:
  template <class LP> void parse();
  template <class Section> void parseSections(ArrayRef<Section>);
  template <class LP>
  void parseSymbols(ArrayRef<typename LP::section> sectionHeaders,
                    ArrayRef<typename LP::nlist> nList, const char *strtab,
                    bool subsectionsViaSymbols);
  template <class NList>
  Symbol *parseNonSectionSymbol(const NList &sym, StringRef name);
  template <class Section>
  void parseRelocations(ArrayRef<Section> sectionHeaders, const Section &,
                        SubsectionMapping &);
  void parseDebugInfo();
};

// command-line -sectcreate file
class OpaqueFile : public InputFile {
public:
  OpaqueFile(MemoryBufferRef mb, StringRef segName, StringRef sectName);
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
  explicit DylibFile(MemoryBufferRef mb, DylibFile *umbrella,
                     bool isBundleLoader = false);

  explicit DylibFile(const llvm::MachO::InterfaceFile &interface,
                     DylibFile *umbrella = nullptr,
                     bool isBundleLoader = false);

  static bool classof(const InputFile *f) { return f->kind() == DylibKind; }

  StringRef dylibName;
  uint32_t compatibilityVersion = 0;
  uint32_t currentVersion = 0;
  int64_t ordinal = 0; // Ordinal numbering starts from 1, so 0 is a sentinel
  RefState refState;
  bool reexport = false;
  bool forceWeakImport = false;

  // An executable can be used as a bundle loader that will load the output
  // file being linked, and that contains symbols referenced, but not
  // implemented in the bundle. When used like this, it is very similar
  // to a Dylib, so we re-used the same class to represent it.
  bool isBundleLoader;

private:
  template <class LP> void parse(DylibFile *umbrella = nullptr);
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

extern llvm::SetVector<InputFile *> inputFiles;

llvm::Optional<MemoryBufferRef> readFile(StringRef path);

template <class CommandType = llvm::MachO::load_command, class Header>
const CommandType *findCommand(const Header *hdr, uint32_t type) {
  const uint8_t *p = reinterpret_cast<const uint8_t *>(hdr) + sizeof(Header);

  for (uint32_t i = 0, n = hdr->ncmds; i < n; ++i) {
    auto *cmd = reinterpret_cast<const CommandType *>(p);
    if (cmd->cmd == type)
      return cmd;
    p += cmd->cmdsize;
  }
  return nullptr;
}

} // namespace macho

std::string toString(const macho::InputFile *file);
} // namespace lld

#endif
