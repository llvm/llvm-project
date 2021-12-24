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
#include "Target.h"

#include "lld/Common/LLVM.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TextAPI/TextAPIReader.h"

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

struct PlatformInfo;
class ConcatInputSection;
class Symbol;
class Defined;
struct Reloc;
enum class RefState : uint8_t;

// If --reproduce option is given, all input files are written
// to this tar archive.
extern std::unique_ptr<llvm::TarWriter> tar;

// If .subsections_via_symbols is set, each InputSection will be split along
// symbol boundaries. The field offset represents the offset of the subsection
// from the start of the original pre-split InputSection.
struct Subsection {
  uint64_t offset = 0;
  InputSection *isec = nullptr;
};

using Subsections = std::vector<Subsection>;

struct Section {
  uint64_t address = 0;
  Subsections subsections;
  Section(uint64_t addr) : address(addr){};
};

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
  static void resetIdCount() { idCount = 0; }

  MemoryBufferRef mb;

  std::vector<Symbol *> symbols;
  std::vector<Section> sections;
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
class ObjFile final : public InputFile {
public:
  ObjFile(MemoryBufferRef mb, uint32_t modTime, StringRef archiveName);
  ArrayRef<llvm::MachO::data_in_code_entry> getDataInCode() const;

  static bool classof(const InputFile *f) { return f->kind() == ObjKind; }

  llvm::DWARFUnit *compileUnit = nullptr;
  const uint32_t modTime;
  std::vector<ConcatInputSection *> debugSections;

private:
  Section *compactUnwindSection = nullptr;

  template <class LP> void parse();
  template <class SectionHeader> void parseSections(ArrayRef<SectionHeader>);
  template <class LP>
  void parseSymbols(ArrayRef<typename LP::section> sectionHeaders,
                    ArrayRef<typename LP::nlist> nList, const char *strtab,
                    bool subsectionsViaSymbols);
  template <class NList>
  Symbol *parseNonSectionSymbol(const NList &sym, StringRef name);
  template <class SectionHeader>
  void parseRelocations(ArrayRef<SectionHeader> sectionHeaders,
                        const SectionHeader &, Subsections &);
  void parseDebugInfo();
  void registerCompactUnwind();
};

// command-line -sectcreate file
class OpaqueFile final : public InputFile {
public:
  OpaqueFile(MemoryBufferRef mb, StringRef segName, StringRef sectName);
  static bool classof(const InputFile *f) { return f->kind() == OpaqueKind; }
};

// .dylib or .tbd file
class DylibFile final : public InputFile {
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

  void parseLoadCommands(MemoryBufferRef mb);
  void parseReexports(const llvm::MachO::InterfaceFile &interface);

  static bool classof(const InputFile *f) { return f->kind() == DylibKind; }

  StringRef installName;
  DylibFile *exportingFile = nullptr;
  DylibFile *umbrella;
  SmallVector<StringRef, 2> rpaths;
  uint32_t compatibilityVersion = 0;
  uint32_t currentVersion = 0;
  int64_t ordinal = 0; // Ordinal numbering starts from 1, so 0 is a sentinel
  RefState refState;
  bool reexport = false;
  bool forceNeeded = false;
  bool forceWeakImport = false;
  bool deadStrippable = false;
  bool explicitlyLinked = false;

  unsigned numReferencedSymbols = 0;

  bool isReferenced() const { return numReferencedSymbols > 0; }

  // An executable can be used as a bundle loader that will load the output
  // file being linked, and that contains symbols referenced, but not
  // implemented in the bundle. When used like this, it is very similar
  // to a Dylib, so we re-used the same class to represent it.
  bool isBundleLoader;

private:
  bool handleLDSymbol(StringRef originalName);
  void handleLDPreviousSymbol(StringRef name, StringRef originalName);
  void handleLDInstallNameSymbol(StringRef name, StringRef originalName);
  void handleLDHideSymbol(StringRef name, StringRef originalName);
  void checkAppExtensionSafety(bool dylibIsAppExtensionSafe) const;

  llvm::DenseSet<llvm::CachedHashStringRef> hiddenSymbols;
};

// .a file
class ArchiveFile final : public InputFile {
public:
  explicit ArchiveFile(std::unique_ptr<llvm::object::Archive> &&file);
  void addLazySymbols();
  void fetch(const llvm::object::Archive::Symbol &);
  // LLD normally doesn't use Error for error-handling, but the underlying
  // Archive library does, so this is the cleanest way to wrap it.
  Error fetch(const llvm::object::Archive::Child &, StringRef reason);
  const llvm::object::Archive &getArchive() const { return *file; };
  static bool classof(const InputFile *f) { return f->kind() == ArchiveKind; }

private:
  std::unique_ptr<llvm::object::Archive> file;
  // Keep track of children fetched from the archive by tracking
  // which address offsets have been fetched already.
  llvm::DenseSet<uint64_t> seen;
};

class BitcodeFile final : public InputFile {
public:
  explicit BitcodeFile(MemoryBufferRef mb, StringRef archiveName,
                       uint64_t offsetInArchive);
  static bool classof(const InputFile *f) { return f->kind() == BitcodeKind; }

  std::unique_ptr<llvm::lto::InputFile> obj;
};

extern llvm::SetVector<InputFile *> inputFiles;
extern llvm::DenseMap<llvm::CachedHashStringRef, MemoryBufferRef> cachedReads;

llvm::Optional<MemoryBufferRef> readFile(StringRef path);

namespace detail {

template <class CommandType, class... Types>
std::vector<const CommandType *>
findCommands(const void *anyHdr, size_t maxCommands, Types... types) {
  std::vector<const CommandType *> cmds;
  std::initializer_list<uint32_t> typesList{types...};
  const auto *hdr = reinterpret_cast<const llvm::MachO::mach_header *>(anyHdr);
  const uint8_t *p =
      reinterpret_cast<const uint8_t *>(hdr) + target->headerSize;
  for (uint32_t i = 0, n = hdr->ncmds; i < n; ++i) {
    auto *cmd = reinterpret_cast<const CommandType *>(p);
    if (llvm::is_contained(typesList, cmd->cmd)) {
      cmds.push_back(cmd);
      if (cmds.size() == maxCommands)
        return cmds;
    }
    p += cmd->cmdsize;
  }
  return cmds;
}

} // namespace detail

// anyHdr should be a pointer to either mach_header or mach_header_64
template <class CommandType = llvm::MachO::load_command, class... Types>
const CommandType *findCommand(const void *anyHdr, Types... types) {
  std::vector<const CommandType *> cmds =
      detail::findCommands<CommandType>(anyHdr, 1, types...);
  return cmds.size() ? cmds[0] : nullptr;
}

template <class CommandType = llvm::MachO::load_command, class... Types>
std::vector<const CommandType *> findCommands(const void *anyHdr,
                                              Types... types) {
  return detail::findCommands<CommandType>(anyHdr, 0, types...);
}

} // namespace macho

std::string toString(const macho::InputFile *file);
} // namespace lld

#endif
