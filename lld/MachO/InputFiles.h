//===- InputFiles.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_INPUT_FILES_H
#define LLD_MACHO_INPUT_FILES_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/MemoryBuffer.h"
#include <vector>

namespace lld {
namespace macho {

class InputSection;
class Symbol;
struct Reloc;

class InputFile {
public:
  enum Kind {
    ObjKind,
    DylibKind,
  };

  virtual ~InputFile() = default;
  Kind kind() const { return fileKind; }
  StringRef getName() const { return mb.getBufferIdentifier(); }

  MemoryBufferRef mb;
  std::vector<Symbol *> symbols;
  std::vector<InputSection *> sections;

protected:
  InputFile(Kind kind, MemoryBufferRef mb) : mb(mb), fileKind(kind) {}

  std::vector<InputSection *> parseSections(ArrayRef<llvm::MachO::section_64>);

  void parseRelocations(const llvm::MachO::section_64 &,
                        std::vector<Reloc> &relocs);

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
  explicit DylibFile(MemoryBufferRef mb);
  static bool classof(const InputFile *f) { return f->kind() == DylibKind; }

  StringRef dylibName;
  uint64_t ordinal = 0; // Ordinal numbering starts from 1, so 0 is a sentinel
};

extern std::vector<InputFile *> inputFiles;

llvm::Optional<MemoryBufferRef> readFile(StringRef path);

} // namespace macho

std::string toString(const macho::InputFile *file);
} // namespace lld

#endif
