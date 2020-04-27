//===- InputSection.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_INPUT_SECTION_H
#define LLD_MACHO_INPUT_SECTION_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/BinaryFormat/MachO.h"

namespace lld {
namespace macho {

class InputFile;
class InputSection;
class OutputSection;
class Symbol;

struct Reloc {
  uint8_t type;
  uint32_t addend;
  uint32_t offset;
  llvm::PointerUnion<Symbol *, InputSection *> target;
};

class InputSection {
public:
  virtual ~InputSection() = default;
  virtual size_t getSize() const { return data.size(); }
  virtual uint64_t getFileSize() const { return getSize(); }
  uint64_t getFileOffset() const;
  uint64_t getVA() const;

  virtual void writeTo(uint8_t *buf);

  InputFile *file = nullptr;
  StringRef name;
  StringRef segname;
  // This provides access to the address of the section in the input file.
  const llvm::MachO::section_64 *header;

  OutputSection *parent = nullptr;
  uint64_t outSecOff = 0;
  uint64_t outSecFileOff = 0;

  uint32_t align = 1;
  uint32_t flags = 0;

  ArrayRef<uint8_t> data;
  std::vector<Reloc> relocs;
};

extern std::vector<InputSection *> inputSections;

} // namespace macho
} // namespace lld

#endif
