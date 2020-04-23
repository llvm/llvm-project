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
class OutputSegment;
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
  virtual void writeTo(uint8_t *buf);
  virtual size_t getSize() const { return data.size(); }

  InputFile *file = nullptr;
  OutputSegment *parent = nullptr;
  StringRef name;
  StringRef segname;

  ArrayRef<uint8_t> data;
  uint64_t addr = 0;
  uint32_t align = 1;
  uint32_t flags = 0;

  std::vector<Reloc> relocs;
};

extern std::vector<InputSection *> inputSections;

} // namespace macho
} // namespace lld

#endif
