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
  bool pcrel;
  uint8_t length;
  // The offset from the start of the subsection that this relocation belongs
  // to.
  uint32_t offset;
  // Adding this offset to the address of the target symbol or subsection gives
  // the destination that this relocation refers to.
  uint64_t addend;
  llvm::PointerUnion<Symbol *, InputSection *> target;
};

inline bool isZeroFill(uint8_t flags) {
  return llvm::MachO::isVirtualSection(flags & llvm::MachO::SECTION_TYPE);
}

inline bool isThreadLocalVariables(uint8_t flags) {
  return (flags & llvm::MachO::SECTION_TYPE) ==
         llvm::MachO::S_THREAD_LOCAL_VARIABLES;
}

class InputSection {
public:
  virtual ~InputSection() = default;
  virtual uint64_t getSize() const { return data.size(); }
  virtual uint64_t getFileSize() const {
    return isZeroFill(flags) ? 0 : getSize();
  }
  uint64_t getFileOffset() const;
  uint64_t getVA() const;

  virtual void writeTo(uint8_t *buf);

  InputFile *file = nullptr;
  StringRef name;
  StringRef segname;

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

std::string toString(const macho::InputSection *);

} // namespace lld

#endif
