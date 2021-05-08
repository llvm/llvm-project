//===- OutputSection.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MergedOutputSection.h"
#include "OutputSegment.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

void MergedOutputSection::mergeInput(InputSection *input) {
  if (inputs.empty()) {
    align = input->align;
    flags = input->flags;
  } else {
    align = std::max(align, input->align);
    mergeFlags(input);
  }

  inputs.push_back(input);
  input->parent = this;
}

void MergedOutputSection::finalize() {
  uint64_t isecAddr = addr;
  uint64_t isecFileOff = fileOff;
  for (InputSection *isec : inputs) {
    isecAddr = alignTo(isecAddr, isec->align);
    isecFileOff = alignTo(isecFileOff, isec->align);
    isec->outSecOff = isecAddr - addr;
    isec->outSecFileOff = isecFileOff - fileOff;
    isecAddr += isec->getSize();
    isecFileOff += isec->getFileSize();
  }
  size = isecAddr - addr;
  fileSize = isecFileOff - fileOff;
}

void MergedOutputSection::writeTo(uint8_t *buf) const {
  for (InputSection *isec : inputs)
    isec->writeTo(buf + isec->outSecFileOff);
}

// TODO: this is most likely wrong; reconsider how section flags
// are actually merged. The logic presented here was written without
// any form of informed research.
void MergedOutputSection::mergeFlags(InputSection *input) {
  uint8_t baseType = flags & SECTION_TYPE;
  uint8_t inputType = input->flags & SECTION_TYPE;
  if (baseType != inputType)
    error("Cannot merge section " + input->name + " (type=0x" +
          to_hexString(inputType) + ") into " + name + " (type=0x" +
          to_hexString(baseType) + "): inconsistent types");

  constexpr uint32_t strictFlags = S_ATTR_DEBUG | S_ATTR_STRIP_STATIC_SYMS |
                                   S_ATTR_NO_DEAD_STRIP | S_ATTR_LIVE_SUPPORT;
  if ((input->flags ^ flags) & strictFlags)
    error("Cannot merge section " + input->name + " (flags=0x" +
          to_hexString(input->flags) + ") into " + name + " (flags=0x" +
          to_hexString(flags) + "): strict flags differ");

  // Negate pure instruction presence if any section isn't pure.
  uint32_t pureMask = ~S_ATTR_PURE_INSTRUCTIONS | (input->flags & flags);

  // Merge the rest
  flags |= input->flags;
  flags &= pureMask;
}
