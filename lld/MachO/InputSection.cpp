//===- InputSection.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputSection.h"
#include "OutputSegment.h"
#include "Symbols.h"
#include "Target.h"
#include "lld/Common/Memory.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::support;
using namespace lld;
using namespace lld::macho;

std::vector<InputSection *> macho::inputSections;

uint64_t InputSection::getFileOffset() const {
  return parent->fileOff + outSecFileOff;
}

uint64_t InputSection::getVA() const { return parent->addr + outSecOff; }

void InputSection::writeTo(uint8_t *buf) {
  if (!data.empty())
    memcpy(buf, data.data(), data.size());

  for (Reloc &r : relocs) {
    uint64_t va = 0;
    if (auto *s = r.target.dyn_cast<Symbol *>()) {
      if (auto *dylibSymbol = dyn_cast<DylibSymbol>(s)) {
        va = target->getDylibSymbolVA(*dylibSymbol, r.type);
      } else {
        va = s->getVA();
      }
    } else if (auto *isec = r.target.dyn_cast<InputSection *>()) {
      va = isec->getVA();
    }

    uint64_t val = va + r.addend;
    if (r.pcrel)
      val -= getVA() + r.offset;
    target->relocateOne(buf + r.offset, r, val);
  }
}
