//===- InputSection.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputSection.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/Memory.h"
#include "llvm/Support/Endian.h"

using namespace llvm::MachO;
using namespace llvm::support;
using namespace lld;
using namespace lld::macho;

std::vector<InputSection *> macho::inputSections;

void InputSection::writeTo(uint8_t *buf) {
  memcpy(buf, data.data(), data.size());

  for (Reloc &r : relocs) {
    uint64_t va = 0;
    if (auto *s = r.target.dyn_cast<Symbol *>()) {
      if (auto *dylibSymbol = dyn_cast<DylibSymbol>(s)) {
        va = in.got->addr - ImageBase + dylibSymbol->gotIndex * WordSize;
      } else {
        va = s->getVA();
      }
    } else if (auto *isec = r.target.dyn_cast<InputSection *>()) {
      va = isec->addr;
    } else {
      llvm_unreachable("Unknown relocation target");
    }

    uint64_t val = va + r.addend;
    if (1) // TODO: handle non-pcrel relocations
      val -= addr - ImageBase + r.offset;
    target->relocateOne(buf + r.offset, r.type, val);
  }
}
