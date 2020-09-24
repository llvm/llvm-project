//===- InputSection.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputSection.h"
#include "InputFiles.h"
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
  if (getFileSize() == 0)
    return;

  memcpy(buf, data.data(), data.size());

  for (Reloc &r : relocs) {
    uint64_t referentVA = 0;
    if (auto *referentSym = r.referent.dyn_cast<Symbol *>()) {
      referentVA =
          target->resolveSymbolVA(buf + r.offset, *referentSym, r.type);

      if (isThreadLocalVariables(flags)) {
        // References from thread-local variable sections are treated
        // as offsets relative to the start of the referent section,
        // instead of as absolute addresses.
        if (auto *defined = dyn_cast<Defined>(referentSym))
          referentVA -= defined->isec->parent->addr;
      }
    } else if (auto *referentIsec = r.referent.dyn_cast<InputSection *>()) {
      referentVA = referentIsec->getVA();
    }

    uint64_t referentVal = referentVA + r.addend;
    if (r.pcrel)
      referentVal -= getVA() + r.offset;
    target->relocateOne(buf + r.offset, r, referentVal);
  }
}

std::string lld::toString(const InputSection *isec) {
  return (toString(isec->file) + ":(" + isec->name + ")").str();
}
