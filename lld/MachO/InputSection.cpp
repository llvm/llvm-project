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
#include "Writer.h"
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

uint64_t InputSection::getFileSize() const {
  return isZeroFill(flags) ? 0 : getSize();
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
        // References from thread-local variable sections are treated as offsets
        // relative to the start of the thread-local data memory area, which
        // is initialized via copying all the TLV data sections (which are all
        // contiguous).
        if (isa<Defined>(referentSym))
          referentVA -= firstTLVDataSection->addr;
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

bool macho::isCodeSection(InputSection *isec) {
  uint32_t type = isec->flags & MachO::SECTION_TYPE;
  if (type != S_REGULAR && type != S_COALESCED)
    return false;

  uint32_t attr = isec->flags & MachO::SECTION_ATTRIBUTES_USR;
  if (attr == S_ATTR_PURE_INSTRUCTIONS)
    return true;

  if (isec->segname == segment_names::text)
    return StringSwitch<bool>(isec->name)
        .Cases("__textcoal_nt", "__StaticInit", true)
        .Default(false);

  return false;
}

std::string lld::toString(const InputSection *isec) {
  return (toString(isec->file) + ":(" + isec->name + ")").str();
}
