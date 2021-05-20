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
#include "SyntheticSections.h"
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

static uint64_t resolveSymbolVA(const Symbol *sym, uint8_t type) {
  const RelocAttrs &relocAttrs = target->getRelocAttrs(type);
  if (relocAttrs.hasAttr(RelocAttrBits::BRANCH))
    return sym->resolveBranchVA();
  else if (relocAttrs.hasAttr(RelocAttrBits::GOT))
    return sym->resolveGotVA();
  else if (relocAttrs.hasAttr(RelocAttrBits::TLV))
    return sym->resolveTlvVA();
  return sym->getVA();
}

void InputSection::writeTo(uint8_t *buf) {
  assert(!shouldOmitFromOutput());

  if (getFileSize() == 0)
    return;

  memcpy(buf, data.data(), data.size());

  for (size_t i = 0; i < relocs.size(); i++) {
    const Reloc &r = relocs[i];
    uint8_t *loc = buf + r.offset;
    uint64_t referentVA = 0;
    if (target->hasAttr(r.type, RelocAttrBits::SUBTRAHEND)) {
      const Symbol *fromSym = r.referent.get<Symbol *>();
      const Reloc &minuend = relocs[++i];
      uint64_t minuendVA;
      if (const Symbol *toSym = minuend.referent.dyn_cast<Symbol *>())
        minuendVA = toSym->getVA();
      else {
        auto *referentIsec = minuend.referent.get<InputSection *>();
        assert(!referentIsec->shouldOmitFromOutput());
        minuendVA = referentIsec->getVA();
      }
      referentVA = minuendVA - fromSym->getVA() + minuend.addend;
    } else if (auto *referentSym = r.referent.dyn_cast<Symbol *>()) {
      if (target->hasAttr(r.type, RelocAttrBits::LOAD) &&
          !referentSym->isInGot())
        target->relaxGotLoad(loc, r.type);
      referentVA = resolveSymbolVA(referentSym, r.type);

      if (isThreadLocalVariables(flags)) {
        // References from thread-local variable sections are treated as offsets
        // relative to the start of the thread-local data memory area, which
        // is initialized via copying all the TLV data sections (which are all
        // contiguous).
        if (isa<Defined>(referentSym))
          referentVA -= firstTLVDataSection->addr;
      }
    } else if (auto *referentIsec = r.referent.dyn_cast<InputSection *>()) {
      assert(!referentIsec->shouldOmitFromOutput());
      referentVA = referentIsec->getVA();
    }
    target->relocateOne(loc, r, referentVA + r.addend, getVA() + r.offset);
  }
}

bool macho::isCodeSection(const InputSection *isec) {
  uint32_t type = isec->flags & SECTION_TYPE;
  if (type != S_REGULAR && type != S_COALESCED)
    return false;

  uint32_t attr = isec->flags & SECTION_ATTRIBUTES_USR;
  if (attr == S_ATTR_PURE_INSTRUCTIONS)
    return true;

  if (isec->segname == segment_names::text)
    return StringSwitch<bool>(isec->name)
        .Cases(section_names::textCoalNt, section_names::staticInit, true)
        .Default(false);

  return false;
}

std::string lld::toString(const InputSection *isec) {
  return (toString(isec->file) + ":(" + isec->name + ")").str();
}
