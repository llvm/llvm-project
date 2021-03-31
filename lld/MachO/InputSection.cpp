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

static uint64_t resolveSymbolVA(uint8_t *loc, const Symbol &sym, uint8_t type) {
  const RelocAttrs &relocAttrs = target->getRelocAttrs(type);
  if (relocAttrs.hasAttr(RelocAttrBits::BRANCH)) {
    if (sym.isInStubs())
      return in.stubs->addr + sym.stubsIndex * target->stubSize;
  } else if (relocAttrs.hasAttr(RelocAttrBits::GOT)) {
    if (sym.isInGot())
      return in.got->addr + sym.gotIndex * WordSize;
  } else if (relocAttrs.hasAttr(RelocAttrBits::TLV)) {
    if (sym.isInGot())
      return in.tlvPointers->addr + sym.gotIndex * WordSize;
    assert(isa<Defined>(&sym));
  }
  return sym.getVA();
}

void InputSection::writeTo(uint8_t *buf) {
  if (getFileSize() == 0)
    return;

  memcpy(buf, data.data(), data.size());

  for (size_t i = 0; i < relocs.size(); i++) {
    const Reloc &r = relocs[i];
    uint8_t *loc = buf + r.offset;
    uint64_t referentVA = 0;
    if (target->hasAttr(r.type, RelocAttrBits::SUBTRAHEND)) {
      const Symbol *fromSym = r.referent.get<Symbol *>();
      const Symbol *toSym = relocs[++i].referent.get<Symbol *>();
      referentVA = toSym->getVA() - fromSym->getVA();
    } else if (auto *referentSym = r.referent.dyn_cast<Symbol *>()) {
      if (target->hasAttr(r.type, RelocAttrBits::LOAD) &&
          !referentSym->isInGot())
        target->relaxGotLoad(loc, r.type);
      referentVA = resolveSymbolVA(loc, *referentSym, r.type);

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
    target->relocateOne(loc, r, referentVA, getVA() + r.offset);
  }
}

bool macho::isCodeSection(InputSection *isec) {
  uint32_t type = isec->flags & SECTION_TYPE;
  if (type != S_REGULAR && type != S_COALESCED)
    return false;

  uint32_t attr = isec->flags & SECTION_ATTRIBUTES_USR;
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
