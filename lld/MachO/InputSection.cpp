//===- InputSection.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputSection.h"
#include "ConcatOutputSection.h"
#include "Config.h"
#include "InputFiles.h"
#include "OutputSegment.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "UnwindInfoSection.h"
#include "Writer.h"
#include "lld/Common/Memory.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/xxhash.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::support;
using namespace lld;
using namespace lld::macho;

// Verify ConcatInputSection's size on 64-bit builds. The size of std::vector
// can differ based on STL debug levels (e.g. iterator debugging on MSVC's STL),
// so account for that.
static_assert(sizeof(void *) != 8 ||
                  sizeof(ConcatInputSection) == sizeof(std::vector<Reloc>) + 88,
              "Try to minimize ConcatInputSection's size, we create many "
              "instances of it");

std::vector<ConcatInputSection *> macho::inputSections;

uint64_t InputSection::getFileSize() const {
  return isZeroFill(getFlags()) ? 0 : getSize();
}

uint64_t InputSection::getVA(uint64_t off) const {
  return parent->addr + getOffset(off);
}

static uint64_t resolveSymbolVA(const Symbol *sym, uint8_t type) {
  const RelocAttrs &relocAttrs = target->getRelocAttrs(type);
  if (relocAttrs.hasAttr(RelocAttrBits::BRANCH))
    return sym->resolveBranchVA();
  if (relocAttrs.hasAttr(RelocAttrBits::GOT))
    return sym->resolveGotVA();
  if (relocAttrs.hasAttr(RelocAttrBits::TLV))
    return sym->resolveTlvVA();
  return sym->getVA();
}

std::string InputSection::getLocation(uint64_t off) const {
  // First, try to find a symbol that's near the offset. Use it as a reference
  // point.
  for (size_t i = 0; i < symbols.size(); ++i)
    if (symbols[i]->value <= off &&
        (i + 1 == symbols.size() || symbols[i + 1]->value > off))
      return (toString(getFile()) + ":(symbol " + symbols.front()->getName() +
              "+0x" + Twine::utohexstr(off - symbols[i]->value) + ")")
          .str();

  // If that fails, use the section itself as a reference point.
  for (const Subsection &subsec : section.subsections) {
    if (subsec.isec == this) {
      off += subsec.offset;
      break;
    }
  }
  return (toString(getFile()) + ":(" + getName() + "+0x" +
          Twine::utohexstr(off) + ")")
      .str();
}

void ConcatInputSection::foldIdentical(ConcatInputSection *copy) {
  align = std::max(align, copy->align);
  copy->live = false;
  copy->wasCoalesced = true;
  copy->replacement = this;

  // Merge the sorted vectors of symbols together.
  auto it = symbols.begin();
  for (auto copyIt = copy->symbols.begin(); copyIt != copy->symbols.end();) {
    if (it == symbols.end()) {
      symbols.push_back(*copyIt++);
      it = symbols.end();
    } else if ((*it)->value > (*copyIt)->value) {
      std::swap(*it++, *copyIt);
    } else {
      ++it;
    }
  }
  copy->symbols.clear();

  // Remove duplicate compact unwind info for symbols at the same address.
  if (symbols.empty())
    return;
  it = symbols.begin();
  uint64_t v = (*it)->value;
  for (++it; it != symbols.end(); ++it) {
    Defined *d = *it;
    if (d->value == v)
      d->unwindEntry = nullptr;
    else
      v = d->value;
  }
}

void ConcatInputSection::writeTo(uint8_t *buf) {
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
        minuendVA = toSym->getVA() + minuend.addend;
      else {
        auto *referentIsec = minuend.referent.get<InputSection *>();
        assert(!::shouldOmitFromOutput(referentIsec));
        minuendVA = referentIsec->getVA(minuend.addend);
      }
      referentVA = minuendVA - fromSym->getVA();
    } else if (auto *referentSym = r.referent.dyn_cast<Symbol *>()) {
      if (target->hasAttr(r.type, RelocAttrBits::LOAD) &&
          !referentSym->isInGot())
        target->relaxGotLoad(loc, r.type);
      referentVA = resolveSymbolVA(referentSym, r.type) + r.addend;

      if (isThreadLocalVariables(getFlags())) {
        // References from thread-local variable sections are treated as offsets
        // relative to the start of the thread-local data memory area, which
        // is initialized via copying all the TLV data sections (which are all
        // contiguous).
        if (isa<Defined>(referentSym))
          referentVA -= firstTLVDataSection->addr;
      }
    } else if (auto *referentIsec = r.referent.dyn_cast<InputSection *>()) {
      assert(!::shouldOmitFromOutput(referentIsec));
      referentVA = referentIsec->getVA(r.addend);
    }
    target->relocateOne(loc, r, referentVA, getVA() + r.offset);
  }
}

ConcatInputSection *macho::makeSyntheticInputSection(StringRef segName,
                                                     StringRef sectName,
                                                     uint32_t flags,
                                                     ArrayRef<uint8_t> data,
                                                     uint32_t align) {
  Section &section =
      *make<Section>(/*file=*/nullptr, segName, sectName, flags, /*addr=*/0);
  auto isec = make<ConcatInputSection>(section, data, align);
  section.subsections.push_back({0, isec});
  return isec;
}

void CStringInputSection::splitIntoPieces() {
  size_t off = 0;
  StringRef s = toStringRef(data);
  while (!s.empty()) {
    size_t end = s.find(0);
    if (end == StringRef::npos)
      fatal(getLocation(off) + ": string is not null terminated");
    size_t size = end + 1;
    uint32_t hash = config->dedupLiterals ? xxHash64(s.substr(0, size)) : 0;
    pieces.emplace_back(off, hash);
    s = s.substr(size);
    off += size;
  }
}

StringPiece &CStringInputSection::getStringPiece(uint64_t off) {
  if (off >= data.size())
    fatal(toString(this) + ": offset is outside the section");

  auto it =
      partition_point(pieces, [=](StringPiece p) { return p.inSecOff <= off; });
  return it[-1];
}

const StringPiece &CStringInputSection::getStringPiece(uint64_t off) const {
  return const_cast<CStringInputSection *>(this)->getStringPiece(off);
}

uint64_t CStringInputSection::getOffset(uint64_t off) const {
  const StringPiece &piece = getStringPiece(off);
  uint64_t addend = off - piece.inSecOff;
  return piece.outSecOff + addend;
}

WordLiteralInputSection::WordLiteralInputSection(const Section &section,
                                                 ArrayRef<uint8_t> data,
                                                 uint32_t align)
    : InputSection(WordLiteralKind, section, data, align) {
  switch (sectionType(getFlags())) {
  case S_4BYTE_LITERALS:
    power2LiteralSize = 2;
    break;
  case S_8BYTE_LITERALS:
    power2LiteralSize = 3;
    break;
  case S_16BYTE_LITERALS:
    power2LiteralSize = 4;
    break;
  default:
    llvm_unreachable("invalid literal section type");
  }

  live.resize(data.size() >> power2LiteralSize, !config->deadStrip);
}

uint64_t WordLiteralInputSection::getOffset(uint64_t off) const {
  auto *osec = cast<WordLiteralSection>(parent);
  const uintptr_t buf = reinterpret_cast<uintptr_t>(data.data());
  switch (sectionType(getFlags())) {
  case S_4BYTE_LITERALS:
    return osec->getLiteral4Offset(buf + (off & ~3LLU)) | (off & 3);
  case S_8BYTE_LITERALS:
    return osec->getLiteral8Offset(buf + (off & ~7LLU)) | (off & 7);
  case S_16BYTE_LITERALS:
    return osec->getLiteral16Offset(buf + (off & ~15LLU)) | (off & 15);
  default:
    llvm_unreachable("invalid literal section type");
  }
}

bool macho::isCodeSection(const InputSection *isec) {
  uint32_t type = sectionType(isec->getFlags());
  if (type != S_REGULAR && type != S_COALESCED)
    return false;

  uint32_t attr = isec->getFlags() & SECTION_ATTRIBUTES_USR;
  if (attr == S_ATTR_PURE_INSTRUCTIONS)
    return true;

  if (isec->getSegName() == segment_names::text)
    return StringSwitch<bool>(isec->getName())
        .Cases(section_names::textCoalNt, section_names::staticInit, true)
        .Default(false);

  return false;
}

bool macho::isCfStringSection(const InputSection *isec) {
  return isec->getName() == section_names::cfString &&
         isec->getSegName() == segment_names::data;
}

bool macho::isClassRefsSection(const InputSection *isec) {
  return isec->getName() == section_names::objcClassRefs &&
         isec->getSegName() == segment_names::data;
}

std::string lld::toString(const InputSection *isec) {
  return (toString(isec->getFile()) + ":(" + isec->getName() + ")").str();
}
