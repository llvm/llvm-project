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

std::vector<InputSection *> macho::inputSections;

uint64_t InputSection::getFileSize() const {
  return isZeroFill(flags) ? 0 : getSize();
}

uint64_t InputSection::getVA(uint64_t off) const {
  return parent->addr + getOffset(off);
}

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

// ICF needs to hash any section that might potentially be duplicated so
// that it can match on content rather than identity.
bool ConcatInputSection::isHashableForICF(bool isText) const {
  if (shouldOmitFromOutput())
    return false;
  switch (sectionType(flags)) {
  case S_REGULAR:
    if (isText)
      return !hasPersonality;
    // One might hope that we could hash __TEXT,__const subsections to fold
    // references to duplicated values, but alas, many tests fail.
    return false;
  case S_CSTRING_LITERALS:
  case S_4BYTE_LITERALS:
  case S_8BYTE_LITERALS:
  case S_16BYTE_LITERALS:
  case S_LITERAL_POINTERS:
    // FIXME(jezng): We should not have any ConcatInputSections of these types
    // when running ICF.
    return false;
  case S_ZEROFILL:
  case S_GB_ZEROFILL:
  case S_NON_LAZY_SYMBOL_POINTERS:
  case S_LAZY_SYMBOL_POINTERS:
  case S_SYMBOL_STUBS:
  case S_MOD_INIT_FUNC_POINTERS:
  case S_MOD_TERM_FUNC_POINTERS:
  case S_COALESCED:
  case S_INTERPOSING:
  case S_DTRACE_DOF:
  case S_LAZY_DYLIB_SYMBOL_POINTERS:
  case S_THREAD_LOCAL_REGULAR:
  case S_THREAD_LOCAL_ZEROFILL:
  case S_THREAD_LOCAL_VARIABLES:
  case S_THREAD_LOCAL_VARIABLE_POINTERS:
  case S_THREAD_LOCAL_INIT_FUNCTION_POINTERS:
    return false;
  default:
    llvm_unreachable("Section type");
  }
}

void ConcatInputSection::hashForICF() {
  assert(data.data()); // zeroFill section data has nullptr with non-zero size
  assert(icfEqClass[0] == 0); // don't overwrite a unique ID!
  // Turn-on the top bit to guarantee that valid hashes have no collisions
  // with the small-integer unique IDs for ICF-ineligible sections
  icfEqClass[0] = xxHash64(data) | (1ull << 63);
}

void ConcatInputSection::foldIdentical(ConcatInputSection *copy) {
  align = std::max(align, copy->align);
  copy->live = false;
  copy->wasCoalesced = true;
  numRefs += copy->numRefs;
  copy->numRefs = 0;
  copy->replacement = this;
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

      if (isThreadLocalVariables(flags)) {
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

void CStringInputSection::splitIntoPieces() {
  size_t off = 0;
  StringRef s = toStringRef(data);
  while (!s.empty()) {
    size_t end = s.find(0);
    if (end == StringRef::npos)
      fatal(toString(this) + ": string is not null terminated");
    size_t size = end + 1;
    pieces.emplace_back(off, xxHash64(s.substr(0, size)));
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

WordLiteralInputSection::WordLiteralInputSection(StringRef segname,
                                                 StringRef name,
                                                 InputFile *file,
                                                 ArrayRef<uint8_t> data,
                                                 uint32_t align, uint32_t flags)
    : InputSection(WordLiteralKind, segname, name, file, data, align, flags) {
  switch (sectionType(flags)) {
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
  const uint8_t *buf = data.data();
  switch (sectionType(flags)) {
  case S_4BYTE_LITERALS:
    return osec->getLiteral4Offset(buf + off);
  case S_8BYTE_LITERALS:
    return osec->getLiteral8Offset(buf + off);
  case S_16BYTE_LITERALS:
    return osec->getLiteral16Offset(buf + off);
  default:
    llvm_unreachable("invalid literal section type");
  }
}

bool macho::isCodeSection(const InputSection *isec) {
  uint32_t type = sectionType(isec->flags);
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
