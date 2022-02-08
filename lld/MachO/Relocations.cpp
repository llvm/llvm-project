//===- Relocations.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Relocations.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"

using namespace llvm;
using namespace lld;
using namespace lld::macho;

static_assert(sizeof(void *) != 8 || sizeof(Reloc) == 24,
              "Try to minimize Reloc's size; we create many instances");

bool macho::validateSymbolRelocation(const Symbol *sym,
                                     const InputSection *isec, const Reloc &r) {
  const RelocAttrs &relocAttrs = target->getRelocAttrs(r.type);
  bool valid = true;
  auto message = [&](const Twine &diagnostic) {
    valid = false;
    return (isec->getLocation(r.offset) + ": " + relocAttrs.name +
            " relocation " + diagnostic)
        .str();
  };

  if (relocAttrs.hasAttr(RelocAttrBits::TLV) != sym->isTlv())
    error(message(Twine("requires that symbol ") + sym->getName() + " " +
                  (sym->isTlv() ? "not " : "") + "be thread-local"));

  return valid;
}

void macho::reportRangeError(const Reloc &r, const Twine &v, uint8_t bits,
                             int64_t min, uint64_t max) {
  std::string hint;
  if (auto *sym = r.referent.dyn_cast<Symbol *>())
    hint = "; references " + toString(*sym);
  // TODO: get location of reloc using something like LLD-ELF's getErrorPlace()
  error("relocation " + target->getRelocAttrs(r.type).name +
        " is out of range: " + v + " is not in [" + Twine(min) + ", " +
        Twine(max) + "]" + hint);
}

void macho::reportRangeError(SymbolDiagnostic d, const Twine &v, uint8_t bits,
                             int64_t min, uint64_t max) {
  std::string hint;
  if (d.symbol)
    hint = "; references " + toString(*d.symbol);
  error(d.reason + " is out of range: " + v + " is not in [" + Twine(min) +
        ", " + Twine(max) + "]" + hint);
}

const RelocAttrs macho::invalidRelocAttrs{"INVALID", RelocAttrBits::_0};
