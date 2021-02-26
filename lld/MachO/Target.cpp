//===- Target.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Target.h"
#include "InputSection.h"
#include "Symbols.h"
#include "SyntheticSections.h"

#include "lld/Common/ErrorHandler.h"

using namespace llvm;
using namespace lld;
using namespace lld::macho;

const TargetInfo::RelocAttrs TargetInfo::invalidRelocAttrs{"INVALID",
                                                           RelocAttrBits::_0};

bool TargetInfo::validateSymbolRelocation(const Symbol *sym,
                                          const InputSection *isec,
                                          const Reloc &r) {
  const RelocAttrs &relocAttrs = getRelocAttrs(r.type);
  bool valid = true;
  auto message = [relocAttrs, sym, isec, &valid](const Twine &diagnostic) {
    valid = false;
    return (relocAttrs.name + " relocation " + diagnostic + " for `" +
            sym->getName() + "' in " + toString(isec))
        .str();
  };

  if (relocAttrs.hasAttr(RelocAttrBits::TLV) != sym->isTlv())
    error(message(Twine("requires that variable ") +
                  (sym->isTlv() ? "not " : "") + "be thread-local"));
  if (relocAttrs.hasAttr(RelocAttrBits::DYSYM8) && isa<DylibSymbol>(sym) &&
      r.length != 3)
    error(message("has width " + std::to_string(1 << r.length) +
                  " bytes, but must be 8 bytes"));

  return valid;
}

TargetInfo *macho::target = nullptr;
