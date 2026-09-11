//===- ElideSwiftForceLoad.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StripSwiftForceLoad.h"

#include "Config.h"
#include "InputSection.h"
#include "OutputSegment.h"
#include "Symbols.h"
#include "Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TimeProfiler.h"

using namespace llvm;
using namespace lld;
using namespace lld::macho;

static constexpr StringRef forceLoadPrefix = "__swift_FORCE_LOAD_$_";

// Returns true if every byte of `isec` is a pointer slot covered by an UNSIGNED
// pointer relocation to an imported `__swift_FORCE_LOAD_$_*` symbol, i.e.
// the section exists only to force-load Swift overlays and holds no other data.
static bool isSwiftForceLoadSection(const ConcatInputSection *isec) {
  if (isec->relocs.empty())
    return false;

  if (isec->getSize() != target->wordSize * isec->relocs.size())
    return false;

  for (const Relocation &r : isec->relocs) {
    auto *sym = dyn_cast_if_present<Symbol *>(r.referent);
    auto *dylibSym = dyn_cast_or_null<DylibSymbol>(sym);
    if (!dylibSym || dylibSym->isDynamicLookup() ||
        !dylibSym->getName().starts_with(forceLoadPrefix))
      return false;
  }
  return true;
}

void macho::stripSwiftForceLoadFixups() {
  if (!config->stripSwiftForceLoad)
    return;

  TimeTraceScope timeScope("Strip Swift FORCE_LOAD fixups");

  for (ConcatInputSection *isec : inputSections) {
    if (isec->shouldOmitFromOutput() || isec->replacement)
      continue;

    if (isec->getSegName() != segment_names::data ||
        isec->getName() != section_names::const_)
      continue;

    // Never drop a section that exports a symbol clients may link against.
    if (llvm::any_of(isec->symbols, [](const Defined *d) {
          return d->isExternal() && !d->privateExtern;
        }))
      continue;

    if (!isSwiftForceLoadSection(isec))
      continue;

    isec->live = false;
    for (Defined *d : isec->symbols)
      d->used = false;
  }
}
