//===- Symbols.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "InputFiles.h"
#include "SyntheticSections.h"

using namespace llvm;
using namespace lld;
using namespace lld::macho;

// Returns a symbol for an error message.
static std::string demangle(StringRef symName) {
  if (config->demangle)
    return demangleItanium(symName);
  return std::string(symName);
}

std::string lld::toString(const Symbol &sym) { return demangle(sym.getName()); }

std::string lld::toMachOString(const object::Archive::Symbol &b) {
  return demangle(b.getName());
}

uint64_t Symbol::getStubVA() const { return in.stubs->getVA(stubsIndex); }
uint64_t Symbol::getGotVA() const { return in.got->getVA(gotIndex); }
uint64_t Symbol::getTlvVA() const { return in.tlvPointers->getVA(gotIndex); }

bool Symbol::isLive() const {
  if (isa<DylibSymbol>(this) || isa<Undefined>(this))
    return used;

  if (auto *d = dyn_cast<Defined>(this)) {
    // Non-absolute symbols might be alive because their section is
    // no_dead_strip or live_support. In that case, the section will know
    // that it's live but `used` might be false. Non-absolute symbols always
    // have to use the section's `live` bit as source of truth.
    if (d->isAbsolute())
      return used;
    return d->isec->canonical()->isLive(d->value);
  }

  assert(!isa<CommonSymbol>(this) &&
         "replaceCommonSymbols() runs before dead code stripping, and isLive() "
         "should only be called after dead code stripping");

  // Assume any other kind of symbol is live.
  return true;
}

uint64_t Defined::getVA() const {
  assert(isLive() && "this should only be called for live symbols");

  if (isAbsolute())
    return value;

  if (!isec->canonical()->isFinal) {
    // A target arch that does not use thunks ought never ask for
    // the address of a function that has not yet been finalized.
    assert(target->usesThunks());

    // ConcatOutputSection::finalize() can seek the address of a
    // function before its address is assigned. The thunking algorithm
    // knows that unfinalized functions will be out of range, so it is
    // expedient to return a contrived out-of-range address.
    return TargetInfo::outOfRangeVA;
  }
  return isec->canonical()->getVA(value);
}

uint64_t DylibSymbol::getVA() const {
  return isInStubs() ? getStubVA() : Symbol::getVA();
}

void LazySymbol::fetchArchiveMember() { getFile()->fetch(sym); }
