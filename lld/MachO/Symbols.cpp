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

Defined::Defined(StringRefZ name, InputFile *file, InputSection *isec,
                 uint64_t value, uint64_t size, bool isWeakDef, bool isExternal,
                 bool isPrivateExtern, bool isThumb,
                 bool isReferencedDynamically, bool noDeadStrip)
    : Symbol(DefinedKind, name, file), isec(isec), value(value), size(size),
      overridesWeakDef(false), privateExtern(isPrivateExtern),
      includeInSymtab(true), thumb(isThumb),
      referencedDynamically(isReferencedDynamically), noDeadStrip(noDeadStrip),
      weakDef(isWeakDef), external(isExternal) {
  if (isec) {
    isec->symbols.push_back(this);
    // Maintain sorted order.
    for (auto it = isec->symbols.rbegin(), rend = isec->symbols.rend();
         it != rend; ++it) {
      auto next = std::next(it);
      if (next == rend)
        break;
      if ((*it)->value < (*next)->value)
        std::swap(*next, *it);
      else
        break;
    }
  }
}

bool Defined::isTlv() const {
  return !isAbsolute() && isThreadLocalVariables(isec->getFlags());
}

uint64_t Defined::getVA() const {
  assert(isLive() && "this should only be called for live symbols");

  if (isAbsolute())
    return value;

  if (!isec->isFinal) {
    // A target arch that does not use thunks ought never ask for
    // the address of a function that has not yet been finalized.
    assert(target->usesThunks());

    // ConcatOutputSection::finalize() can seek the address of a
    // function before its address is assigned. The thunking algorithm
    // knows that unfinalized functions will be out of range, so it is
    // expedient to return a contrived out-of-range address.
    return TargetInfo::outOfRangeVA;
  }
  return isec->getVA(value);
}

void Defined::canonicalize() {
  if (compactUnwind)
    compactUnwind = compactUnwind->canonical();
  if (isec)
    isec = isec->canonical();
}

uint64_t DylibSymbol::getVA() const {
  return isInStubs() ? getStubVA() : Symbol::getVA();
}

void LazySymbol::fetchArchiveMember() { getFile()->fetch(sym); }
