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
#include "lld/Common/Strings.h"

using namespace llvm;
using namespace lld;
using namespace lld::macho;

static_assert(sizeof(void *) != 8 || sizeof(Symbol) == 48,
              "Try to minimize Symbol's size; we create many instances");

// The Microsoft ABI doesn't support using parent class tail padding for child
// members, hence the _MSC_VER check.
#if !defined(_MSC_VER)
static_assert(sizeof(void *) != 8 || sizeof(Defined) == 80,
              "Try to minimize Defined's size; we create many instances");
#endif

static_assert(sizeof(SymbolUnion) == sizeof(Defined),
              "Defined should be the largest Symbol kind");

std::string lld::toString(const Symbol &sym) {
  return demangle(sym.getName(), config->demangle);
}

std::string lld::toMachOString(const object::Archive::Symbol &b) {
  return demangle(b.getName(), config->demangle);
}

uint64_t Symbol::getStubVA() const { return in.stubs->getVA(stubsIndex); }
uint64_t Symbol::getGotVA() const { return in.got->getVA(gotIndex); }
uint64_t Symbol::getTlvVA() const { return in.tlvPointers->getVA(gotIndex); }

Defined::Defined(StringRefZ name, InputFile *file, InputSection *isec,
                 uint64_t value, uint64_t size, bool isWeakDef, bool isExternal,
                 bool isPrivateExtern, bool isThumb,
                 bool isReferencedDynamically, bool noDeadStrip,
                 bool canOverrideWeakDef, bool isWeakDefCanBeHidden)
    : Symbol(DefinedKind, name, file), overridesWeakDef(canOverrideWeakDef),
      privateExtern(isPrivateExtern), includeInSymtab(true), thumb(isThumb),
      referencedDynamically(isReferencedDynamically), noDeadStrip(noDeadStrip),
      weakDefCanBeHidden(isWeakDefCanBeHidden), weakDef(isWeakDef),
      external(isExternal), isec(isec), value(value), size(size) {
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
  if (unwindEntry)
    unwindEntry = unwindEntry->canonical();
  if (isec)
    isec = isec->canonical();
}

uint64_t DylibSymbol::getVA() const {
  return isInStubs() ? getStubVA() : Symbol::getVA();
}

void LazyArchive::fetchArchiveMember() { getFile()->fetch(sym); }
