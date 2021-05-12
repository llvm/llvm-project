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

uint64_t Defined::getVA() const {
  if (isAbsolute())
    return value;

  if (!isec->isFinal) {
    // A target arch that does not use thunks ought never ask for
    // the address of a function that has not yet been finalized.
    assert(target->usesThunks());

    // MergedOutputSection::finalize() can seek the address of a
    // function before its address is assigned. The thunking algorithm
    // knows that unfinalized functions will be out of range, so it is
    // expedient to return a contrived out-of-range address.
    return TargetInfo::outOfRangeVA;
  }
  return isec->getVA() + value;
}

uint64_t Defined::getFileOffset() const {
  if (isAbsolute()) {
    error("absolute symbol " + toString(*this) +
          " does not have a file offset");
    return 0;
  }
  return isec->getFileOffset() + value;
}

uint64_t DylibSymbol::getVA() const {
  return isInStubs() ? getStubVA() : Symbol::getVA();
}

void LazySymbol::fetchArchiveMember() { getFile()->fetch(sym); }
