//===--- Index.cpp -----------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Index.h"

#include "llvm/Support/SHA1.h"

namespace clang {
namespace clangd {

namespace {
ArrayRef<uint8_t> toArrayRef(StringRef S) {
  return {reinterpret_cast<const uint8_t *>(S.data()), S.size()};
}
} // namespace

SymbolID::SymbolID(llvm::StringRef USR)
    : HashValue(llvm::SHA1::hash(toArrayRef(USR))) {}

SymbolSlab::const_iterator SymbolSlab::begin() const {
  return Symbols.begin();
}

SymbolSlab::const_iterator SymbolSlab::end() const {
  return Symbols.end();
}

SymbolSlab::const_iterator SymbolSlab::find(const SymbolID& SymID) const {
  return Symbols.find(SymID);
}

void SymbolSlab::freeze() {
  Frozen = true;
}

void SymbolSlab::insert(Symbol S) {
  assert(!Frozen &&
         "Can't insert a symbol after the slab has been frozen!");
  Symbols[S.ID] = std::move(S);
}

} // namespace clangd
} // namespace clang
