//===--- Index.cpp -----------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Index.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA1.h"

namespace clang {
namespace clangd {

SymbolID::SymbolID(llvm::StringRef USR)
    : HashValue(llvm::SHA1::hash(arrayRefFromStringRef(USR))) {}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolID &ID) {
  OS << toHex(llvm::toStringRef(ID.HashValue));
  return OS;
}

void operator>>(llvm::StringRef Str, SymbolID &ID) {
  std::string HexString = fromHex(Str);
  assert(HexString.size() == 20);
  std::copy(HexString.begin(), HexString.end(), ID.HashValue.begin());
}

SymbolSlab::const_iterator SymbolSlab::begin() const { return Symbols.begin(); }

SymbolSlab::const_iterator SymbolSlab::end() const { return Symbols.end(); }

SymbolSlab::const_iterator SymbolSlab::find(const SymbolID &SymID) const {
  return Symbols.find(SymID);
}

void SymbolSlab::freeze() { Frozen = true; }

void SymbolSlab::insert(Symbol S) {
  assert(!Frozen && "Can't insert a symbol after the slab has been frozen!");
  Symbols[S.ID] = std::move(S);
}

} // namespace clangd
} // namespace clang
