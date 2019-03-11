//===--- Symbol.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Symbol.h"

namespace clang {
namespace clangd {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, Symbol::SymbolFlag F) {
  if (F == Symbol::None)
    return OS << "None";
  std::string S;
  if (F & Symbol::Deprecated)
    S += "deprecated|";
  if (F & Symbol::IndexedForCodeCompletion)
    S += "completion|";
  return OS << llvm::StringRef(S).rtrim('|');
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Symbol &S) {
  return OS << S.Scope << S.Name;
}

float quality(const Symbol &S) {
  // This avoids a sharp gradient for tail symbols, and also neatly avoids the
  // question of whether 0 references means a bad symbol or missing data.
  if (S.References < 3)
    return 1;
  return std::log(S.References);
}

SymbolSlab::const_iterator SymbolSlab::find(const SymbolID &ID) const {
  auto It = std::lower_bound(
      Symbols.begin(), Symbols.end(), ID,
      [](const Symbol &S, const SymbolID &I) { return S.ID < I; });
  if (It != Symbols.end() && It->ID == ID)
    return It;
  return Symbols.end();
}

// Copy the underlying data of the symbol into the owned arena.
static void own(Symbol &S, llvm::UniqueStringSaver &Strings) {
  visitStrings(S, [&](llvm::StringRef &V) { V = Strings.save(V); });
}

void SymbolSlab::Builder::insert(const Symbol &S) {
  auto R = SymbolIndex.try_emplace(S.ID, Symbols.size());
  if (R.second) {
    Symbols.push_back(S);
    own(Symbols.back(), UniqueStrings);
  } else {
    auto &Copy = Symbols[R.first->second] = S;
    own(Copy, UniqueStrings);
  }
}

SymbolSlab SymbolSlab::Builder::build() && {
  Symbols = {Symbols.begin(), Symbols.end()}; // Force shrink-to-fit.
  // Sort symbols so the slab can binary search over them.
  llvm::sort(Symbols,
             [](const Symbol &L, const Symbol &R) { return L.ID < R.ID; });
  // We may have unused strings from overwritten symbols. Build a new arena.
  llvm::BumpPtrAllocator NewArena;
  llvm::UniqueStringSaver Strings(NewArena);
  for (auto &S : Symbols)
    own(S, Strings);
  return SymbolSlab(std::move(NewArena), std::move(Symbols));
}

} // namespace clangd
} // namespace clang
