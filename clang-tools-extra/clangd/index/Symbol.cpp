//===--- Symbol.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Symbol.h"

#include <cmath>

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
  auto It = llvm::partition_point(Symbols,
                                  [&](const Symbol &S) { return S.ID < ID; });
  if (It != Symbols.end() && It->ID == ID)
    return It;
  return Symbols.end();
}

// Copy the underlying data of the symbol into the owned arena.
static void own(Symbol &S, llvm::UniqueStringSaver &Strings) {
  visitStrings(S, [&](llvm::StringRef &V) { V = Strings.save(V); });
}

void SymbolSlab::Builder::insert(const Symbol &S) {
  own(Symbols[S.ID] = S, UniqueStrings);
}

SymbolSlab SymbolSlab::Builder::build() && {
  // Sort symbols into vector so the slab can binary search over them.
  std::vector<Symbol> SortedSymbols;
  SortedSymbols.reserve(Symbols.size());
  for (auto &Entry : Symbols)
    SortedSymbols.push_back(std::move(Entry.second));
  llvm::sort(SortedSymbols,
             [](const Symbol &L, const Symbol &R) { return L.ID < R.ID; });
  // We may have unused strings from overwritten symbols. Build a new arena.
  llvm::BumpPtrAllocator NewArena;
  llvm::UniqueStringSaver Strings(NewArena);
  for (auto &S : SortedSymbols)
    own(S, Strings);
  return SymbolSlab(std::move(NewArena), std::move(SortedSymbols));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolSlab &Slab) {
  OS << "{";
  llvm::StringRef Sep = "";
  for (const auto &S : Slab) {
    OS << Sep << S;
    Sep = ", ";
  }
  OS << "}";
  return OS;
}
} // namespace clangd
} // namespace clang
