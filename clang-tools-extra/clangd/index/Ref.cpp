//===--- Ref.cpp -------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ref.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace clangd {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, RefKind K) {
  if (K == RefKind::Unknown)
    return OS << "Unknown";
  static constexpr std::array<const char *, 4> Messages = {"Decl", "Def", "Ref",
                                                           "Spelled"};
  bool VisitedOnce = false;
  for (unsigned I = 0; I < Messages.size(); ++I) {
    if (static_cast<uint8_t>(K) & 1u << I) {
      if (VisitedOnce)
        OS << ", ";
      OS << Messages[I];
      VisitedOnce = true;
    }
  }
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Ref &R) {
  return OS << R.Location << ":" << R.Kind;
}

void RefSlab::Builder::insert(const SymbolID &ID, const Ref &S) {
  Entry E = {ID, S};
  E.Reference.Location.FileURI = UniqueStrings.save(S.Location.FileURI).data();
  Entries.insert(std::move(E));
}

RefSlab RefSlab::Builder::build() && {
  std::vector<std::pair<SymbolID, llvm::ArrayRef<Ref>>> Result;
  // We'll reuse the arena, as it only has unique strings and we need them all.
  // We need to group refs by symbol and form contiguous arrays on the arena.
  std::vector<std::pair<SymbolID, const Ref *>> Flat;
  Flat.reserve(Entries.size());
  for (const Entry &E : Entries)
    Flat.emplace_back(E.Symbol, &E.Reference);
  // Group by SymbolID.
  llvm::sort(Flat, llvm::less_first());
  std::vector<Ref> Refs;
  // Loop over symbols, copying refs for each onto the arena.
  for (auto I = Flat.begin(), End = Flat.end(); I != End;) {
    SymbolID Sym = I->first;
    Refs.clear();
    do {
      Refs.push_back(*I->second);
      ++I;
    } while (I != End && I->first == Sym);
    llvm::sort(Refs); // By file, affects xrefs display order.
    Result.emplace_back(Sym, llvm::ArrayRef<Ref>(Refs).copy(Arena));
  }
  return RefSlab(std::move(Result), std::move(Arena), Entries.size());
}

} // namespace clangd
} // namespace clang
