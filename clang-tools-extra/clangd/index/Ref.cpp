//===--- Ref.cpp -------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ref.h"

namespace clang {
namespace clangd {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, RefKind K) {
  if (K == RefKind::Unknown)
    return OS << "Unknown";
  static const std::vector<const char *> Messages = {"Decl", "Def", "Ref"};
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
  auto &M = Refs[ID];
  if (M.count(S))
    return;
  Ref R = S;
  R.Location.FileURI =
      UniqueStrings.save(R.Location.FileURI).data();
  M.insert(std::move(R));
}

RefSlab RefSlab::Builder::build() && {
  // We can reuse the arena, as it only has unique strings and we need them all.
  // Reallocate refs on the arena to reduce waste and indirections when reading.
  std::vector<std::pair<SymbolID, llvm::ArrayRef<Ref>>> Result;
  Result.reserve(Refs.size());
  size_t NumRefs = 0;
  for (auto &Sym : Refs) {
    std::vector<Ref> SymRefs(Sym.second.begin(), Sym.second.end());
    NumRefs += SymRefs.size();
    Result.emplace_back(Sym.first, llvm::ArrayRef<Ref>(SymRefs).copy(Arena));
  }
  return RefSlab(std::move(Result), std::move(Arena), NumRefs);
}

} // namespace clangd
} // namespace clang
