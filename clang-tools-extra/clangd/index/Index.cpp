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
using namespace llvm;

SymbolID::SymbolID(StringRef USR)
    : HashValue(SHA1::hash(arrayRefFromStringRef(USR))) {}

raw_ostream &operator<<(raw_ostream &OS, const SymbolID &ID) {
  OS << toHex(toStringRef(ID.HashValue));
  return OS;
}

void operator>>(StringRef Str, SymbolID &ID) {
  std::string HexString = fromHex(Str);
  assert(HexString.size() == ID.HashValue.size());
  std::copy(HexString.begin(), HexString.end(), ID.HashValue.begin());
}

SymbolSlab::const_iterator SymbolSlab::find(const SymbolID &ID) const {
  auto It = std::lower_bound(Symbols.begin(), Symbols.end(), ID,
                             [](const Symbol &S, const SymbolID &I) {
                               return S.ID < I;
                             });
  if (It != Symbols.end() && It->ID == ID)
    return It;
  return Symbols.end();
}

// Copy the underlying data of the symbol into the owned arena.
static void own(Symbol &S, DenseSet<StringRef> &Strings,
                BumpPtrAllocator &Arena) {
  // Intern replaces V with a reference to the same string owned by the arena.
  auto Intern = [&](StringRef &V) {
    auto R = Strings.insert(V);
    if (R.second) { // New entry added to the table, copy the string.
      *R.first = V.copy(Arena);
    }
    V = *R.first;
  };

  // We need to copy every StringRef field onto the arena.
  Intern(S.Name);
  Intern(S.Scope);
  Intern(S.CanonicalDeclaration.FilePath);

  Intern(S.CompletionLabel);
  Intern(S.CompletionFilterText);
  Intern(S.CompletionPlainInsertText);
  Intern(S.CompletionSnippetInsertText);

  if (S.Detail) {
    // Copy values of StringRefs into arena.
    auto *Detail = Arena.Allocate<Symbol::Details>();
    Detail->Documentation = S.Detail->Documentation;
    Detail->CompletionDetail = S.Detail->CompletionDetail;
    S.Detail = Detail;

    // Intern the actual strings.
    Intern(S.Detail->Documentation);
    Intern(S.Detail->CompletionDetail);
  }
}

void SymbolSlab::Builder::insert(const Symbol &S) {
  auto R = SymbolIndex.try_emplace(S.ID, Symbols.size());
  if (R.second) {
    Symbols.push_back(S);
    own(Symbols.back(), Strings, Arena);
  } else {
    auto &Copy = Symbols[R.first->second] = S;
    own(Copy, Strings, Arena);
  }
}

SymbolSlab SymbolSlab::Builder::build() && {
  Symbols = {Symbols.begin(), Symbols.end()}; // Force shrink-to-fit.
  // Sort symbols so the slab can binary search over them.
  std::sort(Symbols.begin(), Symbols.end(),
            [](const Symbol &L, const Symbol &R) { return L.ID < R.ID; });
  // We may have unused strings from overwritten symbols. Build a new arena.
  BumpPtrAllocator NewArena;
  DenseSet<StringRef> Strings;
  for (auto &S : Symbols)
    own(S, Strings, NewArena);
  return SymbolSlab(std::move(NewArena), std::move(Symbols));
}

} // namespace clangd
} // namespace clang
