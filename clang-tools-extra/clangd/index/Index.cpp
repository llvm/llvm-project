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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
using namespace llvm;

raw_ostream &operator<<(raw_ostream &OS, const SymbolLocation &L) {
  if (!L)
    return OS << "(none)";
  return OS << L.FileURI << "[" << L.Start.Line << ":" << L.Start.Column << "-"
            << L.End.Line << ":" << L.End.Column << ")";
}

SymbolID::SymbolID(StringRef USR)
    : HashValue(SHA1::hash(arrayRefFromStringRef(USR))) {}

raw_ostream &operator<<(raw_ostream &OS, const SymbolID &ID) {
  OS << toHex(toStringRef(ID.HashValue));
  return OS;
}

std::string SymbolID::str() const {
  std::string ID;
  llvm::raw_string_ostream OS(ID);
  OS << *this;
  return OS.str();
}

void operator>>(StringRef Str, SymbolID &ID) {
  std::string HexString = fromHex(Str);
  assert(HexString.size() == ID.HashValue.size());
  std::copy(HexString.begin(), HexString.end(), ID.HashValue.begin());
}

raw_ostream &operator<<(raw_ostream &OS, SymbolOrigin O) {
  if (O == SymbolOrigin::Unknown)
    return OS << "unknown";
  constexpr static char Sigils[] = "ADSM4567";
  for (unsigned I = 0; I < sizeof(Sigils); ++I)
    if (static_cast<uint8_t>(O) & 1u << I)
      OS << Sigils[I];
  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const Symbol &S) {
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
  auto It = std::lower_bound(Symbols.begin(), Symbols.end(), ID,
                             [](const Symbol &S, const SymbolID &I) {
                               return S.ID < I;
                             });
  if (It != Symbols.end() && It->ID == ID)
    return It;
  return Symbols.end();
}

// Copy the underlying data of the symbol into the owned arena.
static void own(Symbol &S, llvm::UniqueStringSaver &Strings,
                BumpPtrAllocator &Arena) {
  // Intern replaces V with a reference to the same string owned by the arena.
  auto Intern = [&](StringRef &V) { V = Strings.save(V); };

  // We need to copy every StringRef field onto the arena.
  Intern(S.Name);
  Intern(S.Scope);
  Intern(S.CanonicalDeclaration.FileURI);
  Intern(S.Definition.FileURI);

  Intern(S.Signature);
  Intern(S.CompletionSnippetSuffix);

  Intern(S.Documentation);
  Intern(S.ReturnType);
  for (auto &I : S.IncludeHeaders)
    Intern(I.IncludeHeader);
}

void SymbolSlab::Builder::insert(const Symbol &S) {
  auto R = SymbolIndex.try_emplace(S.ID, Symbols.size());
  if (R.second) {
    Symbols.push_back(S);
    own(Symbols.back(), UniqueStrings, Arena);
  } else {
    auto &Copy = Symbols[R.first->second] = S;
    own(Copy, UniqueStrings, Arena);
  }
}

SymbolSlab SymbolSlab::Builder::build() && {
  Symbols = {Symbols.begin(), Symbols.end()}; // Force shrink-to-fit.
  // Sort symbols so the slab can binary search over them.
  std::sort(Symbols.begin(), Symbols.end(),
            [](const Symbol &L, const Symbol &R) { return L.ID < R.ID; });
  // We may have unused strings from overwritten symbols. Build a new arena.
  BumpPtrAllocator NewArena;
  llvm::UniqueStringSaver Strings(NewArena);
  for (auto &S : Symbols)
    own(S, Strings, NewArena);
  return SymbolSlab(std::move(NewArena), std::move(Symbols));
}

raw_ostream &operator<<(raw_ostream &OS, RefKind K) {
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
  M.push_back(S);
  M.back().Location.FileURI = UniqueStrings.save(M.back().Location.FileURI);
}

RefSlab RefSlab::Builder::build() && {
  // We can reuse the arena, as it only has unique strings and we need them all.
  // Reallocate refs on the arena to reduce waste and indirections when reading.
  std::vector<std::pair<SymbolID, ArrayRef<Ref>>> Result;
  Result.reserve(Refs.size());
  for (auto &Sym : Refs) {
    auto &SymRefs = Sym.second;
    std::sort(SymRefs.begin(), SymRefs.end());
    // TODO: do we really need to dedup?
    SymRefs.erase(std::unique(SymRefs.begin(), SymRefs.end()), SymRefs.end());

    auto *Array = Arena.Allocate<Ref>(SymRefs.size());
    std::uninitialized_copy(SymRefs.begin(), SymRefs.end(), Array);
    Result.emplace_back(Sym.first, ArrayRef<Ref>(Array, SymRefs.size()));
  }
  return RefSlab(std::move(Result), std::move(Arena));
}

void SwapIndex::reset(std::unique_ptr<SymbolIndex> Index) {
  // Keep the old index alive, so we don't destroy it under lock (may be slow).
  std::shared_ptr<SymbolIndex> Pin;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    Pin = std::move(this->Index);
    this->Index = std::move(Index);
  }
}
std::shared_ptr<SymbolIndex> SwapIndex::snapshot() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return Index;
}

bool SwapIndex::fuzzyFind(const FuzzyFindRequest &R,
                          llvm::function_ref<void(const Symbol &)> CB) const {
  return snapshot()->fuzzyFind(R, CB);
}
void SwapIndex::lookup(const LookupRequest &R,
                       llvm::function_ref<void(const Symbol &)> CB) const {
  return snapshot()->lookup(R, CB);
}
void SwapIndex::refs(const RefsRequest &R,
                     llvm::function_ref<void(const Ref &)> CB) const {
  return snapshot()->refs(R, CB);
}
size_t SwapIndex::estimateMemoryUsage() const {
  return snapshot()->estimateMemoryUsage();
}

} // namespace clangd
} // namespace clang
