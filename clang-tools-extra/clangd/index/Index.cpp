//===--- Index.cpp -----------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Index.h"
#include "Logger.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
namespace clang {
namespace clangd {

constexpr uint32_t SymbolLocation::Position::MaxLine;
constexpr uint32_t SymbolLocation::Position::MaxColumn;
void SymbolLocation::Position::setLine(uint32_t L) {
  if (L > MaxLine) {
    Line = MaxLine;
    return;
  }
  Line = L;
}
void SymbolLocation::Position::setColumn(uint32_t Col) {
  if (Col > MaxColumn) {
    Column = MaxColumn;
    return;
  }
  Column = Col;
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolLocation &L) {
  if (!L)
    return OS << "(none)";
  return OS << L.FileURI << "[" << L.Start.line() << ":" << L.Start.column()
            << "-" << L.End.line() << ":" << L.End.column() << ")";
}

SymbolID::SymbolID(StringRef USR) {
  auto Hash = SHA1::hash(arrayRefFromStringRef(USR));
  static_assert(sizeof(Hash) >= RawSize, "RawSize larger than SHA1");
  memcpy(HashValue.data(), Hash.data(), RawSize);
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolID &ID) {
  return OS << toHex(ID.raw());
}

SymbolID SymbolID::fromRaw(StringRef Raw) {
  SymbolID ID;
  assert(Raw.size() == RawSize);
  memcpy(ID.HashValue.data(), Raw.data(), RawSize);
  return ID;
}

std::string SymbolID::str() const { return toHex(raw()); }

Expected<SymbolID> SymbolID::fromStr(StringRef Str) {
  if (Str.size() != RawSize * 2)
    return createStringError(inconvertibleErrorCode(), "Bad ID length");
  for (char C : Str)
    if (!isHexDigit(C))
      return createStringError(inconvertibleErrorCode(), "Bad hex ID");
  return fromRaw(fromHex(Str));
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

raw_ostream &operator<<(raw_ostream &OS, Symbol::SymbolFlag F) {
  if (F == Symbol::None)
    return OS << "None";
  std::string s;
  if (F & Symbol::Deprecated)
    s += "deprecated|";
  if (F & Symbol::IndexedForCodeCompletion)
    s += "completion|";
  return OS << StringRef(s).rtrim('|');
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
  auto It = std::lower_bound(
      Symbols.begin(), Symbols.end(), ID,
      [](const Symbol &S, const SymbolID &I) { return S.ID < I; });
  if (It != Symbols.end() && It->ID == ID)
    return It;
  return Symbols.end();
}

// Copy the underlying data of the symbol into the owned arena.
static void own(Symbol &S, UniqueStringSaver &Strings) {
  visitStrings(S, [&](StringRef &V) { V = Strings.save(V); });
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
  BumpPtrAllocator NewArena;
  UniqueStringSaver Strings(NewArena);
  for (auto &S : Symbols)
    own(S, Strings);
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

raw_ostream &operator<<(raw_ostream &OS, const Ref &R) {
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
  size_t NumRefs = 0;
  for (auto &Sym : Refs) {
    auto &SymRefs = Sym.second;
    llvm::sort(SymRefs);
    // FIXME: do we really need to dedup?
    SymRefs.erase(std::unique(SymRefs.begin(), SymRefs.end()), SymRefs.end());

    NumRefs += SymRefs.size();
    auto *Array = Arena.Allocate<Ref>(SymRefs.size());
    std::uninitialized_copy(SymRefs.begin(), SymRefs.end(), Array);
    Result.emplace_back(Sym.first, ArrayRef<Ref>(Array, SymRefs.size()));
  }
  return RefSlab(std::move(Result), std::move(Arena), NumRefs);
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

bool fromJSON(const json::Value &Parameters, FuzzyFindRequest &Request) {
  json::ObjectMapper O(Parameters);
  int64_t Limit;
  bool OK =
      O && O.map("Query", Request.Query) && O.map("Scopes", Request.Scopes) &&
      O.map("AnyScope", Request.AnyScope) && O.map("Limit", Limit) &&
      O.map("RestrictForCodeCompletion", Request.RestrictForCodeCompletion) &&
      O.map("ProximityPaths", Request.ProximityPaths);
  if (OK && Limit <= std::numeric_limits<uint32_t>::max())
    Request.Limit = Limit;
  return OK;
}

json::Value toJSON(const FuzzyFindRequest &Request) {
  return json::Object{
      {"Query", Request.Query},
      {"Scopes", json::Array{Request.Scopes}},
      {"AnyScope", Request.AnyScope},
      {"Limit", Request.Limit},
      {"RestrictForCodeCompletion", Request.RestrictForCodeCompletion},
      {"ProximityPaths", json::Array{Request.ProximityPaths}},
  };
}

bool SwapIndex::fuzzyFind(const FuzzyFindRequest &R,
                          function_ref<void(const Symbol &)> CB) const {
  return snapshot()->fuzzyFind(R, CB);
}
void SwapIndex::lookup(const LookupRequest &R,
                       function_ref<void(const Symbol &)> CB) const {
  return snapshot()->lookup(R, CB);
}
void SwapIndex::refs(const RefsRequest &R,
                     function_ref<void(const Ref &)> CB) const {
  return snapshot()->refs(R, CB);
}
size_t SwapIndex::estimateMemoryUsage() const {
  return snapshot()->estimateMemoryUsage();
}

} // namespace clangd
} // namespace clang
