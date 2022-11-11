//===--- Dex.cpp - Dex Symbol Index Implementation --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dex.h"
#include "FileDistance.h"
#include "FuzzyMatch.h"
#include "Quality.h"
#include "URI.h"
#include "index/Index.h"
#include "index/dex/Iterator.h"
#include "index/dex/Token.h"
#include "index/dex/Trigram.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace dex {

std::unique_ptr<SymbolIndex> Dex::build(SymbolSlab Symbols, RefSlab Refs,
                                        RelationSlab Rels) {
  auto Size = Symbols.bytes() + Refs.bytes();
  // There is no need to include "Rels" in Data because the relations are self-
  // contained, without references into a backing store.
  auto Data = std::make_pair(std::move(Symbols), std::move(Refs));
  return std::make_unique<Dex>(Data.first, Data.second, Rels, std::move(Data),
                                Size);
}

namespace {

// Mark symbols which are can be used for code completion.
const Token RestrictedForCodeCompletion =
    Token(Token::Kind::Sentinel, "Restricted For Code Completion");

// Helper to efficiently assemble the inverse index (token -> matching docs).
// The output is a nice uniform structure keyed on Token, but constructing
// the Token object every time we want to insert into the map is wasteful.
// Instead we have various maps keyed on things that are cheap to compute,
// and produce the Token keys once at the end.
class IndexBuilder {
  llvm::DenseMap<Trigram, std::vector<DocID>> TrigramDocs;
  std::vector<DocID> RestrictedCCDocs;
  llvm::StringMap<std::vector<DocID>> TypeDocs;
  llvm::StringMap<std::vector<DocID>> ScopeDocs;
  llvm::StringMap<std::vector<DocID>> ProximityDocs;
  std::vector<Trigram> TrigramScratch;

public:
  // Add the tokens which are given symbol's characteristics.
  // This includes fuzzy matching trigrams, symbol's scope, etc.
  // FIXME(kbobyrev): Support more token types:
  // * Namespace proximity
  void add(const Symbol &Sym, DocID D) {
    generateIdentifierTrigrams(Sym.Name, TrigramScratch);
    for (Trigram T : TrigramScratch)
      TrigramDocs[T].push_back(D);
    ScopeDocs[Sym.Scope].push_back(D);
    if (!llvm::StringRef(Sym.CanonicalDeclaration.FileURI).empty())
      for (const auto &ProximityURI :
           generateProximityURIs(Sym.CanonicalDeclaration.FileURI))
        ProximityDocs[ProximityURI].push_back(D);
    if (Sym.Flags & Symbol::IndexedForCodeCompletion)
      RestrictedCCDocs.push_back(D);
    if (!Sym.Type.empty())
      TypeDocs[Sym.Type].push_back(D);
  }

  // Assemble the final compressed posting lists for the added symbols.
  llvm::DenseMap<Token, PostingList> build() && {
    llvm::DenseMap<Token, PostingList> Result(/*InitialReserve=*/
                                              TrigramDocs.size() +
                                              RestrictedCCDocs.size() +
                                              TypeDocs.size() +
                                              ScopeDocs.size() +
                                              ProximityDocs.size());
    // Tear down intermediate structs as we go to reduce memory usage.
    // Since we're trying to get rid of underlying allocations, clearing the
    // containers is not enough.
    auto CreatePostingList =
        [&Result](Token::Kind TK, llvm::StringMap<std::vector<DocID>> &Docs) {
          for (auto &E : Docs) {
            Result.try_emplace(Token(TK, E.first()), E.second);
            E.second = {};
          }
          Docs = {};
        };
    CreatePostingList(Token::Kind::Type, TypeDocs);
    CreatePostingList(Token::Kind::Scope, ScopeDocs);
    CreatePostingList(Token::Kind::ProximityURI, ProximityDocs);

    // TrigramDocs are stored in a DenseMap and RestrictedCCDocs is not even a
    // map, treat them specially.
    for (auto &E : TrigramDocs) {
      Result.try_emplace(Token(Token::Kind::Trigram, E.first.str()), E.second);
      E.second = {};
    }
    TrigramDocs = llvm::DenseMap<Trigram, std::vector<DocID>>{};
    if (!RestrictedCCDocs.empty())
      Result.try_emplace(RestrictedForCodeCompletion,
                         std::move(RestrictedCCDocs));
    return Result;
  }
};

} // namespace

void Dex::buildIndex() {
  this->Corpus = dex::Corpus(Symbols.size());
  std::vector<std::pair<float, const Symbol *>> ScoredSymbols(Symbols.size());

  for (size_t I = 0; I < Symbols.size(); ++I) {
    const Symbol *Sym = Symbols[I];
    LookupTable[Sym->ID] = Sym;
    ScoredSymbols[I] = {quality(*Sym), Sym};
  }

  // Symbols are sorted by symbol qualities so that items in the posting lists
  // are stored in the descending order of symbol quality.
  llvm::sort(ScoredSymbols, std::greater<std::pair<float, const Symbol *>>());

  // SymbolQuality was empty up until now.
  SymbolQuality.resize(Symbols.size());
  // Populate internal storage using Symbol + Score pairs.
  for (size_t I = 0; I < ScoredSymbols.size(); ++I) {
    SymbolQuality[I] = ScoredSymbols[I].first;
    Symbols[I] = ScoredSymbols[I].second;
  }

  // Build posting lists for symbols.
  IndexBuilder Builder;
  for (DocID SymbolRank = 0; SymbolRank < Symbols.size(); ++SymbolRank)
    Builder.add(*Symbols[SymbolRank], SymbolRank);
  InvertedIndex = std::move(Builder).build();
}

std::unique_ptr<Iterator> Dex::iterator(const Token &Tok) const {
  auto It = InvertedIndex.find(Tok);
  return It == InvertedIndex.end() ? Corpus.none()
                                   : It->second.iterator(&It->first);
}

// Constructs BOOST iterators for Path Proximities.
std::unique_ptr<Iterator> Dex::createFileProximityIterator(
    llvm::ArrayRef<std::string> ProximityPaths) const {
  std::vector<std::unique_ptr<Iterator>> BoostingIterators;
  // Deduplicate parent URIs extracted from the ProximityPaths.
  llvm::StringSet<> ParentURIs;
  llvm::StringMap<SourceParams> Sources;
  for (const auto &Path : ProximityPaths) {
    Sources[Path] = SourceParams();
    auto PathURI = URI::create(Path).toString();
    const auto PathProximityURIs = generateProximityURIs(PathURI.c_str());
    for (const auto &ProximityURI : PathProximityURIs)
      ParentURIs.insert(ProximityURI);
  }
  // Use SymbolRelevanceSignals for symbol relevance evaluation: use defaults
  // for all parameters except for Proximity Path distance signal.
  SymbolRelevanceSignals PathProximitySignals;
  // DistanceCalculator will find the shortest distance from ProximityPaths to
  // any URI extracted from the ProximityPaths.
  URIDistance DistanceCalculator(Sources);
  PathProximitySignals.FileProximityMatch = &DistanceCalculator;
  // Try to build BOOST iterator for each Proximity Path provided by
  // ProximityPaths. Boosting factor should depend on the distance to the
  // Proximity Path: the closer processed path is, the higher boosting factor.
  for (const auto &ParentURI : ParentURIs.keys()) {
    // FIXME(kbobyrev): Append LIMIT on top of every BOOST iterator.
    auto It = iterator(Token(Token::Kind::ProximityURI, ParentURI));
    if (It->kind() != Iterator::Kind::False) {
      PathProximitySignals.SymbolURI = ParentURI;
      BoostingIterators.push_back(Corpus.boost(
          std::move(It), PathProximitySignals.evaluateHeuristics()));
    }
  }
  BoostingIterators.push_back(Corpus.all());
  return Corpus.unionOf(std::move(BoostingIterators));
}

// Constructs BOOST iterators for preferred types.
std::unique_ptr<Iterator>
Dex::createTypeBoostingIterator(llvm::ArrayRef<std::string> Types) const {
  std::vector<std::unique_ptr<Iterator>> BoostingIterators;
  SymbolRelevanceSignals PreferredTypeSignals;
  PreferredTypeSignals.TypeMatchesPreferred = true;
  auto Boost = PreferredTypeSignals.evaluateHeuristics();
  for (const auto &T : Types)
    BoostingIterators.push_back(
        Corpus.boost(iterator(Token(Token::Kind::Type, T)), Boost));
  BoostingIterators.push_back(Corpus.all());
  return Corpus.unionOf(std::move(BoostingIterators));
}

/// Constructs iterators over tokens extracted from the query and exhausts it
/// while applying Callback to each symbol in the order of decreasing quality
/// of the matched symbols.
bool Dex::fuzzyFind(const FuzzyFindRequest &Req,
                    llvm::function_ref<void(const Symbol &)> Callback) const {
  assert(!StringRef(Req.Query).contains("::") &&
         "There must be no :: in query.");
  trace::Span Tracer("Dex fuzzyFind");
  FuzzyMatcher Filter(Req.Query);
  // For short queries we use specialized trigrams that don't yield all results.
  // Prevent clients from postfiltering them for longer queries.
  bool More = !Req.Query.empty() && Req.Query.size() < 3;

  std::vector<std::unique_ptr<Iterator>> Criteria;
  const auto TrigramTokens = generateQueryTrigrams(Req.Query);

  // Generate query trigrams and construct AND iterator over all query
  // trigrams.
  std::vector<std::unique_ptr<Iterator>> TrigramIterators;
  for (const auto &Trigram : TrigramTokens)
    TrigramIterators.push_back(iterator(Trigram));
  Criteria.push_back(Corpus.intersect(std::move(TrigramIterators)));

  // Generate scope tokens for search query.
  std::vector<std::unique_ptr<Iterator>> ScopeIterators;
  for (const auto &Scope : Req.Scopes)
    ScopeIterators.push_back(iterator(Token(Token::Kind::Scope, Scope)));
  if (Req.AnyScope)
    ScopeIterators.push_back(
        Corpus.boost(Corpus.all(), ScopeIterators.empty() ? 1.0 : 0.2));
  Criteria.push_back(Corpus.unionOf(std::move(ScopeIterators)));

  // Add proximity paths boosting (all symbols, some boosted).
  Criteria.push_back(createFileProximityIterator(Req.ProximityPaths));
  // Add boosting for preferred types.
  Criteria.push_back(createTypeBoostingIterator(Req.PreferredTypes));

  if (Req.RestrictForCodeCompletion)
    Criteria.push_back(iterator(RestrictedForCodeCompletion));

  // Use TRUE iterator if both trigrams and scopes from the query are not
  // present in the symbol index.
  auto Root = Corpus.intersect(std::move(Criteria));
  // Retrieve more items than it was requested: some of  the items with high
  // final score might not be retrieved otherwise.
  // FIXME(kbobyrev): Tune this ratio.
  if (Req.Limit)
    Root = Corpus.limit(std::move(Root), *Req.Limit * 100);
  SPAN_ATTACH(Tracer, "query", llvm::to_string(*Root));
  vlog("Dex query tree: {0}", *Root);

  using IDAndScore = std::pair<DocID, float>;
  std::vector<IDAndScore> IDAndScores = consume(*Root);

  auto Compare = [](const IDAndScore &LHS, const IDAndScore &RHS) {
    return LHS.second > RHS.second;
  };
  TopN<IDAndScore, decltype(Compare)> Top(
      Req.Limit ? *Req.Limit : std::numeric_limits<size_t>::max(), Compare);
  for (const auto &IDAndScore : IDAndScores) {
    const DocID SymbolDocID = IDAndScore.first;
    const auto *Sym = Symbols[SymbolDocID];
    const llvm::Optional<float> Score = Filter.match(Sym->Name);
    if (!Score)
      continue;
    // Combine Fuzzy Matching score, precomputed symbol quality and boosting
    // score for a cumulative final symbol score.
    const float FinalScore =
        (*Score) * SymbolQuality[SymbolDocID] * IDAndScore.second;
    // If Top.push(...) returns true, it means that it had to pop an item. In
    // this case, it is possible to retrieve more symbols.
    if (Top.push({SymbolDocID, FinalScore}))
      More = true;
  }

  // Apply callback to the top Req.Limit items in the descending
  // order of cumulative score.
  for (const auto &Item : std::move(Top).items())
    Callback(*Symbols[Item.first]);
  return More;
}

void Dex::lookup(const LookupRequest &Req,
                 llvm::function_ref<void(const Symbol &)> Callback) const {
  trace::Span Tracer("Dex lookup");
  for (const auto &ID : Req.IDs) {
    auto I = LookupTable.find(ID);
    if (I != LookupTable.end())
      Callback(*I->second);
  }
}

bool Dex::refs(const RefsRequest &Req,
               llvm::function_ref<void(const Ref &)> Callback) const {
  trace::Span Tracer("Dex refs");
  uint32_t Remaining = Req.Limit.value_or(std::numeric_limits<uint32_t>::max());
  for (const auto &ID : Req.IDs)
    for (const auto &Ref : Refs.lookup(ID)) {
      if (!static_cast<int>(Req.Filter & Ref.Kind))
        continue;
      if (Remaining == 0)
        return true; // More refs were available.
      --Remaining;
      Callback(Ref);
    }
  return false; // We reported all refs.
}

void Dex::relations(
    const RelationsRequest &Req,
    llvm::function_ref<void(const SymbolID &, const Symbol &)> Callback) const {
  trace::Span Tracer("Dex relations");
  uint32_t Remaining = Req.Limit.value_or(std::numeric_limits<uint32_t>::max());
  for (const SymbolID &Subject : Req.Subjects) {
    LookupRequest LookupReq;
    auto It = Relations.find(
        std::make_pair(Subject, static_cast<uint8_t>(Req.Predicate)));
    if (It != Relations.end()) {
      for (const auto &Object : It->second) {
        if (Remaining > 0) {
          --Remaining;
          LookupReq.IDs.insert(Object);
        }
      }
    }
    lookup(LookupReq, [&](const Symbol &Object) { Callback(Subject, Object); });
  }
}

llvm::unique_function<IndexContents(llvm::StringRef) const>
Dex::indexedFiles() const {
  return [this](llvm::StringRef FileURI) {
    return Files.contains(FileURI) ? IdxContents : IndexContents::None;
  };
}

size_t Dex::estimateMemoryUsage() const {
  size_t Bytes = Symbols.size() * sizeof(const Symbol *);
  Bytes += SymbolQuality.size() * sizeof(float);
  Bytes += LookupTable.getMemorySize();
  Bytes += InvertedIndex.getMemorySize();
  for (const auto &TokenToPostingList : InvertedIndex)
    Bytes += TokenToPostingList.second.bytes();
  Bytes += Refs.getMemorySize();
  Bytes += Relations.getMemorySize();
  return Bytes + BackingDataSize;
}

// Given foo://bar/one/two
// Returns        ~~~~~~~~  (or empty for bad URI)
llvm::StringRef findPathInURI(llvm::StringRef S) {
  S = S.split(':').second;   // Eat scheme.
  if (S.consume_front("//")) // Eat authority.
    S = S.drop_until([](char C) { return C == '/'; });
  return S;
}

// FIXME(kbobyrev): Currently, this is a heuristic which defines the maximum
// size of resulting vector. Some projects might want to have higher limit if
// the file hierarchy is deeper. For the generic case, it would be useful to
// calculate Limit in the index build stage by calculating the maximum depth
// of the project source tree at runtime.
constexpr unsigned ProximityURILimit = 5;

llvm::SmallVector<llvm::StringRef, ProximityURILimit>
generateProximityURIs(llvm::StringRef URI) {
  // This function is hot when indexing, so don't parse/reserialize URIPath,
  // just emit substrings of it instead.
  //
  // foo://bar/one/two
  //          ~~~~~~~~
  //            Path
  llvm::StringRef Path = findPathInURI(URI);
  if (Path.empty())
    return {}; // Bad URI.
  assert(Path.begin() >= URI.begin() && Path.begin() < URI.end() &&
         Path.end() == URI.end());

  // The original is a proximity URI.
  llvm::SmallVector<llvm::StringRef, ProximityURILimit> Result = {URI};
  // Form proximity URIs by chopping before each slash in the path part.
  for (auto Slash = Path.rfind('/'); Slash > 0 && Slash != StringRef::npos;
       Slash = Path.rfind('/')) {
    Path = Path.substr(0, Slash);
    Result.push_back(URI.substr(0, Path.end() - URI.data()));
    if (Result.size() == ProximityURILimit)
      return Result;
  }
  // The root foo://bar/ is a proximity URI.
  if (Path.startswith("/"))
    Result.push_back(URI.substr(0, Path.begin() + 1 - URI.data()));
  return Result;
}

} // namespace dex
} // namespace clangd
} // namespace clang
