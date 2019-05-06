//===--- Quality.h - Ranking alternatives for ambiguous queries --*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Some operations such as code completion produce a set of candidates.
/// Usually the user can choose between them, but we should put the best options
/// at the top (they're easier to select, and more likely to be seen).
///
/// This file defines building blocks for ranking candidates.
/// It's used by the features directly and also in the implementation of
/// indexes, as indexes also need to heuristically limit their results.
///
/// The facilities here are:
///   - retrieving scoring signals from e.g. indexes, AST, CodeCompletionString
///     These are structured in a way that they can be debugged, and are fairly
///     consistent regardless of the source.
///   - compute scores from scoring signals. These are suitable for sorting.
///   - sorting utilities like the TopN container.
/// These could be split up further to isolate dependencies if we care.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_QUALITY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_QUALITY_H

#include "ExpectedTypes.h"
#include "FileDistance.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include <algorithm>
#include <functional>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace clang {
class CodeCompletionResult;

namespace clangd {

struct Symbol;
class URIDistance;

// Signals structs are designed to be aggregated from 0 or more sources.
// A default instance has neutral signals, and sources are merged into it.
// They can be dumped for debugging, and evaluate()d into a score.

/// Attributes of a symbol that affect how much we like it.
struct SymbolQualitySignals {
  bool Deprecated = false;
  bool ReservedName = false; // __foo, _Foo are usually implementation details.
                             // FIXME: make these findable once user types _.
  bool ImplementationDetail = false;
  unsigned References = 0;

  enum SymbolCategory {
    Unknown = 0,
    Variable,
    Macro,
    Type,
    Function,
    Constructor,
    Destructor,
    Namespace,
    Keyword,
    Operator,
  } Category = Unknown;

  void merge(const CodeCompletionResult &SemaCCResult);
  void merge(const Symbol &IndexResult);

  // Condense these signals down to a single number, higher is better.
  float evaluate() const;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                              const SymbolQualitySignals &);

/// Attributes of a symbol-query pair that affect how much we like it.
struct SymbolRelevanceSignals {
  /// The name of the symbol (for ContextWords). Must be explicitly assigned.
  llvm::StringRef Name;
  /// 0-1+ fuzzy-match score for unqualified name. Must be explicitly assigned.
  float NameMatch = 1;
  /// Lowercase words relevant to the context (e.g. near the completion point).
  llvm::StringSet<>* ContextWords = nullptr;
  bool Forbidden = false; // Unavailable (e.g const) or inaccessible (private).
  /// Whether fixits needs to be applied for that completion or not.
  bool NeedsFixIts = false;
  bool InBaseClass = false; // A member from base class of the accessed class.

  URIDistance *FileProximityMatch = nullptr;
  /// These are used to calculate proximity between the index symbol and the
  /// query.
  llvm::StringRef SymbolURI;
  /// FIXME: unify with index proximity score - signals should be
  /// source-independent.
  /// Proximity between best declaration and the query. [0-1], 1 is closest.
  float SemaFileProximityScore = 0;

  // Scope proximity is only considered (both index and sema) when this is set.
  ScopeDistance *ScopeProximityMatch = nullptr;
  llvm::Optional<llvm::StringRef> SymbolScope;
  // A symbol from sema should be accessible from the current scope.
  bool SemaSaysInScope = false;

  // An approximate measure of where we expect the symbol to be used.
  enum AccessibleScope {
    FunctionScope,
    ClassScope,
    FileScope,
    GlobalScope,
  } Scope = GlobalScope;

  enum QueryType {
    CodeComplete,
    Generic,
  } Query = Generic;

  CodeCompletionContext::Kind Context = CodeCompletionContext::CCC_Other;

  // Whether symbol is an instance member of a class.
  bool IsInstanceMember = false;

  // Whether clang provided a preferred type in the completion context.
  bool HadContextType = false;
  // Whether a source completion item or a symbol had a type information.
  bool HadSymbolType = false;
  // Whether the item matches the type expected in the completion context.
  bool TypeMatchesPreferred = false;

  void merge(const CodeCompletionResult &SemaResult);
  void merge(const Symbol &IndexResult);

  // Condense these signals down to a single number, higher is better.
  float evaluate() const;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                              const SymbolRelevanceSignals &);

/// Combine symbol quality and relevance into a single score.
float evaluateSymbolAndRelevance(float SymbolQuality, float SymbolRelevance);

/// TopN<T> is a lossy container that preserves only the "best" N elements.
template <typename T, typename Compare = std::greater<T>> class TopN {
public:
  using value_type = T;
  TopN(size_t N, Compare Greater = Compare())
      : N(N), Greater(std::move(Greater)) {}

  // Adds a candidate to the set.
  // Returns true if a candidate was dropped to get back under N.
  bool push(value_type &&V) {
    bool Dropped = false;
    if (Heap.size() >= N) {
      Dropped = true;
      if (N > 0 && Greater(V, Heap.front())) {
        std::pop_heap(Heap.begin(), Heap.end(), Greater);
        Heap.back() = std::move(V);
        std::push_heap(Heap.begin(), Heap.end(), Greater);
      }
    } else {
      Heap.push_back(std::move(V));
      std::push_heap(Heap.begin(), Heap.end(), Greater);
    }
    assert(Heap.size() <= N);
    assert(std::is_heap(Heap.begin(), Heap.end(), Greater));
    return Dropped;
  }

  // Returns candidates from best to worst.
  std::vector<value_type> items() && {
    std::sort_heap(Heap.begin(), Heap.end(), Greater);
    assert(Heap.size() <= N);
    return std::move(Heap);
  }

private:
  const size_t N;
  std::vector<value_type> Heap; // Min-heap, comparator is Greater.
  Compare Greater;
};

/// Returns a string that sorts in the same order as (-Score, Tiebreak), for
/// LSP. (The highest score compares smallest so it sorts at the top).
std::string sortText(float Score, llvm::StringRef Tiebreak = "");

struct SignatureQualitySignals {
  uint32_t NumberOfParameters = 0;
  uint32_t NumberOfOptionalParameters = 0;
  bool ContainsActiveParameter = false;
  CodeCompleteConsumer::OverloadCandidate::CandidateKind Kind =
      CodeCompleteConsumer::OverloadCandidate::CandidateKind::CK_Function;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                              const SignatureQualitySignals &);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_QUALITY_H
