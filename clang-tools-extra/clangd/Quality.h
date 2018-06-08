//===--- Quality.h - Ranking alternatives for ambiguous queries -*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
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
//===---------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_QUALITY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_QUALITY_H
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <functional>
#include <vector>
namespace llvm {
class raw_ostream;
}
namespace clang {
class CodeCompletionResult;
namespace clangd {
struct Symbol;

// Signals structs are designed to be aggregated from 0 or more sources.
// A default instance has neutral signals, and sources are merged into it.
// They can be dumped for debugging, and evaluate()d into a score.

/// Attributes of a symbol that affect how much we like it.
struct SymbolQualitySignals {
  bool Deprecated = false;
  bool ReservedName = false; // __foo, _Foo are usually implementation details.
                             // FIXME: make these findable once user types _.
  unsigned References = 0;

  enum SymbolCategory {
    Variable,
    Macro,
    Type,
    Function,
    Namespace,
    Unknown,
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
  /// 0-1+ fuzzy-match score for unqualified name. Must be explicitly assigned.
  float NameMatch = 1;
  bool Forbidden = false; // Unavailable (e.g const) or inaccessible (private).
  /// Proximity between best declaration and the query. [0-1], 1 is closest.
  float ProximityScore = 0;

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

} // namespace clangd
} // namespace clang

#endif
