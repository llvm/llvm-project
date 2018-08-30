//===--- Iterator.h - Query Symbol Retrieval --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Symbol index queries consist of specific requirements for the requested
// symbol, such as high fuzzy matching score, scope, type etc. The lists of all
// symbols matching some criteria (e.g. belonging to "clang::clangd::" scope)
// are expressed in a form of Search Tokens which are stored in the inverted
// index. Inverted index maps these tokens to the posting lists - sorted (by
// symbol quality) sequences of symbol IDs matching the token, e.g. scope token
// "clangd::clangd::" is mapped to the list of IDs of all symbols which are
// declared in this namespace. Search queries are build from a set of
// requirements which can be combined with each other forming the query trees.
// The leafs of such trees are posting lists, and the nodes are operations on
// these posting lists, e.g. intersection or union. Efficient processing of
// these multi-level queries is handled by Iterators. Iterators advance through
// all leaf posting lists producing the result of search query, which preserves
// the sorted order of IDs. Having the resulting IDs sorted is important,
// because it allows receiving a certain number of the most valuable items (e.g.
// symbols with highest quality which was the sorting key in the first place)
// without processing all items with requested properties (this might not be
// computationally effective if search request is not very restrictive).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_ITERATOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_ITERATOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace clang {
namespace clangd {
namespace dex {

/// Symbol position in the list of all index symbols sorted by a pre-computed
/// symbol quality.
using DocID = uint32_t;
/// Contains sorted sequence of DocIDs all of which belong to symbols matching
/// certain criteria, i.e. containing a Search Token. PostingLists are values
/// for the inverted index.
// FIXME(kbobyrev): Posting lists for incomplete trigrams (one/two symbols) are
// likely to be very dense and hence require special attention so that the index
// doesn't use too much memory. Possible solution would be to construct
// compressed posting lists which consist of ranges of DocIDs instead of
// distinct DocIDs. A special case would be the empty query: for that case
// TrueIterator should be implemented - an iterator which doesn't actually store
// any PostingList within itself, but "contains" all DocIDs in range
// [0, IndexSize).
using PostingList = std::vector<DocID>;
/// Immutable reference to PostingList object.
using PostingListRef = llvm::ArrayRef<DocID>;

/// Iterator is the interface for Query Tree node. The simplest type of Iterator
/// is DocumentIterator which is simply a wrapper around PostingList iterator
/// and serves as the Query Tree leaf. More sophisticated examples of iterators
/// can manage intersection, union of the elements produced by other iterators
/// (their children) to form a multi-level Query Tree. The interface is designed
/// to be extensible in order to support multiple types of iterators.
class Iterator {
  // FIXME(kbobyrev): Implement iterator cost, an estimate of advance() calls
  // before iterator exhaustion.
  // FIXME(kbobyrev): Implement Limit iterator.
public:
  /// Returns true if all valid DocIDs were processed and hence the iterator is
  /// exhausted.
  virtual bool reachedEnd() const = 0;
  /// Moves to next valid DocID. If it doesn't exist, the iterator is exhausted
  /// and proceeds to the END.
  ///
  /// Note: reachedEnd() must be false.
  virtual void advance() = 0;
  /// Moves to the first valid DocID which is equal or higher than given ID. If
  /// it doesn't exist, the iterator is exhausted and proceeds to the END.
  ///
  /// Note: reachedEnd() must be false.
  virtual void advanceTo(DocID ID) = 0;
  /// Returns the current element this iterator points to.
  ///
  /// Note: reachedEnd() must be false.
  virtual DocID peek() const = 0;
  /// Informs the iterator that the current document was consumed, and returns
  /// its boost.
  ///
  /// Note: If this iterator has any child iterators that contain the document,
  /// consume() should be called on those and their boosts incorporated.
  /// consume() must *not* be called on children that don't contain the current
  /// doc.
  virtual float consume() = 0;
  /// Returns an estimate of advance() calls before the iterator is exhausted.
  virtual size_t estimateSize() const = 0;

  virtual ~Iterator() {}

  /// Prints a convenient human-readable iterator representation by recursively
  /// dumping iterators in the following format:
  ///
  /// (Type Child1 Child2 ...)
  ///
  /// Where Type is the iterator type representation: "&" for And, "|" for Or,
  /// ChildN is N-th iterator child. Raw iterators over PostingList are
  /// represented as "[ID1, ID2, ..., {IDN}, ... END]" where IDN is N-th
  /// PostingList entry and the element which is pointed to by the PostingList
  /// iterator is enclosed in {} braces.
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       const Iterator &Iterator) {
    return Iterator.dump(OS);
  }

  constexpr static float DEFAULT_BOOST_SCORE = 1;

private:
  virtual llvm::raw_ostream &dump(llvm::raw_ostream &OS) const = 0;
};

/// Advances the iterator until it is exhausted. Returns pairs of document IDs
/// with the corresponding boosting score.
///
/// Boosting can be seen as a compromise between retrieving too many items and
/// calculating finals score for each of them (which might be very expensive)
/// and not retrieving enough items so that items with very high final score
/// would not be processed. Boosting score is a computationally efficient way
/// to acquire preliminary scores of requested items.
std::vector<std::pair<DocID, float>> consume(Iterator &It);

/// Returns a document iterator over given PostingList.
///
/// DocumentIterator returns DEFAULT_BOOST_SCORE for each processed item.
std::unique_ptr<Iterator> create(PostingListRef Documents);

/// Returns AND Iterator which performs the intersection of the PostingLists of
/// its children.
///
/// consume(): AND Iterator returns the product of Childrens' boosting scores
/// when not exhausted and DEFAULT_BOOST_SCORE otherwise.
std::unique_ptr<Iterator>
createAnd(std::vector<std::unique_ptr<Iterator>> Children);

/// Returns OR Iterator which performs the union of the PostingLists of its
/// children.
///
/// consume(): OR Iterator returns the highest boost value among children
/// pointing to requested item when not exhausted and DEFAULT_BOOST_SCORE
/// otherwise.
std::unique_ptr<Iterator>
createOr(std::vector<std::unique_ptr<Iterator>> Children);

/// Returns TRUE Iterator which iterates over "virtual" PostingList containing
/// all items in range [0, Size) in an efficient manner.
///
/// TRUE returns DEFAULT_BOOST_SCORE for each processed item.
std::unique_ptr<Iterator> createTrue(DocID Size);

/// Returns BOOST iterator which multiplies the score of each item by given
/// factor. Boosting can be used as a computationally inexpensive filtering.
/// Users can return significantly more items using consumeAndBoost() and then
/// trim Top K using retrieval score.
std::unique_ptr<Iterator> createBoost(std::unique_ptr<Iterator> Child,
                                      float Factor);

/// Returns LIMIT iterator, which yields up to N elements of its child iterator.
/// Elements only count towards the limit if they are part of the final result
/// set. Therefore the following iterator (AND (2) (LIMIT (1 2) 1)) yields (2),
/// not ().
std::unique_ptr<Iterator> createLimit(std::unique_ptr<Iterator> Child,
                                      size_t Limit);

/// This allows createAnd(create(...), create(...)) syntax.
template <typename... Args> std::unique_ptr<Iterator> createAnd(Args... args) {
  std::vector<std::unique_ptr<Iterator>> Children;
  populateChildren(Children, args...);
  return createAnd(move(Children));
}

/// This allows createOr(create(...), create(...)) syntax.
template <typename... Args> std::unique_ptr<Iterator> createOr(Args... args) {
  std::vector<std::unique_ptr<Iterator>> Children;
  populateChildren(Children, args...);
  return createOr(move(Children));
}

template <typename HeadT, typename... TailT>
void populateChildren(std::vector<std::unique_ptr<Iterator>> &Children,
                      HeadT &Head, TailT &... Tail) {
  Children.push_back(move(Head));
  populateChildren(Children, Tail...);
}

template <typename HeadT>
void populateChildren(std::vector<std::unique_ptr<Iterator>> &Children,
                      HeadT &Head) {
  Children.push_back(move(Head));
}

} // namespace dex
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_ITERATOR_H
