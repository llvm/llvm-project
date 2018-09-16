//===--- PostingList.cpp - Symbol identifiers storage interface -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PostingList.h"
#include "Iterator.h"

namespace clang {
namespace clangd {
namespace dex {

namespace {

/// Implements Iterator over std::vector<DocID>. This is the most basic
/// iterator and is simply a wrapper around
/// std::vector<DocID>::const_iterator.
class PlainIterator : public Iterator {
public:
  explicit PlainIterator(llvm::ArrayRef<DocID> Documents)
      : Documents(Documents), Index(std::begin(Documents)) {}

  bool reachedEnd() const override { return Index == std::end(Documents); }

  /// Advances cursor to the next item.
  void advance() override {
    assert(!reachedEnd() &&
           "Posting List iterator can't advance() at the end.");
    ++Index;
  }

  /// Applies binary search to advance cursor to the next item with DocID
  /// equal or higher than the given one.
  void advanceTo(DocID ID) override {
    assert(!reachedEnd() &&
           "Posting List iterator can't advance() at the end.");
    // If current ID is beyond requested one, iterator is already in the right
    // state.
    if (peek() < ID)
      Index = std::lower_bound(Index, std::end(Documents), ID);
  }

  DocID peek() const override {
    assert(!reachedEnd() &&
           "Posting List iterator can't peek() at the end.");
    return *Index;
  }

  float consume() override {
    assert(!reachedEnd() &&
           "Posting List iterator can't consume() at the end.");
    return DEFAULT_BOOST_SCORE;
  }

  size_t estimateSize() const override { return Documents.size(); }

private:
  llvm::raw_ostream &dump(llvm::raw_ostream &OS) const override {
    OS << '[';
    if (Index != std::end(Documents))
      OS << *Index;
    else
      OS << "END";
    OS << ']';
    return OS;
  }

  llvm::ArrayRef<DocID> Documents;
  llvm::ArrayRef<DocID>::const_iterator Index;
};

} // namespace

std::unique_ptr<Iterator> PostingList::iterator() const {
  return llvm::make_unique<PlainIterator>(Documents);
}

} // namespace dex
} // namespace clangd
} // namespace clang
