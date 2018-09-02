//===--- Iterator.cpp - Query Symbol Retrieval ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Iterator.h"
#include <algorithm>
#include <cassert>
#include <numeric>

namespace clang {
namespace clangd {
namespace dex {

namespace {

/// Implements Iterator over a PostingList. DocumentIterator is the most basic
/// iterator: it doesn't have any children (hence it is the leaf of iterator
/// tree) and is simply a wrapper around PostingList::const_iterator.
class DocumentIterator : public Iterator {
public:
  DocumentIterator(PostingListRef Documents)
      : Documents(Documents), Index(std::begin(Documents)) {}

  bool reachedEnd() const override { return Index == std::end(Documents); }

  /// Advances cursor to the next item.
  void advance() override {
    assert(!reachedEnd() && "DOCUMENT iterator can't advance() at the end.");
    ++Index;
  }

  /// Applies binary search to advance cursor to the next item with DocID equal
  /// or higher than the given one.
  void advanceTo(DocID ID) override {
    assert(!reachedEnd() && "DOCUMENT iterator can't advance() at the end.");
    Index = std::lower_bound(Index, std::end(Documents), ID);
  }

  DocID peek() const override {
    assert(!reachedEnd() && "DOCUMENT iterator can't peek() at the end.");
    return *Index;
  }

  float consume() override {
    assert(!reachedEnd() && "DOCUMENT iterator can't consume() at the end.");
    return DEFAULT_BOOST_SCORE;
  }

  size_t estimateSize() const override { return Documents.size(); }

private:
  llvm::raw_ostream &dump(llvm::raw_ostream &OS) const override {
    OS << '[';
    auto Separator = "";
    for (auto It = std::begin(Documents); It != std::end(Documents); ++It) {
      OS << Separator;
      if (It == Index)
        OS << '{' << *It << '}';
      else
        OS << *It;
      Separator = ", ";
    }
    OS << Separator;
    if (Index == std::end(Documents))
      OS << "{END}";
    else
      OS << "END";
    OS << ']';
    return OS;
  }

  PostingListRef Documents;
  PostingListRef::const_iterator Index;
};

/// Implements Iterator over the intersection of other iterators.
///
/// AndIterator iterates through common items among all children. It becomes
/// exhausted as soon as any child becomes exhausted. After each mutation, the
/// iterator restores the invariant: all children must point to the same item.
class AndIterator : public Iterator {
public:
  AndIterator(std::vector<std::unique_ptr<Iterator>> AllChildren)
      : Children(std::move(AllChildren)) {
    assert(!Children.empty() && "AND iterator should have at least one child.");
    // Establish invariants.
    sync();
    // When children are sorted by the estimateSize(), sync() calls are more
    // effective. Each sync() starts with the first child and makes sure all
    // children point to the same element. If any child is "above" the previous
    // ones, the algorithm resets and and advances the children to the next
    // highest element starting from the front. When child iterators in the
    // beginning have smaller estimated size, the sync() will have less restarts
    // and become more effective.
    std::sort(begin(Children), end(Children),
              [](const std::unique_ptr<Iterator> &LHS,
                 const std::unique_ptr<Iterator> &RHS) {
                return LHS->estimateSize() < RHS->estimateSize();
              });
  }

  bool reachedEnd() const override { return ReachedEnd; }

  /// Advances all children to the next common item.
  void advance() override {
    assert(!reachedEnd() && "AND iterator can't advance() at the end.");
    Children.front()->advance();
    sync();
  }

  /// Advances all children to the next common item with DocumentID >= ID.
  void advanceTo(DocID ID) override {
    assert(!reachedEnd() && "AND iterator can't advanceTo() at the end.");
    Children.front()->advanceTo(ID);
    sync();
  }

  DocID peek() const override { return Children.front()->peek(); }

  float consume() override {
    assert(!reachedEnd() && "AND iterator can't consume() at the end.");
    return std::accumulate(
        begin(Children), end(Children), DEFAULT_BOOST_SCORE,
        [&](float Current, const std::unique_ptr<Iterator> &Child) {
          return Current * Child->consume();
        });
  }

  size_t estimateSize() const override {
    return Children.front()->estimateSize();
  }

private:
  llvm::raw_ostream &dump(llvm::raw_ostream &OS) const override {
    OS << "(& ";
    auto Separator = "";
    for (const auto &Child : Children) {
      OS << Separator << *Child;
      Separator = " ";
    }
    OS << ')';
    return OS;
  }

  /// Restores class invariants: each child will point to the same element after
  /// sync.
  void sync() {
    ReachedEnd |= Children.front()->reachedEnd();
    if (ReachedEnd)
      return;
    auto SyncID = Children.front()->peek();
    // Indicates whether any child needs to be advanced to new SyncID.
    bool NeedsAdvance = false;
    do {
      NeedsAdvance = false;
      for (auto &Child : Children) {
        Child->advanceTo(SyncID);
        ReachedEnd |= Child->reachedEnd();
        // If any child reaches end And iterator can not match any other items.
        // In this case, just terminate the process.
        if (ReachedEnd)
          return;
        // If any child goes beyond given ID (i.e. ID is not the common item),
        // all children should be advanced to the next common item.
        if (Child->peek() > SyncID) {
          SyncID = Child->peek();
          NeedsAdvance = true;
        }
      }
    } while (NeedsAdvance);
  }

  /// AndIterator owns its children and ensures that all of them point to the
  /// same element. As soon as one child gets exhausted, AndIterator can no
  /// longer advance and has reached its end.
  std::vector<std::unique_ptr<Iterator>> Children;
  /// Indicates whether any child is exhausted. It is cheaper to maintain and
  /// update the field, rather than traversing the whole subtree in each
  /// reachedEnd() call.
  bool ReachedEnd = false;
};

/// Implements Iterator over the union of other iterators.
///
/// OrIterator iterates through all items which can be pointed to by at least
/// one child. To preserve the sorted order, this iterator always advances the
/// child with smallest Child->peek() value. OrIterator becomes exhausted as
/// soon as all of its children are exhausted.
class OrIterator : public Iterator {
public:
  OrIterator(std::vector<std::unique_ptr<Iterator>> AllChildren)
      : Children(std::move(AllChildren)) {
    assert(Children.size() > 0 && "OR iterator must have at least one child.");
  }

  /// Returns true if all children are exhausted.
  bool reachedEnd() const override {
    return std::all_of(begin(Children), end(Children),
                       [](const std::unique_ptr<Iterator> &Child) {
                         return Child->reachedEnd();
                       });
  }

  /// Moves each child pointing to the smallest DocID to the next item.
  void advance() override {
    assert(!reachedEnd() && "OR iterator can't advance() at the end.");
    const auto SmallestID = peek();
    for (const auto &Child : Children)
      if (!Child->reachedEnd() && Child->peek() == SmallestID)
        Child->advance();
  }

  /// Advances each child to the next existing element with DocumentID >= ID.
  void advanceTo(DocID ID) override {
    assert(!reachedEnd() && "OR iterator can't advanceTo() at the end.");
    for (const auto &Child : Children)
      if (!Child->reachedEnd())
        Child->advanceTo(ID);
  }

  /// Returns the element under cursor of the child with smallest Child->peek()
  /// value.
  DocID peek() const override {
    assert(!reachedEnd() && "OR iterator can't peek() at the end.");
    DocID Result = std::numeric_limits<DocID>::max();

    for (const auto &Child : Children)
      if (!Child->reachedEnd())
        Result = std::min(Result, Child->peek());

    return Result;
  }

  // Returns the maximum boosting score among all Children when iterator is not
  // exhausted and points to the given ID, DEFAULT_BOOST_SCORE otherwise.
  float consume() override {
    assert(!reachedEnd() && "OR iterator can't consume() at the end.");
    const DocID ID = peek();
    return std::accumulate(
        begin(Children), end(Children), DEFAULT_BOOST_SCORE,
        [&](float Boost, const std::unique_ptr<Iterator> &Child) {
          return (!Child->reachedEnd() && Child->peek() == ID)
                     ? std::max(Boost, Child->consume())
                     : Boost;
        });
  }

  size_t estimateSize() const override {
    return std::accumulate(
        begin(Children), end(Children), Children.front()->estimateSize(),
        [&](size_t Current, const std::unique_ptr<Iterator> &Child) {
          return std::max(Current, Child->estimateSize());
        });
  }

private:
  llvm::raw_ostream &dump(llvm::raw_ostream &OS) const override {
    OS << "(| ";
    auto Separator = "";
    for (const auto &Child : Children) {
      OS << Separator << *Child;
      Separator = " ";
    }
    OS << ')';
    return OS;
  }

  // FIXME(kbobyrev): Would storing Children in min-heap be faster?
  std::vector<std::unique_ptr<Iterator>> Children;
};

/// TrueIterator handles PostingLists which contain all items of the index. It
/// stores size of the virtual posting list, and all operations are performed
/// in O(1).
class TrueIterator : public Iterator {
public:
  TrueIterator(DocID Size) : Size(Size) {}

  bool reachedEnd() const override { return Index >= Size; }

  void advance() override {
    assert(!reachedEnd() && "TRUE iterator can't advance() at the end.");
    ++Index;
  }

  void advanceTo(DocID ID) override {
    assert(!reachedEnd() && "TRUE iterator can't advanceTo() at the end.");
    Index = std::min(ID, Size);
  }

  DocID peek() const override {
    assert(!reachedEnd() && "TRUE iterator can't peek() at the end.");
    return Index;
  }

  float consume() override {
    assert(!reachedEnd() && "TRUE iterator can't consume() at the end.");
    return DEFAULT_BOOST_SCORE;
  }

  size_t estimateSize() const override { return Size; }

private:
  llvm::raw_ostream &dump(llvm::raw_ostream &OS) const override {
    OS << "(TRUE {" << Index << "} out of " << Size << ")";
    return OS;
  }

  DocID Index = 0;
  /// Size of the underlying virtual PostingList.
  DocID Size;
};

/// Boost iterator is a wrapper around its child which multiplies scores of
/// each retrieved item by a given factor.
class BoostIterator : public Iterator {
public:
  BoostIterator(std::unique_ptr<Iterator> Child, float Factor)
      : Child(move(Child)), Factor(Factor) {}

  bool reachedEnd() const override { return Child->reachedEnd(); }

  void advance() override { Child->advance(); }

  void advanceTo(DocID ID) override { Child->advanceTo(ID); }

  DocID peek() const override { return Child->peek(); }

  float consume() override { return Child->consume() * Factor; }

  size_t estimateSize() const override { return Child->estimateSize(); }

private:
  llvm::raw_ostream &dump(llvm::raw_ostream &OS) const override {
    OS << "(BOOST " << Factor << ' ' << *Child << ')';
    return OS;
  }

  std::unique_ptr<Iterator> Child;
  float Factor;
};

/// This iterator limits the number of items retrieved from the child iterator
/// on top of the query tree. To ensure that query tree with LIMIT iterators
/// inside works correctly, users have to call Root->consume(Root->peek()) each
/// time item is retrieved at the root of query tree.
class LimitIterator : public Iterator {
public:
  LimitIterator(std::unique_ptr<Iterator> Child, size_t Limit)
      : Child(move(Child)), Limit(Limit), ItemsLeft(Limit) {}

  bool reachedEnd() const override {
    return ItemsLeft == 0 || Child->reachedEnd();
  }

  void advance() override { Child->advance(); }

  void advanceTo(DocID ID) override { Child->advanceTo(ID); }

  DocID peek() const override { return Child->peek(); }

  /// Decreases the limit in case the element consumed at top of the query tree
  /// comes from the underlying iterator.
  float consume() override {
    assert(!reachedEnd() && "LimitIterator can't consume() at the end.");
    --ItemsLeft;
    return Child->consume();
  }

  size_t estimateSize() const override {
    return std::min(Child->estimateSize(), Limit);
  }

private:
  llvm::raw_ostream &dump(llvm::raw_ostream &OS) const override {
    OS << "(LIMIT " << Limit << '(' << ItemsLeft << ") " << *Child << ')';
    return OS;
  }

  std::unique_ptr<Iterator> Child;
  size_t Limit;
  size_t ItemsLeft;
};

} // end namespace

std::vector<std::pair<DocID, float>> consume(Iterator &It) {
  std::vector<std::pair<DocID, float>> Result;
  for (; !It.reachedEnd(); It.advance())
    Result.emplace_back(It.peek(), It.consume());
  return Result;
}

std::unique_ptr<Iterator> create(PostingListRef Documents) {
  return llvm::make_unique<DocumentIterator>(Documents);
}

std::unique_ptr<Iterator>
createAnd(std::vector<std::unique_ptr<Iterator>> Children) {
  return llvm::make_unique<AndIterator>(move(Children));
}

std::unique_ptr<Iterator>
createOr(std::vector<std::unique_ptr<Iterator>> Children) {
  return llvm::make_unique<OrIterator>(move(Children));
}

std::unique_ptr<Iterator> createTrue(DocID Size) {
  return llvm::make_unique<TrueIterator>(Size);
}

std::unique_ptr<Iterator> createBoost(std::unique_ptr<Iterator> Child,
                                      float Factor) {
  return llvm::make_unique<BoostIterator>(move(Child), Factor);
}

std::unique_ptr<Iterator> createLimit(std::unique_ptr<Iterator> Child,
                                      size_t Limit) {
  return llvm::make_unique<LimitIterator>(move(Child), Limit);
}

} // namespace dex
} // namespace clangd
} // namespace clang
