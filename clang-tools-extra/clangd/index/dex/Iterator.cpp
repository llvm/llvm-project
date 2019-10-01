//===--- Iterator.cpp - Query Symbol Retrieval ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Iterator.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <numeric>

namespace clang {
namespace clangd {
namespace dex {
namespace {

/// Implements Iterator over the intersection of other iterators.
///
/// AndIterator iterates through common items among all children. It becomes
/// exhausted as soon as any child becomes exhausted. After each mutation, the
/// iterator restores the invariant: all children must point to the same item.
class AndIterator : public Iterator {
public:
  explicit AndIterator(std::vector<std::unique_ptr<Iterator>> AllChildren)
      : Iterator(Kind::And), Children(std::move(AllChildren)) {
    assert(!Children.empty() && "AND iterator should have at least one child.");
    // Establish invariants.
    for (const auto &Child : Children)
      ReachedEnd |= Child->reachedEnd();
    sync();
    // When children are sorted by the estimateSize(), sync() calls are more
    // effective. Each sync() starts with the first child and makes sure all
    // children point to the same element. If any child is "above" the previous
    // ones, the algorithm resets and and advances the children to the next
    // highest element starting from the front. When child iterators in the
    // beginning have smaller estimated size, the sync() will have less restarts
    // and become more effective.
    llvm::sort(Children, [](const std::unique_ptr<Iterator> &LHS,
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
    float Boost = 1;
    for (const auto &Child : Children)
      Boost *= Child->consume();
    return Boost;
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
  friend Corpus; // For optimizations.
};

/// Implements Iterator over the union of other iterators.
///
/// OrIterator iterates through all items which can be pointed to by at least
/// one child. To preserve the sorted order, this iterator always advances the
/// child with smallest Child->peek() value. OrIterator becomes exhausted as
/// soon as all of its children are exhausted.
class OrIterator : public Iterator {
public:
  explicit OrIterator(std::vector<std::unique_ptr<Iterator>> AllChildren)
      : Iterator(Kind::Or), Children(std::move(AllChildren)) {
    assert(!Children.empty() && "OR iterator should have at least one child.");
  }

  /// Returns true if all children are exhausted.
  bool reachedEnd() const override {
    for (const auto &Child : Children)
      if (!Child->reachedEnd())
        return false;
    return true;
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

  // Returns the maximum boosting score among all Children when iterator
  // points to the current ID.
  float consume() override {
    assert(!reachedEnd() && "OR iterator can't consume() at the end.");
    const DocID ID = peek();
    float Boost = 1;
    for (const auto &Child : Children)
      if (!Child->reachedEnd() && Child->peek() == ID)
        Boost = std::max(Boost, Child->consume());
    return Boost;
  }

  size_t estimateSize() const override {
    size_t Size = 0;
    for (const auto &Child : Children)
      Size = std::max(Size, Child->estimateSize());
    return Size;
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
  friend Corpus; // For optimizations.
};

/// TrueIterator handles PostingLists which contain all items of the index. It
/// stores size of the virtual posting list, and all operations are performed
/// in O(1).
class TrueIterator : public Iterator {
public:
  explicit TrueIterator(DocID Size) : Iterator(Kind::True), Size(Size) {}

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
    return 1;
  }

  size_t estimateSize() const override { return Size; }

private:
  llvm::raw_ostream &dump(llvm::raw_ostream &OS) const override {
    return OS << "true";
  }

  DocID Index = 0;
  /// Size of the underlying virtual PostingList.
  DocID Size;
};

/// FalseIterator yields no results.
class FalseIterator : public Iterator {
public:
  FalseIterator() : Iterator(Kind::False) {}
  bool reachedEnd() const override { return true; }
  void advance() override { assert(false); }
  void advanceTo(DocID ID) override { assert(false); }
  DocID peek() const override {
    assert(false);
    return 0;
  }
  float consume() override {
    assert(false);
    return 1;
  }
  size_t estimateSize() const override { return 0; }

private:
  llvm::raw_ostream &dump(llvm::raw_ostream &OS) const override {
    return OS << "false";
  }
};

/// Boost iterator is a wrapper around its child which multiplies scores of
/// each retrieved item by a given factor.
class BoostIterator : public Iterator {
public:
  BoostIterator(std::unique_ptr<Iterator> Child, float Factor)
      : Child(std::move(Child)), Factor(Factor) {}

  bool reachedEnd() const override { return Child->reachedEnd(); }

  void advance() override { Child->advance(); }

  void advanceTo(DocID ID) override { Child->advanceTo(ID); }

  DocID peek() const override { return Child->peek(); }

  float consume() override { return Child->consume() * Factor; }

  size_t estimateSize() const override { return Child->estimateSize(); }

private:
  llvm::raw_ostream &dump(llvm::raw_ostream &OS) const override {
    return OS << "(* " << Factor << ' ' << *Child << ')';
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
      : Child(std::move(Child)), Limit(Limit), ItemsLeft(Limit) {}

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
    return OS << "(LIMIT " << Limit << " " << *Child << ')';
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

std::unique_ptr<Iterator>
Corpus::intersect(std::vector<std::unique_ptr<Iterator>> Children) const {
  std::vector<std::unique_ptr<Iterator>> RealChildren;
  for (auto &Child : Children) {
    switch (Child->kind()) {
    case Iterator::Kind::True:
      break; // No effect, drop the iterator.
    case Iterator::Kind::False:
      return std::move(Child); // Intersection is empty.
    case Iterator::Kind::And: {
      // Inline nested AND into parent AND.
      auto &NewChildren = static_cast<AndIterator *>(Child.get())->Children;
      std::move(NewChildren.begin(), NewChildren.end(),
                std::back_inserter(RealChildren));
      break;
    }
    default:
      RealChildren.push_back(std::move(Child));
    }
  }
  switch (RealChildren.size()) {
  case 0:
    return all();
  case 1:
    return std::move(RealChildren.front());
  default:
    return llvm::make_unique<AndIterator>(std::move(RealChildren));
  }
}

std::unique_ptr<Iterator>
Corpus::unionOf(std::vector<std::unique_ptr<Iterator>> Children) const {
  std::vector<std::unique_ptr<Iterator>> RealChildren;
  for (auto &Child : Children) {
    switch (Child->kind()) {
    case Iterator::Kind::False:
      break; // No effect, drop the iterator.
    case Iterator::Kind::Or: {
      // Inline nested OR into parent OR.
      auto &NewChildren = static_cast<OrIterator *>(Child.get())->Children;
      std::move(NewChildren.begin(), NewChildren.end(),
                std::back_inserter(RealChildren));
      break;
    }
    case Iterator::Kind::True:
      // Don't return all(), which would discard sibling boosts.
    default:
      RealChildren.push_back(std::move(Child));
    }
  }
  switch (RealChildren.size()) {
  case 0:
    return none();
  case 1:
    return std::move(RealChildren.front());
  default:
    return llvm::make_unique<OrIterator>(std::move(RealChildren));
  }
}

std::unique_ptr<Iterator> Corpus::all() const {
  return llvm::make_unique<TrueIterator>(Size);
}

std::unique_ptr<Iterator> Corpus::none() const {
  return llvm::make_unique<FalseIterator>();
}

std::unique_ptr<Iterator> Corpus::boost(std::unique_ptr<Iterator> Child,
                                        float Factor) const {
  if (Factor == 1)
    return Child;
  if (Child->kind() == Iterator::Kind::False)
    return Child;
  return llvm::make_unique<BoostIterator>(std::move(Child), Factor);
}

std::unique_ptr<Iterator> Corpus::limit(std::unique_ptr<Iterator> Child,
                                        size_t Limit) const {
  if (Child->kind() == Iterator::Kind::False)
    return Child;
  return llvm::make_unique<LimitIterator>(std::move(Child), Limit);
}

} // namespace dex
} // namespace clangd
} // namespace clang
