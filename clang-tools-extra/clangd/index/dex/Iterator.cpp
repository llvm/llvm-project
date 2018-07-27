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
    assert(!reachedEnd() && "DocumentIterator can't advance at the end.");
    ++Index;
  }

  /// Applies binary search to advance cursor to the next item with DocID equal
  /// or higher than the given one.
  void advanceTo(DocID ID) override {
    assert(!reachedEnd() && "DocumentIterator can't advance at the end.");
    Index = std::lower_bound(Index, std::end(Documents), ID);
  }

  DocID peek() const override {
    assert(!reachedEnd() && "DocumentIterator can't call peek() at the end.");
    return *Index;
  }

  llvm::raw_ostream &dump(llvm::raw_ostream &OS) const override {
    OS << '[';
    auto Separator = "";
    for (const auto &ID : Documents) {
      OS << Separator << ID;
      Separator = ", ";
    }
    OS << ']';
    return OS;
  }

private:
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
    assert(!Children.empty() && "AndIterator should have at least one child.");
    // Establish invariants.
    sync();
  }

  bool reachedEnd() const override { return ReachedEnd; }

  /// Advances all children to the next common item.
  void advance() override {
    assert(!reachedEnd() && "AndIterator can't call advance() at the end.");
    Children.front()->advance();
    sync();
  }

  /// Advances all children to the next common item with DocumentID >= ID.
  void advanceTo(DocID ID) override {
    assert(!reachedEnd() && "AndIterator can't call advanceTo() at the end.");
    Children.front()->advanceTo(ID);
    sync();
  }

  DocID peek() const override { return Children.front()->peek(); }

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

private:
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
        // FIXME(kbobyrev): This is not a very optimized version; after costs
        // are introduced, cycle should break whenever ID exceeds current one
        // and cheapest children should be advanced over again.
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
    assert(Children.size() > 0 && "Or Iterator must have at least one child.");
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
    assert(!reachedEnd() &&
           "OrIterator must have at least one child to advance().");
    const auto SmallestID = peek();
    for (const auto &Child : Children)
      if (!Child->reachedEnd() && Child->peek() == SmallestID)
        Child->advance();
  }

  /// Advances each child to the next existing element with DocumentID >= ID.
  void advanceTo(DocID ID) override {
    assert(!reachedEnd() && "Can't advance iterator after it reached the end.");
    for (const auto &Child : Children)
      if (!Child->reachedEnd())
        Child->advanceTo(ID);
  }

  /// Returns the element under cursor of the child with smallest Child->peek()
  /// value.
  DocID peek() const override {
    assert(!reachedEnd() &&
           "OrIterator must have at least one child to peek().");
    DocID Result = std::numeric_limits<DocID>::max();

    for (const auto &Child : Children)
      if (!Child->reachedEnd())
        Result = std::min(Result, Child->peek());

    return Result;
  }

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

private:
  // FIXME(kbobyrev): Would storing Children in min-heap be faster?
  std::vector<std::unique_ptr<Iterator>> Children;
};

} // end namespace

std::vector<DocID> consume(Iterator &It) {
  std::vector<DocID> Result;
  for (; !It.reachedEnd(); It.advance())
    Result.push_back(It.peek());
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

} // namespace dex
} // namespace clangd
} // namespace clang
