//===- bolt/Core/FunctionLayout.h - Fragmented Function Layout --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the FunctionLayout class. The layout of
// a function is the order of basic blocks, in which we will arrange them in the
// new binary. Normally, when not optimizing for code layout, the blocks of a
// function are contiguous. However, we can split the layout into multiple
// fragments. The blocks within a fragment are contiguous, but the fragments
// itself are disjoint. Fragments could be used to enhance code layout, e.g. to
// separate the blocks into hot and cold sections.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_FUNCTION_LAYOUT_H
#define BOLT_CORE_FUNCTION_LAYOUT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {
namespace bolt {

class BinaryFunction;
class BinaryBasicBlock;
class FunctionLayout;

class FragmentNum {
  unsigned Value{0};

public:
  constexpr FragmentNum() = default;
  constexpr explicit FragmentNum(unsigned Value) : Value(Value) {}
  constexpr unsigned get() const { return Value; }

  constexpr bool operator==(const FragmentNum Other) const {
    return Value == Other.Value;
  }
  constexpr bool operator!=(const FragmentNum Other) const {
    return Value != Other.Value;
  }
  constexpr bool operator<(const FragmentNum Other) const {
    return Value < Other.Value;
  }
  constexpr bool operator<=(const FragmentNum Other) const {
    return Value <= Other.Value;
  }
  constexpr bool operator>=(const FragmentNum Other) const {
    return Value >= Other.Value;
  }
  constexpr bool operator>(const FragmentNum Other) const {
    return Value > Other.Value;
  }

  static constexpr FragmentNum main() { return FragmentNum(0); }
  static constexpr FragmentNum cold() { return FragmentNum(1); }
};

/// A freestanding subset of contiguous blocks of a function.
class FunctionFragment {
  using BasicBlockListType = SmallVector<BinaryBasicBlock *, 0>;
  using FragmentListType = SmallVector<unsigned, 0>;

public:
  using const_iterator = BasicBlockListType::const_iterator;

private:
  FragmentNum Num;
  const FunctionLayout &Layout;

  FunctionFragment(FragmentNum Num, const FunctionLayout &Layout)
      : Num(Num), Layout(Layout) {}

public:
  FragmentNum getFragmentNum() const { return Num; }
  bool isMainFragment() const { return Num.get() == 0; }
  bool isSplitFragment() const { return Num.get() > 0; }

  unsigned size() const;
  bool empty() const;
  const_iterator begin() const;
  const_iterator end() const;
  BinaryBasicBlock *front() const;

  friend class FunctionLayout;
  friend class FragmentIterator;
};

/// The function layout represents the fragments we split a function into and
/// the order of basic blocks within each fragment.
///
/// Internally, the function layout stores blocks across fragments contiguously.
/// This is necessary to retain compatibility with existing code and tests that
/// iterate  over all blocks of the layout and depend on that order. When
/// writing new code, avoid iterating using FunctionLayout::blocks() by
/// iterating either over fragments or over BinaryFunction::begin()..end().
class FunctionLayout {
private:
  using BasicBlockListType = SmallVector<BinaryBasicBlock *, 0>;
  using block_iterator = BasicBlockListType::iterator;
  using FragmentListType = SmallVector<unsigned, 0>;

public:
  class FragmentIterator;

  class FragmentIterator
      : public iterator_facade_base<
            FragmentIterator, std::bidirectional_iterator_tag, FunctionFragment,
            std::ptrdiff_t, FunctionFragment *, FunctionFragment> {
    FragmentNum Num;
    const FunctionLayout *Layout;

    FragmentIterator(FragmentNum Num, const FunctionLayout *Layout)
        : Num(Num), Layout(Layout) {
      assert(Num.get() <= Layout->fragment_size() &&
             "Initializing iterator out of bounds");
    }

  public:
    bool operator==(const FragmentIterator &Other) const {
      return Num == Other.Num;
    }

    FunctionFragment operator*() const {
      assert(Num.get() < Layout->fragment_size() &&
             "Dereferencing end() iterator (or past it)");
      return FunctionFragment(Num, *Layout);
    }

    FragmentIterator &operator++() {
      assert(Num.get() < Layout->fragment_size() &&
             "Incrementing iterator past end()");
      Num = FragmentNum(Num.get() + 1);
      return *this;
    }

    FragmentIterator &operator--() {
      assert(Num.get() > 0 && "Decrementing iterator past begin()");
      Num = FragmentNum(Num.get() - 1);
      return *this;
    }

    friend class FunctionLayout;
  };

  using const_iterator = FragmentIterator;
  using block_const_iterator = BasicBlockListType::const_iterator;
  using block_const_reverse_iterator =
      BasicBlockListType::const_reverse_iterator;

private:
  BasicBlockListType Blocks;
  /// List of indices dividing block list into fragments. To simplify iteration,
  /// we have `Fragments.back()` equals `Blocks.size()`. Hence,
  /// `Fragments.size()` equals `this->size() + 1`. Always contains at least one
  /// fragment.
  FragmentListType Fragments = {0, 0};

public:
  /// Add an empty fragment.
  FunctionFragment addFragment();

  /// Return the fragment identified by Num.
  FunctionFragment getFragment(FragmentNum Num) const;

  /// Get the fragment that contains all entry blocks and other blocks that
  /// cannot be split.
  FunctionFragment getMainFragment() const {
    return getFragment(FragmentNum::main());
  }

  /// Get the fragment that contains all entry blocks and other blocks that
  /// cannot be split.
  iterator_range<const_iterator> getSplitFragments() const {
    return {++fragment_begin(), fragment_end()};
  }

  /// Find the fragment that contains BB.
  FunctionFragment findFragment(const BinaryBasicBlock *BB) const;

  /// Add BB to the end of the last fragment.
  void addBasicBlock(BinaryBasicBlock *BB);

  /// Insert range of basic blocks after InsertAfter. If InsertAfter is nullptr,
  /// the blocks will be inserted at the start of the function.
  void insertBasicBlocks(BinaryBasicBlock *InsertAfter,
                         ArrayRef<BinaryBasicBlock *> NewBlocks);

  /// Erase all blocks from the layout that are in ToErase. If this method
  /// erases all blocks of a fragment, it will be removed as well.
  void eraseBasicBlocks(const DenseSet<const BinaryBasicBlock *> ToErase);

  /// Make sure fragments' and basic blocks' indices match the current layout.
  void updateLayoutIndices() const;

  /// Replace the current layout with NewLayout. Uses the block's
  /// self-identifying fragment number to assign blocks to infer function
  /// fragments. Returns `true` if the new layout is different from the current
  /// layout.
  bool update(ArrayRef<BinaryBasicBlock *> NewLayout);

  /// Clear layout releasing memory.
  void clear();

  BinaryBasicBlock *getBlock(unsigned Index) const { return Blocks[Index]; }

  /// Returns the basic block after the given basic block in the layout or
  /// nullptr if the last basic block is given.
  BinaryBasicBlock *getBasicBlockAfter(const BinaryBasicBlock *BB,
                                       bool IgnoreSplits = true) const;

  /// True if the layout contains at least two non-empty fragments.
  bool isSplit() const;

  /// Get the edit distance of the new layout with respect to the previous
  /// layout after basic block reordering.
  uint64_t
  getEditDistance(ArrayRef<const BinaryBasicBlock *> OldBlockOrder) const;

  /// True if the function is split into at most 2 fragments. Mostly used for
  /// checking whether a function can be processed in places that do not support
  /// multiple fragments yet.
  bool isHotColdSplit() const { return fragment_size() <= 2; }

  size_t fragment_size() const {
    assert(Fragments.size() >= 2 &&
           "Layout should have at least one fragment.");
    return Fragments.size() - 1;
  }
  bool fragment_empty() const { return Fragments.size() == 1; }
  const_iterator fragment_begin() const { return {FragmentNum(0), this}; }
  const_iterator fragment_end() const {
    return {FragmentNum(fragment_size()), this};
  }
  iterator_range<const_iterator> fragments() const {
    return {fragment_begin(), fragment_end()};
  }

  size_t block_size() const { return Blocks.size(); }
  bool block_empty() const { return Blocks.empty(); }
  BinaryBasicBlock *block_front() const { return Blocks.front(); }
  BinaryBasicBlock *block_back() const { return Blocks.back(); }
  block_const_iterator block_begin() const { return Blocks.begin(); }
  block_const_iterator block_end() const { return Blocks.end(); }
  iterator_range<block_const_iterator> blocks() const {
    return {block_begin(), block_end()};
  }
  block_const_reverse_iterator block_rbegin() const { return Blocks.rbegin(); }
  block_const_reverse_iterator block_rend() const { return Blocks.rend(); }
  iterator_range<block_const_reverse_iterator> rblocks() const {
    return {block_rbegin(), block_rend()};
  }

private:
  block_const_iterator findBasicBlockPos(const BinaryBasicBlock *BB) const;
  block_iterator findBasicBlockPos(const BinaryBasicBlock *BB);

  friend class FunctionFragment;
};

} // namespace bolt
} // namespace llvm

#endif
