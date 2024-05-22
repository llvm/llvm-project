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
#include <iterator>
#include <utility>

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
  static constexpr FragmentNum warm() { return FragmentNum(2); }
};

/// A freestanding subset of contiguous blocks of a function.
class FunctionFragment {
  using BasicBlockListType = SmallVector<BinaryBasicBlock *, 0>;
  using FragmentListType = SmallVector<unsigned, 0>;

public:
  using iterator = BasicBlockListType::iterator;
  using const_iterator = BasicBlockListType::const_iterator;

private:
  FunctionLayout *Layout;
  FragmentNum Num;
  unsigned StartIndex;
  unsigned Size = 0;

  /// Output address for the fragment.
  uint64_t Address = 0;

  /// The address for the code for this fragment in codegen memory. Used for
  /// functions that are emitted in a dedicated section with a fixed address,
  /// e.g. for functions that are overwritten in-place.
  uint64_t ImageAddress = 0;

  /// The size of the code in memory.
  uint64_t ImageSize = 0;

  /// Offset in the file.
  uint64_t FileOffset = 0;

  FunctionFragment(FunctionLayout &Layout, FragmentNum Num);
  FunctionFragment(const FunctionFragment &) = default;
  FunctionFragment(FunctionFragment &&) = default;
  FunctionFragment &operator=(const FunctionFragment &) = default;
  FunctionFragment &operator=(FunctionFragment &&) = default;
  ~FunctionFragment() = default;

public:
  FragmentNum getFragmentNum() const { return Num; }
  bool isMainFragment() const {
    return getFragmentNum() == FragmentNum::main();
  }
  bool isSplitFragment() const { return !isMainFragment(); }

  uint64_t getAddress() const { return Address; }
  void setAddress(uint64_t Value) { Address = Value; }
  uint64_t getImageAddress() const { return ImageAddress; }
  void setImageAddress(uint64_t Address) { ImageAddress = Address; }
  uint64_t getImageSize() const { return ImageSize; }
  void setImageSize(uint64_t Size) { ImageSize = Size; }
  uint64_t getFileOffset() const { return FileOffset; }
  void setFileOffset(uint64_t Offset) { FileOffset = Offset; }

  unsigned size() const { return Size; };
  bool empty() const { return size() == 0; };
  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;
  const BinaryBasicBlock *front() const;

  friend class FunctionLayout;
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
  using FragmentListType = SmallVector<FunctionFragment *, 0>;
  using BasicBlockListType = SmallVector<BinaryBasicBlock *, 0>;

public:
  using fragment_iterator = pointee_iterator<FragmentListType::const_iterator>;
  using fragment_const_iterator =
      pointee_iterator<FragmentListType::const_iterator,
                       const FunctionFragment>;
  using block_iterator = BasicBlockListType::iterator;
  using block_const_iterator = BasicBlockListType::const_iterator;
  using block_reverse_iterator = std::reverse_iterator<block_iterator>;
  using block_const_reverse_iterator =
      std::reverse_iterator<block_const_iterator>;

private:
  FragmentListType Fragments;
  BasicBlockListType Blocks;

public:
  FunctionLayout();
  FunctionLayout(const FunctionLayout &Other);
  FunctionLayout(FunctionLayout &&Other);
  FunctionLayout &operator=(const FunctionLayout &Other);
  FunctionLayout &operator=(FunctionLayout &&Other);
  ~FunctionLayout();

  /// Add an empty fragment.
  FunctionFragment &addFragment();

  /// Return the fragment identified by Num.
  FunctionFragment &getFragment(FragmentNum Num);

  /// Return the fragment identified by Num.
  const FunctionFragment &getFragment(FragmentNum Num) const;

  /// Get the fragment that contains all entry blocks and other blocks that
  /// cannot be split.
  FunctionFragment &getMainFragment() {
    return getFragment(FragmentNum::main());
  }

  /// Get the fragment that contains all entry blocks and other blocks that
  /// cannot be split.
  const FunctionFragment &getMainFragment() const {
    return getFragment(FragmentNum::main());
  }

  /// Get the fragment that contains all entry blocks and other blocks that
  /// cannot be split.
  iterator_range<fragment_iterator> getSplitFragments() {
    return {++fragment_begin(), fragment_end()};
  }

  /// Get the fragment that contains all entry blocks and other blocks that
  /// cannot be split.
  iterator_range<fragment_const_iterator> getSplitFragments() const {
    return {++fragment_begin(), fragment_end()};
  }

  /// Find the fragment that contains BB.
  const FunctionFragment &findFragment(const BinaryBasicBlock *BB) const;

  /// Add BB to the end of the last fragment.
  void addBasicBlock(BinaryBasicBlock *BB);

  /// Insert range of basic blocks after InsertAfter. If InsertAfter is nullptr,
  /// the blocks will be inserted at the start of the function.
  void insertBasicBlocks(const BinaryBasicBlock *InsertAfter,
                         ArrayRef<BinaryBasicBlock *> NewBlocks);

  /// Erase all blocks from the layout that are in ToErase. If this method
  /// erases all blocks of a fragment, it will be removed as well.
  void eraseBasicBlocks(const DenseSet<const BinaryBasicBlock *> ToErase);

  /// Make sure fragments' and basic blocks' indices match the current layout.
  void updateLayoutIndices();

  /// Replace the current layout with NewLayout. Uses the block's
  /// self-identifying fragment number to assign blocks to infer function
  /// fragments. Returns `true` if the new layout is different from the current
  /// layout.
  bool update(ArrayRef<BinaryBasicBlock *> NewLayout);

  /// Clear layout releasing memory.
  void clear();

  BinaryBasicBlock *getBlock(unsigned Index) { return Blocks[Index]; }

  const BinaryBasicBlock *getBlock(unsigned Index) const {
    return Blocks[Index];
  }

  /// Returns the basic block after the given basic block in the layout or
  /// nullptr if the last basic block is given.
  BinaryBasicBlock *getBasicBlockAfter(const BinaryBasicBlock *const BB,
                                       const bool IgnoreSplits = true) {
    return const_cast<BinaryBasicBlock *>(
        static_cast<const FunctionLayout &>(*this).getBasicBlockAfter(
            BB, IgnoreSplits));
  }

  /// Returns the basic block after the given basic block in the layout or
  /// nullptr if the last basic block is given.
  const BinaryBasicBlock *getBasicBlockAfter(const BinaryBasicBlock *BB,
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
    assert(Fragments.size() >= 1 &&
           "Layout should have at least one fragment.");
    return Fragments.size();
  }
  bool fragment_empty() const { return fragment_size() == 0; }

  fragment_iterator fragment_begin() { return Fragments.begin(); }
  fragment_const_iterator fragment_begin() const { return Fragments.begin(); }
  fragment_iterator fragment_end() { return Fragments.end(); }
  fragment_const_iterator fragment_end() const { return Fragments.end(); }
  iterator_range<fragment_iterator> fragments() {
    return {fragment_begin(), fragment_end()};
  }
  iterator_range<fragment_const_iterator> fragments() const {
    return {fragment_begin(), fragment_end()};
  }

  size_t block_size() const { return Blocks.size(); }
  bool block_empty() const { return Blocks.empty(); }

  /// Required to return non-const qualified `BinaryBasicBlock *` for graph
  /// traits.
  BinaryBasicBlock *block_front() const { return Blocks.front(); }
  const BinaryBasicBlock *block_back() const { return Blocks.back(); }

  block_iterator block_begin() { return Blocks.begin(); }
  block_const_iterator block_begin() const {
    return block_const_iterator(Blocks.begin());
  }
  block_iterator block_end() { return Blocks.end(); }
  block_const_iterator block_end() const {
    return block_const_iterator(Blocks.end());
  }
  iterator_range<block_iterator> blocks() {
    return {block_begin(), block_end()};
  }
  iterator_range<block_const_iterator> blocks() const {
    return {block_begin(), block_end()};
  }
  block_reverse_iterator block_rbegin() {
    return block_reverse_iterator(Blocks.rbegin());
  }
  block_const_reverse_iterator block_rbegin() const {
    return block_const_reverse_iterator(
        std::make_reverse_iterator(block_end()));
  }
  block_reverse_iterator block_rend() {
    return block_reverse_iterator(Blocks.rend());
  }
  block_const_reverse_iterator block_rend() const {
    return block_const_reverse_iterator(
        std::make_reverse_iterator(block_begin()));
  }
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
