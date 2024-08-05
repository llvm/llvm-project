//===-- Interface for freelist --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREELIST_H
#define LLVM_LIBC_SRC___SUPPORT_FREELIST_H

#include "block.h"

namespace LIBC_NAMESPACE_DECL {

/// A circularly-linked FIFO list storing free Blocks. All Blocks on a list
/// are the same size.
///
/// Allocating free blocks in FIFO order maximizes the amount of time before a
/// free block is reused. This in turn maximizes the number of opportunities for
/// it to be coalesced with an adjacent block, which tends to reduce heap
/// fragmentation.
class FreeList {
public:
  class Node {
  public:
    /// @returns The block containing this node.
    Block<> *block() const {
      return const_cast<Block<> *>(Block<>::from_usable_space(this));
    }

    /// @returns The inner size of blocks in the list containing this node.
    size_t size() const { return block()->inner_size(); }

  private:
    // Circularly linked pointers to adjacent nodes.
    Node *prev;
    Node *next;
    friend class FreeList;
  };

  constexpr FreeList() : FreeList(nullptr) {}
  constexpr FreeList(Node *begin) : begin_(begin) {}

  bool empty() const { return !begin_; }

  /// @returns The inner size of blocks in the list.
  size_t size() const {
    LIBC_ASSERT(begin_ && "empty lists have no size");
    return begin_->size();
  }

  /// @returns The first node in the list.
  Node *begin() { return begin_; }

  /// @returns The first block in the list.
  Block<> *front() { return begin_->block(); }

  /// Push a block to the back of the list.
  /// The block must be large enough to contain a node.
  void push(Block<> *block);

  /// Push an already-constructed node to the back of the list.
  /// This allows pushing derived node types with additional data.
  void push(Node *node);

  /// Pop the first node from the list.
  void pop();

  /// Remove an arbitrary node from the list.
  void remove(Node *node);

private:
  Node *begin_;
};

LIBC_INLINE void FreeList::push(Block<> *block) {
  LIBC_ASSERT(!block->used() && "only free blocks can be placed on free lists");
  LIBC_ASSERT(block->inner_size_free() >= sizeof(FreeList) &&
              "block too small to accomodate free list node");
  push(new (block->usable_space()) Node);
}

LIBC_INLINE void FreeList::push(Node *node) {
  if (begin_) {
    LIBC_ASSERT(Block<>::from_usable_space(node)->outer_size() ==
                    begin_->block()->outer_size() &&
                "freelist entries must have the same size");
    // Since the list is circular, insert the node immediately before begin_.
    node->prev = begin_->prev;
    node->next = begin_;
    begin_->prev->next = node;
    begin_->prev = node;
  } else {
    begin_ = node->prev = node->next = node;
  }
}

LIBC_INLINE void FreeList::pop() { remove(begin_); }

LIBC_INLINE void FreeList::remove(Node *node) {
  LIBC_ASSERT(begin_ && "cannot remove from empty list");
  if (node == node->next) {
    LIBC_ASSERT(node == begin_ &&
                "a self-referential node must be the only element");
    begin_ = nullptr;
  } else {
    node->prev->next = node->next;
    node->next->prev = node->prev;
    if (begin_ == node)
      begin_ = node->next;
  }
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREELIST_H
