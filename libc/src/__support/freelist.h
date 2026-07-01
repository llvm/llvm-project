//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Interface for freelist.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREELIST_H
#define LLVM_LIBC_SRC___SUPPORT_FREELIST_H

#include "block.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"

#ifndef LIBC_COPT_HARDEN_FREELIST
#define LIBC_COPT_HARDEN_FREELIST false
#endif

#if LIBC_COPT_HARDEN_FREELIST
#define LIBC_HARDENING_ASSERT(cond)                                            \
  do {                                                                         \
    if (LIBC_UNLIKELY(!(cond))) {                                              \
      __builtin_trap();                                                        \
    }                                                                          \
  } while (0)
#else
#define LIBC_HARDENING_ASSERT(cond) LIBC_ASSERT(cond)
#endif

namespace LIBC_NAMESPACE_DECL {

struct FreeListSecrets {
  static constexpr int NODE_PTR_ROTATE_DISTANCE = 17;

#if LIBC_COPT_HARDEN_FREELIST
  uintptr_t k0;
  uintptr_t k1;
  uintptr_t k2;

  LIBC_INLINE constexpr FreeListSecrets(uintptr_t k0, uintptr_t k1,
                                        uintptr_t k2)
      : k0(k0), k1(k1), k2(k2) {}
  LIBC_INLINE constexpr FreeListSecrets() : k0(0), k1(0), k2(0) {}
#else
  LIBC_INLINE constexpr FreeListSecrets() = default;
#endif

  template <typename T> LIBC_INLINE T *decrypt_next(T *next_val) const {
#if LIBC_COPT_HARDEN_FREELIST
    return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(next_val) ^ k0);
#else
    return next_val;
#endif
  }

  template <typename T>
  LIBC_INLINE T *decrypt_prev([[maybe_unused]] const void *node,
                              T *prev_val) const {
#if LIBC_COPT_HARDEN_FREELIST
    uintptr_t val = reinterpret_cast<uintptr_t>(prev_val) ^ k2 ^
                    reinterpret_cast<uintptr_t>(node);
    val = cpp::rotl(val, NODE_PTR_ROTATE_DISTANCE);
    return reinterpret_cast<T *>(val ^ k1);
#else
    return prev_val;
#endif
  }

  template <typename T> LIBC_INLINE T *encrypt_next(T *next_val) const {
#if LIBC_COPT_HARDEN_FREELIST
    return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(next_val) ^ k0);
#else
    return next_val;
#endif
  }

  template <typename T>
  LIBC_INLINE T *encrypt_prev([[maybe_unused]] const void *node,
                              T *prev_val) const {
#if LIBC_COPT_HARDEN_FREELIST
    uintptr_t val = reinterpret_cast<uintptr_t>(prev_val) ^ k1;
    val = cpp::rotr(val, NODE_PTR_ROTATE_DISTANCE);
    return reinterpret_cast<T *>(val ^ reinterpret_cast<uintptr_t>(node) ^ k2);
#else
    return prev_val;
#endif
  }
};

/// A circularly-linked FIFO list storing free Blocks. All Blocks on a list
/// are the same size. The blocks are referenced by Nodes in the list; the list
/// refers to these, but it does not own them.
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
    LIBC_INLINE BlockRef block() const {
      return BlockRef::from_usable_space(this);
    }

    /// @returns The block containing this node.
    LIBC_INLINE BlockRef block() { return BlockRef::from_usable_space(this); }

    /// @returns The inner size of blocks in the list containing this node.
    LIBC_INLINE size_t size() const { return block().inner_size(); }

  private:
    // Circularly linked pointers to adjacent nodes.
    Node *prev;
    Node *next;
    friend class FreeList;
  };

  LIBC_INLINE constexpr FreeList() : FreeList(nullptr) {}
  LIBC_INLINE constexpr FreeList(Node *begin) : begin_(begin) {}

  LIBC_INLINE bool empty() const { return !begin_; }

  /// @returns The inner size of blocks in the list.
  LIBC_INLINE size_t size() const {
    LIBC_ASSERT(begin_ && "empty lists have no size");
    return begin_->size();
  }

  /// @returns The first node in the list.
  LIBC_INLINE Node *begin() { return begin_; }

  /// @returns The first block in the list.
  LIBC_INLINE BlockRef front() { return begin_->block(); }

  LIBC_INLINE Node *next_node(const Node *node,
                              const FreeListSecrets &secrets) const {
    return node ? secrets.decrypt_next(node->next) : nullptr;
  }

  LIBC_INLINE Node *prev_node(const Node *node,
                              const FreeListSecrets &secrets) const {
    return node ? secrets.decrypt_prev(node, node->prev) : nullptr;
  }

  /// Push a block to the back of the list.
  /// The block must be large enough to contain a node.
  LIBC_INLINE void push(BlockRef block, const FreeListSecrets &secrets) {
    LIBC_ASSERT(!block.used() &&
                "only free blocks can be placed on free lists");
    LIBC_ASSERT(block.inner_size_free() >= sizeof(Node) &&
                "block too small to accomodate free list node");
    push(new (block.usable_space()) Node, secrets);
  }

  /// Push an already-constructed node to the back of the list.
  /// This allows pushing derived node types with additional data.
  void push(Node *node, const FreeListSecrets &secrets);

  /// Pop the first node from the list.
  LIBC_INLINE void pop(const FreeListSecrets &secrets) {
    remove(begin_, secrets);
  }

  /// Remove an arbitrary node from the list.
  void remove(Node *node, const FreeListSecrets &secrets);

  /// Verify secret invariants for all nodes in the list.
  void sanitize(const FreeListSecrets &secrets) const;

private:
  Node *begin_;
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREELIST_H
