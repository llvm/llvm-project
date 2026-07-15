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

/// Secrets used to encrypt and verify forward and backward pointers in free
/// lists when hardening is enabled.
///
/// When hardening is enabled (LIBC_COPT_HARDEN_FREELIST), we use XOR and
/// bit-rotation encoding to protect the forward (`next`) and backward (`prev`)
/// pointers in free list nodes. This encoding mechanism protects heap integrity
/// against memory corruption and software bugs.
///
/// Threat Model and Security Guarantees:
/// To effectively protect against intentional attackers who might attempt to
/// infer the secret keys from memory, this hardening mechanism relies on the
/// following assumptions:
///
/// - Header Separation: We assume that the allocator metadata (the heap header
///   and secret keys) is isolated from dynamic data memory. This ensures that a
///   linear buffer overflow read cannot easily dump the secret keys from
///   adjacent memory.
///
/// - Allocation Randomness: We assume that allocation and deallocation patterns
///   have sufficient randomness or diversity such that an attacker cannot
///   reliably predict the exact memory layout of free list nodes. If an
///   attacker could reliably guess node addresses and read forward/backward
///   pointers via local overflow reads, they could XOR known addresses against
///   encoded pointers to derive the secret keys.
struct FreeListSecrets {
  static constexpr int NODE_PTR_ROTATE_DISTANCE = 17;

#if LIBC_COPT_HARDEN_FREELIST
  uintptr_t forward_key;
  uintptr_t backward_key[2];

  LIBC_INLINE constexpr FreeListSecrets(uintptr_t forward_key, uintptr_t k1,
                                        uintptr_t k2)
      : forward_key(forward_key), backward_key{k1, k2} {}
#else
  LIBC_INLINE constexpr FreeListSecrets() = default;
#endif

  template <typename T> LIBC_INLINE T *flip_next(T *next_val) const {
#if LIBC_COPT_HARDEN_FREELIST
    return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(next_val) ^
                                 forward_key);
#else
    return next_val;
#endif
  }

  /// Encrypts/decrypts `prev` using XOR, bitwise rotation, and node address
  /// binding. Pure XOR chaining (A ^ K0 ^ B ^ K1) reduces to A ^ B ^ K due to
  /// commutativity. Bitwise rotation breaks commutativity between key XORs, and
  /// node address binding prevents reusing encoded pointers across different
  /// node locations.
  template <typename T>
  LIBC_INLINE T *decrypt_prev([[maybe_unused]] const void *node,
                              T *prev_val) const {
#if LIBC_COPT_HARDEN_FREELIST
    uintptr_t val = reinterpret_cast<uintptr_t>(prev_val) ^ backward_key[1] ^
                    reinterpret_cast<uintptr_t>(node);
    val = cpp::rotl(val, NODE_PTR_ROTATE_DISTANCE);
    return reinterpret_cast<T *>(val ^ backward_key[0]);
#else
    return prev_val;
#endif
  }

  template <typename T>
  LIBC_INLINE T *encrypt_prev([[maybe_unused]] const void *node,
                              T *prev_val) const {
#if LIBC_COPT_HARDEN_FREELIST
    uintptr_t val = reinterpret_cast<uintptr_t>(prev_val) ^ backward_key[0];
    val = cpp::rotr(val, NODE_PTR_ROTATE_DISTANCE);
    return reinterpret_cast<T *>(val ^ reinterpret_cast<uintptr_t>(node) ^
                                 backward_key[1]);
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
    return node ? secrets.flip_next(node->next) : nullptr;
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
