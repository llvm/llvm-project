//===-- Interface for freelist_malloc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_FREELIST_H
#define LLVM_LIBC_SRC_STDLIB_FREELIST_H

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/cstddef.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/span.h"
#include "src/__support/fixedvector.h"

namespace LIBC_NAMESPACE {

using cpp::span;

/// Basic [freelist](https://en.wikipedia.org/wiki/Free_list) implementation
/// for an allocator. This implementation buckets by chunk size, with a list
/// of user-provided buckets. Each bucket is a linked list of storage chunks.
/// Because this freelist uses the added chunks themselves as list nodes, there
/// is a lower bound of `sizeof(FreeList.FreeListNode)` bytes for chunks which
/// can be added to this freelist. There is also an implicit bucket for
/// "everything else", for chunks which do not fit into a bucket.
///
/// Each added chunk will be added to the smallest bucket under which it fits.
/// If it does not fit into any user-provided bucket, it will be added to the
/// default bucket.
///
/// As an example, assume that the `FreeList` is configured with buckets of
/// sizes {64, 128, 256, and 512} bytes. The internal state may look like the
/// following:
///
/// @code{.unparsed}
/// bucket[0] (64B) --> chunk[12B] --> chunk[42B] --> chunk[64B] --> NULL
/// bucket[1] (128B) --> chunk[65B] --> chunk[72B] --> NULL
/// bucket[2] (256B) --> NULL
/// bucket[3] (512B) --> chunk[312B] --> chunk[512B] --> chunk[416B] --> NULL
/// bucket[4] (implicit) --> chunk[1024B] --> chunk[513B] --> NULL
/// @endcode
///
/// Note that added chunks should be aligned to a 4-byte boundary.
template <size_t NUM_BUCKETS = 6> class FreeList {
public:
  // Remove copy/move ctors
  FreeList(const FreeList &other) = delete;
  FreeList(FreeList &&other) = delete;
  FreeList &operator=(const FreeList &other) = delete;
  FreeList &operator=(FreeList &&other) = delete;

  /// Adds a chunk to this freelist.
  bool add_chunk(cpp::span<cpp::byte> chunk);

  /// Finds an eligible chunk for an allocation of size `size`.
  ///
  /// @note This returns the first allocation possible within a given bucket;
  /// It does not currently optimize for finding the smallest chunk.
  ///
  /// @returns
  /// * On success - A span representing the chunk.
  /// * On failure (e.g. there were no chunks available for that allocation) -
  ///   A span with a size of 0.
  cpp::span<cpp::byte> find_chunk(size_t size) const;

  /// Removes a chunk from this freelist.
  bool remove_chunk(cpp::span<cpp::byte> chunk);

  /// For a given size, find which index into chunks_ the node should be written
  /// to.
  constexpr size_t find_chunk_ptr_for_size(size_t size, bool non_null) const;

  struct FreeListNode {
    FreeListNode *next;
    size_t size;
  };

  constexpr void set_freelist_node(FreeListNode &node,
                                   cpp::span<cpp::byte> chunk);

  constexpr explicit FreeList(const cpp::array<size_t, NUM_BUCKETS> &sizes)
      : chunks_(NUM_BUCKETS + 1, 0), sizes_(sizes.begin(), sizes.end()) {}

private:
  FixedVector<FreeList::FreeListNode *, NUM_BUCKETS + 1> chunks_;
  FixedVector<size_t, NUM_BUCKETS> sizes_;
};

template <size_t NUM_BUCKETS>
constexpr void FreeList<NUM_BUCKETS>::set_freelist_node(FreeListNode &node,
                                                        span<cpp::byte> chunk) {
  // Add it to the correct list.
  size_t chunk_ptr = find_chunk_ptr_for_size(chunk.size(), false);
  node.size = chunk.size();
  node.next = chunks_[chunk_ptr];
  chunks_[chunk_ptr] = &node;
}

template <size_t NUM_BUCKETS>
bool FreeList<NUM_BUCKETS>::add_chunk(span<cpp::byte> chunk) {
  // Check that the size is enough to actually store what we need
  if (chunk.size() < sizeof(FreeListNode))
    return false;

  FreeListNode *node = ::new (chunk.data()) FreeListNode;
  set_freelist_node(*node, chunk);

  return true;
}

template <size_t NUM_BUCKETS>
span<cpp::byte> FreeList<NUM_BUCKETS>::find_chunk(size_t size) const {
  if (size == 0)
    return span<cpp::byte>();

  size_t chunk_ptr = find_chunk_ptr_for_size(size, true);

  // Check that there's data. This catches the case where we run off the
  // end of the array
  if (chunks_[chunk_ptr] == nullptr)
    return span<cpp::byte>();

  // Now iterate up the buckets, walking each list to find a good candidate
  for (size_t i = chunk_ptr; i < chunks_.size(); i++) {
    FreeListNode *node = chunks_[static_cast<unsigned short>(i)];

    while (node != nullptr) {
      if (node->size >= size)
        return span<cpp::byte>(reinterpret_cast<cpp::byte *>(node), node->size);

      node = node->next;
    }
  }

  // If we get here, we've checked every block in every bucket. There's
  // nothing that can support this allocation.
  return span<cpp::byte>();
}

template <size_t NUM_BUCKETS>
bool FreeList<NUM_BUCKETS>::remove_chunk(span<cpp::byte> chunk) {
  size_t chunk_ptr = find_chunk_ptr_for_size(chunk.size(), true);

  // Check head first.
  if (chunks_[chunk_ptr] == nullptr)
    return false;

  FreeListNode *node = chunks_[chunk_ptr];
  if (reinterpret_cast<cpp::byte *>(node) == chunk.data()) {
    chunks_[chunk_ptr] = node->next;
    return true;
  }

  // No? Walk the nodes.
  node = chunks_[chunk_ptr];

  while (node->next != nullptr) {
    if (reinterpret_cast<cpp::byte *>(node->next) == chunk.data()) {
      // Found it, remove this node out of the chain
      node->next = node->next->next;
      return true;
    }

    node = node->next;
  }

  return false;
}

template <size_t NUM_BUCKETS>
constexpr size_t
FreeList<NUM_BUCKETS>::find_chunk_ptr_for_size(size_t size,
                                               bool non_null) const {
  size_t chunk_ptr = 0;
  for (chunk_ptr = 0u; chunk_ptr < sizes_.size(); chunk_ptr++) {
    if (sizes_[chunk_ptr] >= size &&
        (!non_null || chunks_[chunk_ptr] != nullptr)) {
      break;
    }
  }

  return chunk_ptr;
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_FREELIST_H
