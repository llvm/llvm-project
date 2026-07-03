//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains a two-level segregated fit free block store.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREESTORE_H
#define LLVM_LIBC_SRC___SUPPORT_FREESTORE_H

#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/block.h"
#include "src/__support/freelist.h"
#include "src/__support/freetrie.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

/// Configuration for TLSFFreeStore.
template <size_t UNIT_SIZE_VAL, size_t STEP_SIZE_BITS_VAL,
          size_t NUM_STEP_BITS_VAL, size_t NUM_TABLE_ENTRIES_VAL,
          bool USE_TRIE_FOR_OVERFLOW_BIN_VAL = false>
struct TLSFFreeStoreConfig {
  static constexpr size_t UNIT_SIZE = UNIT_SIZE_VAL;
  static constexpr size_t STEP_SIZE_BITS = STEP_SIZE_BITS_VAL;
  static constexpr size_t NUM_STEP_BITS = NUM_STEP_BITS_VAL;
  static constexpr size_t NUM_TABLE_ENTRIES = NUM_TABLE_ENTRIES_VAL;
  static constexpr bool USE_TRIE_FOR_OVERFLOW_BIN =
      USE_TRIE_FOR_OVERFLOW_BIN_VAL;
};

// A two-level segregated fit store for free blocks.
//
// The store starts with small lists that grow linearly for small sizes, which
// covers [0, ... UNIT_SIZE * EXP_BASE]. For larger sizes, the bits are managed
// in a 2-D table. One can think of each row containing NUM_STEPS lists. Along
// the row, the size grows by 2 exponentially; along the column, the size
// increases by STEP_SIZE linearly.
//
// Mathematical layout:
//   STEP_SIZE = 1 << STEP_SIZE_BITS
//   NUM_STEPS = 1 << NUM_STEP_BITS
//   EXP_BASE = STEP_SIZE * NUM_STEPS
//   LARGE_SIZE_THRESHOLD = UNIT_SIZE * EXP_BASE
//
// Visual representation with example parameters:
//   UNIT_SIZE = 32, STEP_SIZE = 8, NUM_STEPS = 4
//   EXP_BASE = 32, THRESHOLD = 1024 B (1 KiB)
//
// 1. Small Sizes (Linear Array):
//    Covers [0, ... 1024 B] growing directly by UNIT_SIZE = 32 B
//   +-------+-------+-------+-------+-------+-----------+---------------+
//   | [0 B] | [32B] | [64B] | [96B] |  ...  | [992 B]   | [1024 B (Th)] |
//   +-------+-------+-------+-------+-------+-----------+---------------+
//
// 2. Large Sizes (2-D Table):
//    Rows = FL (Exponential growth), Columns = SL (Linear steps)
//    One can think of each Row containing NUM_STEPS (4) lists.
//
//                       LINEAR INCREASE ALONG COLUMN (SL) --->
//             +---------------+---------------+---------------+---------------+
//             |    Col = 0    |    Col = 1    |    Col = 2    |    Col = 3    |
//             |    (Base)     |   (+25% FL)   |   (+50% FL)   |   (+75% FL)   |
//   +---------+---------------+---------------+---------------+---------------+
// E | Row = 0 |    1024 B     |    1280 B     |    1536 B     |    1792 B     |
// X |(Base 1K)| [1024 - 1279] | [1280 - 1535] | [1536 - 1791] | [1792 - 2047] |
// P +---------+---------------+---------------+---------------+---------------+
// O | Row = 1 |    2048 B     |    2560 B     |    3072 B     |    3584 B     |
// N |(Base 2K)| [2048 - 2559] | [2560 - 3071] | [3072 - 3583] | [3584 - 4095] |
// E +---------+---------------+---------------+---------------+---------------+
// N | Row = 2 |    4096 B     |    5120 B     |    6144 B     |    7168 B     |
// T |(Base 4K)| [4096 - 5119] | [5120 - 6143] | [6144 - 7167] | [7168 - 8191] |
// I +---------+---------------+---------------+---------------+---------------+
// A | Row = 3 |    8192 B     |   10240 B     |   12288 B     |   14336 B     |
// L |(Base 8K)|[8192 - 10239]|[10240 - 12287]|[12288 - 14335]|[14336 - 16383]|
//   +---------+---------------+---------------+---------------+---------------+
//
// Note: For the real implementation, we don't actually store the lists in a
// 2-D structure. Instead, we flatten the entire 2-D layout into a single
// flat 1-D array of size TOTAL_BITS (free_lists), and map sizes directly to
// a continuous 1-D index using size_to_bit_index. The allocation state is
// tracked compactly in the lookup_table bitmask array.
template <typename CONFIG> class TLSFFreeStoreImpl {
protected:
  static_assert(cpp::has_single_bit(CONFIG::UNIT_SIZE),
                "unit size must be a power of two");
  static_assert(CONFIG::NUM_TABLE_ENTRIES > 0,
                "the lookup table must have at least one entry");

  static constexpr size_t STEP_SIZE = size_t(1) << CONFIG::STEP_SIZE_BITS;
  static constexpr size_t NUM_STEPS = size_t(1) << CONFIG::NUM_STEP_BITS;
  static constexpr size_t EXP_BASE = STEP_SIZE * NUM_STEPS;
  static constexpr int UNIT_SIZE_LOG2 = cpp::bit_width(CONFIG::UNIT_SIZE) - 1;
  static constexpr int EXP_BASE_LOG2 =
      CONFIG::STEP_SIZE_BITS + CONFIG::NUM_STEP_BITS;
  static constexpr size_t BITS_PER_ENTRY =
      cpp::numeric_limits<uintptr_t>::digits;
  static constexpr size_t TOTAL_BITS =
      CONFIG::NUM_TABLE_ENTRIES * BITS_PER_ENTRY;
  static constexpr bool USE_TRIE = CONFIG::USE_TRIE_FOR_OVERFLOW_BIN;
  static constexpr size_t OVERFLOW_WIDTH =
      size_t(1) << (cpp::numeric_limits<size_t>::digits - 2);

public:
  static constexpr size_t MIN_OUTER_SIZE = align_up(
      BlockRef::HEADER_SIZE + sizeof(FreeList::Node), BlockRef::MIN_ALIGN);

  LIBC_INLINE TLSFFreeStoreImpl() = default;
  LIBC_INLINE TLSFFreeStoreImpl(const TLSFFreeStoreImpl &other) = delete;
  LIBC_INLINE TLSFFreeStoreImpl &
  operator=(const TLSFFreeStoreImpl &other) = delete;

  LIBC_INLINE void insert(BlockRef block);
  LIBC_INLINE void remove(BlockRef block);
  LIBC_INLINE BlockRef remove_best_fit(size_t size) {
    return find_and_remove_fit(size);
  }
  LIBC_INLINE BlockRef find_and_remove_fit(size_t size);

protected:
  LIBC_INLINE static bool too_small(BlockRef block) {
    return block.outer_size() < MIN_OUTER_SIZE;
  }

  union ListOrTrie {
    FreeList list;
    FreeTrie::Node *trie_root;

    LIBC_INLINE constexpr ListOrTrie() : trie_root(nullptr) {}
  };

  cpp::array<uintptr_t, CONFIG::NUM_TABLE_ENTRIES> lookup_table{};
  cpp::array<ListOrTrie, TOTAL_BITS> free_lists{};

  LIBC_INLINE static constexpr size_t size_to_bit_index(size_t size);
  LIBC_INLINE void set_bit(size_t bit_index);
  LIBC_INLINE void clear_bit(size_t bit_index);
  LIBC_INLINE bool get_bit(size_t bit_index) const;
  LIBC_INLINE size_t find_first_bit_set_after(size_t bit_index) const;
  LIBC_INLINE BlockRef remove_first_fit_in_list(size_t index, size_t size);
  LIBC_INLINE FreeTrie get_trie();
  LIBC_INLINE BlockRef find_and_remove_fit_in_trie(size_t size);
  LIBC_INLINE BlockRef pop_min_in_trie();
};

template <typename CONFIG>
LIBC_INLINE constexpr size_t
TLSFFreeStoreImpl<CONFIG>::size_to_bit_index(size_t size) {
  if (size <= (EXP_BASE << UNIT_SIZE_LOG2))
    return size >> UNIT_SIZE_LOG2;

  size_t size_ilog2 = static_cast<size_t>(cpp::bit_width(size) - 1);
  size_t exp_offset = (size_ilog2 - UNIT_SIZE_LOG2 - EXP_BASE_LOG2 - 1)
                      << CONFIG::NUM_STEP_BITS;
  size_t step_index = size >> (size_ilog2 - CONFIG::NUM_STEP_BITS);
  size_t index = EXP_BASE + exp_offset + step_index;

  return index < TOTAL_BITS ? index : TOTAL_BITS - 1;
}

template <typename CONFIG>
LIBC_INLINE void TLSFFreeStoreImpl<CONFIG>::set_bit(size_t bit_index) {
  size_t entry_index = bit_index / BITS_PER_ENTRY;
  size_t bit_offset = bit_index % BITS_PER_ENTRY;
  lookup_table[entry_index] |= uintptr_t(1) << bit_offset;
}

template <typename CONFIG>
LIBC_INLINE void TLSFFreeStoreImpl<CONFIG>::clear_bit(size_t bit_index) {
  size_t entry_index = bit_index / BITS_PER_ENTRY;
  size_t bit_offset = bit_index % BITS_PER_ENTRY;
  lookup_table[entry_index] &= ~(uintptr_t(1) << bit_offset);
}

template <typename CONFIG>
LIBC_INLINE bool TLSFFreeStoreImpl<CONFIG>::get_bit(size_t bit_index) const {
  size_t entry_index = bit_index / BITS_PER_ENTRY;
  size_t bit_offset = bit_index % BITS_PER_ENTRY;
  return (lookup_table[entry_index] & (uintptr_t(1) << bit_offset)) != 0;
}

template <typename CONFIG>
LIBC_INLINE size_t
TLSFFreeStoreImpl<CONFIG>::find_first_bit_set_after(size_t bit_index) const {
  if (bit_index >= TOTAL_BITS - 1)
    return TOTAL_BITS;

  size_t target_index = bit_index + 1;
  size_t start_entry = target_index / BITS_PER_ENTRY;
  size_t bit_offset = target_index % BITS_PER_ENTRY;

  uintptr_t value = lookup_table[start_entry] & (~uintptr_t(0) << bit_offset);
  if (value != 0)
    return start_entry * BITS_PER_ENTRY +
           static_cast<size_t>(cpp::countr_zero(value));

  for (size_t i = start_entry + 1; i < CONFIG::NUM_TABLE_ENTRIES; ++i) {
    value = lookup_table[i];
    if (value != 0)
      return i * BITS_PER_ENTRY + static_cast<size_t>(cpp::countr_zero(value));
  }
  return TOTAL_BITS;
}

template <typename CONFIG>
LIBC_INLINE FreeTrie TLSFFreeStoreImpl<CONFIG>::get_trie() {
  return FreeTrie(free_lists[TOTAL_BITS - 1].trie_root,
                  FreeTrie::SizeRange(0, OVERFLOW_WIDTH));
}

template <typename CONFIG>
LIBC_INLINE BlockRef
TLSFFreeStoreImpl<CONFIG>::find_and_remove_fit_in_trie(size_t size) {
  FreeTrie trie = get_trie();
  if (FreeTrie::Node *best_fit = trie.find_best_fit(size)) {
    BlockRef block = best_fit->block();
    trie.remove(best_fit);
    if (trie.empty())
      clear_bit(TOTAL_BITS - 1);
    return block;
  }
  return BlockRef();
}

template <typename CONFIG>
LIBC_INLINE BlockRef TLSFFreeStoreImpl<CONFIG>::pop_min_in_trie() {
  FreeTrie trie = get_trie();
  FreeTrie::Node *min_node = trie.pop_min();
  LIBC_ASSERT(min_node && "bit was set but trie is empty");
  BlockRef block = min_node->block();
  if (trie.empty())
    clear_bit(TOTAL_BITS - 1);
  return block;
}

template <typename CONFIG>
LIBC_INLINE void TLSFFreeStoreImpl<CONFIG>::insert(BlockRef block) {
  if (too_small(block))
    return;
  size_t bit_index = size_to_bit_index(block.inner_size());

  if constexpr (USE_TRIE)
    if (bit_index == TOTAL_BITS - 1) {
      get_trie().push(block);
      set_bit(bit_index);
      return;
    }

  free_lists[bit_index].list.push(block);
  set_bit(bit_index);
}

template <typename CONFIG>
LIBC_INLINE void TLSFFreeStoreImpl<CONFIG>::remove(BlockRef block) {
  if (too_small(block))
    return;
  size_t bit_index = size_to_bit_index(block.inner_size());

  if constexpr (USE_TRIE)
    if (bit_index == TOTAL_BITS - 1) {
      FreeTrie trie = get_trie();
      trie.remove(reinterpret_cast<FreeTrie::Node *>(block.usable_space()));
      if (trie.empty())
        clear_bit(bit_index);
      return;
    }

  free_lists[bit_index].list.remove(
      reinterpret_cast<FreeList::Node *>(block.usable_space()));
  if (free_lists[bit_index].list.empty())
    clear_bit(bit_index);
}

template <typename CONFIG>
LIBC_INLINE BlockRef
TLSFFreeStoreImpl<CONFIG>::remove_first_fit_in_list(size_t index, size_t size) {
  FreeList::Node *begin_node = free_lists[index].list.begin();
  if (begin_node == nullptr)
    return BlockRef();

  FreeList::Node *cur = begin_node;
  do {
    if (cur->size() >= size) {
      free_lists[index].list.remove(cur);
      if (free_lists[index].list.empty())
        clear_bit(index);
      return cur->block();
    }
    cur = cur->next_node();
  } while (cur != begin_node);

  return BlockRef();
}

template <typename CONFIG>
LIBC_INLINE BlockRef
TLSFFreeStoreImpl<CONFIG>::find_and_remove_fit(size_t size) {
  size_t bit_index = size_to_bit_index(size);

  if (LIBC_UNLIKELY(bit_index >= TOTAL_BITS - 1)) {
    if constexpr (USE_TRIE)
      return find_and_remove_fit_in_trie(size);
    else
      return remove_first_fit_in_list(TOTAL_BITS - 1, size);
  }

  // 1. Try oversized bins (guaranteed fit, but larger).
  size_t oversized_bit = find_first_bit_set_after(bit_index);
  if (LIBC_LIKELY(oversized_bit < TOTAL_BITS)) {
    if constexpr (USE_TRIE) {
      if (oversized_bit == TOTAL_BITS - 1)
        return pop_min_in_trie();
    }

    BlockRef block = free_lists[oversized_bit].list.front();
    free_lists[oversized_bit].list.pop();
    if (free_lists[oversized_bit].list.empty())
      clear_bit(oversized_bit);
    return block;
  }

  // 2. Try exact fit (fallback).
  if (get_bit(bit_index)) {
    if (BlockRef block = remove_first_fit_in_list(bit_index, size))
      return block;
  }

  return BlockRef();
}

template <size_t UNIT_SIZE, size_t STEP_SIZE_BITS, size_t NUM_STEP_BITS,
          size_t NUM_TABLE_ENTRIES, bool USE_TRIE = false>
using TLSFFreeStore = TLSFFreeStoreImpl<TLSFFreeStoreConfig<
    UNIT_SIZE, STEP_SIZE_BITS, NUM_STEP_BITS, NUM_TABLE_ENTRIES, USE_TRIE>>;

#ifndef LIBC_COPT_USE_TRIE_FOR_OVERFLOW_BIN
#define LIBC_COPT_USE_TRIE_FOR_OVERFLOW_BIN false
#endif

using FreeStore =
    TLSFFreeStore<BlockRef::MIN_ALIGN, 3, 2, (sizeof(uintptr_t) == 8 ? 3 : 6),
                  LIBC_COPT_USE_TRIE_FOR_OVERFLOW_BIN>;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREESTORE_H
