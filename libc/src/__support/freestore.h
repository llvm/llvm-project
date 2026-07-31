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

/// Default configuration for TLSFFreeStore.
struct DefaultFreeStoreConfig {
  static constexpr size_t UNIT_SIZE = BlockRef::MIN_ALIGN;
  static constexpr size_t STEP_SIZE_BITS = 3;
  static constexpr size_t NUM_STEP_BITS = 2;
  static constexpr size_t NUM_TABLE_ENTRIES = sizeof(uintptr_t) == 8 ? 3 : 6;
  static constexpr bool USE_TRIE_FOR_OVERFLOW_BIN = true;
  static constexpr size_t LINEAR_SCAN_LIMIT = 16;
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

public:
  static constexpr size_t MIN_OUTER_SIZE = align_up(
      BlockRef::HEADER_SIZE + sizeof(FreeList::Node), BlockRef::MIN_ALIGN);
  static constexpr size_t MIN_INNER_SIZE =
      MIN_OUTER_SIZE - BlockRef::HEADER_SIZE + BlockRef::PREV_FIELD_SIZE;
  static constexpr size_t LINEAR_BINS =
      (CONFIG::UNIT_SIZE == BlockRef::MIN_ALIGN)
          ? ((EXP_BASE << UNIT_SIZE_LOG2) - MIN_INNER_SIZE) /
                    CONFIG::UNIT_SIZE +
                1
          : EXP_BASE;

  LIBC_INLINE TLSFFreeStoreImpl() = default;
  LIBC_INLINE TLSFFreeStoreImpl(const TLSFFreeStoreImpl &other) = delete;
  LIBC_INLINE TLSFFreeStoreImpl &
  operator=(const TLSFFreeStoreImpl &other) = delete;

  LIBC_INLINE static constexpr size_t index_to_min_size(size_t index);
  LIBC_INLINE void set_range(FreeTrie::SizeRange range);
  LIBC_INLINE void insert(BlockRef block);
  LIBC_INLINE void remove(BlockRef block);
  LIBC_INLINE BlockRef remove_best_fit(size_t size) {
    return find_and_remove_fit(size);
  }
  LIBC_INLINE BlockRef find_and_remove_fit(size_t size);
  LIBC_INLINE static constexpr size_t size_to_bit_index(size_t size);

protected:
  LIBC_INLINE static bool too_small(BlockRef block) {
    return block.outer_size() < MIN_OUTER_SIZE;
  }

  cpp::array<uintptr_t, CONFIG::NUM_TABLE_ENTRIES> lookup_table{};
  cpp::array<FreeList, TOTAL_BITS - 1> free_lists{};
  FreeTrie trie{};
  FreeList overflow_list{};

  LIBC_INLINE void set_bit(size_t bit_index);
  LIBC_INLINE void clear_bit(size_t bit_index);
  LIBC_INLINE bool get_bit(size_t bit_index) const;
  LIBC_INLINE size_t find_first_bit_set_after(size_t bit_index) const;
  LIBC_INLINE BlockRef remove_first_fit_in_list(size_t index, size_t size);
  LIBC_INLINE BlockRef find_and_remove_fit_in_trie(size_t size);
};

template <typename CONFIG>
LIBC_INLINE constexpr size_t
TLSFFreeStoreImpl<CONFIG>::size_to_bit_index(size_t size) {
  if (size <= (EXP_BASE << UNIT_SIZE_LOG2)) {
    if constexpr (CONFIG::UNIT_SIZE == BlockRef::MIN_ALIGN) {
      if (size <= MIN_INNER_SIZE)
        return 0;
      return ((size - MIN_INNER_SIZE - 1) >> UNIT_SIZE_LOG2) + 1;
    }
    return size >> UNIT_SIZE_LOG2;
  }

  size_t size_ilog2 = static_cast<size_t>(cpp::bit_width(size) - 1);
  size_t exp_offset = (size_ilog2 - UNIT_SIZE_LOG2 - EXP_BASE_LOG2 - 1)
                      << CONFIG::NUM_STEP_BITS;
  size_t step_index = size >> (size_ilog2 - CONFIG::NUM_STEP_BITS);
  size_t index = LINEAR_BINS + exp_offset + step_index;

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
LIBC_INLINE constexpr size_t
TLSFFreeStoreImpl<CONFIG>::index_to_min_size(size_t index) {
  if (index < LINEAR_BINS) {
    if constexpr (CONFIG::UNIT_SIZE == BlockRef::MIN_ALIGN) {
      if (index == 0)
        return 0;
      return MIN_INNER_SIZE + (index - 1) * CONFIG::UNIT_SIZE + 1;
    }
    return index << UNIT_SIZE_LOG2;
  }

  size_t local_index = index - LINEAR_BINS;
  size_t exp_index = local_index >> CONFIG::NUM_STEP_BITS;
  size_t linear_index = local_index & (NUM_STEPS - 1);

  size_t row_base = (EXP_BASE << exp_index) << UNIT_SIZE_LOG2;
  size_t step_size = (STEP_SIZE << exp_index) << UNIT_SIZE_LOG2;
  return row_base + linear_index * step_size;
}

template <typename CONFIG>
LIBC_INLINE void
TLSFFreeStoreImpl<CONFIG>::set_range(FreeTrie::SizeRange range) {
  if constexpr (USE_TRIE) {
    size_t heap_max = range.min + range.width;
    size_t overflow_min = index_to_min_size(TOTAL_BITS - 1);
    size_t width = 1;
    if (heap_max > overflow_min)
      width = cpp::bit_ceil(heap_max - overflow_min);
    trie.set_range(FreeTrie::SizeRange(overflow_min, width));
  }
}

template <typename CONFIG>
LIBC_INLINE BlockRef
TLSFFreeStoreImpl<CONFIG>::find_and_remove_fit_in_trie(size_t size) {
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
LIBC_INLINE void TLSFFreeStoreImpl<CONFIG>::insert(BlockRef block) {
  if (too_small(block))
    return;
  size_t bit_index = size_to_bit_index(block.inner_size());

  if (bit_index == TOTAL_BITS - 1) {
    if constexpr (USE_TRIE)
      trie.push(block);
    else
      overflow_list.push(block);
    set_bit(bit_index);
    return;
  }

  free_lists[bit_index].push(block);
  set_bit(bit_index);
}

template <typename CONFIG>
LIBC_INLINE void TLSFFreeStoreImpl<CONFIG>::remove(BlockRef block) {
  if (too_small(block))
    return;
  size_t bit_index = size_to_bit_index(block.inner_size());

  if (bit_index == TOTAL_BITS - 1) {
    if constexpr (USE_TRIE) {
      trie.remove(reinterpret_cast<FreeTrie::Node *>(block.usable_space()));
      if (trie.empty())
        clear_bit(bit_index);
    } else {
      overflow_list.remove(
          reinterpret_cast<FreeList::Node *>(block.usable_space()));
      if (overflow_list.empty())
        clear_bit(bit_index);
    }
    return;
  }

  free_lists[bit_index].remove(
      reinterpret_cast<FreeList::Node *>(block.usable_space()));
  if (free_lists[bit_index].empty())
    clear_bit(bit_index);
}

template <typename CONFIG>
LIBC_INLINE BlockRef
TLSFFreeStoreImpl<CONFIG>::remove_first_fit_in_list(size_t index, size_t size) {
  FreeList &list =
      (index == TOTAL_BITS - 1) ? overflow_list : free_lists[index];
  FreeList::Node *begin_node = list.begin();
  if (begin_node == nullptr)
    return BlockRef();

  FreeList::Node *cur = begin_node;
  size_t count = 0;
  do {
    if (cur->size() >= size) {
      list.remove(cur);
      if (list.empty())
        clear_bit(index);
      return cur->block();
    }
    cur = cur->next();
    ++count;
  } while (cur != begin_node && count < CONFIG::LINEAR_SCAN_LIMIT);

  return BlockRef();
}

template <typename CONFIG>
LIBC_INLINE BlockRef
TLSFFreeStoreImpl<CONFIG>::find_and_remove_fit(size_t size) {

  size_t bit_index = size_to_bit_index(size);

  // Fast path for small linear bins if UNIT_SIZE == MIN_ALIGN
  if constexpr (CONFIG::UNIT_SIZE == BlockRef::MIN_ALIGN) {
    if (LIBC_LIKELY(bit_index < LINEAR_BINS && get_bit(bit_index))) {
      BlockRef block = free_lists[bit_index].front();
      free_lists[bit_index].pop();
      if (free_lists[bit_index].empty())
        clear_bit(bit_index);
      return block;
    }
  }

  if (LIBC_UNLIKELY(bit_index >= TOTAL_BITS - 1)) {
    if constexpr (USE_TRIE)
      return find_and_remove_fit_in_trie(size);
    else
      return remove_first_fit_in_list(TOTAL_BITS - 1, size);
  }

  // 1. Try oversized bins (guaranteed fit, but larger).
  size_t oversized_bit = find_first_bit_set_after(bit_index);
  if (LIBC_LIKELY(oversized_bit < TOTAL_BITS)) {
    if (LIBC_UNLIKELY(oversized_bit == TOTAL_BITS - 1)) {
      if constexpr (USE_TRIE)
        return find_and_remove_fit_in_trie(size);
      else
        return remove_first_fit_in_list(TOTAL_BITS - 1, size);
    }

    BlockRef block = free_lists[oversized_bit].front();
    free_lists[oversized_bit].pop();
    if (free_lists[oversized_bit].empty())
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

template <typename CONFIG = DefaultFreeStoreConfig>
using TLSFFreeStore = TLSFFreeStoreImpl<CONFIG>;

using FreeStore = TLSFFreeStore<DefaultFreeStoreConfig>;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREESTORE_H
