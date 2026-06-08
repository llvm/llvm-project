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

#ifndef LLVM_LIBC_SRC___SUPPORT_TLSF_FREESTORE_H
#define LLVM_LIBC_SRC___SUPPORT_TLSF_FREESTORE_H

#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/block.h"
#include "src/__support/freelist.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

/// Configuration for TLSFFreeStore.
template <size_t UNIT_SIZE_VAL, size_t STEP_SIZE_BITS_VAL,
          size_t NUM_STEP_BITS_VAL, size_t NUM_TABLE_ENTRIES_VAL>
struct TLSFFreeStoreConfig {
  static constexpr size_t UNIT_SIZE = UNIT_SIZE_VAL;
  static constexpr size_t STEP_SIZE_BITS = STEP_SIZE_BITS_VAL;
  static constexpr size_t NUM_STEP_BITS = NUM_STEP_BITS_VAL;
  static constexpr size_t NUM_TABLE_ENTRIES = NUM_TABLE_ENTRIES_VAL;
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

public:
  static constexpr size_t MIN_OUTER_SIZE =
      align_up(sizeof(Block) + sizeof(FreeList::Node), Block::MIN_ALIGN);

  LIBC_INLINE TLSFFreeStoreImpl() = default;
  LIBC_INLINE TLSFFreeStoreImpl(const TLSFFreeStoreImpl &other) = delete;
  LIBC_INLINE TLSFFreeStoreImpl &
  operator=(const TLSFFreeStoreImpl &other) = delete;

  LIBC_INLINE void insert(Block *block);
  LIBC_INLINE void remove(Block *block);
  LIBC_INLINE Block *find_and_remove_fit(size_t size);

protected:
  LIBC_INLINE static bool too_small(Block *block) {
    return block->outer_size() < MIN_OUTER_SIZE;
  }

  cpp::array<uintptr_t, CONFIG::NUM_TABLE_ENTRIES> lookup_table{};
  cpp::array<FreeList, TOTAL_BITS> free_lists{};

  LIBC_INLINE static constexpr size_t size_to_bit_index(size_t size);
  LIBC_INLINE void set_bit(size_t bit_index);
  LIBC_INLINE void clear_bit(size_t bit_index);
  LIBC_INLINE bool get_bit(size_t bit_index) const;
  LIBC_INLINE size_t find_first_bit_set_after(size_t bit_index) const;
  LIBC_INLINE Block *remove_first_fit_in_list(size_t index, size_t size);
};

template <typename CONFIG>
LIBC_INLINE constexpr size_t
TLSFFreeStoreImpl<CONFIG>::size_to_bit_index(size_t size) {
  // Original branchy form, kept here to make the mapping easy to audit:
  //   size_t shifted_size = size / CONFIG::UNIT_SIZE;
  //   size_t index = shifted_size;
  //   if (shifted_size > EXP_BASE) {
  //     size_t exp_index = cpp::bit_width(shifted_size / EXP_BASE) - 1;
  //     size_t base_shifted = EXP_BASE << exp_index;
  //     size_t step_shifted = base_shifted >> CONFIG::NUM_STEP_BITS;
  //     size_t linear_index = (shifted_size - base_shifted) / step_shifted;
  //     index = EXP_BASE + NUM_STEPS * exp_index + linear_index;
  //   }
  //   return index < TOTAL_BITS ? index : TOTAL_BITS - 1;

  // =========================================================================
  // --- Optimized Shift-and-Add Indexing ---
  // =========================================================================
  // Intuition:
  // For sizes above EXP_BASE, the flat index is composed of an exponential
  // row index (determined by the highest set bit, size_ilog2) and a linear
  // column index (determined by the next NUM_STEP_BITS bits directly below
  // the highest set bit).
  //
  // Instead of subtracting the power-of-two row base and then shifting to
  // find the column index, we shift the entire value so the highest set bit
  // aligns with the NUM_STEPS boundary. This single shift operation extracts
  // both the column index and a fixed offset simultaneously:
  //
  //   size >> (size_ilog2 - NUM_STEP_BITS) = NUM_STEPS + column_index
  //
  // Substituting this into the flat index equation collapses the multi-stage
  // pipeline into a single, highly parallelizable shift-and-add sequence:
  //
  //   index = EXP_BASE
  //           + NUM_STEPS * (size_ilog2 - UNIT_SIZE_LOG2 - EXP_BASE_LOG2 - 1)
  //           + (size >> (size_ilog2 - NUM_STEP_BITS))
  //
  // Key Properties:
  // 1. Direct Leading Zero: Operates directly on the raw size parameter to
  //    count leading zeros, saving one shift instruction on the critical path.
  // 2. No Base Subtractions: Avoids subtracting the power-of-two row base,
  //    reducing dynamic data hazards.
  // 3. Underflow Safety: When (size_ilog2 - UNIT_SIZE_LOG2) == EXP_BASE_LOG2,
  //    the subtraction term underflows to standard two's complement, which
  //    shifts left to -NUM_STEPS, perfectly acting as subtraction in standard
  //    unsigned flat index calculations.
  //
  // Example (UNIT_SIZE=16, STEP_SIZE_BITS=2, NUM_STEP_BITS=4):
  //   - EXP_BASE = 64 (EXP_BASE_LOG2 = 6), NUM_STEPS = 16 (S = 4)
  //   - Let size = 2100 bytes
  //   - Raw version:
  //       shifted_size = 131
  //       131 / 64 = 2 -> exp_index = 1 (row base = 128, step = 8)
  //       linear_index = (131 - 128) / 8 = 0
  //       index = 64 + 16 * 1 + 0 = 80
  //   - Optimized version:
  //       size_ilog2 = 11 (bit_width(2100) - 1)
  //       exp_offset = (11 - 4 - 6 - 1) << 4 = 0
  //       step_index = 2100 >> (11 - 4) = 16
  //       index = 64 + 0 + 16 = 80
  // =========================================================================
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
LIBC_INLINE void TLSFFreeStoreImpl<CONFIG>::insert(Block *block) {
  if (too_small(block))
    return;
  size_t bit_index = size_to_bit_index(block->inner_size());
  free_lists[bit_index].push(block);
  set_bit(bit_index);
}

template <typename CONFIG>
LIBC_INLINE void TLSFFreeStoreImpl<CONFIG>::remove(Block *block) {
  if (too_small(block))
    return;
  size_t bit_index = size_to_bit_index(block->inner_size());
  free_lists[bit_index].remove(
      reinterpret_cast<FreeList::Node *>(block->usable_space()));
  if (free_lists[bit_index].empty())
    clear_bit(bit_index);
}

template <typename CONFIG>
LIBC_INLINE Block *
TLSFFreeStoreImpl<CONFIG>::remove_first_fit_in_list(size_t index, size_t size) {
  // Performs a linear search on the free list to find the first block that
  // fits. Note that this linear search only ever happens when searching the
  // exact-fit size class (or the overflow bin). For larger size classes found
  // through the bitmap, any block inside the list is guaranteed to fit.
  FreeList::Node *begin_node = free_lists[index].begin();
  if (begin_node == nullptr)
    return nullptr;

  FreeList::Node *cur = begin_node;
  do {
    if (cur->size() >= size) {
      free_lists[index].remove(cur);
      if (free_lists[index].empty())
        clear_bit(index);
      return cur->block();
    }
    cur = cur->next_node();
  } while (cur != begin_node);

  return nullptr;
}

template <typename CONFIG>
LIBC_INLINE Block *TLSFFreeStoreImpl<CONFIG>::find_and_remove_fit(size_t size) {
  size_t bit_index = size_to_bit_index(size);
  // If the computed bit index overflows the table structure, fallback to
  // searching the last freelist (which serves as the remainder/overflow bin).
  if (LIBC_UNLIKELY(bit_index >= TOTAL_BITS - 1))
    return remove_first_fit_in_list(TOTAL_BITS - 1, size);

  // Search for the first oversized free list that can guarantee a fit in O(1).
  size_t oversized_bit = find_first_bit_set_after(bit_index);
  if (LIBC_LIKELY(oversized_bit < TOTAL_BITS)) {
    Block *block = free_lists[oversized_bit].front();
    free_lists[oversized_bit].pop();
    if (free_lists[oversized_bit].empty())
      clear_bit(oversized_bit);
    return block;
  }

  // Fallback: Search the exact size class (requires a linear scan since the
  // bin contains a range of sizes and not all blocks are guaranteed to fit).
  // Check the bitmap bit first to bypass useless list head loading.
  if (get_bit(bit_index))
    if (Block *block = remove_first_fit_in_list(bit_index, size))
      return block;

  return nullptr;
}

template <size_t UNIT_SIZE, size_t STEP_SIZE_BITS, size_t NUM_STEP_BITS,
          size_t NUM_TABLE_ENTRIES>
using TLSFFreeStore =
    TLSFFreeStoreImpl<TLSFFreeStoreConfig<UNIT_SIZE, STEP_SIZE_BITS,
                                          NUM_STEP_BITS, NUM_TABLE_ENTRIES>>;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_TLSF_FREESTORE_H
