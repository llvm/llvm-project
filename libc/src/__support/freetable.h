//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the FreeTable class which represents
/// a table of free lists.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREETABLE_H
#define LLVM_LIBC_SRC___SUPPORT_FREETABLE_H

#include "hdr/types/size_t.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/block.h"
#include "src/__support/common.h"
#include "src/__support/freelist.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math_extras.h"

namespace LIBC_NAMESPACE_DECL {

// A two-level segregate table for free lists.
// The table starts with small lists that grows linearly for small sizes, which
// covers [0, ... UNIT * EXP_BASE].
// For larger sizes, the bits are managed in a 2-D table.
// One can think of each row containing NUM_STEPS lists.
// Along the row, the size grows by 2 exponentially; along the column,
// the size increases by STEP_SIZE linearly.
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
//   ┌───────┬───────┬───────┬───────┬───────┬───────────┬───────────────┐
//   │ [0 B] │ [32B] │ [64B] │ [96B] │  ...  │ [992 B]   │ [1024 B (Th)] │
//   └───────┴───────┴───────┴───────┴───────┴───────────┴───────────────┘
//
// 2. Large Sizes (2-D Table):
//    Rows = FL (Exponential growth), Columns = SL (Linear steps)
//    One can think of each Row containing NUM_STEPS (4) lists.
//
//                       LINEAR INCREASE ALONG COLUMN (SL) --->
//             ┌───────────────┬───────────────┬───────────────┬───────────────┐
//             │    Col = 0    │    Col = 1    │    Col = 2    │    Col = 3    │
//             │    (Base)     │   (+25% FL)   │   (+50% FL)   │   (+75% FL)   │
//   ┌─────────┼───────────────┼───────────────┼───────────────┼───────────────┤
// E │ Row = 0 │    1024 B     │    1280 B     │    1536 B     │    1792 B     │
// X │(Base 1K)│ [1024 - 1279] │ [1280 - 1535] │ [1536 - 1791] │ [1792 - 2047] │
// P ├─────────┼───────────────┼───────────────┼───────────────┼───────────────┤
// O │ Row = 1 │    2048 B     │    2560 B     │    3072 B     │    3584 B     │
// N │(Base 2K)│ [2048 - 2559] │ [2560 - 3071] │ [3072 - 3583] │ [3584 - 4095] │
// E ├─────────┼───────────────┼───────────────┼───────────────┼───────────────┤
// N │ Row = 2 │    4096 B     │    5120 B     │    6144 B     │    7168 B     │
// T │(Base 4K)│ [4096 - 5119] │ [5120 - 6143] │ [6144 - 7167] │ [7168 - 8191] │
// I ├─────────┼───────────────┼───────────────┼───────────────┼───────────────┤
// A │ Row = 3 │    8192 B     │   10240 B     │   12288 B     │   14336 B     │
// L │(Base 8K)│ [8192 - 10239]│[10240 - 12287]│[12288 - 14335]│[14336 - 16383]│
//   └─────────┴───────────────┴───────────────┴───────────────┴───────────────┘
//
// Note: For the real implementation, we don't actually store the lists in a 2-D
// structure. Instead, we flatten the entire 2-D layout into a single flat 1-D
// array of size `TOTAL_BITS` (`free_lists`), and map sizes directly to a
// continuous 1-D index using `size_to_bit_index`. The allocation state is
// tracked compactly in the `lookup_table` bitmask array.

// Separate config object to avoid repeated template parameters.
template <size_t UNIT_SIZE_VAL, size_t STEP_SIZE_BITS_VAL,
          size_t NUM_STEP_BITS_VAL, size_t NUM_TABLE_ENTRIES_VAL>
struct FreeTableConfig {
  // size of basic allocation units, should be power of 2
  constexpr static size_t UNIT_SIZE = UNIT_SIZE_VAL;
  // log2 of linear division step size
  constexpr static size_t STEP_SIZE_BITS = STEP_SIZE_BITS_VAL;
  // log2 of number of steps per exponential level
  constexpr static size_t NUM_STEP_BITS = NUM_STEP_BITS_VAL;
  // total number of entries in the table
  constexpr static size_t NUM_TABLE_ENTRIES = NUM_TABLE_ENTRIES_VAL;
};

template <typename CONFIG> struct FreeTableImpl {
protected:
  constexpr static size_t STEP_SIZE = size_t(1) << CONFIG::STEP_SIZE_BITS;
  constexpr static size_t NUM_STEPS = size_t(1) << CONFIG::NUM_STEP_BITS;
  constexpr static size_t EXP_BASE = STEP_SIZE * NUM_STEPS;
  constexpr static size_t LARGE_SIZE_THRESHOLD = CONFIG::UNIT_SIZE * EXP_BASE;
  constexpr static size_t BITS_PER_ENTRY =
      cpp::numeric_limits<uintptr_t>::digits;
  constexpr static size_t TOTAL_BITS =
      CONFIG::NUM_TABLE_ENTRIES * BITS_PER_ENTRY;

public:
  constexpr static size_t MIN_OUTER_SIZE =
      align_up(sizeof(Block) + sizeof(FreeList::Node), Block::MIN_ALIGN);

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

public:
  LIBC_INLINE void insert(Block *block);
  LIBC_INLINE void remove(Block *block);
  LIBC_INLINE Block *find_and_remove_fit(size_t size);
  LIBC_INLINE FreeTableImpl() = default;
};

template <typename CONFIG>
LIBC_INLINE constexpr size_t
FreeTableImpl<CONFIG>::size_to_bit_index(size_t size) {
  size_t shifted_size = size / CONFIG::UNIT_SIZE;
  size_t index = shifted_size;
  if (shifted_size > EXP_BASE) {
    size_t exp_index = floor_ilog2(shifted_size / EXP_BASE);
    size_t base_shifted = EXP_BASE << exp_index;
    size_t step_shifted = base_shifted >> CONFIG::NUM_STEP_BITS;
    size_t linear_index = (shifted_size - base_shifted) / step_shifted;
    index = EXP_BASE + NUM_STEPS * exp_index + linear_index;
  }
  return index < TOTAL_BITS ? index : TOTAL_BITS - 1;
}

template <typename CONFIG>
LIBC_INLINE void FreeTableImpl<CONFIG>::set_bit(size_t bit_index) {
  size_t entry_index = bit_index / BITS_PER_ENTRY;
  size_t bit_offset = bit_index % BITS_PER_ENTRY;
  lookup_table[entry_index] |= (uintptr_t(1) << bit_offset);
}

template <typename CONFIG>
LIBC_INLINE void FreeTableImpl<CONFIG>::clear_bit(size_t bit_index) {
  size_t entry_index = bit_index / BITS_PER_ENTRY;
  size_t bit_offset = bit_index % BITS_PER_ENTRY;
  lookup_table[entry_index] &= ~(uintptr_t(1) << bit_offset);
}

template <typename CONFIG>
LIBC_INLINE bool FreeTableImpl<CONFIG>::get_bit(size_t bit_index) const {
  size_t entry_index = bit_index / BITS_PER_ENTRY;
  size_t bit_offset = bit_index % BITS_PER_ENTRY;
  return (lookup_table[entry_index] & (uintptr_t(1) << bit_offset)) != 0;
}

template <typename CONFIG>
LIBC_INLINE size_t
FreeTableImpl<CONFIG>::find_first_bit_set_after(size_t bit_index) const {
  size_t target_index = bit_index + 1;
  if (target_index >= TOTAL_BITS)
    return TOTAL_BITS;
  size_t start_entry = target_index / BITS_PER_ENTRY;
  size_t bit_offset = target_index % BITS_PER_ENTRY;

  uintptr_t val = lookup_table[start_entry] & (~uintptr_t(0) << bit_offset);
  if (val != 0)
    return start_entry * BITS_PER_ENTRY +
           static_cast<size_t>(cpp::countr_zero(val));

  for (size_t i = start_entry + 1; i < CONFIG::NUM_TABLE_ENTRIES; ++i) {
    uintptr_t v = lookup_table[i];
    if (v != 0)
      return i * BITS_PER_ENTRY + static_cast<size_t>(cpp::countr_zero(v));
  }
  return TOTAL_BITS;
}

template <typename CONFIG>
LIBC_INLINE void FreeTableImpl<CONFIG>::insert(Block *block) {
  if (too_small(block))
    return;
  size_t bit_index = size_to_bit_index(block->inner_size());
  free_lists[bit_index].push(block);
  set_bit(bit_index);
}

template <typename CONFIG>
LIBC_INLINE void FreeTableImpl<CONFIG>::remove(Block *block) {
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
FreeTableImpl<CONFIG>::remove_first_fit_in_list(size_t index, size_t size) {
  // Performs a linear search on the free list to find the first block that
  // fits. Note that this linear search only ever happens during large
  // allocations when falling back to searching the exact-fit size class (or the
  // overflow bin). For standard small allocations, we always find a guaranteed
  // fit in a larger oversized freelist, allowing a one-shot O(1) pop.
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
LIBC_INLINE Block *FreeTableImpl<CONFIG>::find_and_remove_fit(size_t size) {
  size_t bit_index = size_to_bit_index(size);
  // If the computed bit index overflows the table structure, fallback to
  // searching the last freelist (which serves as the remainder/overflow bin).
  if (LIBC_UNLIKELY(bit_index >= TOTAL_BITS - 1))
    return remove_first_fit_in_list(TOTAL_BITS - 1, size);

  // Search for the first oversized free list that can guarantee a fit.
  size_t first_oversized_bit = find_first_bit_set_after(bit_index);
  // If no larger free list has blocks, fallback to searching the exact free
  // list.
  if (LIBC_UNLIKELY(first_oversized_bit >= TOTAL_BITS))
    return remove_first_fit_in_list(bit_index, size);

  // If a larger free list is found, any block inside it is guaranteed to fit
  // the requested size. Pop the first block in FIFO order.
  Block *block = free_lists[first_oversized_bit].front();
  free_lists[first_oversized_bit].pop();
  if (free_lists[first_oversized_bit].empty())
    clear_bit(first_oversized_bit);
  return block;
}

template <size_t UNIT_SIZE, size_t STEP_SIZE_BITS, size_t NUM_STEP_BITS,
          size_t NUM_TABLE_ENTRIES>
using FreeTable =
    FreeTableImpl<FreeTableConfig<UNIT_SIZE, STEP_SIZE_BITS, NUM_STEP_BITS,
                                  NUM_TABLE_ENTRIES>>;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREETABLE_H
