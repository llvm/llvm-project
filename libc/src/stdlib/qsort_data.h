//===-- Data structures for sorting routines --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_QSORT_DATA_H
#define LLVM_LIBC_SRC_STDLIB_QSORT_DATA_H

#include "src/__support/CPP/cstddef.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memmove.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Returns the max amount of bytes deemed reasonable - based on the target
// properties - for use in local stack arrays.
constexpr size_t max_stack_array_size() {
  // Uses target pointer size as heuristic how much memory is available and
  // unlikely to run into stack overflow and perf problems.
  constexpr size_t ptr_diff_size = sizeof(ptrdiff_t);

  if constexpr (ptr_diff_size >= 8) {
    return 4096;
  }

  if constexpr (ptr_diff_size == 4) {
    return 512;
  }

  // 8-bit platforms are just not gonna work well with libc, qsort
  // won't be the problem.
  // 16-bit platforms ought to be able to store 64 bytes on the stack.
  return 64;
}

class ArrayGenericSize {
  uint8_t *array_base;
  size_t array_len;
  size_t elem_size;

  uint8_t *get_internal(size_t i) const noexcept {
    return array_base + (i * elem_size);
  }

public:
  ArrayGenericSize(uint8_t *a, size_t s, size_t e) noexcept
      : array_base(a), array_len(s), elem_size(e) {}

  static constexpr bool has_fixed_size() { return false; }

  void *get(size_t i) const noexcept {
    return reinterpret_cast<void *>(get_internal(i));
  }

  void swap(size_t i, size_t j) const noexcept {
    // It's possible to use 8 byte blocks with `uint64_t`, but that
    // generates more machine code as the remainder loop gets
    // unrolled, plus 4 byte operations are more likely to be
    // efficient on a wider variety of hardware. On x86 LLVM tends
    // to unroll the block loop again into 2 16 byte swaps per
    // iteration which is another reason that 4 byte blocks yields
    // good performance even for big types.
    using block_t = uint32_t;
    constexpr size_t BLOCK_SIZE = sizeof(block_t);

    uint8_t *elem_i = get_internal(i);
    uint8_t *elem_j = get_internal(j);

    const size_t elem_size_rem = elem_size % BLOCK_SIZE;
    const block_t *elem_i_block_end =
        reinterpret_cast<block_t *>(elem_i + (elem_size - elem_size_rem));

    block_t *elem_i_block = reinterpret_cast<block_t *>(elem_i);
    block_t *elem_j_block = reinterpret_cast<block_t *>(elem_j);

    while (elem_i_block != elem_i_block_end) {
      block_t tmp = *elem_i_block;
      *elem_i_block = *elem_j_block;
      *elem_j_block = tmp;
      elem_i_block += 1;
      elem_j_block += 1;
    }

    elem_i = reinterpret_cast<uint8_t *>(elem_i_block);
    elem_j = reinterpret_cast<uint8_t *>(elem_j_block);
    for (size_t n = 0; n < elem_size_rem; ++n) {
      uint8_t tmp = elem_i[n];
      elem_i[n] = elem_j[n];
      elem_j[n] = tmp;
    }
  }

  size_t len() const noexcept { return array_len; }

  // Make an Array starting at index |i| and length |s|.
  ArrayGenericSize make_array(size_t i, size_t s) const noexcept {
    return ArrayGenericSize(get_internal(i), s, elem_size);
  }

  // Reset this Array to point at a different interval of the same
  // items starting at index |i|.
  void reset_bounds(size_t i, size_t s) noexcept {
    array_base = get_internal(i);
    array_len = s;
  }
};

// Having a specialized Array type for sorting that knowns at
// compile-time what the size of the element is, allows for much more
// efficient swapping and for cheaper offset calculations.
template <size_t ELEM_SIZE> class ArrayFixedSize {
  uint8_t *array_base;
  size_t array_len;

  uint8_t *get_internal(size_t i) const noexcept {
    return array_base + (i * ELEM_SIZE);
  }

public:
  ArrayFixedSize(uint8_t *a, size_t s) noexcept : array_base(a), array_len(s) {}

  // Beware this function is used a heuristic for cheap to swap types, so
  // instantiating `ArrayFixedSize` with `ELEM_SIZE > 100` is probably a bad
  // idea perf wise.
  static constexpr bool has_fixed_size() { return true; }

  void *get(size_t i) const noexcept {
    return reinterpret_cast<void *>(get_internal(i));
  }

  void swap(size_t i, size_t j) const noexcept {
    alignas(32) uint8_t tmp[ELEM_SIZE];

    uint8_t *elem_i = get_internal(i);
    uint8_t *elem_j = get_internal(j);

    inline_memcpy(tmp, elem_i, ELEM_SIZE);
    inline_memmove(elem_i, elem_j, ELEM_SIZE);
    inline_memcpy(elem_j, tmp, ELEM_SIZE);
  }

  size_t len() const noexcept { return array_len; }

  // Make an Array starting at index |i| and length |s|.
  ArrayFixedSize<ELEM_SIZE> make_array(size_t i, size_t s) const noexcept {
    return ArrayFixedSize<ELEM_SIZE>(get_internal(i), s);
  }

  // Reset this Array to point at a different interval of the same
  // items starting at index |i|.
  void reset_bounds(size_t i, size_t s) noexcept {
    array_base = get_internal(i);
    array_len = s;
  }
};

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_QSORT_DATA_H
