//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_INLINE_STRCMP_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_INLINE_STRCMP_H

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

#include <stddef.h>

#if LIBC_HAS_VECTOR_TYPE

namespace LIBC_NAMESPACE_DECL {
namespace clang_vector {

LIBC_INLINE_VAR constexpr size_t VEC_SIZE = sizeof(cpp::simd<unsigned char>);
LIBC_INLINE_VAR constexpr size_t LOOP_UNROLL_FACTOR = 4;
LIBC_INLINE_VAR constexpr size_t LOOP_BLOCK_SIZE =
    LOOP_UNROLL_FACTOR * VEC_SIZE;

LIBC_NO_SANITIZE_OOB_ACCESS LIBC_INLINE static cpp::simd_mask<unsigned char>
are_equal_and_nonzero(const unsigned char *lhs, const unsigned char *rhs) {
  constexpr cpp::simd<unsigned char> zero = cpp::splat<unsigned char>(0);
  cpp::simd<unsigned char> a =
      cpp::load<cpp::simd<unsigned char>>(lhs, /*aligned=*/false);
  cpp::simd<unsigned char> b =
      cpp::load<cpp::simd<unsigned char>>(rhs, /*aligned=*/false);
  return (a == b) & ~(a == zero);
}

// Returns the comparison result for a single vector width, assuming
// that at least one lane is zero/not equal.
LIBC_INLINE static int
cmp_return_diff(const unsigned char *lhs, const unsigned char *rhs,
                cpp::simd_mask<unsigned char> eq_and_nonzero) {
  using bitmask_t = cpp::simd_bitmask<unsigned char>;
  uint64_t stop_bits = static_cast<uint64_t>(
      static_cast<bitmask_t>(~cpp::to_bitmask(eq_and_nonzero)));
  size_t idx = static_cast<size_t>(cpp::countr_zero(stop_bits));
  return static_cast<int>(lhs[idx]) - static_cast<int>(rhs[idx]);
}

// Performs n bytes of comparison and returns an optional representing whether
// there was a comparison, and if so, what the value of the comparison was. This
// is used for aligning up to page boundaries.
LIBC_INLINE static cpp::optional<int>
cmp_scalar_n(const unsigned char *lhs, const unsigned char *rhs, size_t n) {
  LIBC_LOOP_NOUNROLL
  for (size_t i = 0; i < n; ++i) {
    unsigned char lc = lhs[i];
    unsigned char rc = rhs[i];
    if (lc != rc || lc == 0)
      return static_cast<int>(lc) - static_cast<int>(rc);
  }
  return cpp::nullopt;
}

// Compares LOOP_BLOCK_SIZE (4 * VEC_SIZE) bytes per iteration while at least
// LOOP_BLOCK_SIZE bytes remain before either pointer's page boundary. Uses
// unsigned min reduction to perform a single null check per 4-vector block.
LIBC_NO_SANITIZE_OOB_ACCESS LIBC_ALWAYS_INLINE static cpp::optional<int>
cmp_block_loop(const unsigned char *&lhs, const unsigned char *&rhs,
               size_t &safe_bytes) {
  LIBC_LOOP_NOUNROLL
  while (safe_bytes >= LOOP_BLOCK_SIZE) {
    cpp::simd<unsigned char> a0 =
        cpp::load<cpp::simd<unsigned char>>(lhs, /*aligned=*/false);
    cpp::simd<unsigned char> b0 =
        cpp::load<cpp::simd<unsigned char>>(rhs, /*aligned=*/false);
    cpp::simd<unsigned char> a1 =
        cpp::load<cpp::simd<unsigned char>>(lhs + VEC_SIZE, /*aligned=*/false);
    cpp::simd<unsigned char> b1 =
        cpp::load<cpp::simd<unsigned char>>(rhs + VEC_SIZE, /*aligned=*/false);
    cpp::simd<unsigned char> a2 = cpp::load<cpp::simd<unsigned char>>(
        lhs + 2 * VEC_SIZE, /*aligned=*/false);
    cpp::simd<unsigned char> b2 = cpp::load<cpp::simd<unsigned char>>(
        rhs + 2 * VEC_SIZE, /*aligned=*/false);
    cpp::simd<unsigned char> a3 = cpp::load<cpp::simd<unsigned char>>(
        lhs + 3 * VEC_SIZE, /*aligned=*/false);
    cpp::simd<unsigned char> b3 = cpp::load<cpp::simd<unsigned char>>(
        rhs + 3 * VEC_SIZE, /*aligned=*/false);

    auto eq = ((a0 == b0) & (a1 == b1)) & ((a2 == b2) & (a3 == b3));
    cpp::simd<unsigned char> min_a =
        cpp::min(cpp::min(a0, a1), cpp::min(a2, a3));
    constexpr cpp::simd<unsigned char> zero = cpp::splat<unsigned char>(0);
    cpp::simd_mask<unsigned char> ok = eq & ~(min_a == zero);
    if (LIBC_UNLIKELY(!cpp::all_of(ok))) {
      LIBC_LOOP_UNROLL
      for (size_t i = 0; i < LOOP_UNROLL_FACTOR; ++i) {
        const unsigned char *l = lhs + i * VEC_SIZE;
        const unsigned char *r = rhs + i * VEC_SIZE;
        cpp::simd_mask<unsigned char> m = are_equal_and_nonzero(l, r);
        if (!cpp::all_of(m))
          return cmp_return_diff(l, r, m);
      }
      __builtin_unreachable();
    }
    lhs += LOOP_BLOCK_SIZE;
    rhs += LOOP_BLOCK_SIZE;
    safe_bytes -= LOOP_BLOCK_SIZE;
  }
  return cpp::nullopt;
}

LIBC_NO_SANITIZE_OOB_ACCESS LIBC_ALWAYS_INLINE int strcmp(const char *left,
                                                          const char *right) {
  constexpr size_t min_page_size = 4096;
  static_assert(cpp::has_single_bit(min_page_size) &&
                    cpp::has_single_bit(LOOP_BLOCK_SIZE) &&
                    min_page_size % LOOP_BLOCK_SIZE == 0,
                "4x vector block must cleanly divide a page.");

  const unsigned char *orig_lhs = reinterpret_cast<const unsigned char *>(left);
  const unsigned char *lhs = orig_lhs;
  const unsigned char *rhs = reinterpret_cast<const unsigned char *>(right);

  constexpr uintptr_t page_mask = min_page_size - 1;
  const uintptr_t page_offset_or =
      (reinterpret_cast<uintptr_t>(lhs) | reinterpret_cast<uintptr_t>(rhs)) &
      page_mask;

  // Fast path for short-to-medium strings (up to 4 * VEC_SIZE bytes) with
  // per-vector page headroom checks. We use the or of the two page offsets
  // as it is a cheap way to guarantee that both sides fit into the page
  // with rare (harmless) false negatives.
  size_t step;
  if (LIBC_LIKELY(page_offset_or <= min_page_size - VEC_SIZE)) {
    size_t checked = 0;
    LIBC_LOOP_UNROLL
    for (size_t i = 0; i < LOOP_UNROLL_FACTOR; ++i) {
      if (i > 0 &&
          LIBC_UNLIKELY(page_offset_or > min_page_size - (i + 1) * VEC_SIZE))
        break;
      const unsigned char *l = lhs + i * VEC_SIZE;
      const unsigned char *r = rhs + i * VEC_SIZE;
      cpp::simd_mask<unsigned char> eq_and_nonzero =
          are_equal_and_nonzero(l, r);
      if (LIBC_LIKELY(!cpp::all_of(eq_and_nonzero)))
        return cmp_return_diff(l, r, eq_and_nonzero);
      checked += VEC_SIZE;
    }
    step = checked - (reinterpret_cast<uintptr_t>(lhs) & (VEC_SIZE - 1));
  } else {
    // Cold path: at least one pointer is within VEC_SIZE of a page boundary,
    // run a scalar loop until we are aligned.
    size_t lhs_rem =
        min_page_size - (reinterpret_cast<uintptr_t>(lhs) & page_mask);
    size_t rhs_rem =
        min_page_size - (reinterpret_cast<uintptr_t>(rhs) & page_mask);
    size_t n = cpp::min(cpp::min(lhs_rem, rhs_rem), VEC_SIZE);
    if (auto diff = cmp_scalar_n(lhs, rhs, n))
      return *diff;
    step = n;
  }

  lhs += step;
  rhs += step;

  // Page-chunked 4x-unrolled main loop.
  for (;;) {
    size_t lhs_to_page =
        min_page_size - (reinterpret_cast<uintptr_t>(lhs) & page_mask);
    size_t rhs_to_page =
        min_page_size - (reinterpret_cast<uintptr_t>(rhs) & page_mask);
    size_t safe_bytes = cpp::min(lhs_to_page, rhs_to_page);

    // Run the main loop until we run out of safe bytes and need to realign
    // to a page boundary.
    if (auto diff = cmp_block_loop(lhs, rhs, safe_bytes))
      return *diff;

    // While safe_bytes is less than the main loop stride but bigger than the
    // vector size, we can do vector comparisons.
    while (safe_bytes >= VEC_SIZE) {
      cpp::simd_mask<unsigned char> eq_and_nonzero =
          are_equal_and_nonzero(lhs, rhs);
      if (LIBC_UNLIKELY(!cpp::all_of(eq_and_nonzero)))
        return cmp_return_diff(lhs, rhs, eq_and_nonzero);
      lhs += VEC_SIZE;
      rhs += VEC_SIZE;
      safe_bytes -= VEC_SIZE;
    }

    // Now we are guaranteed to have less safe bytes than a vector width. Use
    // vectors that overlap with already checked regions when possible,
    // otherwise fallback to scalar comparison until we are aligned again.
    if (safe_bytes > 0) {
      if (LIBC_LIKELY(static_cast<size_t>(lhs - orig_lhs) >= VEC_SIZE)) {
        size_t back = VEC_SIZE - safe_bytes;
        cpp::simd_mask<unsigned char> eq_and_nonzero =
            are_equal_and_nonzero(lhs - back, rhs - back);
        if (LIBC_UNLIKELY(!cpp::all_of(eq_and_nonzero)))
          return cmp_return_diff(lhs - back, rhs - back, eq_and_nonzero);
      } else {
        if (auto diff = cmp_scalar_n(lhs, rhs, safe_bytes))
          return *diff;
      }
      lhs += safe_bytes;
      rhs += safe_bytes;
    }
  }
}

} // namespace clang_vector
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_HAS_VECTOR_TYPE

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_INLINE_STRCMP_H
