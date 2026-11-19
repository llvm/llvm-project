//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of inline_memmem.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_MEMMEM_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_MEMMEM_H

#include "src/__support/CPP/span.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include "src/__support/macros/config.h"     // LIBC_NAMESPACE_DECL

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {

struct MaximalSuffix {
  size_t index;
  size_t period;
};

// Finds the lexicographically largest suffix of `needle` under `comp` (or the
// reversed order when `Reverse` is true) and its smallest period. Taking the
// later split from both orderings gives a critical factorization of `needle`,
// where the local period at the split equals the global period of `needle`.
template <bool Reverse, typename Comp>
LIBC_INLINE constexpr MaximalSuffix
maximal_suffix(cpp::span<const unsigned char> needle, Comp &comp) {
  size_t max_suffix = 0;
  size_t candidate = 1;
  size_t offset = 0;
  size_t period = 1;

  while (candidate + offset < needle.size()) {
    int cmp = 0;
    if constexpr (Reverse)
      cmp = comp(needle[max_suffix + offset], needle[candidate + offset]);
    else
      cmp = comp(needle[candidate + offset], needle[max_suffix + offset]);

    if (cmp < 0) {
      // Candidate is smaller; skip past it and extend the current period.
      candidate += offset + 1;
      offset = 0;
      period = candidate - max_suffix;
    } else if (cmp == 0) {
      if (offset + 1 != period) {
        ++offset;
      } else {
        // Completed one full period; advance to the next repetition.
        candidate += period;
        offset = 0;
      }
    } else {
      // Candidate is larger; it becomes the new maximal suffix.
      max_suffix = candidate;
      candidate = max_suffix + 1;
      offset = 0;
      period = 1;
    }
  }

  return {max_suffix, period};
}

template <typename Comp>
LIBC_INLINE constexpr bool bytes_equal(cpp::span<const unsigned char> lhs,
                                       cpp::span<const unsigned char> rhs,
                                       Comp &comp) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (comp(lhs[i], rhs[i]) != 0)
      return false;
  }
  return true;
}

} // namespace internal

// Two-Way string matching (Crochemore & Perrin, 1991): worst-case O(n + m)
// time and O(1) space. `comp` must define a three-way total preorder (< 0,
// == 0, > 0).
//
// The needle is split at a critical factorization `needle = u + v` (`split` =
// |u|). At each alignment, `v` is matched left-to-right and then `u` is matched
// right-to-left. A mismatch in `v` skips past the matched prefix of `v`; a
// mismatch in `u` skips by `needle`'s period (or a safe lower bound on it when
// `u` does not share `v`'s period).
// TODO: Consider a fast path for short needles (e.g., Horspool / pair-hash)
// or SIMD/memchr acceleration.
template <typename Comp>
LIBC_INLINE constexpr static void *
inline_memmem(const void *haystack, size_t haystack_len, const void *needle,
              size_t needle_len, Comp &&comp) {
  cpp::span<const unsigned char> h(static_cast<const unsigned char *>(haystack),
                                   haystack_len);
  cpp::span<const unsigned char> n(static_cast<const unsigned char *>(needle),
                                   needle_len);

  if (n.empty())
    return const_cast<void *>(haystack);

  if (n.size() > h.size())
    return nullptr;

  if (n.size() == 1) {
    for (size_t i = 0; i < h.size(); ++i) {
      if (comp(h[i], n[0]) == 0)
        return const_cast<unsigned char *>(h.data() + i);
    }
    return nullptr;
  }

  internal::MaximalSuffix normal =
      internal::maximal_suffix</*Reverse=*/false>(n, comp);
  internal::MaximalSuffix reversed =
      internal::maximal_suffix</*Reverse=*/true>(n, comp);
  internal::MaximalSuffix critical =
      (normal.index >= reversed.index) ? normal : reversed;

  size_t split = critical.index;
  size_t period = critical.period;
  size_t max_pos = h.size() - n.size();

  if (internal::bytes_equal(n.first(split), n.subspan(period, split), comp)) {
    // `u` repeats at `period`, so `period` is the global period of `needle`.
    // Track `memory` (already-matched prefix bytes) to avoid re-scanning.
    size_t pos = 0;
    size_t memory = 0;
    while (pos <= max_pos) {
      size_t j = (split > memory) ? split : memory;
      while (j < n.size() && comp(h[pos + j], n[j]) == 0)
        ++j;
      if (j < n.size()) {
        pos += j - split + 1;
        memory = 0;
      } else {
        size_t k = split;
        while (k > memory && comp(h[pos + k - 1], n[k - 1]) == 0)
          --k;
        if (k <= memory)
          return const_cast<unsigned char *>(h.data() + pos);
        pos += period;
        memory = n.size() - period;
      }
    }
  } else {
    // `period` does not extend to `u`, so the global period of `needle` is
    // strictly greater than max(|u|, |v|); no `memory` tracking is needed.
    size_t right_len = n.size() - split;
    size_t step = (split > right_len ? split : right_len) + 1;
    size_t pos = 0;
    while (pos <= max_pos) {
      size_t j = split;
      while (j < n.size() && comp(h[pos + j], n[j]) == 0)
        ++j;
      if (j < n.size()) {
        pos += j - split + 1;
      } else {
        size_t k = split;
        while (k > 0 && comp(h[pos + k - 1], n[k - 1]) == 0)
          --k;
        if (k == 0)
          return const_cast<unsigned char *>(h.data() + pos);
        pos += step;
      }
    }
  }

  return nullptr;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_MEMMEM_H
