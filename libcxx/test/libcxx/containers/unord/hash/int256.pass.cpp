//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Test std::hash specialization for __int256_t / __uint256_t.
//
// The generic __hash_impl dispatches to __scalar_hash<_Tp, N> where
// N = sizeof(_Tp) / sizeof(size_t). For __int256_t on 64-bit platforms,
// N = 32/8 = 4, using __scalar_hash<_Tp, 4> which hashes via __hash_memory.

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "test_macros.h"

#ifdef TEST_HAS_NO_INT256
int main(int, char**) { return 0; }
#else

int main(int, char**) {
  std::hash<__int256_t> h_s;
  std::hash<__uint256_t> h_u;

  // --- Basic consistency: same input always gives same output ---
  {
    __int256_t a = 42;
    if (h_s(a) != h_s(a))
      return 1;

    __uint256_t b = 42;
    if (h_u(b) != h_u(b))
      return 2;
  }

  // --- Different values should (usually) give different hashes ---
  {
    __uint256_t a = 0;
    __uint256_t b = 1;
    __uint256_t c = (__uint256_t)1 << 128;
    __uint256_t d = (__uint256_t)1 << 255;

    // We can't guarantee different hashes for all pairs (pigeonhole),
    // but for these carefully chosen values it's astronomically unlikely
    // that all four hash to the same value.
    size_t ha = h_u(a);
    size_t hb = h_u(b);
    size_t hc = h_u(c);
    size_t hd = h_u(d);

    // At least 2 of the 4 hashes should be distinct
    int distinct = 1;
    if (hb != ha)
      ++distinct;
    if (hc != ha && hc != hb)
      ++distinct;
    if (hd != ha && hd != hb && hd != hc)
      ++distinct;
    if (distinct < 2)
      return 3;
  }

  // --- Zero and negative values ---
  {
    __int256_t zero = 0;
    __int256_t neg  = -1;
    // Hash of 0 and -1 should differ (very high probability)
    if (h_s(zero) == h_s(neg)) {
      // Allow this in theory, but verify the hash function is callable
      (void)h_s(zero);
    }
  }

  // --- Large values near max ---
  {
    __uint256_t max_val     = ~(__uint256_t)0;
    __uint256_t max_minus_1 = max_val - 1;
    // These should produce valid hash values (no crash)
    size_t h1 = h_u(max_val);
    size_t h2 = h_u(max_minus_1);
    (void)h1;
    (void)h2;
  }

  // --- std::unordered_set with __uint256_t keys ---
  {
    std::unordered_set<__uint256_t> s;
    s.insert(0);
    s.insert(1);
    s.insert((__uint256_t)1 << 128);
    s.insert(~(__uint256_t)0);

    if (s.size() != 4)
      return 4;
    if (s.count(0) != 1)
      return 5;
    if (s.count(1) != 1)
      return 6;
    if (s.count(2) != 0)
      return 7;
  }

  // --- std::unordered_map with __int256_t keys ---
  {
    std::unordered_map<__int256_t, int> m;
    m[0]                    = 10;
    m[-1]                   = 20;
    m[(__int256_t)1 << 200] = 30;

    if (m.size() != 3)
      return 8;
    if (m[0] != 10)
      return 9;
    if (m[-1] != 20)
      return 10;
  }

  // --- Signed and unsigned hash independence ---
  // hash<__int256_t>(42) and hash<__uint256_t>(42) may or may not be equal
  // (implementation defined), but both must be callable
  {
    __int256_t sv  = 42;
    __uint256_t uv = 42;
    (void)h_s(sv);
    (void)h_u(uv);
  }

  return 0;
}
#endif
