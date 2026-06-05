//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <functional>

// std::hash<_BitInt(N)>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <type_traits>

#include "test_macros.h"

#if TEST_HAS_EXTENSION(bit_int)

template <class T>
void test_basic() {
  using H = std::hash<T>;
  ASSERT_NOEXCEPT(H{}(T{}));
  static_assert(std::is_same_v<decltype(H{}(T{})), std::size_t>);

  // Same value -> same hash, every time.
  H h;
  for (int v = 0; v <= 16; ++v) {
    T t1(static_cast<T>(v));
    T t2(static_cast<T>(v));
    assert(h(t1) == h(t2));
  }
}

// Distinct value bits should not collide trivially. The hash function is
// allowed to collide, but adjacent small integers should not all hash to
// the same value -- that would be a sign that the value was not being
// included in the hash at all (e.g. hashing only padding).
template <class T>
void test_distinct_values_distinct_hashes() {
  std::hash<T> h;
  std::size_t hashes[16];
  for (int i = 0; i < 16; ++i)
    hashes[i] = h(static_cast<T>(i));
  // At least 8 of 16 must be unique. (Pigeonhole gives a much stronger
  // bound for any reasonable hash, but be conservative for exotic hashes.)
  int unique = 0;
  for (int i = 0; i < 16; ++i) {
    bool seen = false;
    for (int j = 0; j < i; ++j)
      if (hashes[i] == hashes[j])
        seen = true;
    if (!seen)
      ++unique;
  }
  assert(unique >= 8);
}

// The standard guarantees `a == b` implies `hash(a) == hash(b)`. For
// _BitInt(N), two values that compare equal might be reached via different
// expression chains. Round-tripping through a wider unsigned type and
// truncating must not change the hash.
template <class T>
void test_equal_values_same_hash_via_different_paths() {
  std::hash<T> h;
  T direct = static_cast<T>(42);
  T via_wider;
  {
    using U64 = unsigned long long;
    U64 wide  = 42;
    via_wider = static_cast<T>(wide);
  }
  assert(direct == via_wider);
  assert(h(direct) == h(via_wider));

  // Bitwise-NOT of zero is the unsigned all-ones value. For the maximum
  // representable value, two paths must agree.
  if constexpr (!std::is_signed_v<T>) {
    T max1 = static_cast<T>(~T(0));
    T max2 = T(0);
    for (int i = 0; i < (int)(sizeof(T) * 8); ++i)
      max2 = static_cast<T>((max2 << 1) | T(1));
    assert(h(max1) == h(max2));
  }
}

// Probe [hash.requirements] `a == b => hash(a) == hash(b)` directly by
// injecting different bits at padding-bit storage positions.
template <class T>
void test_padding_bits_dont_break_equivalence() {
  constexpr int value_bits   = std::numeric_limits<T>::digits + std::is_signed_v<T>;
  constexpr int storage_bits = static_cast<int>(sizeof(T)) * 8;
  if constexpr (storage_bits > value_bits) {
    std::hash<T> h;
    T clean(static_cast<T>(5));
    T dirty;
    std::memcpy(&dirty, &clean, sizeof(T));
    // Flip every padding-bit position (little-endian _BitInt: high bits of high bytes).
    auto* bytes = reinterpret_cast<unsigned char*>(&dirty);
    for (int i = value_bits; i < storage_bits; ++i)
      bytes[i / 8] ^= static_cast<unsigned char>(1u << (i % 8));
    assert(clean == dirty);
    assert(h(clean) == h(dirty));
  }
}

template <class T>
void test_all() {
  test_basic<T>();
  test_distinct_values_distinct_hashes<T>();
  test_equal_values_same_hash_via_different_paths<T>();
  test_padding_bits_dont_break_equivalence<T>();
}

#endif // TEST_HAS_EXTENSION(bit_int)

int main(int, char**) {
#if TEST_HAS_EXTENSION(bit_int)
  // Byte-aligned widths.
  test_all<unsigned _BitInt(8)>();
  test_all<signed _BitInt(8)>();
  test_all<unsigned _BitInt(32)>();
  test_all<signed _BitInt(32)>();
  test_all<unsigned _BitInt(64)>();
  test_all<signed _BitInt(64)>();

  // Non-byte-aligned widths -- exercise types whose sizeof * CHAR_BIT
  // exceeds the value-bit count.
  test_all<unsigned _BitInt(7)>();
  test_all<signed _BitInt(7)>();
  test_all<unsigned _BitInt(13)>();
  test_all<signed _BitInt(13)>();
  test_all<unsigned _BitInt(37)>();
  test_all<signed _BitInt(37)>();

#  if __BITINT_MAXWIDTH__ >= 128
  test_all<unsigned _BitInt(77)>();
  test_all<signed _BitInt(77)>();
  test_all<unsigned _BitInt(128)>();
  test_all<signed _BitInt(128)>();
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
  test_all<unsigned _BitInt(129)>();
  test_all<signed _BitInt(129)>();
  test_all<unsigned _BitInt(255)>();
  test_all<signed _BitInt(255)>();
  test_all<unsigned _BitInt(256)>();
  test_all<signed _BitInt(256)>();
#  endif
  // Widths that exercise the new primary template (sizeof > 4 * sizeof(size_t)).
#  if __BITINT_MAXWIDTH__ >= 257
  test_all<unsigned _BitInt(257)>();
  test_all<signed _BitInt(257)>();
#  endif
#  if __BITINT_MAXWIDTH__ >= 1024
  test_all<unsigned _BitInt(1024)>();
  test_all<signed _BitInt(1024)>();
#  endif
#  if __BITINT_MAXWIDTH__ >= 4096
  test_all<unsigned _BitInt(4096)>();
  test_all<signed _BitInt(4096)>();
#  endif
#endif // TEST_HAS_EXTENSION(bit_int)

  return 0;
}
