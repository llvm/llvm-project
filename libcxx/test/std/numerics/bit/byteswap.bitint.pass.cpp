//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <bit>

// std::byteswap for _BitInt(N).
//
// Byte-aligned widths (N % CHAR_BIT == 0) work via the existing builtins
// for sizeof <= 16 and via the new generic loop for sizeof > 16. Non-byte-
// aligned widths are rejected by static_assert; that case is covered in
// byteswap.bitint.verify.cpp.

#include <bit>
#include <cassert>
#include <cstdint>

#include "test_macros.h"

#if TEST_HAS_EXTENSION(bit_int)

template <class T>
constexpr void test_roundtrip(T v) {
  assert(std::byteswap(std::byteswap(v)) == v);
  ASSERT_SAME_TYPE(decltype(std::byteswap(v)), T);
  ASSERT_NOEXCEPT(std::byteswap(v));
}

constexpr bool test() {
  // sizeof == 1: identity. The size-1 branch returns the input unchanged
  // and the padding-bit static_assert is bypassed -- no bytes move and no
  // padding gets shuffled into significant positions, so non-byte-aligned
  // widths up to CHAR_BIT (e.g. _BitInt(7)) are also identity.
  assert(std::byteswap(static_cast<unsigned _BitInt(8)>(0xAB)) == static_cast<unsigned _BitInt(8)>(0xAB));
  test_roundtrip<unsigned _BitInt(8)>(0xAB);
  test_roundtrip<signed _BitInt(8)>(0x12);
  // _BitInt(7) signed has a padding bit but stays identity at sizeof == 1.
  assert(std::byteswap(static_cast<signed _BitInt(7)>(42)) == static_cast<signed _BitInt(7)>(42));
  assert(std::byteswap(static_cast<unsigned _BitInt(7)>(42)) == static_cast<unsigned _BitInt(7)>(42));

  // sizeof == 2: __builtin_bswap16
  assert(std::byteswap(static_cast<unsigned _BitInt(16)>(0xCDEF)) == static_cast<unsigned _BitInt(16)>(0xEFCD));
  test_roundtrip<unsigned _BitInt(16)>(0xCDEF);
  test_roundtrip<signed _BitInt(16)>(0x1234);

  // sizeof == 4: __builtin_bswap32
  assert(std::byteswap(static_cast<unsigned _BitInt(32)>(0x01234567U)) ==
         static_cast<unsigned _BitInt(32)>(0x67452301U));
  test_roundtrip<unsigned _BitInt(32)>(0x01234567U);
  test_roundtrip<signed _BitInt(32)>(0x01234567);

  // sizeof == 8: __builtin_bswap64
  assert(std::byteswap(static_cast<unsigned _BitInt(64)>(0x0123456789ABCDEFULL)) ==
         static_cast<unsigned _BitInt(64)>(0xEFCDAB8967452301ULL));
  test_roundtrip<unsigned _BitInt(64)>(0x0123456789ABCDEFULL);
  test_roundtrip<signed _BitInt(64)>(0x0123456789ABCDEFLL);

#  if __BITINT_MAXWIDTH__ >= 128
  // sizeof == 16: __builtin_bswap128 (or 2x bswap64 fallback). Same path
  // as the existing __int128_t / __uint128_t coverage in byteswap.pass.cpp.
  unsigned _BitInt(128) v128 =
      (static_cast<unsigned _BitInt(128)>(0x0123456789ABCDEFULL) << 64) |
      static_cast<unsigned _BitInt(128)>(0x13579BDF02468ACEULL);
  test_roundtrip<unsigned _BitInt(128)>(v128);
  test_roundtrip<signed _BitInt(128)>(static_cast<signed _BitInt(128)>(v128));
#  endif

#  if __BITINT_MAXWIDTH__ >= 256
  // sizeof == 32: hits the new generic loop fallback.
  unsigned _BitInt(256) v256 =
      (static_cast<unsigned _BitInt(256)>(0xDEADBEEFCAFEBABEULL) << 128) |
      (static_cast<unsigned _BitInt(256)>(0x1234567890ABCDEFULL) << 64) |
      static_cast<unsigned _BitInt(256)>(0xFEDCBA9876543210ULL);
  test_roundtrip<unsigned _BitInt(256)>(v256);
  test_roundtrip<signed _BitInt(256)>(static_cast<signed _BitInt(256)>(v256));

  // Spot check for the wide loop: low byte of input must end up as the
  // high byte of the output, and high byte of input as the low byte.
  unsigned _BitInt(256) lo_only = 0xAB;
  auto lo_swapped               = std::byteswap(lo_only);
  assert(static_cast<unsigned char>(lo_swapped >> ((sizeof(lo_swapped) - 1) * CHAR_BIT)) == 0xAB);
  unsigned _BitInt(256) hi_only =
      static_cast<unsigned _BitInt(256)>(0xCD) << ((sizeof(unsigned _BitInt(256)) - 1) * CHAR_BIT);
  auto hi_swapped = std::byteswap(hi_only);
  assert(static_cast<unsigned char>(hi_swapped) == 0xCD);

  // Mid-value test: distinct byte at every position so an off-by-one in
  // the loop indexing surfaces directly. Build 0x00010203...1F at bytes
  // 0..31 then verify byteswap reverses the byte sequence.
  unsigned _BitInt(256) ramp = 0;
  for (int __i = 0; __i < 32; ++__i)
    ramp |= static_cast<unsigned _BitInt(256)>(__i) << (__i * CHAR_BIT);
  auto ramp_swapped = std::byteswap(ramp);
  for (int __i = 0; __i < 32; ++__i)
    assert(static_cast<unsigned char>(ramp_swapped >> ((31 - __i) * CHAR_BIT)) == __i);
#  endif

#  if __BITINT_MAXWIDTH__ >= 1024
  // Larger width still in the generic loop.
  unsigned _BitInt(1024) v1024 = static_cast<unsigned _BitInt(1024)>(0xAB) << ((128 - 1) * CHAR_BIT);
  test_roundtrip<unsigned _BitInt(1024)>(v1024);
#  endif

#  if __BITINT_MAXWIDTH__ >= 4096
  // Largest width tested. Picked to cover the upper end of what the
  // dev-branch experiments exercised; values larger than this take a
  // long time to constexpr-evaluate without adding much coverage.
  unsigned _BitInt(4096) v4096 = static_cast<unsigned _BitInt(4096)>(0xAB) << ((512 - 1) * CHAR_BIT);
  test_roundtrip<unsigned _BitInt(4096)>(v4096);
#  endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

#else

int main(int, char**) { return 0; }

#endif
