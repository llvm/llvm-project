//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <bit>

// std::byteswap rejects integer types that have padding bits per
// [bit.byteswap]/Mandates. The implementation uses static_assert; the
// diagnostic comes from <__bit/byteswap.h>.

#include <bit>

#include "test_macros.h"

#if TEST_HAS_EXTENSION(bit_int)

// Sub-byte widths (sizeof == 1 but bit width below CHAR_BIT)
void test_unsigned_1() {
  unsigned _BitInt(1) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}

void test_unsigned_7() {
  unsigned _BitInt(7) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}

void test_signed_7() {
  signed _BitInt(7) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}

// Non-byte-aligned widths
void test_unsigned_13() {
  unsigned _BitInt(13) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}

void test_unsigned_17() {
  unsigned _BitInt(17) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}

void test_signed_33() {
  signed _BitInt(33) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}

void test_unsigned_65() {
  unsigned _BitInt(65) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}

// Byte-aligned widths whose value bits don't fill the object representation.
// On platforms where sizeof(_BitInt(N)) rounds up to a power of two, these
// types have padding bits in the high-order storage positions even though
// their value width is a multiple of CHAR_BIT.
void test_unsigned_24() {
  // sizeof(_BitInt(24)) == 4 on x86_64; 8 padding bits.
  unsigned _BitInt(24) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}

void test_unsigned_40() {
  // sizeof(_BitInt(40)) == 8 on x86_64; 24 padding bits.
  unsigned _BitInt(40) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}

void test_unsigned_48() {
  // sizeof(_BitInt(48)) == 8 on x86_64; 16 padding bits. Note that the value
  // bit width (48) is a multiple of 16, so __builtin_bswapg accepts it -- the
  // libc++ static_assert is what actually catches this case.
  unsigned _BitInt(48) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}

void test_unsigned_56() {
  // sizeof(_BitInt(56)) == 8 on x86_64; 8 padding bits.
  unsigned _BitInt(56) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}

#  if __BITINT_MAXWIDTH__ >= 80
void test_unsigned_80() {
  // sizeof(_BitInt(80)) == 16 on x86_64; 48 padding bits. Width 80 is also
  // a multiple of 16, so bswapg would accept it without the static_assert.
  unsigned _BitInt(80) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}
#  endif

#  if __BITINT_MAXWIDTH__ >= 96
void test_unsigned_96() {
  // sizeof(_BitInt(96)) == 16 on x86_64; 32 padding bits.
  unsigned _BitInt(96) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}
#  endif

#  if __BITINT_MAXWIDTH__ >= 112
void test_unsigned_112() {
  // sizeof(_BitInt(112)) == 16 on x86_64; 16 padding bits.
  unsigned _BitInt(112) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}
#  endif

#  if __BITINT_MAXWIDTH__ >= 256
void test_unsigned_192() {
  // sizeof(_BitInt(192)) == 32 on x86_64; 64 padding bits. Multiple of 16
  // but not of the storage size.
  unsigned _BitInt(192) v = 0;
  // expected-error@*:* {{static assertion failed{{.*}}std::byteswap requires T to have no padding bits}}
  (void)std::byteswap(v);
}
#  endif

#else
// expected-no-diagnostics
#endif
