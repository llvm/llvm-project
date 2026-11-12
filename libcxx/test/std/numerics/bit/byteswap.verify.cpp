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

#if TEST_HAS_BITINT

// Sub-byte widths (sizeof == 1 but bit width below CHAR_BIT).
// _BitInt(1) is excluded because make_unsigned on _BitInt(1) triggers a
// separate static_assert that's unrelated to byteswap's padding-bit Mandate.
void test_unsigned_7() {
  unsigned _BitInt(7) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}

void test_signed_7() {
  signed _BitInt(7) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}

// Non-byte-aligned widths
void test_unsigned_13() {
  unsigned _BitInt(13) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}

void test_unsigned_17() {
  unsigned _BitInt(17) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}

void test_signed_33() {
  signed _BitInt(33) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}

// Widths with sizeof == 16 land on the libc++ 128-bit dispatch path, which is
// gated on _LIBCPP_HAS_INT128 or __builtin_bswapg. On platforms without
// either, the size-dispatch static_assert fires alongside the padding-bit
// one, doubling the diagnostic count and breaking 1-to-1 directive matching.
// Restrict 65/80/96/112 to platforms that have one path. TEST_HAS_NO_INT128
// mirrors libc++'s _LIBCPP_HAS_INT128 (also false on _MSC_VER).
#  if TEST_HAS_BUILTIN(__builtin_bswapg) || !defined(TEST_HAS_NO_INT128)
void test_unsigned_65() {
  unsigned _BitInt(65) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}
#  endif

// Byte-aligned widths whose value bits don't fill the object representation,
// so the high-order storage holds padding bits even though the value width is
// a multiple of CHAR_BIT.
void test_unsigned_24() {
  // sizeof(_BitInt(24)) == 4 on x86_64; 8 padding bits.
  unsigned _BitInt(24) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}

void test_unsigned_40() {
  // sizeof(_BitInt(40)) == 8 on x86_64; 24 padding bits.
  unsigned _BitInt(40) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}

void test_unsigned_48() {
  // sizeof(_BitInt(48)) == 8 on x86_64; 16 padding bits. Note that the value
  // bit width (48) is a multiple of 16, so __builtin_bswapg accepts it -- the
  // libc++ static_assert is what actually catches this case.
  unsigned _BitInt(48) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}

void test_unsigned_56() {
  // sizeof(_BitInt(56)) == 8 on x86_64; 8 padding bits.
  unsigned _BitInt(56) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}

// Same dispatch-availability guard as test_unsigned_65 above.
#  if TEST_HAS_BUILTIN(__builtin_bswapg) || !defined(TEST_HAS_NO_INT128)
// 72 value bits leave padding on every ABI.
#    if __BITINT_MAXWIDTH__ >= 72
void test_unsigned_72() {
  unsigned _BitInt(72) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}
#    endif

#    if __BITINT_MAXWIDTH__ >= 80
void test_unsigned_80() {
  // sizeof(_BitInt(80)) == 16 on x86_64; 48 padding bits. Width 80 is also
  // a multiple of 16, so bswapg would accept it without the static_assert.
  unsigned _BitInt(80) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}
#    endif

#    if __BITINT_MAXWIDTH__ >= 112
void test_unsigned_112() {
  // sizeof(_BitInt(112)) == 16 on x86_64; 16 padding bits.
  unsigned _BitInt(112) v = 0;
  // expected-error-re@*:* {{{{(std::byteswap requires T to have no padding bits|byteswap is unimplemented for integral types of this size)}}}}
  (void)std::byteswap(v);
}
#    endif
#  endif

// Widths above 128 bits drop out: Clang's sizeof for those widths matches the
// value width on x86_64 (e.g., sizeof(_BitInt(192)) == 24), so there are no
// padding bits to reject.

#else
// expected-no-diagnostics
#endif
