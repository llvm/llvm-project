//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// __builtin_bswapg was added in Clang 22. std::byteswap for _BitInt(N)
// defers to that builtin, so the rejection diagnostic for non-multiple-
// of-16-bit widths only fires on Clang 22+. Skip on Clang 21.
// UNSUPPORTED: clang-21

// <bit>

// std::byteswap rejects _BitInt(N) where the bit width is not a multiple
// of 16. The diagnostic comes from Clang's __builtin_bswapg sema check.

#include <bit>

#include "test_macros.h"

#if TEST_HAS_EXTENSION(bit_int)

void f_unsigned_13() {
  unsigned _BitInt(13) v = 0;
  // expected-error-re@*:* {{{{_BitInt type.*must be a multiple of 16 bits}}}}
  (void)std::byteswap(v);
}

void f_signed_13() {
  signed _BitInt(13) v = 0;
  // expected-error-re@*:* {{{{_BitInt type.*must be a multiple of 16 bits}}}}
  (void)std::byteswap(v);
}

void f_unsigned_17() {
  unsigned _BitInt(17) v = 0;
  // expected-error-re@*:* {{{{_BitInt type.*must be a multiple of 16 bits}}}}
  (void)std::byteswap(v);
}

void f_signed_33() {
  signed _BitInt(33) v = 0;
  // expected-error-re@*:* {{{{_BitInt type.*must be a multiple of 16 bits}}}}
  (void)std::byteswap(v);
}

void f_unsigned_65() {
  unsigned _BitInt(65) v = 0;
  // expected-error-re@*:* {{{{_BitInt type.*must be a multiple of 16 bits}}}}
  (void)std::byteswap(v);
}

#  if __BITINT_MAXWIDTH__ >= 129
// _BitInt(129) is wider than __int128 and only available where
// __BITINT_MAXWIDTH__ supports it (x86 / RISC-V 64). The wide-type
// generic loop also relies on the rejection; cover it here.
void f_unsigned_129() {
  unsigned _BitInt(129) v = 0;
  // expected-error-re@*:* {{{{_BitInt type.*must be a multiple of 16 bits}}}}
  (void)std::byteswap(v);
}

void f_signed_255() {
  signed _BitInt(255) v = 0;
  // expected-error-re@*:* {{{{_BitInt type.*must be a multiple of 16 bits}}}}
  (void)std::byteswap(v);
}
#  endif

#else
// expected-no-diagnostics
#endif
