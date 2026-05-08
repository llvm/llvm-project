//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <bit>

// std::byteswap rejects _BitInt(N) where N is not a multiple of CHAR_BIT.
//
// The byte-level builtins (and the generic loop fallback) treat the
// storage representation as the value, so for a type with padding bits
// they would shuffle padding into significant positions and produce a
// value whose semantic meaning is unspecified. The static_assert added
// in [libc++] Reject byteswap of types with padding bits pins the
// rejection so the diagnostic does not regress silently into a wrong
// value.

#include <bit>

#include "test_macros.h"

#if TEST_HAS_EXTENSION(bit_int)

void f_unsigned_13() {
  unsigned _BitInt(13) v = 0;
  // expected-error-re@*:* {{{{(static assertion|static_assert)}} failed{{.*}}"std::byteswap requires{{.*}}"}}
  (void)std::byteswap(v);
}

void f_signed_13() {
  signed _BitInt(13) v = 0;
  // expected-error-re@*:* {{{{(static assertion|static_assert)}} failed{{.*}}"std::byteswap requires{{.*}}"}}
  (void)std::byteswap(v);
}

void f_unsigned_17() {
  unsigned _BitInt(17) v = 0;
  // expected-error-re@*:* {{{{(static assertion|static_assert)}} failed{{.*}}"std::byteswap requires{{.*}}"}}
  (void)std::byteswap(v);
}

void f_signed_33() {
  signed _BitInt(33) v = 0;
  // expected-error-re@*:* {{{{(static assertion|static_assert)}} failed{{.*}}"std::byteswap requires{{.*}}"}}
  (void)std::byteswap(v);
}

void f_unsigned_65() {
  unsigned _BitInt(65) v = 0;
  // expected-error-re@*:* {{{{(static assertion|static_assert)}} failed{{.*}}"std::byteswap requires{{.*}}"}}
  (void)std::byteswap(v);
}

#  if __BITINT_MAXWIDTH__ >= 129
// _BitInt(129) is wider than __int128 and only available where
// __BITINT_MAXWIDTH__ supports it (x86 / RISC-V 64). The wide-type
// generic loop also relies on the rejection; cover it here.
void f_unsigned_129() {
  unsigned _BitInt(129) v = 0;
  // expected-error-re@*:* {{{{(static assertion|static_assert)}} failed{{.*}}"std::byteswap requires{{.*}}"}}
  (void)std::byteswap(v);
}

void f_signed_255() {
  signed _BitInt(255) v = 0;
  // expected-error-re@*:* {{{{(static assertion|static_assert)}} failed{{.*}}"std::byteswap requires{{.*}}"}}
  (void)std::byteswap(v);
}
#  endif

#else
// expected-no-diagnostics
#endif
