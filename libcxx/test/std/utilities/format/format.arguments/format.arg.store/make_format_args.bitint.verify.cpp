//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// make_format_args with _BitInt(N) wider than __int128 is unsupported.
//
// After [libc++] recognized _BitInt as an integer type in
// __type_traits/integer_traits.h, format_arg_store's __determine_arg_t
// dispatches on sizeof(_Tp) and maps _BitInt up to sizeof(__int128) onto
// the i128 storage slot. For wider _BitInt (sizeof > sizeof(__int128)),
// no storage slot exists and a static_assert fires.
//
// This test pins down that diagnostic so that if the dispatch ever changes
// to silently accept a wider type (or drops the diagnostic), the test
// breaks and forces a reconsideration.

#include <format>

#include "test_macros.h"

#if TEST_HAS_EXTENSION(bit_int) && __BITINT_MAXWIDTH__ >= 129

void f_signed() {
  // _BitInt(129) has sizeof == 32 on x86-64 (first size wider than __int128).
  _BitInt(129) value = 0;
  // expected-error-re@*:* {{{{(static assertion|static_assert)}} failed{{.*}}"an unsupported signed integer was used"}}
  (void)std::make_format_args(value);
}

void f_unsigned() {
  unsigned _BitInt(129) value = 0;
  // expected-error-re@*:* {{{{(static assertion|static_assert)}} failed{{.*}}"an unsupported unsigned integer was used"}}
  (void)std::make_format_args(value);
}

#  if __BITINT_MAXWIDTH__ >= 256
void f_signed_256() {
  _BitInt(256) value = 0;
  // expected-error-re@*:* {{{{(static assertion|static_assert)}} failed{{.*}}"an unsupported signed integer was used"}}
  (void)std::make_format_args(value);
}
#  endif

#else
// When _BitInt is unavailable or the implementation limits preclude the
// test, keep the file well-formed with a trivial positive expectation so
// the driver does not fail.
// expected-no-diagnostics
#endif
