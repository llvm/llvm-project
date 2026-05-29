//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// make_format_args of a cv-qualified integer.
//
// `__create_format_arg` applies `remove_const_t` before dispatch via
// `__determine_arg_t`, so a `const int` lvalue is still classified as a
// signed integer. `volatile` is not stripped: with the no-cv guard in
// `__signed_integer`/`__unsigned_integer`, `volatile int` no longer maps
// to a storage slot and trips the "not formattable" static_assert chain.
// Pin down both behaviors so a future refactor does not silently change
// them.

#include <format>

#include "test_macros.h"

// Three volatile lvalues below each trigger:
//   - format_arg_store.h: "the supplied type is not formattable"
//   - format_arg_store.h: a follow-on static_assert
//   - constructible.h: implicitly-deleted copy constructor cascade
// expected-error@*:* 3 {{the supplied type is not formattable}}
// expected-error@*:* 3+ {{static assertion failed}}
// expected-error@*:* 3+ {{implicitly-deleted copy constructor}}

void f_const_int_ok() {
  const int value = 0;
  (void)std::make_format_args(value); // ok: const stripped before classification
}

void f_volatile_int() {
  volatile int value = 0;
  (void)std::make_format_args(value);
}

void f_const_volatile_int() {
  const volatile int value = 0;
  (void)std::make_format_args(value);
}

void f_volatile_unsigned() {
  volatile unsigned value = 0;
  (void)std::make_format_args(value);
}
