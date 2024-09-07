//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that both `_LIBCPP_ASSERT(false, ...)` and `_LIBCPP_ASSUME(false)`
// mean that a constant expression cannot be formed.

#include <__assert>
#include "test_macros.h"

// expected-note@*:* 0+ {{expanded from macro}}

static_assert((_LIBCPP_ASSERT(false, "message"), true), "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}

static_assert((_LIBCPP_ASSUME(false), true), "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}

const int i = (_LIBCPP_ASSERT(false, "message"), 1);
const int j = (_LIBCPP_ASSUME(false), 1);

static_assert(i, "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
static_assert(j, "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}

#if TEST_STD_VER >= 11
constexpr int f(int x) {
  return (_LIBCPP_ASSERT(x != 6, "message"), x);
  // expected-note@-1 {{subexpression not valid in a constant expression}}
}
constexpr int g(int x) {
  return (_LIBCPP_ASSUME(x != 6), x);
  // expected-note@-1 {{subexpression not valid in a constant expression}}
}
static_assert(f(6) == 6, "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
static_assert(g(6) == 6, "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
#endif

#if TEST_STD_VER >= 14
constexpr int ff(int x) {
  _LIBCPP_ASSERT(x != 6, "message");
  // expected-note@-1 {{subexpression not valid in a constant expression}}
  return x;
}
constexpr int gg(int x) {
  _LIBCPP_ASSUME(x != 6);
  // expected-note@-1 {{subexpression not valid in a constant expression}}
  return x;
}
static_assert(ff(6) == 6, "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
static_assert(gg(6) == 6, "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
#endif
