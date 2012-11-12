// RUN: %clang_cc1 -std=c1x -fsyntax-only -verify %s

_Static_assert("foo", "string is nonzero"); // expected-error {{static_assert expression is not an integral constant expression}}

_Static_assert(1, "1 is nonzero");
_Static_assert(0, "0 is nonzero"); // expected-error {{static_assert failed "0 is nonzero"}}

void foo(void) {
  _Static_assert(1, "1 is nonzero");
  _Static_assert(0, "0 is nonzero"); // expected-error {{static_assert failed "0 is nonzero"}}
}
