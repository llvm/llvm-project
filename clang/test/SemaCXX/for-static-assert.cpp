// RUN: %clang_cc1 -fsyntax-only -verify %s

// C permits a 'static_assert' in the first part of a 'for' loop
// whereas C++ does not.
void f() {
  for(static_assert(true);;) {} // expected-error {{expected expression}}
}
