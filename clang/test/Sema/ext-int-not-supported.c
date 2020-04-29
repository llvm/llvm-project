// RUN: %clang_cc1 -triple armv7 -fsyntax-only -verify %s

void foo() {
  _ExtInt(33) a; // expected-error{{_ExtInt is not supported on this target}}
}
