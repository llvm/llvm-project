// RUN: %clang_cc1 -fsyntax-only %s -verify
// expected-no-diagnostics

typedef _Atomic char atomic_char;

atomic_char counter;

char load_plus_one() {
  return ({ counter; }) + 1;
}