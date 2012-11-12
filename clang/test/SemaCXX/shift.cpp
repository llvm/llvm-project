// RUN: %clang_cc1 -Wall -Wshift-sign-overflow -ffreestanding -fsyntax-only -verify %s

#include <limits.h>

#define WORD_BIT (sizeof(int) * CHAR_BIT)

template <int N> void f() {
  (void)(N << 30); // expected-warning {{bits to represent, but 'int' only has}}
  (void)(30 << N); // expected-warning {{bits to represent, but 'int' only has}}
}

void test() {
  f<30>(); // expected-note {{instantiation}}
}
