// RUN: %clang_cc1 -fsyntax-only -verify -Wsign-conversion %s

// PR9345: make a subgroup of -Wconversion for signedness changes

void test(int x) {
  unsigned t0 = x; // expected-warning {{implicit conversion changes signedness}}
  unsigned t1 = (t0 == 5 ? x : 0); // expected-warning {{operand of ? changes signedness}}
}
