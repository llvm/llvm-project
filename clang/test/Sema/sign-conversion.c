// RUN: %clang_cc1 -fsyntax-only -verify -Wsign-conversion %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify -Wsign-conversion %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsyntax-only -verify -Wsign-conversion %s

// PR9345: make a subgroup of -Wconversion for signedness changes

void test(int x) {
  unsigned t0 = x; // expected-warning {{implicit conversion changes signedness}}
  unsigned t1 = (t0 == 5 ? x : 0); // expected-warning {{operand of ? changes signedness}}

  // Clang has special treatment for left shift of literal '1'.
  // Make sure there is no diagnostics.
  long t2 = 1LL << x;
}
