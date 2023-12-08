// RUN: %clang_cc1 %s -verify -fsyntax-only

kernel void test(global int* buf) {
  buf[0] = sizeof(void); // expected-error {{invalid application of 'sizeof' to a void type}}
}
