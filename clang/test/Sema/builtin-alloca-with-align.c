// RUN: %clang_cc1 -fsyntax-only -verify %s

void test1(int a) {
  __builtin_alloca_with_align(a, 32);
}

void test2(int a) {
  __builtin_alloca_with_align(a, -32); // expected-error {{requested alignment is not a power of 2}}
}

void test3(unsigned *b) {
  __builtin_alloca_with_align(b, 32); // expected-warning {{incompatible pointer to integer conversion passing 'unsigned int *' to parameter of type}}
}

void test4(int a) {
  __builtin_alloca_with_align(a, 32, 0); // expected-error {{too many arguments to function call, expected 2, have 3}}
}

void test5(int a) {
  __builtin_alloca_with_align(a, 31); // expected-error {{requested alignment is not a power of 2}}
}

void test6(int a, int j) {
  __builtin_alloca_with_align(a, j); // expected-error {{must be a constant integer}}
}

void test7(int a) {
  __builtin_alloca_with_align(a, 2); // expected-error {{requested alignment must be 8 or greater}}
}

void test8(void) {
  __builtin_alloca_with_align(sizeof(__INT64_TYPE__), __alignof__(__INT64_TYPE__)); // expected-warning {{second argument to __builtin_alloca_with_align is supposed to be in bits}}
#if defined(__csky__)
  // expected-error@-2 {{requested alignment must be 8 or greater}}
  // Because the alignment of long long is 4 in CSKY target
#endif
}
