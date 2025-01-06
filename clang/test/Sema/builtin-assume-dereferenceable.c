// RUN: %clang_cc1 -DSIZE_T_64 -fsyntax-only -Wno-strict-prototypes -triple x86_64-linux -verify %s

int test1(int *a) {
  __builtin_assume_dereferenceable(a, 32);
  return a[0];
}

int test2(int *a) {
  __builtin_assume_dereferenceable(a, 32ull);
  return a[0];
}

int test3(int *a) {
  __builtin_assume_dereferenceable(a, 32u);
  return a[0];
}

int test4(int *a, unsigned size) {
  a = __builtin_assume_dereferenceable(a, size); // expected-error {{argument to '__builtin_assume_dereferenceable' must be a constant integer}}
  return a[0];
}

int test5(int *a, unsigned long long size) {
  a = __builtin_assume_dereferenceable(a, size); // expected-error {{argument to '__builtin_assume_dereferenceable' must be a constant integer}}
  return a[0];
}

int test6(float a) {
  __builtin_assume_dereferenceable(a, 2); // expected-error {{passing 'float' to parameter of incompatible type 'const void *'}}
  return 0;;
}

int test7(int *a) {
  __builtin_assume_dereferenceable(a, 32, 1); // expected-error {{too many arguments to function call, expected 2, have 3}}
  return a[0];
}

int test8(int *a) {
  __builtin_assume_dereferenceable(a); // expected-error {{too few arguments to function call, expected 2, have 1}}
  return a[0];
}
