// RUN: %clang_cc1 -DSIZE_T_64 -fsyntax-only -verify -std=c++11 -triple x86_64-linux-gnu %s
// RUN: %clang_cc1 -DSIZE_T_64 -fsyntax-only -verify -std=c++11 -triple x86_64-linux-gnu %s -fexperimental-new-constant-interpreter


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
  __builtin_assume_dereferenceable(a, size);
  return a[0];
}

int test5(int *a, unsigned long long size) {
  __builtin_assume_dereferenceable(a, size);
  return a[0];
}

int test6(float a) {
  __builtin_assume_dereferenceable(a, 2); // expected-error {{cannot initialize a parameter of type 'const void *' with an lvalue of type 'float'}}
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

int test9(int *a) {
  a[0] = __builtin_assume_dereferenceable(a, 32); // expected-error {{assigning to 'int' from incompatible type 'void'}}
  return a[0];
}

constexpr int *p = 0;
constexpr void *l = __builtin_assume_dereferenceable(p, 4); // expected-error {{cannot initialize a variable of type 'void *const' with an rvalue of type 'void'}}

void *foo() {
  return l;
}

int test10(int *a) {
  __builtin_assume_dereferenceable(a, a); // expected-error {{cannot initialize a parameter of type '__size_t' (aka 'unsigned long') with an lvalue of type 'int *'}}
  return a[0];
}
