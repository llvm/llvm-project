// RUN: %clang_cc1 %s -triple %itanium_abi_triple -fsyntax-only -verify -Wc++-compat

// Note: Empty C structs are 4 bytes in the Microsoft ABI.

struct emp_1 { // expected-warning {{empty struct has size 0 in C, size 1 in C++}}
};

union emp_2 { // expected-warning {{empty union has size 0 in C, size 1 in C++}}
};

struct emp_3 { // expected-warning {{struct has size 0 in C, size 1 in C++}}
  int : 0;
};

union emp_4 { // expected-warning {{union has size 0 in C, size 1 in C++}}
  int : 0;
};

struct emp_5 { // expected-warning {{struct has size 0 in C, size 1 in C++}}
  int : 0;
  int : 0;
};

union emp_6 { // expected-warning {{union has size 0 in C, size 1 in C++}}
  int : 0;
  int : 0;
};

struct emp_7 { // expected-warning {{struct has size 0 in C, size 1 in C++}}
  struct emp_1 f1;
};

union emp_8 { // expected-warning {{union has size 0 in C, size 1 in C++}}
  struct emp_1 f1;
};

struct emp_9 { // expected-warning {{struct has size 0 in C, non-zero size in C++}}
  struct emp_1 f1;
  union emp_2 f2;
};

// Checks for pointer subtraction (PR15683)
struct emp_1 *func_1p(struct emp_1 *x) { return x - 5; }

int func_1(void) {
  struct emp_1 v[1];
  return v - v; // expected-warning {{subtraction of pointers to type 'struct emp_1' of zero size has undefined behavior}}
}

int func_2(struct emp_1 *x) {
  return 1 + x - x; // expected-warning {{subtraction of pointers to type 'struct emp_1' of zero size has undefined behavior}}
}

int func_3(struct emp_1 *x, struct emp_1 *y) {
  return x - y; // expected-warning {{subtraction of pointers to type 'struct emp_1' of zero size has undefined behavior}}
}

int func_4(struct emp_1 *x, const struct emp_1 *y) {
  return x - y; // expected-warning {{subtraction of pointers to type 'struct emp_1' of zero size has undefined behavior}}
}

int func_5(volatile struct emp_1 *x, const struct emp_1 *y) {
  return x - y; // expected-warning {{subtraction of pointers to type 'struct emp_1' of zero size has undefined behavior}}
}

int func_6(void) {
  union emp_2 v[1];
  return v - v; // expected-warning {{subtraction of pointers to type 'union emp_2' of zero size has undefined behavior}}
}

struct A; // expected-note {{forward declaration of 'struct A'}}

int func_7(struct A *x, struct A *y) {
  return x - y; // expected-error {{arithmetic on a pointer to an incomplete type 'struct A'}}
}

int func_8(struct emp_1 (*x)[10], struct emp_1 (*y)[10]) {
  return x - y; // expected-warning {{subtraction of pointers to type 'struct emp_1[10]' of zero size has undefined behavior}}
}

int func_9(struct emp_1 (*x)[], struct emp_1 (*y)[]) {
  return x - y; // expected-error {{arithmetic on a pointer to an incomplete type 'struct emp_1[]'}}
}

int func_10(int (*x)[0], int (*y)[0]) {
  return x - y; // expected-warning {{subtraction of pointers to type 'int[0]' of zero size has undefined behavior}}
}
