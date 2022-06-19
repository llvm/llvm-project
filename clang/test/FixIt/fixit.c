// RUN: %clang_cc1 -pedantic -Wunused-label -Wno-deprecated-non-prototype -verify -x c %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -pedantic -Wunused-label -fixit -x c %t
// RUN: %clang_cc1 -pedantic -Wunused-label -Wno-deprecated-non-prototype -Werror -x c %t
// RUN: grep -v CHECK %t | FileCheck %t

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

// FIXME: FIX-IT should add #include <string.h>?
int strcmp(const char *s1, const char *s2);

void f0(void) { }; // expected-warning {{';'}}

struct s {
  int x, y;; // expected-warning {{extra ';'}}
};

// CHECK: _Complex double cd;
_Complex cd; // expected-warning {{assuming '_Complex double'}}

// CHECK: struct s s0 = { .y = 5 };
struct s s0 = { y: 5 }; // expected-warning {{GNU old-style}}

// CHECK: int array0[5] = { [3] = 3 };
int array0[5] = { [3] 3 }; // expected-warning {{GNU 'missing ='}}

// CHECK: int x
// CHECK: int y
void f1(x, y) // expected-error 2{{was not declared, defaults to 'int'; ISO C99 and later do not support implicit int}}
{
}

int i0 = { 17 };

#define ONE 1
#define TWO 2

int test_cond(int y, int fooBar) { // expected-note {{here}}
// CHECK: int x = y ? 1 : 4+fooBar;
  int x = y ? 1 4+foobar; // expected-error {{expected ':'}} expected-error {{undeclared identifier}} expected-note {{to match}}
// CHECK: x = y ? ONE : TWO;
  x = y ? ONE TWO; // expected-error {{':'}} expected-note {{to match}}
  return x;
}

// CHECK: const typedef int int_t;
const typedef typedef int int_t; // expected-warning {{duplicate 'typedef'}}

// <rdar://problem/7159693>
enum Color {
  Red // expected-error{{missing ',' between enumerators}}
  Green = 17 // expected-error{{missing ',' between enumerators}}
  Blue,
};

// rdar://9295072
struct test_struct {
  // CHECK: struct test_struct *struct_ptr;
  test_struct *struct_ptr; // expected-error {{must use 'struct' tag to refer to type 'test_struct'}}
};

void removeUnusedLabels(char c) {
  L0 /*removed comment*/:        c++; // expected-warning {{unused label}}
  removeUnusedLabels(c);
  L1: // expected-warning {{unused label}}
  c++;
  /*preserved comment*/ L2  :        c++; // expected-warning {{unused label}}
  LL // expected-warning {{unused label}}
  : c++;
  c = c + 3; L4: return; // expected-warning {{unused label}}
}

int oopsAComma = 0, // expected-error {{';'}}
void oopsMoreCommas(void) {
  static int a[] = { 0, 1, 2 }, // expected-error {{';'}}
  static int b[] = { 3, 4, 5 }, // expected-error {{';'}}
  &a == &b ? oopsMoreCommas() : removeUnusedLabels(a[0]);
}

int commaAtEndOfStatement(void) {
  int a = 1;
  a = 5, // expected-error {{';'}}
  int m = 5, // expected-error {{';'}}
  return 0, // expected-error {{';'}}
}

int noSemiAfterLabel(int n) {
  switch (n) {
    default:
      return n % 4;
    case 0:
    case 1:
    case 2:
    // CHECK: /*FOO*/ case 3: ;
    /*FOO*/ case 3: // expected-error {{expected statement}}
  }
  switch (n) {
    case 1:
    case 2:
      return 0;
    // CHECK: /*BAR*/ default: ;
    /*BAR*/ default: // expected-error {{expected statement}}
  }
  return 1;
}

struct noSemiAfterStruct // expected-error {{expected ';' after struct}}
struct noSemiAfterStruct {
  int n // expected-warning {{';'}}
} // expected-error {{expected ';' after struct}}
enum noSemiAfterEnum {
  e1
} // expected-error {{expected ';' after enum}}

int PR17175 __attribute__((visibility(hidden))); // expected-error {{'visibility' attribute requires a string}}
