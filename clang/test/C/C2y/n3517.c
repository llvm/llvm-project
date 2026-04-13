// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic -Wno-unused %s

/* WG14 N3517: No
 * Array subscripting without decay
 *
 * 1. Unconventional subscripting like 0[a] is made obsolescent.
 * 2. A negative index in subscripting array violates constraints.
 * 3. Subscripting a non-lvalue array member results in a non-lvalue.
 *
 * FIXME: Clang doesn't yet implement this paper.
 */

struct S {
  int a[1];
};

struct S get_value();

// FIXME: Should diagnose these.
void test_constraint_violation() {  
  &(get_value().a[0]);
  get_value().a[0] = 42;

  struct S s = {{0}};
  &((0, s).a[0]);
  (0, s).a[0] = 42;

  int arr[1] = {0}; // expected-note {{declared here}}

  s.a[-1];
  (0, s).a[-1];
  get_value().a[-1];
  arr[-1]; // expected-warning {{before the beginning of the array}}
}

// FIXME: Should diagnose these.
void test_deprecation() {
  int arr[1] = {0};
  0[arr];

  int* ptr = arr;
  0[ptr];

  struct S s = {{0}};
  0[s.a];
}
