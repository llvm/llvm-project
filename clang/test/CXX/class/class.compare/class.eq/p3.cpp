// RUN: %clang_cc1 -std=c++2a -verify %s

struct A {
  int a, b, c;
  bool operator==(const A&) const = default;
};

static_assert(A{1, 2, 3} == A{1, 2, 3});
static_assert(A{1, 2, 3} == A{0, 2, 3}); // expected-error {{failed}}
static_assert(A{1, 2, 3} == A{1, 0, 3}); // expected-error {{failed}}
static_assert(A{1, 2, 3} == A{1, 2, 0}); // expected-error {{failed}}
