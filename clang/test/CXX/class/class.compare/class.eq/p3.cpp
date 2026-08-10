// RUN: %clang_cc1 -std=c++2a -verify %s

struct A {
  int a, b[3], c;
  bool operator==(const A&) const = default;
};

static_assert(A{1, 2, 3, 4, 5} == A{1, 2, 3, 4, 5});
static_assert(A{1, 2, 3, 4, 5} == A{0, 2, 3, 4, 5}); // expected-error {{failed}}
static_assert(A{1, 2, 3, 4, 5} == A{1, 0, 3, 4, 5}); // expected-error {{failed}}
static_assert(A{1, 2, 3, 4, 5} == A{1, 2, 0, 4, 5}); // expected-error {{failed}}
static_assert(A{1, 2, 3, 4, 5} == A{1, 2, 3, 0, 5}); // expected-error {{failed}}
static_assert(A{1, 2, 3, 4, 5} == A{1, 2, 3, 4, 0}); // expected-error {{failed}}

struct B {
  int a, b[3], c;
  friend bool operator==(B, B) = default;
};

static_assert(B{1, 2, 3, 4, 5} == B{1, 2, 3, 4, 5});
static_assert(B{1, 2, 3, 4, 5} == B{0, 2, 3, 4, 5}); // expected-error {{failed}}
static_assert(B{1, 2, 3, 4, 5} == B{1, 0, 3, 4, 5}); // expected-error {{failed}}
static_assert(B{1, 2, 3, 4, 5} == B{1, 2, 0, 4, 5}); // expected-error {{failed}}
static_assert(B{1, 2, 3, 4, 5} == B{1, 2, 3, 0, 5}); // expected-error {{failed}}
static_assert(B{1, 2, 3, 4, 5} == B{1, 2, 3, 4, 0}); // expected-error {{failed}}
