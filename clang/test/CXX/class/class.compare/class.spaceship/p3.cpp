// RUN: %clang_cc1 -std=c++2a -verify %s

namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less{-1}, strong_ordering::equal{0}, strong_ordering::greater{1};
}

struct A {
  int a, b[3], c;
  std::strong_ordering operator<=>(const A&) const = default;
};

static_assert(A{1, 2, 3, 4, 5} <= A{1, 2, 3, 4, 5});
static_assert(A{1, 2, 3, 4, 5} <= A{0, 20, 3, 4, 5}); // expected-error {{failed}}
static_assert(A{1, 2, 3, 4, 5} <= A{1, 0, 30, 4, 5}); // expected-error {{failed}}
static_assert(A{1, 2, 3, 4, 5} <= A{1, 2, 0, 40, 5}); // expected-error {{failed}}
static_assert(A{1, 2, 3, 4, 5} <= A{1, 2, 3, 0, 50}); // expected-error {{failed}}
static_assert(A{1, 2, 3, 4, 5} <= A{1, 2, 3, 4, 0}); // expected-error {{failed}}

struct reverse_compare {
  int n;
  constexpr explicit reverse_compare(std::strong_ordering o) : n(-o.n) {}
  constexpr operator int() const { return n; }
};

struct B {
  int a, b[3], c;
  friend reverse_compare operator<=>(const B&, const B&) = default;
};
static_assert(B{1, 2, 3, 4, 5} >= B{1, 2, 3, 4, 5});
static_assert(B{1, 2, 3, 4, 5} >= B{0, 20, 3, 4, 5}); // expected-error {{failed}}
static_assert(B{1, 2, 3, 4, 5} >= B{1, 0, 30, 4, 5}); // expected-error {{failed}}
static_assert(B{1, 2, 3, 4, 5} >= B{1, 2, 0, 40, 5}); // expected-error {{failed}}
static_assert(B{1, 2, 3, 4, 5} >= B{1, 2, 3, 0, 50}); // expected-error {{failed}}
static_assert(B{1, 2, 3, 4, 5} >= B{1, 2, 3, 4, 0}); // expected-error {{failed}}
