// RUN: %clang_cc1 -std=c++1z -verify %s

template<typename ...T> constexpr auto sum(T ...t) { return (... + t); }
template<typename ...T> constexpr auto product(T ...t) { return (t * ...); }
template<typename ...T> constexpr auto all(T ...t) { return (true && ... && t); }
template<typename ...T> constexpr auto dumb(T ...t) { return (false && ... && t); }

static_assert(sum(1, 2, 3, 4, 5) == 15);
static_assert(product(1, 2, 3, 4, 5) == 120);
static_assert(!all(true, true, false, true, false));
static_assert(all(true, true, true, true, true));
static_assert(!dumb(true, true, true, true, true));

struct S {
  int a, b, c, d, e;
};
template<typename ...T> constexpr auto increment_all(T &...t) {
  (++t, ...);
}
constexpr bool check() {
  S s = { 1, 2, 3, 4, 5 };
  increment_all(s.a, s.b, s.c, s.d, s.e);
  return s.a == 2 && s.b == 3 && s.c == 4 && s.d == 5 && s.e == 6;
}
static_assert(check());

template<int ...N> void empty() {
  static_assert((N || ...) == false);
  static_assert((N && ...) == true);
  (N, ...);
}
template void empty<>();

// An empty fold-expression isn't a null pointer just because it's an integer
// with value 0. (This is no longer an issue since empty pack expansions don't
// produce integers any more.)
template<int ...N> void null_ptr() {
  void *p = (N || ...); // expected-error {{rvalue of type 'bool'}}
  void *q = (N , ...); // expected-error {{rvalue of type 'void'}}
}
template void null_ptr<>(); // expected-note {{in instantiation of}}

template<int ...N> void bad_empty() {
  (N + ...); // expected-error {{empty expansion for operator '+' with no fallback}}
  (N * ...); // expected-error {{empty expansion for operator '*' with no fallback}}
  (N | ...); // expected-error {{empty expansion for operator '|' with no fallback}}
  (N & ...); // expected-error {{empty expansion for operator '&' with no fallback}}
  (N - ...); // expected-error {{empty expansion for operator '-' with no fallback}}
  (N / ...); // expected-error {{empty expansion for operator '/' with no fallback}}
  (N % ...); // expected-error {{empty expansion for operator '%' with no fallback}}
  (N = ...); // expected-error {{empty expansion for operator '=' with no fallback}}
}
template void bad_empty<>(); // expected-note {{in instantiation of}}

template<int ...N> void empty_with_base() {
  extern int k;
  (k = ... = N); // expected-warning{{unused}}

  void (k = ... = N); // expected-error {{expected ')'}} expected-note {{to match}}
  void ((k = ... = N));
  (void) (k = ... = N);
}
template void empty_with_base<>(); // expected-note {{in instantiation of}}
template void empty_with_base<1>();

struct A {
  struct B {
    struct C {
      struct D {
        int e;
      } d;
    } c;
  } b;
} a;
template<typename T, typename ...Ts> constexpr decltype(auto) apply(T &t, Ts ...ts) {
  return (t.*....*ts);
}
static_assert(&apply(a, &A::b, &A::B::c, &A::B::C::d, &A::B::C::D::e) == &a.b.c.d.e);
