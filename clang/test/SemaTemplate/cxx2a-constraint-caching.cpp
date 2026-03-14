// RUN:  %clang_cc1 -std=c++2a -verify %s
// expected-no-diagnostics

template<typename T>
concept C = (f(T()), true);

template<typename T>
constexpr bool foo() { return false; }

template<typename T>
  requires (f(T()), true)
constexpr bool foo() requires (f(T()), true) { return true; }

namespace a {
  struct A {};
  constexpr void f(A a) {}
}

static_assert(C<a::A>);
static_assert(foo<a::A>());

namespace a {
  // This makes calls to f ambiguous, but the second check will still succeed
  // because the constraint satisfaction results are cached.
  constexpr void f(A a, int = 2) {}
}
static_assert(C<a::A>);
static_assert(foo<a::A>());
