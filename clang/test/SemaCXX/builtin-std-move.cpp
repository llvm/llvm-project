// RUN: %clang_cc1 -std=c++17 -verify %s
// RUN: %clang_cc1 -std=c++17 -verify %s -DNO_CONSTEXPR
// RUN: %clang_cc1 -std=c++20 -verify %s

namespace std {
#ifndef NO_CONSTEXPR
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

  template<typename T> CONSTEXPR T &&move(T &x) {
    static_assert(T::moveable, "instantiated move"); // expected-error {{no member named 'moveable' in 'B'}}
                                                     // expected-error@-1 {{no member named 'moveable' in 'C'}}
    return static_cast<T&&>(x);
  }

  // Unrelated move functions are not the builtin.
  template<typename T> CONSTEXPR int move(T, T) { return 5; }

  template<typename T, bool Rref> struct ref { using type = T&; };
  template<typename T> struct ref<T, true> { using type = T&&; };

  template<typename T> CONSTEXPR auto move_if_noexcept(T &x) -> typename ref<T, noexcept(T(static_cast<T&&>(x)))>::type {
    static_assert(T::moveable, "instantiated move_if_noexcept"); // expected-error {{no member named 'moveable' in 'B'}}
    return static_cast<typename ref<T, noexcept(T(static_cast<T&&>(x)))>::type>(x);
  }

  template<typename T> struct remove_reference { using type = T; };
  template<typename T> struct remove_reference<T&> { using type = T; };
  template<typename T> struct remove_reference<T&&> { using type = T; };

  template<typename T> CONSTEXPR T &&forward(typename remove_reference<T>::type &x) {
    static_assert(T::moveable, "instantiated forward"); // expected-error {{no member named 'moveable' in 'B'}}
                                                        // expected-error@-1 {{no member named 'moveable' in 'C'}}
    return static_cast<T&&>(x);
  }
}

// Note: this doesn't have a 'moveable' member. Instantiation of the above
// functions will fail if it's attempted.
struct A {};
constexpr bool f(A a) { // #f
  A &&move = std::move(a); // #call
  A &&move_if_noexcept = std::move_if_noexcept(a);
  A &&forward1 = std::forward<A>(a);
  A &forward2 = std::forward<A&>(a);
  return &move == &a && &move_if_noexcept == &a &&
         &forward1 == &a && &forward2 == &a &&
         std::move(a, a) == 5;
}

#ifndef NO_CONSTEXPR
static_assert(f({}), "should be constexpr");
#else
// expected-error@#f {{never produces a constant expression}}
// expected-note@#call {{}}
#endif

struct B {};
B &&(*pMove)(B&) = std::move; // #1 expected-note {{instantiation of}}
B &&(*pMoveIfNoexcept)(B&) = &std::move_if_noexcept; // #2 expected-note {{instantiation of}}
B &&(*pForward)(B&) = &std::forward<B>; // #3 expected-note {{instantiation of}}
int (*pUnrelatedMove)(B, B) = std::move;

struct C {};
C &&(&rMove)(C&) = std::move; // #4 expected-note {{instantiation of}}
C &&(&rForward)(C&) = std::forward<C>; // #5 expected-note {{instantiation of}}
int (&rUnrelatedMove)(B, B) = std::move;

#if __cplusplus <= 201703L
// expected-warning@#1 {{non-addressable}}
// expected-warning@#2 {{non-addressable}}
// expected-warning@#3 {{non-addressable}}
// expected-warning@#4 {{non-addressable}}
// expected-warning@#5 {{non-addressable}}
#else
// expected-error@#1 {{non-addressable}}
// expected-error@#2 {{non-addressable}}
// expected-error@#3 {{non-addressable}}
// expected-error@#4 {{non-addressable}}
// expected-error@#5 {{non-addressable}}
#endif

void attribute_const() {
  int n;
  std::move(n); // expected-warning {{ignoring return value}}
  std::move_if_noexcept(n); // expected-warning {{ignoring return value}}
  std::forward<int>(n); // expected-warning {{ignoring return value}}
}
