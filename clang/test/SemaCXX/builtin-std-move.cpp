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

  template<typename T> struct is_lvalue_reference { static constexpr bool value = false; };
  template<typename T> struct is_lvalue_reference<T&> { static constexpr bool value = true; };

  template<typename T> CONSTEXPR T &&forward(typename remove_reference<T>::type &x) {
    static_assert(T::moveable, "instantiated forward"); // expected-error {{no member named 'moveable' in 'B'}}
                                                        // expected-error@-1 {{no member named 'moveable' in 'C'}}
    return static_cast<T&&>(x);
  }
  template<typename T> CONSTEXPR T &&forward(typename remove_reference<T>::type &&x) {
    static_assert(!is_lvalue_reference<T>::value, "should not forward rval as lval"); // expected-error {{static assertion failed}}
    return static_cast<T&&>(x);
  }

  template <class T> struct is_const { static constexpr bool value = false; };
  template <class T> struct is_const<const T> { static constexpr bool value = true; };

  template <bool B, class T, class F> struct conditional { using type = T; };
  template <class T, class F> struct conditional<false, T, F> { using type = F; };

  template <class U, class T>
  using CopyConst = typename conditional<
          is_const<remove_reference<U>>::value,
          const T, T>::type;

  template <class U, class T>
  using OverrideRef = typename conditional<
          is_lvalue_reference<U &&>::value,
          typename remove_reference<T>::type &,
          typename remove_reference<T>::type &&>::type;

  template <class U, class T>
  using ForwardLikeRetType = OverrideRef<U &&, CopyConst<U, T>>;

  template <class U, class T>
  CONSTEXPR auto forward_like(T &&t) -> ForwardLikeRetType<U, T> {
    using TT = typename remove_reference<T>::type;
    static_assert(TT::moveable, "instantiated as_const"); // expected-error {{no member named 'moveable' in 'B'}}
    return static_cast<ForwardLikeRetType<U, T>>(t);
  }

  template<typename T> CONSTEXPR const T &as_const(T &x) {
    static_assert(T::moveable, "instantiated as_const"); // expected-error {{no member named 'moveable' in 'B'}}
    return x;
  }

  template<typename T> CONSTEXPR T *addressof(T &x) {
    static_assert(T::moveable, "instantiated addressof"); // expected-error {{no member named 'moveable' in 'B'}}
    return __builtin_addressof(x);
  }

  template<typename T> CONSTEXPR T *__addressof(T &x) {
    static_assert(T::moveable, "instantiated __addressof"); // expected-error {{no member named 'moveable' in 'B'}}
    return __builtin_addressof(x);
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
  const A &as_const = std::as_const(a);
  A *addressof = std::addressof(a);
  A *addressof2 = std::__addressof(a);
  return &move == &a && &move_if_noexcept == &a &&
         &forward1 == &a && &forward2 == &a &&
         &as_const == &a && addressof == &a &&
         addressof2 == &a && std::move(a, a) == 5;
}

#ifndef NO_CONSTEXPR
static_assert(f({}), "should be constexpr");
#else
// expected-error@#f {{never produces a constant expression}}
// expected-note@#call {{}}
#endif

A &forward_rval_as_lval() {
  std::forward<A&&>(A()); // expected-warning {{const attribute}}
  return std::forward<A&>(A()); // expected-note {{instantiation of}} expected-warning {{returning reference}}
}

struct B {};
B &&(*pMove)(B&) = std::move; // #1 expected-note {{instantiation of}}
B &&(*pMoveIfNoexcept)(B&) = &std::move_if_noexcept; // #2 expected-note {{instantiation of}}
B &&(*pForward)(B&) = &std::forward<B>; // #3 expected-note {{instantiation of}}
B &&(*pForwardLike)(B&) = &std::forward_like<int&&, B&>; // #4 expected-note {{instantiation of}}
const B &(*pAsConst)(B&) = &std::as_const; // #5 expected-note {{instantiation of}}
B *(*pAddressof)(B&) = &std::addressof; // #6 expected-note {{instantiation of}}
B *(*pUnderUnderAddressof)(B&) = &std::__addressof; // #7 expected-note {{instantiation of}}
int (*pUnrelatedMove)(B, B) = std::move;

struct C {};
C &&(&rMove)(C&) = std::move; // #8 expected-note {{instantiation of}}
C &&(&rForward)(C&) = std::forward<C>; // #9 expected-note {{instantiation of}}
int (&rUnrelatedMove)(B, B) = std::move;

#if __cplusplus <= 201703L
// expected-warning@#1 {{non-addressable}}
// expected-warning@#2 {{non-addressable}}
// expected-warning@#3 {{non-addressable}}
// expected-warning@#4 {{non-addressable}}
// expected-warning@#5 {{non-addressable}}
// expected-warning@#6 {{non-addressable}}
// expected-warning@#7 {{non-addressable}}
// expected-warning@#8 {{non-addressable}}
// expected-warning@#9 {{non-addressable}}
#else
// expected-error@#1 {{non-addressable}}
// expected-error@#2 {{non-addressable}}
// expected-error@#3 {{non-addressable}}
// expected-error@#4 {{non-addressable}}
// expected-error@#5 {{non-addressable}}
// expected-error@#6 {{non-addressable}}
// expected-error@#7 {{non-addressable}}
// expected-error@#8 {{non-addressable}}
// expected-error@#9 {{non-addressable}}
#endif

void attribute_const() {
  int n;
  std::move(n); // expected-warning {{ignoring return value}}
  std::move_if_noexcept(n); // expected-warning {{ignoring return value}}
  std::forward<int>(n); // expected-warning {{ignoring return value}}
  std::forward_like<float&&>(n); // expected-warning {{ignoring return value}}
  std::addressof(n); // expected-warning {{ignoring return value}}
  std::__addressof(n); // expected-warning {{ignoring return value}}
  std::as_const(n); // expected-warning {{ignoring return value}}
}

namespace std {
  template<typename T> int &move(T);
}
int bad_signature = std::move(0); // expected-error {{unsupported signature for 'std::move<int>'}}
