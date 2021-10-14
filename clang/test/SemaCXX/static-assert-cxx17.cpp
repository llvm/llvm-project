// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++1z -triple=x86_64-linux-gnu

template <typename U, typename V>
struct S1 {
  static constexpr const bool value = false;
};

template <typename U, typename V>
inline constexpr bool global_inline_var = S1<U, V>::value;

template <typename T>
struct S2 {
  template <typename U, typename V>
  static inline constexpr bool var = global_inline_var<U, V>;
};

template <typename U, typename V>
inline constexpr bool constexpr_return_false() {
  return false;
}

template <typename U, typename V>
void foo() {
  static_assert(S1<U, V>::value);
  // expected-error@-1{{static_assert failed due to requirement 'S1<int, float>::value'}}
}
template void foo<int, float>();
// expected-note@-1{{in instantiation of function template specialization 'foo<int, float>' requested here}}

template <typename U, typename V>
void foo2() {
  static_assert(global_inline_var<U, V>);
  // expected-error@-1{{static_assert failed due to requirement 'global_inline_var<int, float>'}}
}
template void foo2<int, float>();
// expected-note@-1{{in instantiation of function template specialization 'foo2<int, float>' requested here}}

template <typename T, typename U, typename V>
void foo3() {
  static_assert(T::template var<U, V>);
  // expected-error@-1{{static_assert failed due to requirement 'S2<long>::var<int, float>'}}
}
template void foo3<S2<long>, int, float>();
// expected-note@-1{{in instantiation of function template specialization 'foo3<S2<long>, int, float>' requested here}}

template <typename T>
void foo4() {
  static_assert(S1<T[sizeof(T)], int[4]>::value, "");
  // expected-error@-1{{static_assert failed due to requirement 'S1<float [4], int [4]>::value'}}
};
template void foo4<float>();
// expected-note@-1{{in instantiation of function template specialization 'foo4<float>' requested here}}


template <typename U, typename V>
void foo5() {
  static_assert(!!(global_inline_var<U, V>));
  // expected-error@-1{{static_assert failed due to requirement '!!(global_inline_var<int, float>)'}}
}
template void foo5<int, float>();
// expected-note@-1{{in instantiation of function template specialization 'foo5<int, float>' requested here}}

struct ExampleTypes {
  explicit ExampleTypes(int);
  using T = int;
  using U = float;
};

template <class T>
struct X {
  int i = 0;
  int j = 0;
  constexpr operator bool() const { return false; }
};

template <class T>
void foo6() {
  static_assert(X<typename T::T>());
  // expected-error@-1{{static_assert failed due to requirement 'X<int>()'}}
  static_assert(X<typename T::T>{});
  // expected-error@-1{{static_assert failed due to requirement 'X<int>{}'}}
  static_assert(X<typename T::T>{1, 2});
  // expected-error@-1{{static_assert failed due to requirement 'X<int>{1, 2}'}}
  static_assert(X<typename T::T>({1, 2}));
  // expected-error@-1{{static_assert failed due to requirement 'X<int>({1, 2})'}}
  static_assert(typename T::T{0});
  // expected-error@-1{{static_assert failed due to requirement 'int{0}'}}
  static_assert(typename T::T(0));
  // expected-error@-1{{static_assert failed due to requirement 'int(0)'}}
  static_assert(sizeof(X<typename T::T>) == 0);
  // expected-error@-1{{static_assert failed due to requirement 'sizeof(X<int>) == 0'}}
  static_assert((const X<typename T::T> *)nullptr);
  // expected-error@-1{{static_assert failed due to requirement '(const X<int> *)nullptr'}}
  static_assert(static_cast<const X<typename T::T> *>(nullptr));
  // expected-error@-1{{static_assert failed due to requirement 'static_cast<const X<int> *>(nullptr)'}}
  static_assert((const X<typename T::T>[]){} == nullptr);
  // expected-error@-1{{static_assert failed due to requirement '(const X<int> [0]){} == nullptr'}}
  static_assert(sizeof(X<decltype(X<typename T::T>().X<typename T::T>::~X())>) == 0);
  // expected-error@-1{{static_assert failed due to requirement 'sizeof(X<void>) == 0'}}
  static_assert(constexpr_return_false<typename T::T, typename T::U>());
  // expected-error@-1{{static_assert failed due to requirement 'constexpr_return_false<int, float>()'}}
}
template void foo6<ExampleTypes>();
// expected-note@-1{{in instantiation of function template specialization 'foo6<ExampleTypes>' requested here}}
