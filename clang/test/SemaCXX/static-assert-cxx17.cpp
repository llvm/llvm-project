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
