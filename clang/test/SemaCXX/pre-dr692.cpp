// RUN: %clang_cc1 %s -std=c++11 -verify -fexceptions -fcxx-exceptions -pedantic-errors -fno-spell-checking -fclang-abi-compat=15

template <typename... T> struct A1 {};
template <typename U, typename... T> struct A2 {};
template <class T1, class... U> void e1(A1<T1, U...>);  // expected-note {{candidate}}
template <class T1> void e1(A1<T1>);  // expected-note {{candidate}}
template <class T1, class... U> void e2(A2<T1, U...>);  // expected-note {{candidate}}
template <class T1> void e2(A2<T1>);  // expected-note {{candidate}}
void h() {
  A1<int> b1;
  e1(b1); // expected-error{{call to 'e1' is ambiguous}}
  A2<int> b2;
  e2(b2); // expected-error{{call to 'e2' is ambiguous}}
}
