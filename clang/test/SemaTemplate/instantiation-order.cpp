// RUN: %clang_cc1 -std=c++11 -verify %s

// From core issue 1227.

template <class T> struct A { using X = typename T::X; }; // expected-error {{no members}}
template <class T> typename T::X f(typename A<T>::X);
template <class T> void f(...) {}
template <class T> auto g(typename A<T>::X) -> typename T::X; // expected-note {{here}} expected-note {{substituting}}
template <class T> void g(...) {}

void h()
{
  f<int>(0); // ok, SFINAE in return type
  g<int>(0); // not ok, substitution inside A<int> is a hard error
}
