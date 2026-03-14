// RUN: %clang_cc1 -verify -std=c++17 %s

template<typename T> constexpr int f() { return T::value; } // expected-error {{'::'}}
template<bool B, typename T> void g(decltype(B ? f<T>() : 0));
template<bool B, typename T> void g(...);
template<bool B, typename T> void h(decltype(int{B ? f<T>() : 0})); // expected-note {{instantiation of}}
template<bool B, typename T> void h(...);
void x() {
  g<false, int>(0); // ok
  g<true, int>(0); // ok
  h<false, int>(0); // expected-note {{while substituting}}
}
