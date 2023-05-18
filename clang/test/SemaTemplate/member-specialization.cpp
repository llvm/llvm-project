// RUN: %clang_cc1 -std=c++17 -verify %s
// expected-no-diagnostics

template<typename T, typename U> struct X {
  template<typename V> const V &as() { return V::error; }
  template<> const U &as<U>() { return u; }
  U u;
};
int f(X<int, int> x) {
  return x.as<int>();
}
