// RUN: %clang_cc1 -std=c++17 -verify %s

template<typename T, typename U> struct X {
  template<typename V> const V &as() { return V::error; }
  template<> const U &as<U>() { return u; }
  U u;
};
int f(X<int, int> x) {
  return x.as<int>();
}

namespace GH205971 {
  template<class> struct A {};

  template<>
  template<class>
  struct A<int>::B;
  // expected-error@-1 {{out-of-line definition of 'B' does not match any declaration}}

  template<>
  template<class>
  struct A<int>::B;
  // expected-error@-1 {{out-of-line definition of 'B' does not match any declaration}}
} // namespace GH205971
