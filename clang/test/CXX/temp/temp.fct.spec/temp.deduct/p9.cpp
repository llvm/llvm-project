// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template <int> int f(int);  // expected-note {{candidate function}}
#if __cplusplus <= 199711L
// expected-note@-2 {{candidate function}}
#endif

template <signed char> int f(int); // expected-note {{candidate function}}
#if __cplusplus <= 199711L
// expected-note@-2 {{candidate function}}
#endif

int i1 = f<1>(0); // expected-error{{call to 'f' is ambiguous}}
int i2 = f<1000>(0);
#if __cplusplus <= 199711L
// expected-error@-2{{call to 'f' is ambiguous}}
#endif

namespace PR6707 {
  template<typename T, T Value>
  struct X { };

  template<typename T, T Value>
  void f(X<T, Value>);

  void g(X<int, 10> x) {
    f(x);
  }

  static const unsigned char ten = 10;
  template<typename T, T Value, typename U>
  void f2(X<T, Value>, X<U, Value>);
  // expected-note@-1 {{candidate template ignored: deduced values of conflicting types for parameter 'Value' (10 of type 'int' vs. 10 of type 'char')}}
  // expected-note@-2 {{candidate template ignored: deduced values of conflicting types for parameter 'Value' (10 of type 'char' vs. 10 of type 'int')}}

  void g2() {
    f2(X<int, 10>(), X<char, ten>()); // expected-error {{no matching}}
    f2(X<char, 10>(), X<int, ten>()); // expected-error {{no matching}}
  }
}
