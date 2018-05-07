// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace PR15360 {
  template<typename R, typename U, R F>
  U f() { return &F; } // expected-error{{cannot take the address of an rvalue of type 'int (*)(int)'}} expected-error{{cannot take the address of an rvalue of type 'int *'}}
  void test() {
    f<int(int), int(*)(int), nullptr>(); // expected-note{{in instantiation of}}
    f<int[3], int*, nullptr>(); // expected-note{{in instantiation of}}
  }
}

namespace CanonicalNullptr {
  template<typename T> struct get { typedef T type; };
  struct X {};
  template<typename T, typename get<T *>::type P = nullptr> struct A {};
  template<typename T, typename get<decltype((T(), nullptr))>::type P = nullptr> struct B {};
  template<typename T, typename get<T X::*>::type P = nullptr> struct C {};

  template<typename T> A<T> MakeA();
  template<typename T> B<T> MakeB();
  template<typename T> C<T> MakeC();
  A<int> a = MakeA<int>();
  B<int> b = MakeB<int>();
  C<int> c = MakeC<int>();
}

namespace Auto {
  template<auto> struct A { };  // expected-error {{until C++17}}
}

namespace check_conversion_early {
  struct X {};
  template<int> struct A {};
  template<X &x> struct A<x> {}; // expected-error {{not implicitly convertible}}

  struct Y { constexpr operator int() const { return 0; } };
  template<Y &y> struct A<y> {}; // expected-error {{cannot be deduced}} expected-note {{'y'}}
}

namespace ReportCorrectParam {
template <int a, unsigned b, int c>
void TempFunc() {}

void Useage() {
  //expected-error@+2 {{no matching function}}
  //expected-note@-4 {{candidate template ignored: invalid explicitly-specified argument for template parameter 'b'}}
  TempFunc<1, -1, 1>();
}
}

