// RUN: %clang_cc1 -std=c++11 -verify %s

template<typename ...T> struct X {};

template<typename T, typename U> struct P {};

namespace Nested {
  template<typename ...T> int f1(X<T, T...>... a); // expected-note +{{conflicting types for parameter 'T'}}
  template<typename ...T> int f2(P<X<T...>, T> ...a); // expected-note +{{conflicting types for parameter 'T'}}

  int a1 = f1(X<int, int, double>(), X<double, int, double>());
  int a2 = f1(X<int, int>());
  int a3 = f1(X<int>(), X<double>()); // expected-error {{no matching}}
  int a4 = f1(X<int, int>(), X<int>()); // expected-error {{no matching}}
  int a5 = f1(X<int>(), X<int, int>()); // expected-error {{no matching}}
  int a6 = f1(X<int, int, int>(), X<int, int, int>(), X<int, int, int, int>()); // expected-error {{no matching}}

  int b1 = f2(P<X<int, double>, int>(), P<X<int, double>, double>());
  int b2 = f2(P<X<int, double>, int>(), P<X<int, double>, double>(), P<X<int, double>, char>()); // expected-error {{no matching}}
}

namespace PR14841 {
  template<typename T, typename U> struct A {};
  template<typename ...Ts> void f(A<Ts...>); // expected-note {{substitution failure [with Ts = <char, short, int>]: too many template arg}}

  void g(A<char, short> a) {
    f(a);
    f<char>(a);
    f<char, short>(a);
    f<char, short, int>(a); // expected-error {{no matching function}}
  }
}

namespace RetainExprPacks {
  int f(int a, int b, int c);
  template<typename ...Ts> struct X {};
  template<typename ...Ts> int g(X<Ts...>, decltype(f(Ts()...)));
  int n = g<int, int>(X<int, int, int>(), 0);
}

namespace PR14615 {
  namespace comment0 {
    template <class A, class...> struct X {};
    template <class... B> struct X<int, B...> {
      typedef int type;
      struct valid {};
    };
    template <typename A, typename... B, typename T = X<A, B...>,
              typename = typename T::valid>
    typename T::type check(int);
    int i = check<int, char>(1);
  }

  namespace comment2 {
    template <class...> struct X;
    template <typename... B, typename X<B...>::type I = 0>
    char check(B...); // expected-note {{undefined template 'PR14615::comment2::X<char, int>'}}
    void f() { check<char>(1, 2); } // expected-error {{no matching function}}
  }

  namespace comment3 {
    template <class...> struct X;
    template <typename... B, typename X<B...>::type I = (typename X<B...>::type)0>
    char check(B...); // expected-note {{undefined template 'PR14615::comment3::X<char, int>'}}
    void f() { check<char>(1, 2); } // expected-error {{no matching function}}
  }
}
