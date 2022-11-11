// RUN: %clang_cc1 -std=c++11 -verify %s

template<typename ...T> struct X {};

template<typename T, typename U> struct P {};

namespace Nested {
  template<typename ...T> int f1(X<T, T...>... a); // expected-note +{{packs of different lengths for parameter 'T'}}
  template<typename ...T> int f2(P<X<T...>, T> ...a); // expected-note +{{packs of different lengths for parameter 'T'}}

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

namespace fully_expanded_packs {
  template<typename ...T> struct A {
    template<T ...X> static constexpr int f() {
      // expected-note@-1 1+{{deduced too few arguments for expanded pack 'X'}}
      // expected-note@-2 1+{{too many template arguments}}
      return (X + ... + 0); // expected-warning {{extension}}
    }

    template<T ...X, int Y> static constexpr int g() {
      // expected-note@-1 1+{{deduced too few arguments for expanded pack 'X'}}
      // expected-note@-2 1+{{couldn't infer template argument 'Y'}}
      // expected-note@-3 1+{{too many template arguments}}
      return (X + ... + (1000 * Y)); // expected-warning {{extension}}
    }

    template<T ...X, int Y, T ...Z> static constexpr int h() {
      // expected-note@-1 1+{{deduced too few arguments for expanded pack 'X'}}
      // expected-note@-2 1+{{couldn't infer template argument 'Y'}}
      // expected-note@-3 1+{{deduced too few arguments for expanded pack 'Z'}}
      // expected-note@-4 1+{{too many template arguments}}
      return (X + ... + (1000 * Y)) + 1000000 * (Z + ... + 0); // expected-warning 2{{extension}}
    }

    template<T ...X, int ...Z> static constexpr int i() {
      return (X + ... + 0) + 1000 * (Z + ... + 0); // expected-warning 2{{extension}}
    }

    template<T ...X, int Y, int ...Z> static constexpr int j() {
      return (X + ... + (1000 * Y)) + 1000000 * (Z + ... + 0); // expected-warning 2{{extension}}
    }
  };

  void check_invalid_calls() {
    A<int, int>::f(); // expected-error {{no matching function}}
    A<int, int>::f<>(); // expected-error {{no matching function}}
    A<int, int>::f<0>(); // expected-error {{no matching function}}
    A<int, int>::g(); // expected-error {{no matching function}}
    A<int, int>::g<>(); // expected-error {{no matching function}}
    A<int, int>::g<0>(); // expected-error {{no matching function}}
    A<int, int>::g<0, 0>(); // expected-error {{no matching function}}
    A<>::f<0>(); // expected-error {{no matching function}}
    A<>::g(); // expected-error {{no matching function}}
    A<>::g<>(); // expected-error {{no matching function}}
    A<>::g<0, 0>(); // expected-error {{no matching function}}
    A<>::h<>(); // expected-error {{no matching function}}
    A<int>::h<>(); // expected-error {{no matching function}}
    A<int>::h<0, 0>(); // expected-error {{no matching function}}
    A<>::h<0, 0>(); // expected-error {{no matching function}}
  }

  static_assert(A<>::f() == 0, "");
  static_assert(A<int>::f<1>() == 1, "");
  static_assert(A<>::g<1>() == 1000, "");
  static_assert(A<int>::g<1, 2>() == 2001, "");
  static_assert(A<>::h<1>() == 1000, "");
  static_assert(A<int>::h<1, 2, 3>() == 3002001, "");
  static_assert(A<int, int>::h<1, 20, 3, 4, 50>() == 54003021, "");
  static_assert(A<>::i<1>() == 1000, "");
  static_assert(A<int>::i<1>() == 1, "");
  static_assert(A<>::j<1, 2, 30>() == 32001000, "");
  static_assert(A<int>::j<1, 2, 3, 40>() == 43002001, "");
}

namespace partial_full_mix {
  template<typename T, typename U> struct pair {};
  template<typename ...T> struct tuple {};
  template<typename ...T> struct A {
    template<typename ...U> static pair<tuple<T...>, tuple<U...>> f(pair<T, U> ...p);
    // expected-note@-1 {{[with U = <char, double, long>]: pack expansion contains parameter packs 'T' and 'U' that have different lengths (2 vs. 3)}}
    // expected-note@-2 {{[with U = <char, double, void>]: pack expansion contains parameter pack 'U' that has a different length (at least 3 vs. 2) from outer parameter packs}}

    template<typename ...U> static pair<tuple<T...>, tuple<U...>> g(pair<T, U> ...p, ...);
    // expected-note@-1 {{[with U = <char, double, long>]: pack expansion contains parameter packs 'T' and 'U' that have different lengths (2 vs. 3)}}

    template<typename ...U> static tuple<U...> h(tuple<pair<T, U>..., pair<int, int>>);
    // expected-note@-1 {{[with U = <int[2]>]: pack expansion contains parameter packs 'T' and 'U' that have different lengths (2 vs. 1)}}
  };

  pair<tuple<int, float>, tuple<char, double>> k1 = A<int, float>().f<char>(pair<int, char>(), pair<float, double>());
  pair<tuple<int, float>, tuple<char, double>> k2 = A<int, float>().f<char>(pair<int, char>(), pair<float, double>(), pair<void, long>()); // expected-error {{no match}}
  pair<tuple<int, float>, tuple<char, double>> k3 = A<int, float>().f<char, double, void>(pair<int, char>(), pair<float, double>()); // expected-error {{no match}}

  // FIXME: We should accept this by treating the pack 'p' as having a fixed length of 2 here.
  pair<tuple<int, float>, tuple<char, double>> k4 = A<int, float>().g<char>(pair<int, char>(), pair<float, double>(), pair<void, long>()); // expected-error {{no match}}

  // FIXME: We should accept this by treating the pack of pairs as having a fixed length of 2 here.
  tuple<int[2], int[4]> k5 = A<int[1], int[3]>::h<int[2]>(tuple<pair<int[1], int[2]>, pair<int[3], int[4]>, pair<int, int>>()); // expected-error {{no match}}
}

namespace substitution_vs_function_deduction {
  template <typename... T> struct A {
    template <typename... U> void f(void(*...)(T, U)); // expected-warning {{ISO C++11 requires a parenthesized pack declaration to have a name}}
    template <typename... U> void g(void...(T, U)); // expected-note {{could not match 'void (T, U)' against 'void (*)(int, int)'}}
  };
  void f(int, int) {
    A<int>().f(f);
    // FIXME: We fail to decay the parameter to a pointer type.
    A<int>().g(f); // expected-error {{no match}}
  }
}

namespace Nested_Explicit_Specialization {
template <typename>
struct Outer {

  template <int>
  struct Inner;

  template <>
  struct Inner<0> {
    template <typename... Args>
    void Test(Args...) {}
  };
};

void Run() {
  Outer<void>::Inner<0>().Test(1,1);
}
}
