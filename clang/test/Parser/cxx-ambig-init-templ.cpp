// RUN: %clang_cc1 -Wno-uninitialized -std=c++11 -verify %s

template<int> struct c { c(int) = delete; typedef void val; operator int() const; };

int val;
int foobar;
struct S {
  int k1 = a < b < c, d > ::val, e1;
  int k2 = a < b, c < d > ::val, e2;
  int k3 = b < a < c, d > ::val, e3;
  int k4 = b < c, x, y = d > ::val, e4;
  int k5 = T1 < b, &S::operator=(int); // expected-error {{extra qualification}}
  int k6 = T2 < b, &S::operator= >::val;
  int k7 = T1 < b, &S::operator>(int); // expected-error {{extra qualification}}
  int k8 = T2 < b, &S::operator> >::val;
  int k9 = T3 < a < b, c >> (d), e5 = 1 > (e4);
  int k10 = 0 < T3 < a < b, c >> (d
      ) // expected-error {{expected ';' at end of declaration}}
      , a > (e4);
  int k11 = 0 < 1, c<3>::*ptr;
  int k12 = e < 0, int a<b<c>::* >(), e11;

  void f1(
    int k1 = a < b < c, d > ::val,
    int k2 = b < a < c, d > ::val,
    int k3 = b < c, int x = 0 > ::val,
    int k4 = a < b, T3 < int > >(), // expected-error {{must be an expression}}
    int k5 = a < b, c < d > ::val,
    int k6 = a < b, c < d > (n) // expected-error {{undeclared identifier 'n'}}
  );

  void f2a(
    // T3<int> here is a parameter type, so must be declared before it is used.
    int k1 = c < b, T3 < int > x = 0 // expected-error {{unexpected end of default argument expression}}
  );

  template<typename, int=0> struct T3 { T3(int); operator int(); };

  void f2b(
    int k1 = c < b, T3 < int > x  = 0 // ok
  );

  // This is a one-parameter function. Ensure we don't typo-correct it to
  //     int = a < b, c < foobar > ()
  // ... which would be a function with two parameters.
  int f3(int = a < b, c < goobar > ());
  static constexpr int (S::*f3_test)(int) = &S::f3;

  void f4(
    int k1 = a<1,2>::val,
    int missing_default // expected-error {{missing default argument on parameter}}
  );

  void f5(
    int k1 = b < c,
    int missing_default // expected-error {{missing default argument on parameter}}
  );

  void f6(
    int k = b < c,
    unsigned int (missing_default) // expected-error {{missing default argument on parameter}}
  );

  template<int, int=0> struct a { static const int val = 0; operator int(); }; // expected-note {{here}}
  static const int b = 0, c = 1, d = 2, goobar = 3;
  template<int, typename> struct e { operator int(); };

  int mp1 = 0 < 1,
      a<b<c,b<c>::*mp2,
      mp3 = 0 > a<b<c>::val,
      a<b<c,b<c>::*mp4 = 0,
      a<b<c,b<c>::*mp5 {0},
      a<b<c,b<c>::*mp6;

  int np1 = e<0, int a<b<c,b<c>::*>();

  static const int T1 = 4;
  template<int, int &(S::*)(int)> struct T2 { static const int val = 0; };
};

namespace NoAnnotationTokens {
  template<bool> struct Bool { Bool(int); };
  static const bool in_class = false;

  struct Test {
    // Check we don't keep around a Bool<false> annotation token here.
    int f(Bool<true> = X<Y, Bool<in_class> >(0));

    // But it's OK if we do here.
    int g(Bool<true> = Z<Y, Bool<in_class> = Bool<false>(0));

    static const bool in_class = true;
    template<int, typename U> using X = U;
    static const int Y = 0, Z = 0;
  };
}

namespace ImplicitInstantiation {
  template<typename T> struct HasError { typename T::error error; }; // expected-error {{has no members}}

  struct S {
    // This triggers the instantiation of the outer HasError<int> during
    // disambiguation, even though it uses the inner HasError<int>.
    void f(int a = X<Y, HasError<int>::Z >()); // expected-note {{in instantiation of}}

    template<typename, typename> struct X { operator int(); };
    typedef int Y;
    template<typename> struct HasError { typedef int Z; };
  };

  HasError<int> hei;
}

namespace CWG325 {
  template <int A, typename B> struct T { static int i; operator int(); };
  class C {
    int Foo (int i = T<1, int>::i);
  };

  class D {
    int Foo (int i = T<1, int>::i);
    template <int A, typename B> struct T {static int i;};
  };

  const int a = 0;
  typedef int b;
  T<a,b> c;
  struct E {
    int n = T<a,b>(c);
  };
}

namespace Operators {
  struct Y {};
  constexpr int operator,(const Y&, const Y&) { return 8; }
  constexpr int operator>(const Y&, const Y&) { return 8; }
  constexpr int operator<(const Y&, const Y&) { return 8; }
  constexpr int operator>>(const Y&, const Y&) { return 8; }

  struct X {
    typedef int (*Fn)(const Y&, const Y&);

    Fn a = operator,, b = operator<, c = operator>;
    void f(Fn a = operator,, Fn b = operator<, Fn c = operator>);

    int k1 = T1<0, operator<, operator>, operator<>::val, l1;
    int k2 = T1<0, operator>, operator,, operator,>::val, l2;
    int k3 = T2<0, operator,(Y{}, Y{}),  operator<(Y{}, Y{})>::val, l3;
    int k4 = T2<0, operator>(Y{}, Y{}),  operator,(Y{}, Y{})>::val, l4;
    int k5 = T3<0, operator>>>::val, l5;
    int k6 = T4<0, T3<0, operator>>>>::val, l6;

    template<int, Fn, Fn, Fn> struct T1 { enum { val }; };
    template<int, int, int> struct T2 { enum { val }; };
    template<int, Fn> struct T3 { enum { val }; };
    template<int, typename T> struct T4 : T {};
  };
}

namespace ElaboratedTypeSpecifiers {
  struct S {
    int f(int x = T<a, struct S>());
    int h(int x = T<a, union __attribute__(()) U>());
    int i(int x = T<a, enum E>());
    int j(int x = T<a, struct S::template T<0, enum E>>());
    template <int, typename> struct T { operator int(); };
    static const int a = 0;
    enum E {};
  };
}

namespace PR20459 {
  template <typename EncTraits> struct A {
     void foo(int = EncTraits::template TypeEnc<int, int>::val); // ok
  };
}
