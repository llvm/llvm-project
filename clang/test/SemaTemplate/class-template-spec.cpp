// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
template<typename T, typename U = int> struct A; // expected-note {{template is declared here}} \
                                                 // expected-note{{explicitly specialized}}

template<> struct A<double, double>; // expected-note{{forward declaration}}

template<> struct A<float, float> {  // expected-note{{previous definition}}
  int x;
};

template<> struct A<float> { // expected-note{{previous definition}}
  int y;
};

int test_specs(A<float, float> *a1, A<float, int> *a2) {
  return a1->x + a2->y;
}

int test_incomplete_specs(A<double, double> *a1, 
                          A<double> *a2)
{
  (void)a1->x; // expected-error{{member access into incomplete type}}
  (void)a2->x; // expected-error{{implicit instantiation of undefined template 'A<double, int>'}}
}

typedef float FLOAT;

template<> struct A<float, FLOAT>;

template<> struct A<FLOAT, float> { }; // expected-error{{redefinition}}

template<> struct A<float, int> { }; // expected-error{{redefinition}}

template<typename T, typename U = int> struct X;

template <> struct X<int, int> { int foo(); }; // #1
template <> struct X<float> { int bar(); };  // #2

typedef int int_type;
void testme(X<int_type> *x1, X<float, int> *x2) { 
  (void)x1->foo(); // okay: refers to #1
  (void)x2->bar(); // okay: refers to #2
}

// Make sure specializations are proper classes.
template<>
struct A<char> {
  A();
};

A<char>::A() { }

// Make sure we can see specializations defined before the primary template.
namespace N{ 
  template<typename T> struct A0;
}

namespace N {
  template<>
  struct A0<void> {
    typedef void* pointer;
  };
}

namespace N {
  template<typename T>
  struct A0 {
    void foo(A0<void>::pointer p = 0);
  };
}

// Diagnose specialization errors
struct A<double> { }; // expected-error{{template specialization requires 'template<>'}}

template<> struct ::A<double>;

namespace N {
  template<typename T> struct B; // expected-note {{explicitly specialized}}
#if __cplusplus <= 199711L
  // expected-note@-2 {{explicitly specialized}}
#endif

  template<> struct ::N::B<char>; // okay
  template<> struct ::N::B<short>; // okay
  template<> struct ::N::B<int>; // okay

  int f(int);
}

template<> struct N::B<int> { }; // okay

template<> struct N::B<float> { };
#if __cplusplus <= 199711L
// expected-warning@-2 {{first declaration of class template specialization of 'B' outside namespace 'N' is a C++11 extension}}
#endif


namespace M {
  template<> struct ::N::B<short> { }; // expected-error{{class template specialization of 'B' not in a namespace enclosing 'N'}}

  template<> struct ::A<long double>; // expected-error{{must occur at global scope}}
}

template<> struct N::B<char> { 
  int testf(int x) { return f(x); }
};

// PR5264
template <typename T> class Foo;
Foo<int>* v;
Foo<int>& F() { return *v; }
template <typename T> class Foo {};
Foo<int> x;


// Template template parameters
template<template<class T> class Wibble>
class Wibble<int> { }; // expected-error{{cannot specialize a template template parameter}}

namespace rdar9676205 {
  template<typename T>
  struct X {
    template<typename U>
    struct X<U*> { // expected-error{{explicit specialization of 'X' in class scope}}
    };
  };

}

namespace PR18009 {
  template <typename T> struct A {
    template <int N, int M> struct S;
    template <int N> struct S<N, sizeof(T)> {};
  };
  A<int>::S<8, sizeof(int)> a; // ok

  template <typename T> struct B {
    template <int N, int M> struct S; // expected-note {{declared here}}
    template <int N> struct S<N, sizeof(T) +
        N // expected-error {{non-type template argument depends on a template parameter of the partial specialization}}
        > {};
  };
  B<int>::S<8, sizeof(int) + 8> s; // expected-error {{undefined}}

  template<int A> struct outer {
    template<int B, int C> struct inner {};
    template<int C> struct inner<A * 2, C> {};
  };
}

namespace PR16519 {
  template<typename T, T...N> struct integer_sequence { typedef T value_type; };
#if __cplusplus <= 199711L
  // expected-warning@-2 {{variadic templates are a C++11 extension}}
#endif

  template<typename T> struct __make_integer_sequence;
  template<typename T, T N> using make_integer_sequence = typename __make_integer_sequence<T>::template make<N, N % 2>::type;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{alias declarations are a C++11 extension}}
#endif

  template<typename T, typename T::value_type ...Extra> struct __make_integer_sequence_impl;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{variadic templates are a C++11 extension}}
#endif

  template<typename T, T ...N, T ...Extra> struct __make_integer_sequence_impl<integer_sequence<T, N...>, Extra...> {
#if __cplusplus <= 199711L
  // expected-warning@-2 2 {{variadic templates are a C++11 extension}}
#endif
    typedef integer_sequence<T, N..., sizeof...(N) + N..., Extra...> type;
  };

  template<typename T> struct __make_integer_sequence {
    template<T N, T Parity, typename = void> struct make;
    template<typename Dummy> struct make<0, 0, Dummy> { typedef integer_sequence<T> type; };
    template<typename Dummy> struct make<1, 1, Dummy> { typedef integer_sequence<T, 0> type; };
    template<T N, typename Dummy> struct make<N, 0, Dummy> : __make_integer_sequence_impl<make_integer_sequence<T, N/2> > {};
    template<T N, typename Dummy> struct make<N, 1, Dummy> : __make_integer_sequence_impl<make_integer_sequence<T, N/2>, N - 1> {};
  };

  using X = make_integer_sequence<int, 5>;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{alias declarations are a C++11 extension}}
#endif

  using X = integer_sequence<int, 0, 1, 2, 3, 4>;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{alias declarations are a C++11 extension}}
#endif
}

namespace DefaultArgVsPartialSpec {
  // Check that the diagnostic points at the partial specialization, not just at
  // the default argument.
  template<typename T, int N =
      sizeof(T) // expected-note {{template parameter is used in default argument declared here}}
  > struct X {};
  template<typename T> struct X<T> {}; // expected-error {{non-type template argument depends on a template parameter of the partial specialization}}

  template<typename T,
      T N = 0 // expected-note {{template parameter is declared here}}
  > struct S;
  template<typename T> struct S<T> {}; // expected-error {{non-type template argument specializes a template parameter with dependent type 'T'}}
}
