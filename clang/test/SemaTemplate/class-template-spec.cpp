// RUN: %clang_cc1 -fsyntax-only -verify %s
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
  template<typename T> struct B; // expected-note 2{{explicitly specialized}}

  template<> struct ::N::B<char>; // okay
  template<> struct ::N::B<short>; // okay
  template<> struct ::N::B<int>; // okay

  int f(int);
}

template<> struct N::B<int> { }; // okay

template<> struct N::B<float> { }; // expected-warning{{C++11 extension}}

namespace M {
  template<> struct ::N::B<short> { }; // expected-error{{class template specialization of 'B' not in a namespace enclosing 'N'}}

  template<> struct ::A<long double>; // expected-error{{originally}}
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
