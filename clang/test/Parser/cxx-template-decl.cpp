// RUN: %clang_cc1 -fsyntax-only -verify %s

// Errors
export class foo { };   // expected-error {{expected template}}
template  x;            // expected-error {{C++ requires a type specifier for all declarations}} \
                        // expected-error {{does not refer}}
export template x;      // expected-error {{expected '<' after 'template'}}
export template<class T> class x0; // expected-warning {{exported templates are unsupported}}
template < ;            // expected-error {{expected template parameter}} \
// expected-error{{expected ',' or '>' in template-parameter-list}} \
// expected-warning {{declaration does not declare anything}}
template <int +> struct x1; // expected-error {{expected ',' or '>' in template-parameter-list}}

// verifies that we only walk to the ',' & still produce errors on the rest of the template parameters
template <int +, T> struct x2; // expected-error {{expected ',' or '>' in template-parameter-list}} \
                                expected-error {{expected unqualified-id}}
template<template<int+>> struct x3; // expected-error {{expected ',' or '>' in template-parameter-list}} \
                                         expected-error {{template template parameter requires 'class' after the parameter list}}
template <template X> struct Err1; // expected-error {{expected '<' after 'template'}} \
// expected-error{{extraneous}}
template <template <typename> > struct Err2;       // expected-error {{template template parameter requires 'class' after the parameter list}}
template <template <typename> Foo> struct Err3;    // expected-error {{template template parameter requires 'class' after the parameter list}}

// Template function declarations
template <typename T> void foo();
template <typename T, typename U> void foo();

// Template function definitions.
template <typename T> void foo() { }

// Template class (forward) declarations
template <typename T> struct A;
template <typename T, typename U> struct b;
template <typename> struct C;
template <typename, typename> struct D;

// Forward declarations with default parameters?
template <typename T = int> class X1;
template <typename = int> class X2;

// Forward declarations w/template template parameters
template <template <typename> class T> class TTP1;
template <template <typename> class> class TTP2;
template <template <typename> class T = foo> class TTP3; // expected-error{{must be a class template}}
template <template <typename> class = foo> class TTP3; // expected-error{{must be a class template}}
template <template <typename X, typename Y> class T> class TTP5;

// Forward declarations with non-type params
template <int> class NTP0;
template <int N> class NTP1;
template <int N = 5> class NTP2;
template <int = 10> class NTP3;
template <unsigned int N = 12u> class NTP4; 
template <unsigned int = 12u> class NTP5;
template <unsigned = 15u> class NTP6;
template <typename T, T Obj> class NTP7;

// Template class declarations
template <typename T> struct A { };
template <typename T, typename U> struct B { };

// Template parameter shadowing
template<typename T, // expected-note{{template parameter is declared here}}
         typename T> // expected-error{{declaration of 'T' shadows template parameter}}
  void shadow1();

template<typename T> // expected-note{{template parameter is declared here}}
void shadow2(int T); // expected-error{{declaration of 'T' shadows template parameter}}

template<typename T> // expected-note{{template parameter is declared here}}
class T { // expected-error{{declaration of 'T' shadows template parameter}}
};

template<int Size> // expected-note{{template parameter is declared here}}
void shadow3(int Size); // expected-error{{declaration of 'Size' shadows template parameter}}

// <rdar://problem/6952203>
template<typename T> // expected-note{{here}}
struct shadow4 {
  int T; // expected-error{{shadows}}
};

template<typename T> // expected-note{{here}}
struct shadow5 {
  int T(int, float); // expected-error{{shadows}}
};

// Non-type template parameters in scope
template<int Size> 
void f(int& i) {
  i = Size;
  Size = i; // expected-error{{expression is not assignable}}
}

template<typename T>
const T& min(const T&, const T&);

void f2() {
  int x;
  A< typeof(x>1) > a;
}


// PR3844
template <> struct S<int> { }; // expected-error{{explicit specialization of non-template struct 'S'}}

namespace PR6184 {
  namespace N {
    template <typename T>
    void bar(typename T::x);
  }
  
  template <typename T>
  void N::bar(typename T::x) { }
}
