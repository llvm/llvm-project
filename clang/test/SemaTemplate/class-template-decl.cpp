// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> class A;

extern "C++" {
  template<typename T> class B;
}

namespace N {
  template<typename T> class C;
}

extern "C" {
  template<typename T> class D; // expected-error{{templates must have C++ linkage}}
}

template<class U> class A; // expected-note{{previous template declaration is here}}

template<int N> class A; // expected-error{{template parameter has a different kind in template redeclaration}}

template<int N> class NonTypeTemplateParm;

typedef int INT;

template<INT M> class NonTypeTemplateParm; // expected-note{{previous non-type template parameter with type 'INT' (aka 'int') is here}}

template<long> class NonTypeTemplateParm; // expected-error{{template non-type parameter has a different type 'long' in template redeclaration}}

template<template<typename T> class X> class TemplateTemplateParm;

template<template<class> class Y> class TemplateTemplateParm; // expected-note{{previous template declaration is here}} \
      // expected-note{{previous template template parameter is here}}

template<typename> class TemplateTemplateParm; // expected-error{{template parameter has a different kind in template redeclaration}}

template<template<typename T, int> class X> class TemplateTemplateParm; // expected-error{{too many template parameters in template template parameter redeclaration}}

template<typename T>
struct test {}; // expected-note{{previous definition}}

template<typename T>
struct test : T {}; // expected-error{{redefinition}}

class X {
public:
  template<typename T> class C;
};

void f() {
  template<typename T> class X; // expected-error{{expression}}
}

template<typename T> class X1 var; // expected-error{{declared as a template}}

namespace M {
}

template<typename T> class M::C3 { }; // expected-error{{out-of-line definition of 'C3' does not match any declaration in namespace 'M'}}

namespace PR8001 {
  template<typename T1>
  struct Foo {
    template<typename T2> class Bar;
    typedef Bar<T1> Baz;

   template<typename T2>
   struct Bar {
     Bar() {}
   };
  };

  void pr8001() {
    Foo<int>::Baz x;
    Foo<int>::Bar<int> y(x);
  }
}

namespace rdar9676205 {
  template <unsigned, class _Tp> class tuple_element;

  template <class _T1, class _T2> class pair;

  template <class _T1, class _T2>
  class tuple_element<0, pair<_T1, _T2> >
  {
    template <class _Tp>
    struct X
    {
      template <class _Up, bool = X<_Up>::value>
      struct Y
        : public X<_Up>,
          public Y<_Up>
      { };
    };
  };
}

namespace redecl {
  int A; // expected-note {{here}}
  template<typename T> struct A; // expected-error {{different kind of symbol}}

  int B; // expected-note {{here}}
  template<typename T> struct B { // expected-error {{different kind of symbol}}
  };

  template<typename T> struct F;
  template<typename T> struct K;

  int G, H; // expected-note {{here}}

  struct S {
    int C; // expected-note {{here}}
    template<typename T> struct C; // expected-error {{different kind of symbol}}

    int D; // expected-note {{here}}
    template<typename T> struct D { // expected-error {{different kind of symbol}}
    };

    int E;
    template<typename T> friend struct E { // expected-error {{cannot define a type in a friend}}
    };

    int F;
    template<typename T> friend struct F; // ok, redecl::F

    template<typename T> struct G; // ok

    template<typename T> friend struct H; // expected-error {{different kind of symbol}}

    int I, J, K;

    struct U {
      template<typename T> struct I; // ok
      template<typename T> struct J { // ok
      };
      template<typename T> friend struct K; // ok, redecl::K
    };
  };
}
