// RUN: %clang_cc1 -fsyntax-only -verify %s
template<template<typename T> class MetaFun, typename Value>
struct apply {
  typedef typename MetaFun<Value>::type type;
};

template<class T>
struct add_pointer {
  typedef T* type;
};

template<class T>
struct add_reference {
  typedef T& type;
};

int i;
apply<add_pointer, int>::type ip = &i;
apply<add_reference, int>::type ir = i;
apply<add_reference, float>::type fr = i; // expected-error{{non-const lvalue reference to type 'float' cannot bind to a value of unrelated type 'int'}}

// Template template parameters
template<int> struct B; // expected-note {{template parameter is declared here}}

template<typename T,
         template<T Value> class X> // expected-error{{cannot have type 'float'}}
                                    // expected-error@-1 {{cannot be narrowed from type 'long long' to 'int'}}
                                    // expected-note@-2 2{{template parameter is declared here}}
struct X0 { };

X0<int, B> x0b1;
X0<float, B> x0b2; // expected-note{{while substituting}}
X0<long long, B> x0b3; // expected-note {{template template argument is incompatible}}

template<template<int V> class TT>
struct X1 { };

template<typename T, template<T V> class TT>
struct X2 {
  X1<TT> x1;
};

template<int V> struct X3i { };
template<long V> struct X3l { };

X2<int, X3i> x2okay;
X2<long, X3l> x2okay2;

template <typename T, template <T, T> class TT, class R = TT<1, 2> >
struct Comp {
  typedef R r1;
  template <T x, T y> struct gt {
    static const bool result = x > y;
  };
  typedef gt<2, 1> r2;
};

template <int x, int y> struct lt {
  static const bool result = x < y;
};

Comp<int, lt> c0;

namespace PR8629 {
  template<template<int> class TT> struct X0
  {
    static void apply();
  };
  template<int> struct Type { };

  template<class T> struct X1
  {
    template<class U> struct Inner;

    template<class U> void g()
    {
      typedef Inner<U> Init;
      X0<Init::template VeryInner>::apply();
    }
    template<int N> void f ()
    {
      g<Type<N> >();
    }
  };
  template<class T> template<class U> struct X1<T>::Inner
  {
    template<int> struct VeryInner {
    };
  };
  struct X1Container
  {
    X1Container()
    {
      simplex_.f<0>();
    }
    X1<double> simplex_;
  };
}
