// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template<typename T, typename U> struct pair { };
template<typename ...Types> struct tuple { };

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

namespace ExpandIntoFixed {
  template<typename T, 
           typename U, 
           typename V = pair<T, U>, 
           typename W = V*> 
  class X0 { };

  template<typename ...Ts>
  class X1 {
  public:
    typedef X0<Ts...> type;
  };

  static_assert(is_same<X1<int, int>::type, 
                        X0<int, int, pair<int, int>, pair<int, int>*>>::value,
                "fails with two default arguments");

  static_assert(is_same<X1<int, int, float>::type, 
                        X0<int, int, float, float*>>::value,
                "fails with one default argument");

  static_assert(is_same<X1<int, int, float, double>::type, 
                        X0<int, int, float, double>>::value,
                "fails with no default arguments");
}

namespace ExpandIntoFixedShifted {
  template<typename T, 
           typename U, 
           typename V = pair<T, U>, 
           typename W = V*> 
  class X0 { };

  template<typename ...Ts>
  class X1 {
  public:
    typedef X0<char, Ts...> type;
  };

  static_assert(is_same<X1<int>::type, 
                        X0<char, int, pair<char, int>, pair<char, int>*>>::value,
                "fails with two default arguments");

  static_assert(is_same<X1<int, float>::type, 
                        X0<char, int, float, float*>>::value,
                "fails with one default argument");

  static_assert(is_same<X1<int, float, double>::type, 
                        X0<char, int, float, double>>::value,
                "fails with no default arguments");
}

namespace Deduction {
  template <typename X, typename Y = double> struct Foo {};
  template <typename ...Args> tuple<Args...> &foo(Foo<Args...>);

  void call_foo(Foo<int, float> foo_if, Foo<int> foo_i) {
    tuple<int, float> &t1 = foo(foo_if);
    tuple<int, double> &t2 = foo(foo_i);
  }
}

namespace PR9021a {
  template<typename, typename> 
  struct A { };

  template<typename ...T>
  struct B { 
    A<T...> a1;
  };

  void test() {
    B<int, int> c;
  }
}

namespace PR9021b {
  template<class, class>
  struct t2
  {
    
  };
  
  template<template<class...> class M>
  struct m
  {
    template<class... B>
    using inner = M<B...>;
  };

  m<t2> sta2;
}

namespace PartialSpecialization {
  template<typename T, typename U, typename V = U>
  struct X0; // expected-note{{template is declared here}}

  template<typename ...Ts>
  struct X0<Ts...> {
  };

  X0<int> x0i; // expected-error{{too few template arguments for class template 'X0'}}
  X0<int, float> x0if;
  X0<int, float, double> x0ifd;
}

namespace FixedAliasTemplate {
  template<typename,typename,typename> struct S {};
  template<typename T, typename U> using U = S<T, int, U>;
  template<typename...Ts> U<Ts...> &f(U<Ts...>, Ts...);
  S<int, int, double> &s1 = f({}, 0, 0.0);
}
