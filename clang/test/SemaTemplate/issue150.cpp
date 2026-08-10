// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// Core issue 150: Template template parameters and default arguments

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

namespace PR9353 {
  template<class _T, class Traits> class IM;

  template <class T, class Trt, 
            template<class _T, class Traits = int> class IntervalMap>
  void foo(IntervalMap<T,Trt>* m) { typedef IntervalMap<int> type; }

  void f(IM<int, int>* m) { foo(m); }
}

namespace PR9400 {
  template<template <typename T, typename = T > class U> struct A
  {
    template<int> U<int> foo();
  };

  template <typename T, typename = T>
  struct s {
  };

  void f() {
    A<s> x;
    x.foo<2>();
  }
}

namespace MultiReplace {
  template<typename Z, 
           template<typename T, typename U = T *, typename V = U const> class TT>
  struct X {
    typedef TT<Z> type;
  };

  template<typename T, typename = int, typename = float> 
  struct Y { };

  int check0[is_same<X<int, Y>::type, Y<int, int*, int* const> >::value? 1 : -1];
}

namespace MultiReplacePartial {
  template<typename First, typename Z, 
           template<typename T, typename U = T *, typename V = U const> class TT>
  struct X {
    typedef TT<Z> type;
  };

  template<typename Z, 
           template<typename T, typename U = T *, typename V = U const> class TT>
  struct X<int, Z, TT> {
    typedef TT<Z> type;
  };

  template<typename T, typename = int, typename = float> 
  struct Y { };

  int check0[is_same<X<int, int, Y>::type, Y<int, int*, int* const> >::value? 1 : -1];
}

namespace PR9016 {
  template<typename > struct allocator ;
  template<typename > struct less ;

  template<class T, template<class> class Compare, class Default,
           template<class> class Alloc>
  struct interval_set { };

  template <class X, template<class> class = less> struct interval_type_default {
    typedef X type;
  };

  template <class T,
            template<class _T, template<class> class Compare = PR9016::less,
                     class = typename interval_type_default<_T,Compare>::type,
                     template<class> class = allocator> class IntervalSet>
  struct ZZZ
  {
    IntervalSet<T> IntervalSetT;
  };
  
  template <class T, 
            template<class _T, template<class> class Compare = PR9016::less,
                     class = typename interval_type_default<_T,Compare>::type,
                     template<class> class = allocator> class IntervalSet>
  void int40()
  {
    IntervalSet<T> IntervalSetT;
  }

  void test() {
    ZZZ<int, interval_set> zzz;
    int40<int, interval_set>();
  }
}
