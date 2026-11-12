// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// This is a collection of various template metafunctions involving
// variadic templates, which are meant to exercise common use cases.
template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

template<typename...> struct tuple { };
template<int ...> struct int_tuple { };
template<typename T, typename U> struct pair { };

namespace Count {
  template<typename Head, typename ...Tail>
  struct count {
    static const unsigned value = 1 + count<Tail...>::value;
  };

  template<typename T>
  struct count<T> {
    static const unsigned value = 1;
  };

  int check1[count<int>::value == 1? 1 : -1];
  int check2[count<float, double>::value == 2? 1 : -1];
  int check3[count<char, signed char, unsigned char>::value == 3? 1 : -1];
}

namespace CountWithPackExpansion {
  template<typename ...> struct count;

  template<typename Head, typename ...Tail>
  struct count<Head, Tail...> {
    static const unsigned value = 1 + count<Tail...>::value;
  };

  template<>
  struct count<> {
    static const unsigned value = 0;
  };

  int check0[count<>::value == 0? 1 : -1];
  int check1[count<int>::value == 1? 1 : -1];
  int check2[count<float, double>::value == 2? 1 : -1];
  int check3[count<char, signed char, unsigned char>::value == 3? 1 : -1];
}

namespace Replace {
  // Simple metafunction that replaces the template arguments of
  // template template parameters with 'int'.
  template<typename T>
  struct EverythingToInt;

  template<template<typename ...> class TT, typename T1, typename T2>
  struct EverythingToInt<TT<T1, T2> > {
    typedef TT<int, int> type;
  };

  int check0[is_same<EverythingToInt<tuple<double, float>>::type, 
             tuple<int, int>>::value? 1 : -1];
}

namespace Math {
  template<int ...Values>
  struct double_values {
    typedef int_tuple<Values*2 ...> type;
  };

  int check0[is_same<double_values<1, 2, -3>::type, 
                     int_tuple<2, 4, -6>>::value? 1 : -1];

  template<int ...Values>
  struct square {
    typedef int_tuple<(Values*Values)...> type;
  };

  int check1[is_same<square<1, 2, -3>::type, 
                     int_tuple<1, 4, 9>>::value? 1 : -1];

  template<typename IntTuple> struct square_tuple;

  template<int ...Values>
  struct square_tuple<int_tuple<Values...>> {
    typedef int_tuple<(Values*Values)...> type;
  };

  int check2[is_same<square_tuple<int_tuple<1, 2, -3> >::type, 
                     int_tuple<1, 4, 9>>::value? 1 : -1];

  template<int ...Values> struct sum;

  template<int First, int ...Rest> 
  struct sum<First, Rest...> {
    static const int value = First + sum<Rest...>::value;
  };

  template<>
  struct sum<> {
    static const int value = 0;
  };

  int check3[sum<1, 2, 3, 4, 5>::value == 15? 1 : -1];

  template<int ... Values>
  struct lazy_sum {
    int operator()() {
      return sum<Values...>::value;
    }
  };

  void f() {
    lazy_sum<1, 2, 3, 4, 5>()();
  }
}

namespace ListMath {
  template<typename T, T ... V> struct add;

  template<typename T, T i, T ... V>
  struct add<T, i, V...> {
    static const T value = i + add<T, V...>::value; 
  };

  template<typename T>
  struct add<T> {
    static const T value = T(); 
  };

  template<typename T, T ... V>
  struct List {
    struct sum {
      static const T value = add<T, V...>::value;
    };
  };

  template<int ... V>
  struct ListI : public List<int, V...> {
  };

  int check0[ListI<1, 2, 3>::sum::value == 6? 1 : -1];
}

namespace Indices {
  template<unsigned I, unsigned N, typename IntTuple>
  struct build_indices_impl;

  template<unsigned I, unsigned N, int ...Indices>
  struct build_indices_impl<I, N, int_tuple<Indices...> >
    : build_indices_impl<I+1, N, int_tuple<Indices..., I> > {
  };

  template<unsigned N, int ...Indices> 
  struct build_indices_impl<N, N, int_tuple<Indices...> > {
    typedef int_tuple<Indices...> type;
  };

  template<unsigned N>
  struct build_indices : build_indices_impl<0, N, int_tuple<> > { };

  int check0[is_same<build_indices<5>::type,
                     int_tuple<0, 1, 2, 3, 4>>::value? 1 : -1];
}

namespace TemplateTemplateApply {
  template<typename T, template<class> class ...Meta>
  struct apply_each {
    typedef tuple<typename Meta<T>::type...> type;
  };

  template<typename T> 
  struct add_reference {
    typedef T& type;
  };

  template<typename T> 
  struct add_pointer {
    typedef T* type;
  };

  template<typename T> 
  struct add_const {
    typedef const T type;
  };

  int check0[is_same<apply_each<int, 
                                add_reference, add_pointer, add_const>::type,
                     tuple<int&, int*, int const>>::value? 1 : -1];

  template<typename T, template<class> class ...Meta>
  struct apply_each_indirect {
    typedef typename apply_each<T, Meta...>::type type;
  };

  int check1[is_same<apply_each_indirect<int, add_reference, add_pointer, 
                                         add_const>::type,
                     tuple<int&, int*, int const>>::value? 1 : -1];

  template<typename T, typename ...Meta>
  struct apply_each_nested {
    typedef typename apply_each<T, Meta::template apply...>::type type;
  };

  struct add_reference_meta {
    template<typename T>
    struct apply {
      typedef T& type;
    };
  };

  struct add_pointer_meta {
    template<typename T>
    struct apply {
      typedef T* type;
    };
  };

  struct add_const_meta {
    template<typename T>
    struct apply {
      typedef const T type;
    };
  };

  int check2[is_same<apply_each_nested<int, add_reference_meta, 
                                       add_pointer_meta, add_const_meta>::type,
                     tuple<int&, int*, int const>>::value? 1 : -1];

}

namespace FunctionTypes {
  template<typename FunctionType>
  struct Arity;

  template<typename R, typename ...Types>
  struct Arity<R(Types...)> {
    static const unsigned value = sizeof...(Types);
  };

  template<typename R, typename ...Types>
  struct Arity<R(Types......)> { // expected-warning {{varargs}} expected-note {{pack}} expected-note {{insert ','}}
    static const unsigned value = sizeof...(Types);
  };

  template<typename R, typename T1, typename T2, typename T3, typename T4>
  struct Arity<R(T1, T2, T3, T4)>; // expected-note{{template is declared here}}

  int check0[Arity<int()>::value == 0? 1 : -1];
  int check1[Arity<int(float, double)>::value == 2? 1 : -1];
  int check2[Arity<int(float...)>::value == 1? 1 : -1];
  int check3[Arity<int(float, double, long double...)>::value == 3? 1 : -1];
  Arity<int(float, double, long double, char)> check4; // expected-error{{implicit instantiation of undefined template 'FunctionTypes::Arity<int (float, double, long double, char)>'}}
}

namespace SuperReplace {
  template<typename T>
  struct replace_with_int {
    typedef int type;
  };
  
  template<template<typename ...> class TT, typename ...Types>
  struct replace_with_int<TT<Types...>> {
    typedef TT<typename replace_with_int<Types>::type...> type;
  };
  
  int check0[is_same<replace_with_int<pair<tuple<float, double, short>,
                                           pair<char, unsigned char>>>::type,
                     pair<tuple<int, int, int>, pair<int, int>>>::value? 1 : -1];
}
