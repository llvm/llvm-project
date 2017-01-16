// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<typename ...Types> struct tuple;
template<unsigned> struct unsigned_c;

template<typename T, typename U> 
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

namespace PackExpansionNotAtEnd {
  template<typename T, typename U>
  struct tuple_same_with_int {
    static const bool value = false;
  };

  template<typename ...Types>
  struct tuple_same_with_int<tuple<Types...>, tuple<Types..., int>> {
    static const bool value = true;
  };

  int tuple_same_with_int_1[tuple_same_with_int<tuple<int, float, double>,
                                                tuple<int, float, double, int>
                                                >::value? 1 : -1];

  template<typename ... Types> struct UselessPartialSpec;

  template<typename ... Types, // expected-note{{non-deducible template parameter 'Types'}}
           typename Tail> // expected-note{{non-deducible template parameter 'Tail'}}
  struct UselessPartialSpec<Types..., Tail>; // expected-error{{class template partial specialization contains template parameters that cannot be deduced; this partial specialization will never be used}}
}

namespace DeduceNonTypeTemplateArgsInArray {
  template<typename ...ArrayTypes>
  struct split_arrays;

  template<typename ...ElementTypes, unsigned ...Bounds>
  struct split_arrays<ElementTypes[Bounds]...> {
    typedef tuple<ElementTypes...> element_types;

    // FIXME: Would like to have unsigned_tuple<Bounds...> here.
    typedef tuple<unsigned_c<Bounds>...> bounds_types;
  };

  int check1[is_same<split_arrays<int[1], float[2], double[3]>::element_types,
                     tuple<int, float, double>>::value? 1 : -1];
  int check2[is_same<split_arrays<int[1], float[2], double[3]>::bounds_types,
                     tuple<unsigned_c<1>, unsigned_c<2>, unsigned_c<3>>
                     >::value? 1 : -1];
}

namespace DeduceWithDefaultArgs {
  template<template<typename...> class Container> void f(Container<int>); // expected-note {{deduced type 'X<[...], (default) int>' of 1st parameter does not match adjusted type 'X<[...], double>' of argument [with Container = X]}}
  template<typename, typename = int> struct X {};
  void g() {
    // OK, use default argument for the second template parameter.
    f(X<int>{});
    f(X<int, int>{});

    // Not OK.
    f(X<int, double>{}); // expected-error {{no matching function for call to 'f'}}
  }
}
