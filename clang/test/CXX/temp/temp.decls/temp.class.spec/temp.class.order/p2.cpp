// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
template<int I, int J, class T> struct X { 
  static const int value = 0;
};

template<int I, int J> struct X<I, J, int> { 
  static const int value = 1;
};

template<int I> struct X<I, I, int> { 
  static const int value = 2;
};

int array0[X<0, 0, float>::value == 0? 1 : -1];
int array1[X<0, 1, int>::value == 1? 1 : -1];
int array2[X<0, 0, int>::value == 2? 1 : -1];

namespace DependentSubstPartialOrdering {
  template<typename T, typename U = void, typename V = void>
  struct X { 
    static const unsigned value = 1;
  };

  template<typename T, typename U>
  struct X<T, U, typename T::is_b> {
    static const unsigned value = 2;
  };

  template<typename T>
  struct X<T, typename T::is_a, typename T::is_b> {
    static const unsigned value = 3;
  };

  struct X1 { };

  struct X2 { 
    typedef void is_b;
  };

  struct X3 {
    typedef void is_a;
    typedef void is_b;
  };

  int check_X1[X<X1, void, void>::value == 1? 1 : -1];
  int check_X2[X<X2, void, void>::value == 2? 1 : -1];
  int check_X3[X<X3, void, void>::value == 3? 1 : -1];
}
