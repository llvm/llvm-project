// RUN: %clang_cc1 -verify %s
// expected-no-diagnostics

typedef char one_byte;
struct two_bytes { char data[2]; };

template<typename T> one_byte __is_class_check(int T::*);
template<typename T> two_bytes __is_class_check(...);

template<typename T> struct is_class {
  static const bool value = sizeof(__is_class_check<T>(0)) == 1;
};

struct X { };

int array0[is_class<X>::value? 1 : -1];
int array1[is_class<int>::value? -1 : 1];
int array2[is_class<char[3]>::value? -1 : 1];

namespace instantiation_order1 {
  template<typename T>
  struct it_is_a_trap { 
    typedef typename T::trap type;
  };

  template<bool, typename T = void>
  struct enable_if {
    typedef T type;
  };

  template<typename T>
  struct enable_if<false, T> { };

  template<typename T>
  typename enable_if<sizeof(T) == 17>::type 
  f(const T&, typename it_is_a_trap<T>::type* = 0);

  void f(...);

  void test_f() {
    f('a');
  }
}
