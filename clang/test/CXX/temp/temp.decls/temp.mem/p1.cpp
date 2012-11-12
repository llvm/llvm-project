// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template <class T> struct A {
  static T cond;
  
  template <class U> struct B {
    static T twice(U value) {
      return (cond ? value + value : value);
    }
  };
};

int foo() {
  A<bool>::cond = true;
  return A<bool>::B<int>::twice(4);
}

namespace PR6376 {
  template<typename T>
  struct X {
    template<typename Y>
    struct Y1 { }; //
  };

  template<>
  struct X<float> {
    template<typename Y>
    struct Y1 { };
  };

  template<typename T, typename U>
  struct Z : public X<T>::template Y1<U> { };

  Z<float, int> z0;
}
