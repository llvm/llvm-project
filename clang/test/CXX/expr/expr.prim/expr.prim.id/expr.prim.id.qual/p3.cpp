// RUN: %clang_cc1 -verify %s

template<typename T>
struct A {
  void f();
};

template<typename T>
using B = A<T>;

template<typename T>
void B<T>::f() { } // expected-warning {{a declarative nested name specifier cannot name an alias template}}

template<>
void B<int>::f() { } // ok, template argument list of simple-template-id doesn't involve template parameters

namespace N {

  template<typename T>
  struct D {
    void f();
  };

  template<typename T>
  using E = D<T>;
}

template<typename T>
void N::E<T>::f() { } // expected-warning {{a declarative nested name specifier cannot name an alias template}}
