// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -std=c++2c -verify %s

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

#if __cplusplus > 202302L
template<typename... Ts>
struct A {
  // FIXME: The nested-name-specifier in the following friend declarations are declarative,
  // but we don't treat them as such (yet).
  friend void Ts...[0]::f();
  template<typename U>
  friend void Ts...[0]::g();

  friend struct Ts...[0]::B;
  template<typename U>
  friend struct Ts...[0]::C; // expected-warning{{is not supported; ignoring this friend declaration}}
};
#endif
