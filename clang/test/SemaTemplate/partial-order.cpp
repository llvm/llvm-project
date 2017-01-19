// RUN: %clang_cc1 -std=c++1z %s -verify

// expected-no-diagnostics

namespace hana_enable_if_idiom {
  template<bool> struct A {};
  template<typename, typename = A<true>> struct B;
  template<typename T, bool N> struct B<T, A<N>> {};
  template<typename T> struct B<T, A<T::value>> {};
  struct C {
    static const bool value = true;
  };
  B<C> b;
}
