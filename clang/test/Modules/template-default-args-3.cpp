// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -verify -std=c++20 -emit-module -fmodules -fmodule-name=B -fmodules-cache-path=%t -xc++ module.modulemap

//--- module.modulemap
module A {
  header "A.h"
}
module B {
  module X {
    header "B.h"
  }
  header "C.h"
}

//--- A.h
template <class _Tp, class = void> struct A {};

//--- B.h
template <class> struct C;
template <class T> void f() {
  C<T> a;
}
void g() {
  f<int>();
}

//--- C.h
// expected-no-diagnostics
#pragma clang module import A
template <class T> struct C {
  using X = A<T>;
};
