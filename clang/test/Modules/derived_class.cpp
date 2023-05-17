// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/foo.cppm -emit-module-interface -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only -verify
//
//--- bar.h
struct bar_base {
  enum A {
    a,
    b,
    c,
    d
  };
  constexpr static bool value = false;
  static bool get() { return false; }
  bool member_value = false;
  bool get_func() { return false; }
};

template <typename T>
struct bar : public bar_base {
};

//--- foo.cppm
module;
#include "bar.h"
export module foo;
export template <typename T>
int foo() {
  bool a = bar<T>::value;
  bar<T>::get();
  bar<T> b;
  b.member_value = a;
  bool c = b.get_func();
  return bar<T>::a;
}

//--- Use.cpp
// expected-no-diagnostics
import foo;
void test() {
  foo<int>();
}
