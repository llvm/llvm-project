// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-reduced-module-interface -o %t/b.pcm -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -emit-reduced-module-interface -o %t/c.pcm -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/d.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

//--- a.h
template <typename T>
struct A {
  static const T value0;
  static const T value1;

  constexpr T get0() {
    return value0;
  }

  constexpr T get1() {
    return value1;
  }
};

template <typename T>
const T A<T>::value0 = T(43);
template <typename T>
const T A<T>::value1 = T(44);

//--- a.cppm
module;
#include "a.h"
export module a;
export using ::A;

//--- b.cppm
export module b;
export import a;

export constexpr int bar() {
    return A<int>().get0();
}

//--- c.cppm
export module c;
export import b;

export constexpr int foo() {
    return A<int>().get1() + A<int>().get0();
}

//--- d.cpp
// expected-no-diagnostics

import c;

static_assert(bar() + foo() == 130);

