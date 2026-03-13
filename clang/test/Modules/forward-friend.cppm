// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -fsyntax-only -verify

//--- foo.h

template <typename... U>
static void foo(U...) noexcept;

class A {
  template <typename... U>
  friend void foo(U...) noexcept;
};

//--- m.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module m;
export using ::A;
