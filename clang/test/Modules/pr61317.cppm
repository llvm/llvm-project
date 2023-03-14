// From https://github.com/llvm/llvm-project/issues/61317
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-module-interface -o %t/B.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/Use.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

//--- foo.h
#ifndef _FOO
#define _FOO

template <typename T> struct Foo {
  Foo(T) {}
};

template <typename T> Foo(T&) -> Foo<T>;

struct Bar {
  template <typename T>
    requires requires { Foo{T()}; }
  void baz() const {}
};

#endif

//--- A.cppm
module;
#include "foo.h"
export module A;
export using ::Foo;
export using ::Bar;

//--- B.cppm
module;
#include "foo.h"
export module B;
export import A;

//--- Use.cpp
// expected-no-diagnostics
import A;
import B;
void use() {
  Bar _; 
  _.baz<int>();
}
