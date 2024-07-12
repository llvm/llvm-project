// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/X.cppm -emit-module-interface -o %t/X.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/X.cppm -emit-reduced-module-interface -o %t/X.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only -verify
//
//--- foo.h
#ifndef FOO_H
#define FOO_H
template <typename T>
struct base {};

template <typename T>
struct foo;

template <typename T>
struct foo {};

template <>
struct foo<int> : base<int> {
  int getInt();
};
#endif // FOO_H

//--- X.cppm
module;
#include "foo.h"
export module X;
export template <class T>
class X {
  foo<int> x;

public:
  int print() {
    return x.getInt();
  }
};

//--- Use.cpp
import X;
foo<int> f; // expected-error {{'foo' must be declared before it is used}}
            // expected-note@* {{declaration here is not visible}}
int bar() {
  X<int> x;
  return x.print();
}
