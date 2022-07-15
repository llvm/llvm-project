// Checks for reachability for C++11 enum class properly
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only

//--- foo.h
enum class foo {
    a, b, c
};

//--- A.cppm
module;
#include "foo.h"
export module A;
export foo func();

//--- Use.cpp
// expected-no-diagnostics
import A;
void bar() {
    auto f = func();
}
