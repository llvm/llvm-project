// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -fsyntax-only -verify

//--- foo.h
extern "C++" void c_func();

//--- a.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module a;
export extern "C++" void foo() {}
extern "C++" void bar() {}
export extern "C" void foo_c() {}
extern "C" void bar_c() {}
export void a() {
    foo();
    bar();
    foo_c();
    bar_c();
    c_func();
}
