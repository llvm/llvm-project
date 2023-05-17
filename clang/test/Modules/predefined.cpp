// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -x c++ -std=c++20 -emit-module-interface a.h -o a.pcm -fms-extensions -verify
// RUN: %clang_cc1 -std=c++20 a.cpp -fmodule-file=A=a.pcm -fms-extensions -fsyntax-only -verify

//--- a.h

// expected-no-diagnostics

export module A;

export template <typename T>
void f() {
    char a[] = __func__;
}

//--- a.cpp

// expected-warning@a.h:8 {{initializing an array from a '__func__' predefined identifier is a Microsoft extension}}

import A;

void g() {
    f<int>(); // expected-note {{in instantiation of function template specialization 'f<int>' requested here}}
}
