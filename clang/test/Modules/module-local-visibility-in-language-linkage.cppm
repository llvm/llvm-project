// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/m.a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/m.b.cppm -fmodule-file=m:a=%t/a.pcm -fsyntax-only -verify

//--- m.a.cppm
export module m:a;
int a;

//--- m.b.cppm
// expected-no-diagnostics
module m:b;
import :a;
extern "C++" int get_a() { return a; }
