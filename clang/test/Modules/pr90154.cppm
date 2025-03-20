// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cc -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
//
// Test again with reduced BMI
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cc -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

//--- a.cppm
export module a;
int b = 99;
namespace a { int a = 43; }

//--- use.cc
// expected-no-diagnostics
import a;

namespace a {
    double a = 43.0;
}

int b = 883;
