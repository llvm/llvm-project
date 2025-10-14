// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

//--- lambda.h
auto cmp = [](auto l, auto r) {
  return l < r;
};

//--- a.cppm
module;
#include "lambda.h"
export module a;
export using ::cmp;

//--- use.cpp
// expected-no-diagnostics
import a;
auto x = cmp;
