// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/template_lambdas.cppm -emit-module-interface \
// RUN:    -o %t/lambdas.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only \
// RUN:    -verify
//
// RUN: %clang_cc1 -std=c++20 %t/template_lambdas2.cppm -emit-module-interface \
// RUN:    -o %t/lambdas2.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only \
// RUN:    -verify -DUSE_LAMBDA2

// Test again with reduced BMI
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/template_lambdas.cppm -emit-reduced-module-interface \
// RUN:    -o %t/lambdas.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only \
// RUN:    -verify
//
// RUN: %clang_cc1 -std=c++20 %t/template_lambdas2.cppm -emit-reduced-module-interface \
// RUN:    -o %t/lambdas2.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only \
// RUN:    -verify -DUSE_LAMBDA2

//--- lambdas.h
auto l1 = []<int I>() constexpr -> int {
    return I;
};

auto l2 = []<auto I>() constexpr -> decltype(I) {
    return I;
};

auto l3 = []<class T>(auto i) constexpr -> T {
  return T(i);
};

auto l4 = []<template<class> class T, class U>(T<U>, auto i) constexpr -> U {
  return U(i);
};

//--- template_lambdas.cppm
module;
#include "lambdas.h"
export module lambdas;
export using ::l1;
export using ::l2;
export using ::l3;
export using ::l4;

//--- template_lambdas2.cppm
export module lambdas2;
export {
#include "lambdas.h"  
}

//--- Use.cpp
// expected-no-diagnostics
#ifndef USE_LAMBDA2
import lambdas;
#else
import lambdas2;
#endif

static_assert(l1.operator()<5>() == 5);
static_assert(l1.operator()<6>() == 6);

static_assert(l2.operator()<7>() == 7);
static_assert(l2.operator()<nullptr>() == nullptr);

static_assert(l3.operator()<int>(8.4) == 8);
static_assert(l3.operator()<int>(9.9) == 9);

template<typename T>
struct DummyTemplate { };

static_assert(l4(DummyTemplate<float>(), 12) == 12.0);
static_assert(l4(DummyTemplate<int>(), 19.8) == 19);
