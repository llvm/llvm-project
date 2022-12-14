// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/lambdas.cppm -emit-module-interface \
// RUN:    -o %t/lambdas.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only \
// RUN:    -verify
//
// RUN: %clang_cc1 -std=c++20 %t/lambdas2.cppm -emit-module-interface \
// RUN:    -o %t/lambdas2.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only \
// RUN:    -verify -DUSE_LAMBDA2

//--- lambdas.h
auto l1 = []() constexpr -> int {
    return 43;
};
// 
auto l2 = []() constexpr -> double {
    return 3.0;
};
// 
auto l3 = [](auto i) constexpr -> int {
  return int(i);
};
// 
auto l4 = [](auto i, auto u) constexpr -> int {
  return i + u;
};

//--- lambdas.cppm
module;
#include "lambdas.h"
export module lambdas;
export using ::l1;
export using ::l2;
export using ::l3;
export using ::l4;

//--- lambdas2.cppm
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

static_assert(l1.operator()() == 43);

static_assert(l2.operator()() == 3.0);

static_assert(l3.operator()(8.4) == 8);
 
static_assert(l4(4, 12) == 16);
static_assert(l4(5, 20) == 25);
