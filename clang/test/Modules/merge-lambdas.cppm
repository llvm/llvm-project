// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cppm -fprebuilt-module-path=%t -fsyntax-only -verify

//--- lambda.h
inline auto cmp = [](auto l, auto r) {
  return l < r;
};

//--- A.cppm
module;
#include "lambda.h"

export module A;
export auto c1 = cmp;
export using ::cmp;

//--- B.cppm
module;
#include "lambda.h"

export module B;
export auto c2 = cmp;
export using ::cmp;

//--- use.cppm
// expected-no-diagnostics
module;

export module use;

import A;
import B;

static_assert(__is_same(decltype(c1), decltype(c2))); // should succeed.
auto x = cmp; // cmp must not be ambiguous,
