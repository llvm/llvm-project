//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <type_traits>

// template<class T> struct is_implicit_lifetime;

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"

struct IncompleteStruct;

// expected-error@*:* {{incomplete type 'IncompleteStruct' used in type trait expression}}
static_assert(!std::is_implicit_lifetime<IncompleteStruct>::value);

// expected-error@*:* {{atomic types are not supported in '__builtin_is_implicit_lifetime'}}
static_assert(!std::is_implicit_lifetime<_Atomic int>::value);

#if 0
// FIXME: "variable length arrays in C++ are a Clang extension"
void test(int n) {
  int varArr[n];
  using VarArrT = decltype(varArr);
  // expected-error@*:* {{variable length arrays are not supported in '__builtin_is_implicit_lifetime'}}
  static_assert(std::is_implicit_lifetime<VarArrT>::value);
}
#endif
