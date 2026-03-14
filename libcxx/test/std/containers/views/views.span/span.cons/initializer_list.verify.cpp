//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// constexpr explicit(extent != dynamic_extent) span(std::initializer_list<value_type> il); // Since C++26

#include <span>
#include <utility>

#include "test_macros.h"

// Test P2447R4 "Annex C examples"

void one(std::pair<int, int>);
void one(std::span<const int>);

void two(std::span<const int, 2>);

void test_P2447R4_annex_c_examples() {
  // 1. Overload resolution is affected
#if TEST_STD_VER >= 26
  // expected-error@+1 {{call to 'one' is ambiguous}}
  one({1, 2});
#else
  // expected-no-diagnostics
  one({1, 2});
#endif

// 2. The `initializer_list` ctor has high precedence
#if TEST_STD_VER >= 26
  // expected-error@+1 {{chosen constructor is explicit in copy-initialization}}
  two({{1, 2}});
#else
  // expected-no-diagnostics
  two({{1, 2}});
#endif

  // 3. Implicit two-argument construction with a highly convertible value_type
  // --> tested in "initializer_list.pass.cpp"
}
