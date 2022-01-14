//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// clang-cl and cl currently don't support [[no_unique_address]]
// XFAIL: msvc

// [algorithms.results]/1
//   Each of the class templates specified in this subclause has the template parameters,
//   data members, and special members specified below, and has no base classes or members
//   other than those specified.

#include <algorithm>

struct Empty {};

static_assert(sizeof(std::ranges::in_in_result<Empty, int>) == sizeof(int));
static_assert(sizeof(std::ranges::in_in_result<int, Empty>) == sizeof(int));
static_assert(sizeof(std::ranges::in_in_result<Empty, Empty>) == 2 * sizeof(Empty));
