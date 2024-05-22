//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// XFAIL: msvc

// This test ensures that we use `[[no_unique_address]]` in `repeat_view`.

#include <ranges>

struct Empty {};

struct Test {
  [[no_unique_address]] std::ranges::repeat_view<Empty> v;
  bool b;
};

static_assert(sizeof(Test) == sizeof(bool));
