//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// struct from_range_t { explicit from_range_t() = default; };  // Since C++23
// inline constexpr from_range_t from_range{};                  // Since C++23

#include <ranges>

template <class T>
void check(std::from_range_t);

template <class T>
concept IsCtrNonexplicit = requires {
  check<T>({});
};

// Verify that the constructor is `explicit`.
static_assert(!IsCtrNonexplicit<std::from_range_t>);
