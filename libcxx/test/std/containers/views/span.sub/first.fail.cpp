//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// AppleClang 12.0.0 doesn't fully support ranges/concepts
// XFAIL: apple-clang-12.0.0

// <span>

// template<size_t Count>
//  constexpr span<element_type, Count> first() const;
//
// constexpr span<element_type, dynamic_extent> first(size_type count) const;
//
//  Requires: Count <= size().

#include <span>

#include <cstddef>

#include "test_macros.h"

constexpr int carr[] = {1, 2, 3, 4};

int main(int, char**) {
  std::span<const int, 4> sp(carr);

  //  Count too large
  {
    [[maybe_unused]] auto s1 = sp.first<5>(); // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Count out of range in span::first()"}}
  }

  //  Count numeric_limits
  {
    [[maybe_unused]] auto s1 = sp.first<std::size_t(-1)>(); // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Count out of range in span::first()"}}
  }

  return 0;
}
