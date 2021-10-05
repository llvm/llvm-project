//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// template<size_t Count>
//  constexpr span<element_type, Count> last() const;
//
// constexpr span<element_type, dynamic_extent> last(size_type count) const;
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
    [[maybe_unused]] auto s1 = sp.last<5>(); // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Count out of range in span::last()"}}
  }

  //  Count numeric_limits
  {
    [[maybe_unused]] auto s1 = sp.last<std::size_t(-1)>(); // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Count out of range in span::last()"}}
  }

  return 0;
}
