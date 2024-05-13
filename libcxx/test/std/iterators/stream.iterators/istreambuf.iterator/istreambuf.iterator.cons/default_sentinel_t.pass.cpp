//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// constexpr istreambuf_iterator(default_sentinel_t) noexcept; // since C++20

#include <iterator>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  using T = std::istreambuf_iterator<char>;

  {
    T it(std::default_sentinel);
    assert(it == T());
  }

  {
    T it = std::default_sentinel;
    assert(it == T());
  }

  {
    constexpr T it(std::default_sentinel);
    (void)it;
  }

  ASSERT_NOEXCEPT(T(std::default_sentinel));

  return 0;
}
