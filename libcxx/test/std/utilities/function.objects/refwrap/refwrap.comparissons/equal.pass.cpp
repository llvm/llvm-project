//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <functional>

// class reference_wrapper

// // [refwrap.comparisons], comparisons
// friend constexpr bool operator==(reference_wrapper, reference_wrapper);                                // Since C++26
// friend constexpr bool operator==(reference_wrapper, const T&);                                         // Since C++26
// friend constexpr bool operator==(reference_wrapper, reference_wrapper<const T>);                       // Since C++26

#include <cassert>
#include <functional>

#include "test_macros.h"

constexpr bool test() {
  {
    int i = 92;
    std::reference_wrapper<int> lhs{i};
    std::reference_wrapper<int> rhs = lhs;
    assert(lhs == rhs);
  }
  {
    int i = 92;
    std::reference_wrapper<int> lhs{i};
    int j = 84;
    std::reference_wrapper<int> rhs{j};
    assert(lhs != rhs);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
