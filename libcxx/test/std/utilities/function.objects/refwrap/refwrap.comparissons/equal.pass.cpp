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
  int i = 92;
  int j = 84;

  // ==
  {
    // refwrap, refwrap
    std::reference_wrapper<int> rw1{i};
    std::reference_wrapper<int> rw2 = rw1;
    assert(rw1 == rw2);
    assert(rw2 == rw1);
  }
  {
    // refwrap, const&
    std::reference_wrapper<int> rw{i};
    assert(rw == i);
    assert(i == rw);
  }
  {
    // refwrap, refwrap<const>
    std::reference_wrapper<int> rw1{i};
    std::reference_wrapper<const int> rw2 = rw1;
    assert(rw1 == rw2);
    assert(rw2 == rw1);
  }

  // !=
  {
    // refwrap, refwrap
    std::reference_wrapper<int> rw1{i};
    std::reference_wrapper<int> rw2{j};
    assert(rw1 != rw2);
    assert(rw2 != rw1);
  }
  {
    // refwrap, const&
    std::reference_wrapper<int> rw{i};
    assert(rw != j);
    assert(j != rw);
  }
  {
    // refwrap, refwrap<const>
    std::reference_wrapper<int> rw1{i};
    std::reference_wrapper<const int> rw2{j};
    assert(rw1 != rw2);
    assert(rw2 != rw1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
