//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <span>

// constexpr reference at(size_type idx) const; // since C++26

#include <array>
#include <cassert>
#include <concepts>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "test_macros.h"

constexpr void testSpanAt(auto& container, bool hasDynamicExtent, int index, int expectedValue) {
  std::span anySpan{container};

  if (hasDynamicExtent) {
    assert(std::dynamic_extent == anySpan.extent);
  } else {
    assert(std::dynamic_extent != anySpan.extent);
  }

  // non-const
  {
    std::same_as<typename decltype(anySpan)::reference> decltype(auto) elem = anySpan.at(index);
    assert(elem == expectedValue);
  }

  // const
  {
    std::same_as<typename decltype(anySpan)::reference> decltype(auto) elem = std::as_const(anySpan).at(index);
    assert(elem == expectedValue);
  }
}

constexpr bool test() {
  // With static extent
  {
    std::array arr{0, 1, 2, 3, 4, 5, 9084, std::numeric_limits<int>::max()};

    testSpanAt(arr, false, 0, 0);
    testSpanAt(arr, false, 1, 1);
    testSpanAt(arr, false, 6, 9084);
    testSpanAt(arr, false, 7, std::numeric_limits<int>::max());
  }

  // With dynamic extent
  {
    std::vector vec{0, 1, 2, 3, 4, 5, 9084, std::numeric_limits<int>::max()};

    testSpanAt(vec, true, 0, 0);
    testSpanAt(vec, true, 1, 1);
    testSpanAt(vec, true, 6, 9084);
    testSpanAt(vec, true, 7, std::numeric_limits<int>::max());
  }

  return true;
}

void test_exceptions() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using namespace std::string_literals;

  // With static extent
  {
    std::array arr{0, 1, 2, 3, 4, 5, 9084, std::numeric_limits<int>::max()};
    const std::span arrSpan{arr};

    try {
      std::ignore = arrSpan.at(arr.size());
      assert(false);
    } catch (const std::out_of_range& e) {
      // pass
      assert(e.what() == "span"s);
    } catch (...) {
      assert(false);
    }
  }

  {
    std::array<int, 0> arr{};
    const std::span arrSpan{arr};

    try {
      std::ignore = arrSpan.at(0);
      assert(false);
    } catch (const std::out_of_range& e) {
      // pass
      assert(e.what() == "span"s);
    } catch (...) {
      assert(false);
    }
  }

  // With dynamic extent

  {
    std::vector vec{0, 1, 2, 3, 4, 5, 9084, std::numeric_limits<int>::max()};
    const std::span vecSpan{vec};

    try {
      std::ignore = vecSpan.at(vec.size());
      assert(false);
    } catch (const std::out_of_range& e) {
      // pass
      assert(e.what() == "span"s);
    } catch (...) {
      assert(false);
    }
  }

  {
    std::vector<int> vec{};
    const std::span vecSpan{vec};

    try {
      std::ignore = vecSpan.at(0);
      assert(false);
    } catch (const std::out_of_range& e) {
      // pass
      assert(e.what() == "span"s);
    } catch (...) {
      assert(false);
    }
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
  static_assert(test());

  test_exceptions();

  return 0;
}
