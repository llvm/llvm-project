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
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "test_macros.h"

constexpr void testSpan(auto span, int idx, int expectedValue) {
  std::same_as<typename decltype(span)::reference> decltype(auto) elem = span.at(idx);
  assert(elem == expectedValue);
}

constexpr bool test() {
  // With static extent

  std::array arr{0, 1, 2, 3, 4, 5, 9084};
  std::span arrSpan{arr};

  assert(std::dynamic_extent != arrSpan.extent);

  testSpan(arrSpan, 0, 0);
  testSpan(arrSpan, 1, 1);
  testSpan(arrSpan, 6, 9084);

  {
    std::same_as<typename decltype(arrSpan)::reference> decltype(auto) arrElem = arrSpan.at(1);
    assert(arrElem == 1);
  }

  {
    std::same_as<typename decltype(arrSpan)::reference> decltype(auto) arrElem = std::as_const(arrSpan).at(1);
    assert(arrElem == 1);
  }

  // With dynamic extent

  std::vector vec{0, 1, 2, 3, 4, 5};
  std::span vecSpan{vec};

  assert(std::dynamic_extent == vecSpan.extent);

  {
    std::same_as<typename decltype(vecSpan)::reference> decltype(auto) vecElem = vecSpan.at(1);
    assert(vecElem == 1);
  }

  {
    std::same_as<typename decltype(vecSpan)::reference> decltype(auto) vecElem = std::as_const(vecSpan).at(1);
    assert(vecElem == 1);
  }

  return true;
}

void test_exceptions() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  // With static extent
  {
    std::array arr{1, 2, 3, 4};
    const std::span arrSpan{arr};

    try {
      TEST_IGNORE_NODISCARD arrSpan.at(4);
      assert(false);
    } catch (std::out_of_range const&) {
      // pass
    } catch (...) {
      assert(false);
    }
  }

  {
    std::array<int, 0> arr{};
    const std::span arrSpan{arr};

    try {
      TEST_IGNORE_NODISCARD arrSpan.at(0);
      assert(false);
    } catch (std::out_of_range const&) {
      // pass
    } catch (...) {
      assert(false);
    }
  }

  // With dynamic extent

  {
    std::vector vec{1, 2, 3, 4};
    const std::span vecSpan{vec};

    try {
      TEST_IGNORE_NODISCARD vec.at(4);
      assert(false);
    } catch (std::out_of_range const&) {
      // pass
    } catch (...) {
      assert(false);
    }
  }

  {
    std::vector<int> vec{};
    const std::span vecSpan{vec};

    try {
      TEST_IGNORE_NODISCARD vec.at(0);
      assert(false);
    } catch (std::out_of_range const&) {
      // pass
    } catch (...) {
      assert(false);
    }
  }
#endif
}

int main(int, char**) {
  test();
  test_exceptions();
  static_assert(test());

  return 0;
}
