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
#include <tuple>
#include <utility>
#include <vector>

#include "test_macros.h"

template <typename ReferenceT>
constexpr void testSpanAt(auto&& anySpan, int index, int expectedValue) {
  // non-const
  {
    std::same_as<ReferenceT> decltype(auto) elem = anySpan.at(index);
    assert(elem == expectedValue);
  }

  // const
  {
    std::same_as<ReferenceT> decltype(auto) elem = std::as_const(anySpan).at(index);
    assert(elem == expectedValue);
  }
}

constexpr bool test() {
  // With static extent
  {
    std::array arr{0, 1, 2, 3, 4, 5, 9084};
    std::span arrSpan{arr};

    assert(std::dynamic_extent != arrSpan.extent);

    using ReferenceT = typename decltype(arrSpan)::reference;

    testSpanAt<ReferenceT>(arrSpan, 0, 0);
    testSpanAt<ReferenceT>(arrSpan, 1, 1);
    testSpanAt<ReferenceT>(arrSpan, 6, 9084);
  }

  // With dynamic extent
  {
    std::vector vec{0, 1, 2, 3, 4, 5, 9084};
    std::span vecSpan{vec};

    assert(std::dynamic_extent == vecSpan.extent);

    using ReferenceT = typename decltype(vecSpan)::reference;

    testSpanAt<ReferenceT>(vecSpan, 0, 0);
    testSpanAt<ReferenceT>(vecSpan, 1, 1);
    testSpanAt<ReferenceT>(vecSpan, 6, 9084);
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
      using SizeT = typename decltype(arrSpan)::size_type;
      std::ignore = arrSpan.at(std::numeric_limits<SizeT>::max());
      assert(false);
    } catch ([[maybe_unused]] const std::out_of_range& e) {
      // pass
      LIBCPP_ASSERT(e.what() == "span"s);
    } catch (...) {
      assert(false);
    }

    try {
      std::ignore = arrSpan.at(arr.size());
      assert(false);
    } catch ([[maybe_unused]] const std::out_of_range& e) {
      // pass
      LIBCPP_ASSERT(e.what() == "span"s);
    } catch (...) {
      assert(false);
    }

    try {
      std::ignore = arrSpan.at(arr.size() - 1);
      // pass
      assert(arrSpan.at(arr.size() - 1) == std::numeric_limits<int>::max());
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
    } catch ([[maybe_unused]] const std::out_of_range& e) {
      // pass
      LIBCPP_ASSERT(e.what() == "span"s);
    } catch (...) {
      assert(false);
    }
  }

  // With dynamic extent

  {
    std::vector vec{0, 1, 2, 3, 4, 5, 9084, std::numeric_limits<int>::max()};
    const std::span vecSpan{vec};

    try {
      using SizeT = typename decltype(vecSpan)::size_type;
      std::ignore = vecSpan.at(std::numeric_limits<SizeT>::max());
      assert(false);
    } catch ([[maybe_unused]] const std::out_of_range& e) {
      // pass
      LIBCPP_ASSERT(e.what() == "span"s);
    } catch (...) {
      assert(false);
    }

    try {
      std::ignore = vecSpan.at(vec.size());
      assert(false);
    } catch (const std::out_of_range& e) {
      // pass
      LIBCPP_ASSERT(e.what() == "span"s);
    } catch (...) {
      assert(false);
    }

    try {
      std::ignore = vecSpan.at(vec.size() - 1);
      assert(vecSpan.at(vec.size() - 1) == std::numeric_limits<int>::max());
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
    } catch ([[maybe_unused]] const std::out_of_range& e) {
      // pass
      LIBCPP_ASSERT(e.what() == "span"s);
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
