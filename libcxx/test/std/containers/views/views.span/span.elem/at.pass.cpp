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
#include <vector>

#include "test_macros.h"

// template <typename Span>
// constexpr bool testConstexprSpan(Span sp, std::size_t idx)
// {
//     LIBCPP_ASSERT(noexcept(sp[idx]));

//     typename Span::reference r1 = sp[idx];
//     typename Span::reference r2 = *(sp.data() + idx);

//     return r1 == r2;
// }

// template <typename Span>
// void testRuntimeSpan(Span sp, std::size_t idx)
// {
//     LIBCPP_ASSERT(noexcept(sp[idx]));

//     typename Span::reference r1 = sp[idx];
//     typename Span::reference r2 = *(sp.data() + idx);

//     assert(r1 == r2);
// }

// struct A{};
// constexpr int iArr1[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};
//           int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

// int main(int, char**)
// {
//     static_assert(testConstexprSpan(std::span<const int>(iArr1, 1), 0), "");

//     static_assert(testConstexprSpan(std::span<const int>(iArr1, 2), 0), "");
//     static_assert(testConstexprSpan(std::span<const int>(iArr1, 2), 1), "");

//     static_assert(testConstexprSpan(std::span<const int>(iArr1, 3), 0), "");
//     static_assert(testConstexprSpan(std::span<const int>(iArr1, 3), 1), "");
//     static_assert(testConstexprSpan(std::span<const int>(iArr1, 3), 2), "");

//     static_assert(testConstexprSpan(std::span<const int>(iArr1, 4), 0), "");
//     static_assert(testConstexprSpan(std::span<const int>(iArr1, 4), 1), "");
//     static_assert(testConstexprSpan(std::span<const int>(iArr1, 4), 2), "");
//     static_assert(testConstexprSpan(std::span<const int>(iArr1, 4), 3), "");

//     static_assert(testConstexprSpan(std::span<const int, 1>(iArr1, 1), 0), "");

//     static_assert(testConstexprSpan(std::span<const int, 2>(iArr1, 2), 0), "");
//     static_assert(testConstexprSpan(std::span<const int, 2>(iArr1, 2), 1), "");

//     static_assert(testConstexprSpan(std::span<const int, 3>(iArr1, 3), 0), "");
//     static_assert(testConstexprSpan(std::span<const int, 3>(iArr1, 3), 1), "");
//     static_assert(testConstexprSpan(std::span<const int, 3>(iArr1, 3), 2), "");

//     static_assert(testConstexprSpan(std::span<const int, 4>(iArr1, 4), 0), "");
//     static_assert(testConstexprSpan(std::span<const int, 4>(iArr1, 4), 1), "");
//     static_assert(testConstexprSpan(std::span<const int, 4>(iArr1, 4), 2), "");
//     static_assert(testConstexprSpan(std::span<const int, 4>(iArr1, 4), 3), "");

//     testRuntimeSpan(std::span<int>(iArr2, 1), 0);

//     testRuntimeSpan(std::span<int>(iArr2, 2), 0);
//     testRuntimeSpan(std::span<int>(iArr2, 2), 1);

//     testRuntimeSpan(std::span<int>(iArr2, 3), 0);
//     testRuntimeSpan(std::span<int>(iArr2, 3), 1);
//     testRuntimeSpan(std::span<int>(iArr2, 3), 2);

//     testRuntimeSpan(std::span<int>(iArr2, 4), 0);
//     testRuntimeSpan(std::span<int>(iArr2, 4), 1);
//     testRuntimeSpan(std::span<int>(iArr2, 4), 2);
//     testRuntimeSpan(std::span<int>(iArr2, 4), 3);

//     testRuntimeSpan(std::span<int, 1>(iArr2, 1), 0);

//     testRuntimeSpan(std::span<int, 2>(iArr2, 2), 0);
//     testRuntimeSpan(std::span<int, 2>(iArr2, 2), 1);

//     testRuntimeSpan(std::span<int, 3>(iArr2, 3), 0);
//     testRuntimeSpan(std::span<int, 3>(iArr2, 3), 1);
//     testRuntimeSpan(std::span<int, 3>(iArr2, 3), 2);

//     testRuntimeSpan(std::span<int, 4>(iArr2, 4), 0);
//     testRuntimeSpan(std::span<int, 4>(iArr2, 4), 1);
//     testRuntimeSpan(std::span<int, 4>(iArr2, 4), 2);
//     testRuntimeSpan(std::span<int, 4>(iArr2, 4), 3);

//     std::string s;
//     testRuntimeSpan(std::span<std::string>   (&s, 1), 0);
//     testRuntimeSpan(std::span<std::string, 1>(&s, 1), 0);

//   return 0;
// }

constexpr bool test() {
  //   {
  //     typedef double T;
  //     typedef std::array<T, 3> C;
  //     C const c                      = {1, 2, 3.5};
  //     typename C::const_reference r1 = c.at(0);
  //     assert(r1 == 1);

  //     typename C::const_reference r2 = c.at(2);
  //     assert(r2 == 3.5);
  //   }

  const auto testSpan =
      [](auto span, int idx, int expectedValue) {
        {
          std::same_as<decltype(span)::reference> elem = span.at(idx);
          assert(elem == expectedValue);
        }

        {
          std::same_as<decltype(span)::const_reference> elem = std::as_const(span).at(idx);
          assert(elem == expectedValue);
        }
      }

  // With static extent

  std::array arr{0, 1, 2, 3, 4, 5, 9084};
  std::span arrSpan{ar};

  assert(std::dynamic_extent != arrSpan.extent);

  testSpan(arrSpan, 0, 0);
  testSpan(arrSpan, 1, 1);
  testSpan(arrSpan, 5, 9084);

  {
    std::same_as<decltype(arrSpan)::reference> arrElem = arrSpan.at(1);
    assert(arrElem == 1);
  }

  {
    std::same_as<decltype(arrSpan)::const_reference> arrElem = std::as_const(arrSpan).at(1);
    assert(arrElem == 1);
  }

  // With dynamic extent

  std::vector vec{0, 1, 2, 3, 4, 5};
  std::span vecSpan{vec};
  
  assert(std::dynamic_extent == vecSpan.extent)

  {
    std::same_as<decltype(vecSpan)::reference> vecElem = vecSpan.at(1);
    assert(vec_elem == 1);
  }

  {
    std::same_as<decltype(vecSpan)::reference> vecElem = std::as_const(vecSpan).at(1);
    assert(vec_elem == 1);
  }

  return true;
}

void test_exceptions() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  // With static extent
  {
    const std::array arr{1, 2, 3, 4};

    try {
      TEST_IGNORE_NODISCARD arr.at(4);
      assert(false);
    } catch (std::out_of_range const&) {
      // pass
    } catch (...) {
      assert(false);
    }
  }

  {
    const std::array<int, 0> arr{};

    try {
      TEST_IGNORE_NODISCARD arr.at(0);
      assert(false);
    } catch (std::out_of_range const&) {
      // pass
    } catch (...) {
      assert(false);
    }
  }

  // With dynamic extent

  {
    const std::vector vec{1, 2, 3, 4};

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
    const std::vector<int> vec{};

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