//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <span>

// template<class It, class EndOrSize>
//   span(It, EndOrSize) -> span<remove_reference_t<iter_reference_t<It>>, maybe-static-ext<EndOrSize>>;
// template<class T, size_t N>
//   span(T (&)[N]) -> span<T, N>;
// template<class T, size_t N>
//   span(array<T, N>&) -> span<T, N>;
// template<class T, size_t N>
//   span(const array<T, N>&) -> span<const T, N>;
// template<class R>
//   span(R&&) -> span<remove_reference_t<ranges::range_reference_t<R>>>;

#include <span>
#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <type_traits>

#include "test_macros.h"

void test_iterator_sentinel() {
  int arr[] = {1, 2, 3};
  {
    std::span s{std::begin(arr), std::end(arr)};
    ASSERT_SAME_TYPE(decltype(s), std::span<int>);
    assert(s.size() == std::size(arr));
    assert(s.data() == std::data(arr));
  }
  {
    std::span s{std::begin(arr), 3};
    ASSERT_SAME_TYPE(decltype(s), std::span<int>);
    assert(s.size() == std::size(arr));
    assert(s.data() == std::data(arr));
  }

  // P3029R1: deduction from `integral_constant`
  {
    std::span s{std::begin(arr), std::integral_constant<size_t, 3>{}};
    ASSERT_SAME_TYPE(decltype(s), std::span<int, 3>);
    assert(s.size() == 3);
    assert(s.data() == std::data(arr));
  }
  // support for `T::value` of a reference type from P2781R9 `std::constant_wrapper`
  {
    using size_3_ref_type = std::integral_constant<const std::size_t&, std::integral_constant<std::size_t, 3>::value>;
    std::span s{std::begin(arr), size_3_ref_type{}};
    ASSERT_SAME_TYPE(decltype(s), std::span<int, 3>);
    assert(s.size() == 3);
    assert(s.data() == std::data(arr));
  }
#if TEST_STD_VER >= 26
  {
    std::span s{std::begin(arr), std::cw<3>};
    ASSERT_SAME_TYPE(decltype(s), std::span<int, 3>);
    assert(s.size() == 3);
    assert(s.data() == std::data(arr));
  }
#endif
  {
    // LWG4351 integral-constant-like needs more remove_cvref_t
    using true_ref_type = std::integral_constant<const bool&, std::true_type::value>;
    LIBCPP_STATIC_ASSERT(!std::__integral_constant_like<true_ref_type>);
    std::span s(std::begin(arr), true_ref_type{});
    ASSERT_SAME_TYPE(decltype(s), std::span<int, std::dynamic_extent>);
    assert(s.size() == 1);
    assert(s.data() == std::data(arr));
  }
#if TEST_STD_VER >= 26
  {
    LIBCPP_STATIC_ASSERT(!std::__integral_constant_like<decltype(std::cw<true>)>);
    std::span s(std::begin(arr), std::cw<true>);
    ASSERT_SAME_TYPE(decltype(s), std::span<int, std::dynamic_extent>);
    assert(s.size() == 1);
    assert(s.data() == std::data(arr));
  }
#endif
}

void test_c_array() {
  {
    int arr[] = {1, 2, 3};
    std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), std::span<int, 3>);
    assert(s.size() == std::size(arr));
    assert(s.data() == std::data(arr));
  }

  {
    const int arr[] = {1, 2, 3};
    std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), std::span<const int, 3>);
    assert(s.size() == std::size(arr));
    assert(s.data() == std::data(arr));
  }
}

void test_std_array() {
  {
    std::array<double, 4> arr = {1.0, 2.0, 3.0, 4.0};
    std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), std::span<double, 4>);
    assert(s.size() == arr.size());
    assert(s.data() == arr.data());
  }

  {
    const std::array<long, 5> arr = {4, 5, 6, 7, 8};
    std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), std::span<const long, 5>);
    assert(s.size() == arr.size());
    assert(s.data() == arr.data());
  }
}

void test_range_std_container() {
  {
    std::string str{"ABCDE"};
    std::span s{str};
    ASSERT_SAME_TYPE(decltype(s), std::span<char>);
    assert(s.size() == str.size());
    assert(s.data() == str.data());
  }

  {
    const std::string str{"QWERTYUIOP"};
    std::span s{str};
    ASSERT_SAME_TYPE(decltype(s), std::span<const char>);
    assert(s.size() == str.size());
    assert(s.data() == str.data());
  }
}

int main(int, char**) {
  test_iterator_sentinel();
  test_c_array();
  test_std_array();
  test_range_std_container();

  return 0;
}
