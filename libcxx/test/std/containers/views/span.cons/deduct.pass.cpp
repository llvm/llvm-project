//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <span>

//   template<class It, class EndOrSize>
//     span(It, EndOrSize) -> span<remove_reference_t<iter_reference_t<_It>>>;
//
//   template<class T, size_t N>
//     span(T (&)[N]) -> span<T, N>;
//
//   template<class T, size_t N>
//     span(array<T, N>&) -> span<T, N>;
//
//   template<class T, size_t N>
//     span(const array<T, N>&) -> span<const T, N>;
//
//   template<class R>
//     span(R&&) -> span<remove_reference_t<ranges::range_reference_t<R>>>;


#include <span>
#include <array>
#include <cassert>
#include <memory>
#include <string>

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
    const int arr[] = {1,2,3};
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

int main(int, char**)
{
  test_iterator_sentinel();
  test_c_array();
  test_std_array();
  test_range_std_container();

  return 0;
}
