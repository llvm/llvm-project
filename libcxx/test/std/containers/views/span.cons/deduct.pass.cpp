//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

//   template<class T, size_t N>
//     span(T (&)[N]) -> span<T, N>;
//
//   template<class T, size_t N>
//     span(array<T, N>&) -> span<T, N>;
//
//   template<class T, size_t N>
//     span(const array<T, N>&) -> span<const T, N>;
//
//   template<class Container>
//     span(Container&) -> span<typename Container::value_type>;
//
//   template<class Container>
//     span(const Container&) -> span<const typename Container::value_type>;



#include <span>
#include <array>
#include <cassert>
#include <memory>
#include <string>

#include "test_macros.h"

int main(int, char**)
{
    {
    int arr[] = {1,2,3};
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

  return 0;
}
