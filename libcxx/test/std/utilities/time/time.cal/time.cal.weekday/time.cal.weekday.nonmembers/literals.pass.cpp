//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <chrono>

// inline constexpr weekday   Sunday{0};
// inline constexpr weekday   Monday{1};
// inline constexpr weekday   Tuesday{2};
// inline constexpr weekday   Wednesday{3};
// inline constexpr weekday   Thursday{4};
// inline constexpr weekday   Friday{5};
// inline constexpr weekday   Saturday{6};


#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{

    ASSERT_SAME_TYPE(const std::chrono::weekday, decltype(std::chrono::Sunday));
    ASSERT_SAME_TYPE(const std::chrono::weekday, decltype(std::chrono::Monday));
    ASSERT_SAME_TYPE(const std::chrono::weekday, decltype(std::chrono::Tuesday));
    ASSERT_SAME_TYPE(const std::chrono::weekday, decltype(std::chrono::Wednesday));
    ASSERT_SAME_TYPE(const std::chrono::weekday, decltype(std::chrono::Thursday));
    ASSERT_SAME_TYPE(const std::chrono::weekday, decltype(std::chrono::Friday));
    ASSERT_SAME_TYPE(const std::chrono::weekday, decltype(std::chrono::Saturday));

    static_assert( std::chrono::Sunday    == std::chrono::weekday(0),  "");
    static_assert( std::chrono::Monday    == std::chrono::weekday(1),  "");
    static_assert( std::chrono::Tuesday   == std::chrono::weekday(2),  "");
    static_assert( std::chrono::Wednesday == std::chrono::weekday(3),  "");
    static_assert( std::chrono::Thursday  == std::chrono::weekday(4),  "");
    static_assert( std::chrono::Friday    == std::chrono::weekday(5),  "");
    static_assert( std::chrono::Saturday  == std::chrono::weekday(6),  "");

    assert(std::chrono::Sunday    == std::chrono::weekday(0));
    assert(std::chrono::Monday    == std::chrono::weekday(1));
    assert(std::chrono::Tuesday   == std::chrono::weekday(2));
    assert(std::chrono::Wednesday == std::chrono::weekday(3));
    assert(std::chrono::Thursday  == std::chrono::weekday(4));
    assert(std::chrono::Friday    == std::chrono::weekday(5));
    assert(std::chrono::Saturday  == std::chrono::weekday(6));

    assert(std::chrono::Sunday.c_encoding()    ==  0);
    assert(std::chrono::Monday.c_encoding()    ==  1);
    assert(std::chrono::Tuesday.c_encoding()   ==  2);
    assert(std::chrono::Wednesday.c_encoding() ==  3);
    assert(std::chrono::Thursday.c_encoding()  ==  4);
    assert(std::chrono::Friday.c_encoding()    ==  5);
    assert(std::chrono::Saturday.c_encoding()  ==  6);

  return 0;
}
