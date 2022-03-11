//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

//  constexpr span& operator=(const span& other) noexcept = default;

#include <span>
#include <cassert>
#include <string>
#include <utility>

#include "test_macros.h"

template <typename T>
constexpr bool doAssign(T lhs, T rhs)
{
    ASSERT_NOEXCEPT(std::declval<T&>() = rhs);
    lhs = rhs;
    return lhs.data() == rhs.data()
     &&    lhs.size() == rhs.size();
}

struct A{};

constexpr int carr1[] = {1,2,3,4};
constexpr int carr2[] = {3,4,5};
constexpr int carr3[] = {7,8};
          int   arr[] = {5,6,7,9};
std::string strs[] = {"ABC", "DEF", "GHI"};


int main(int, char**)
{

//  constexpr dynamically sized assignment
    {
//  On systems where 'ptrdiff_t' is a synonym for 'int',
//  the call span(ptr, 0) selects the (pointer, size_type) constructor.
//  On systems where 'ptrdiff_t' is NOT a synonym for 'int',
//  it is ambiguous, because of 0 also being convertible to a null pointer
//  and so the compiler can't choose between:
//      span(pointer, size_type)
//  and span(pointer, pointer)
//  We cast zero to std::ptrdiff_t to remove that ambiguity.
//  Example:
//      On darwin x86_64, ptrdiff_t is the same as long int.
//      On darwin i386, ptrdiff_t is the same as int.
        constexpr std::span<const int> spans[] = {
            {},
            {carr1, static_cast<std::size_t>(0)},
            {carr1,     1U},
            {carr1,     2U},
            {carr1,     3U},
            {carr1,     4U},
            {carr2, static_cast<std::size_t>(0)},
            {carr2,     1U},
            {carr2,     2U},
            {carr2,     3U},
            {carr3, static_cast<std::size_t>(0)},
            {carr3,     1U},
            {carr3,     2U}
            };

        static_assert(std::size(spans) == 13, "" );

//  No for loops in constexpr land :-(
        static_assert(doAssign(spans[0], spans[0]), "");
        static_assert(doAssign(spans[0], spans[1]), "");
        static_assert(doAssign(spans[0], spans[2]), "");
        static_assert(doAssign(spans[0], spans[3]), "");
        static_assert(doAssign(spans[0], spans[4]), "");
        static_assert(doAssign(spans[0], spans[5]), "");
        static_assert(doAssign(spans[0], spans[6]), "");
        static_assert(doAssign(spans[0], spans[7]), "");
        static_assert(doAssign(spans[0], spans[8]), "");
        static_assert(doAssign(spans[0], spans[9]), "");
        static_assert(doAssign(spans[0], spans[10]), "");
        static_assert(doAssign(spans[0], spans[11]), "");
        static_assert(doAssign(spans[0], spans[12]), "");

        static_assert(doAssign(spans[1], spans[1]), "");
        static_assert(doAssign(spans[1], spans[2]), "");
        static_assert(doAssign(spans[1], spans[3]), "");
        static_assert(doAssign(spans[1], spans[4]), "");
        static_assert(doAssign(spans[1], spans[5]), "");
        static_assert(doAssign(spans[1], spans[6]), "");
        static_assert(doAssign(spans[1], spans[7]), "");
        static_assert(doAssign(spans[1], spans[8]), "");
        static_assert(doAssign(spans[1], spans[9]), "");
        static_assert(doAssign(spans[1], spans[10]), "");
        static_assert(doAssign(spans[1], spans[11]), "");
        static_assert(doAssign(spans[1], spans[12]), "");

        static_assert(doAssign(spans[2], spans[2]), "");
        static_assert(doAssign(spans[2], spans[3]), "");
        static_assert(doAssign(spans[2], spans[4]), "");
        static_assert(doAssign(spans[2], spans[5]), "");
        static_assert(doAssign(spans[2], spans[6]), "");
        static_assert(doAssign(spans[2], spans[7]), "");
        static_assert(doAssign(spans[2], spans[8]), "");
        static_assert(doAssign(spans[2], spans[9]), "");
        static_assert(doAssign(spans[2], spans[10]), "");
        static_assert(doAssign(spans[2], spans[11]), "");
        static_assert(doAssign(spans[2], spans[12]), "");

        static_assert(doAssign(spans[3], spans[3]), "");
        static_assert(doAssign(spans[3], spans[4]), "");
        static_assert(doAssign(spans[3], spans[4]), "");
        static_assert(doAssign(spans[3], spans[4]), "");
        static_assert(doAssign(spans[3], spans[4]), "");
        static_assert(doAssign(spans[3], spans[4]), "");
        static_assert(doAssign(spans[3], spans[4]), "");
        static_assert(doAssign(spans[3], spans[4]), "");
        static_assert(doAssign(spans[3], spans[4]), "");
        static_assert(doAssign(spans[3], spans[10]), "");
        static_assert(doAssign(spans[3], spans[11]), "");
        static_assert(doAssign(spans[3], spans[12]), "");

        static_assert(doAssign(spans[4], spans[4]), "");
        static_assert(doAssign(spans[4], spans[5]), "");
        static_assert(doAssign(spans[4], spans[6]), "");
        static_assert(doAssign(spans[4], spans[7]), "");
        static_assert(doAssign(spans[4], spans[8]), "");
        static_assert(doAssign(spans[4], spans[9]), "");
        static_assert(doAssign(spans[4], spans[10]), "");
        static_assert(doAssign(spans[4], spans[11]), "");
        static_assert(doAssign(spans[4], spans[12]), "");

        static_assert(doAssign(spans[5], spans[5]), "");
        static_assert(doAssign(spans[5], spans[6]), "");
        static_assert(doAssign(spans[5], spans[7]), "");
        static_assert(doAssign(spans[5], spans[8]), "");
        static_assert(doAssign(spans[5], spans[9]), "");
        static_assert(doAssign(spans[5], spans[10]), "");
        static_assert(doAssign(spans[5], spans[11]), "");
        static_assert(doAssign(spans[5], spans[12]), "");

        static_assert(doAssign(spans[6], spans[6]), "");
        static_assert(doAssign(spans[6], spans[7]), "");
        static_assert(doAssign(spans[6], spans[8]), "");
        static_assert(doAssign(spans[6], spans[9]), "");
        static_assert(doAssign(spans[6], spans[10]), "");
        static_assert(doAssign(spans[6], spans[11]), "");
        static_assert(doAssign(spans[6], spans[12]), "");

        static_assert(doAssign(spans[7], spans[7]), "");
        static_assert(doAssign(spans[7], spans[8]), "");
        static_assert(doAssign(spans[7], spans[9]), "");
        static_assert(doAssign(spans[7], spans[10]), "");
        static_assert(doAssign(spans[7], spans[11]), "");
        static_assert(doAssign(spans[7], spans[12]), "");

        static_assert(doAssign(spans[8], spans[8]), "");
        static_assert(doAssign(spans[8], spans[9]), "");
        static_assert(doAssign(spans[8], spans[10]), "");
        static_assert(doAssign(spans[8], spans[11]), "");
        static_assert(doAssign(spans[8], spans[12]), "");

        static_assert(doAssign(spans[9], spans[9]), "");
        static_assert(doAssign(spans[9], spans[10]), "");
        static_assert(doAssign(spans[9], spans[11]), "");
        static_assert(doAssign(spans[9], spans[12]), "");

        static_assert(doAssign(spans[10], spans[10]), "");
        static_assert(doAssign(spans[10], spans[11]), "");
        static_assert(doAssign(spans[10], spans[12]), "");

        static_assert(doAssign(spans[11], spans[11]), "");
        static_assert(doAssign(spans[11], spans[12]), "");

        static_assert(doAssign(spans[12], spans[12]), "");

//      for (size_t i = 0; i < std::size(spans); ++i)
//          for (size_t j = i; j < std::size(spans); ++j)
//              static_assert(doAssign(spans[i], spans[j]), "");
    }

//  constexpr statically sized assignment
    {
        using spanType = std::span<const int,2>;
        constexpr spanType spans[] = {
            spanType{carr1, 2},
            spanType{carr1 + 1, 2},
            spanType{carr1 + 2, 2},
            spanType{carr2, 2},
            spanType{carr2 + 1, 2},
            spanType{carr3, 2}
            };

        static_assert(std::size(spans) == 6, "" );

//  No for loops in constexpr land :-(
        static_assert(doAssign(spans[0], spans[0]), "");
        static_assert(doAssign(spans[0], spans[1]), "");
        static_assert(doAssign(spans[0], spans[2]), "");
        static_assert(doAssign(spans[0], spans[3]), "");
        static_assert(doAssign(spans[0], spans[4]), "");
        static_assert(doAssign(spans[0], spans[5]), "");

        static_assert(doAssign(spans[1], spans[1]), "");
        static_assert(doAssign(spans[1], spans[2]), "");
        static_assert(doAssign(spans[1], spans[3]), "");
        static_assert(doAssign(spans[1], spans[4]), "");
        static_assert(doAssign(spans[1], spans[5]), "");

        static_assert(doAssign(spans[2], spans[2]), "");
        static_assert(doAssign(spans[2], spans[3]), "");
        static_assert(doAssign(spans[2], spans[4]), "");
        static_assert(doAssign(spans[2], spans[5]), "");

        static_assert(doAssign(spans[3], spans[3]), "");
        static_assert(doAssign(spans[3], spans[4]), "");
        static_assert(doAssign(spans[3], spans[5]), "");

        static_assert(doAssign(spans[4], spans[4]), "");
        static_assert(doAssign(spans[4], spans[5]), "");

        static_assert(doAssign(spans[5], spans[5]), "");

//      for (size_t i = 0; i < std::size(spans); ++i)
//          for (size_t j = i; j < std::size(spans); ++j)
//              static_assert(doAssign(spans[i], spans[j]), "");
    }


//  dynamically sized assignment
    {
        std::span<int> spans[] = {
            {},
            {arr,     arr + 1},
            {arr,     arr + 2},
            {arr,     arr + 3},
            {arr + 1, arr + 3} // same size as s2
            };

        for (size_t i = 0; i < std::size(spans); ++i)
            for (size_t j = i; j < std::size(spans); ++j)
                assert((doAssign(spans[i], spans[j])));
    }

//  statically sized assignment
    {
        using spanType = std::span<int,2>;
        spanType spans[] = {
            spanType{arr,     arr + 2},
            spanType{arr + 1, arr + 3},
            spanType{arr + 2, arr + 4}
            };

        for (size_t i = 0; i < std::size(spans); ++i)
            for (size_t j = i; j < std::size(spans); ++j)
                assert((doAssign(spans[i], spans[j])));
    }

//  dynamically sized assignment
    {
    std::span<std::string> spans[] = {
            {strs,     strs},
            {strs,     strs + 1},
            {strs,     strs + 2},
            {strs,     strs + 3},
            {strs + 1, strs + 1},
            {strs + 1, strs + 2},
            {strs + 1, strs + 3},
            {strs + 2, strs + 2},
            {strs + 2, strs + 3},
            {strs + 3, strs + 3}
            };

        for (size_t i = 0; i < std::size(spans); ++i)
            for (size_t j = i; j < std::size(spans); ++j)
                assert((doAssign(spans[i], spans[j])));
    }

    {
    using spanType = std::span<std::string, 1>;
    spanType spans[] = {
            spanType{strs,     strs + 1},
            spanType{strs + 1, strs + 2},
            spanType{strs + 2, strs + 3}
            };

        for (size_t i = 0; i < std::size(spans); ++i)
            for (size_t j = i; j < std::size(spans); ++j)
                assert((doAssign(spans[i], spans[j])));
    }

  return 0;
}
