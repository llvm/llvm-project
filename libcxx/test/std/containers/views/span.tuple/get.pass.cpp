// -*- C++ -*-
//===------------------------------ span ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <span>

// template <size_t _Ip, ElementType, size_t Extent>
//   constexpr ElementType& get(span<ElementType, Extent> s) noexcept;
//


#include <span>
#include <cassert>
#include <string>

#include "test_macros.h"


template <size_t Idx, typename Span>
constexpr bool testConstexprSpan(Span sp, typename Span::pointer ptr)
{
    ASSERT_NOEXCEPT(std::get<Idx>(sp));
    return std::addressof(std::get<Idx>(sp)) == ptr;
}


template <size_t Idx, typename Span>
void testRuntimeSpan(Span sp, typename Span::pointer ptr)
{
    ASSERT_NOEXCEPT(std::get<Idx>(sp));
    assert(std::addressof(std::get<Idx>(sp)) == ptr);
}

struct A{};
constexpr int iArr1[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};
          int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**)
{

//  static size
    static_assert(testConstexprSpan<0>(std::span<const int, 4>(iArr1, 4), iArr1 + 0), "");
    static_assert(testConstexprSpan<1>(std::span<const int, 4>(iArr1, 4), iArr1 + 1), "");
    static_assert(testConstexprSpan<2>(std::span<const int, 4>(iArr1, 4), iArr1 + 2), "");
    static_assert(testConstexprSpan<3>(std::span<const int, 4>(iArr1, 4), iArr1 + 3), "");

    static_assert(testConstexprSpan<0>(std::span<const int, 1>(iArr1 + 1, 1), iArr1 + 1), "");
    static_assert(testConstexprSpan<1>(std::span<const int, 2>(iArr1 + 2, 2), iArr1 + 3), "");
    static_assert(testConstexprSpan<2>(std::span<const int, 3>(iArr1 + 3, 3), iArr1 + 5), "");
    static_assert(testConstexprSpan<3>(std::span<const int, 4>(iArr1 + 4, 4), iArr1 + 7), "");

//  static size
    testRuntimeSpan<0>(std::span<int, 4>(iArr2, 4), iArr2);
    testRuntimeSpan<1>(std::span<int, 4>(iArr2, 4), iArr2 + 1);
    testRuntimeSpan<2>(std::span<int, 4>(iArr2, 4), iArr2 + 2);
    testRuntimeSpan<3>(std::span<int, 4>(iArr2, 4), iArr2 + 3);

    testRuntimeSpan<0>(std::span<int, 1>(iArr2 + 1, 1), iArr2 + 1);
    testRuntimeSpan<1>(std::span<int, 2>(iArr2 + 2, 2), iArr2 + 3);
    testRuntimeSpan<2>(std::span<int, 3>(iArr2 + 3, 3), iArr2 + 5);
    testRuntimeSpan<3>(std::span<int, 4>(iArr2 + 4, 4), iArr2 + 7);


    std::string s;
    testRuntimeSpan<0>(std::span<std::string, 1>(&s, 1), &s);


  return 0;
}
