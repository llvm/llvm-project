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

// constexpr index_type size_bytes() const noexcept;
//
//  Effects: Equivalent to: return size() * sizeof(element_type);


#include <span>
#include <cassert>
#include <string>

#include "test_macros.h"


template <typename Span>
constexpr bool testConstexprSpan(Span sp, size_t sz)
{
    ASSERT_NOEXCEPT(sp.size_bytes());
    return (size_t) sp.size_bytes() == sz * sizeof(typename Span::element_type);
}


template <typename Span>
void testRuntimeSpan(Span sp, size_t sz)
{
    ASSERT_NOEXCEPT(sp.size_bytes());
    assert((size_t) sp.size_bytes() == sz * sizeof(typename Span::element_type));
}

struct A{};
constexpr int iArr1[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};
          int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**)
{
    static_assert(testConstexprSpan(std::span<int>(), 0),            "");
    static_assert(testConstexprSpan(std::span<long>(), 0),           "");
    static_assert(testConstexprSpan(std::span<double>(), 0),         "");
    static_assert(testConstexprSpan(std::span<A>(), 0),              "");
    static_assert(testConstexprSpan(std::span<std::string>(), 0),    "");

    static_assert(testConstexprSpan(std::span<int, 0>(), 0),         "");
    static_assert(testConstexprSpan(std::span<long, 0>(), 0),        "");
    static_assert(testConstexprSpan(std::span<double, 0>(), 0),      "");
    static_assert(testConstexprSpan(std::span<A, 0>(), 0),           "");
    static_assert(testConstexprSpan(std::span<std::string, 0>(), 0), "");

    static_assert(testConstexprSpan(std::span<const int>(iArr1, 1), 1),    "");
    static_assert(testConstexprSpan(std::span<const int>(iArr1, 2), 2),    "");
    static_assert(testConstexprSpan(std::span<const int>(iArr1, 3), 3),    "");
    static_assert(testConstexprSpan(std::span<const int>(iArr1, 4), 4),    "");
    static_assert(testConstexprSpan(std::span<const int>(iArr1, 5), 5),    "");

    testRuntimeSpan(std::span<int>        (), 0);
    testRuntimeSpan(std::span<long>       (), 0);
    testRuntimeSpan(std::span<double>     (), 0);
    testRuntimeSpan(std::span<A>          (), 0);
    testRuntimeSpan(std::span<std::string>(), 0);

    testRuntimeSpan(std::span<int, 0>        (), 0);
    testRuntimeSpan(std::span<long, 0>       (), 0);
    testRuntimeSpan(std::span<double, 0>     (), 0);
    testRuntimeSpan(std::span<A, 0>          (), 0);
    testRuntimeSpan(std::span<std::string, 0>(), 0);

    testRuntimeSpan(std::span<int>(iArr2, 1), 1);
    testRuntimeSpan(std::span<int>(iArr2, 2), 2);
    testRuntimeSpan(std::span<int>(iArr2, 3), 3);
    testRuntimeSpan(std::span<int>(iArr2, 4), 4);
    testRuntimeSpan(std::span<int>(iArr2, 5), 5);

    testRuntimeSpan(std::span<int, 1>(iArr2 + 5, 1), 1);
    testRuntimeSpan(std::span<int, 2>(iArr2 + 4, 2), 2);
    testRuntimeSpan(std::span<int, 3>(iArr2 + 3, 3), 3);
    testRuntimeSpan(std::span<int, 4>(iArr2 + 2, 4), 4);
    testRuntimeSpan(std::span<int, 5>(iArr2 + 1, 5), 5);

    std::string s;
    testRuntimeSpan(std::span<std::string>(&s, (std::size_t) 0), 0);
    testRuntimeSpan(std::span<std::string>(&s, 1), 1);

  return 0;
}
