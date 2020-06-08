//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// constexpr       reverse_iterator  rbegin() const noexcept;
// constexpr const_reverse_iterator crbegin() const noexcept;

#include <span>
#include <cassert>
#include <string>

#include "test_macros.h"

template <class Span>
constexpr bool testConstexprSpan(Span s)
{
    bool ret = true;
    typename Span::reverse_iterator b = s.rbegin();
    if (s.empty())
    {
        ret = ret &&  ( b ==  s.rend());
    }
    else
    {
        const typename Span::size_type last = s.size() - 1;
        ret = ret && ( *b ==  s[last]);
        ret = ret && (&*b == &s[last]);
    }
    return ret;
}


template <class Span>
void testRuntimeSpan(Span s)
{
    typename Span::reverse_iterator b = s.rbegin();
    if (s.empty())
    {
        assert(b == s.rend());
    }
    else
    {
        const typename Span::size_type last = s.size() - 1;
        assert( *b ==  s[last]);
        assert(&*b == &s[last]);
    }
}


struct A{};
bool operator==(A, A) {return true;}

constexpr int iArr1[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};
          int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};


int main(int, char**)
{
    static_assert(testConstexprSpan(std::span<int>()),            "");
    static_assert(testConstexprSpan(std::span<long>()),           "");
    static_assert(testConstexprSpan(std::span<double>()),         "");
    static_assert(testConstexprSpan(std::span<A>()),              "");
    static_assert(testConstexprSpan(std::span<std::string>()),    "");

    static_assert(testConstexprSpan(std::span<int, 0>()),         "");
    static_assert(testConstexprSpan(std::span<long, 0>()),        "");
    static_assert(testConstexprSpan(std::span<double, 0>()),      "");
    static_assert(testConstexprSpan(std::span<A, 0>()),           "");
    static_assert(testConstexprSpan(std::span<std::string, 0>()), "");

    static_assert(testConstexprSpan(std::span<const int>(iArr1, 1)),    "");
    static_assert(testConstexprSpan(std::span<const int>(iArr1, 2)),    "");
    static_assert(testConstexprSpan(std::span<const int>(iArr1, 3)),    "");
    static_assert(testConstexprSpan(std::span<const int>(iArr1, 4)),    "");
    static_assert(testConstexprSpan(std::span<const int>(iArr1, 5)),    "");


    testRuntimeSpan(std::span<int>        ());
    testRuntimeSpan(std::span<long>       ());
    testRuntimeSpan(std::span<double>     ());
    testRuntimeSpan(std::span<A>          ());
    testRuntimeSpan(std::span<std::string>());

    testRuntimeSpan(std::span<int, 0>        ());
    testRuntimeSpan(std::span<long, 0>       ());
    testRuntimeSpan(std::span<double, 0>     ());
    testRuntimeSpan(std::span<A, 0>          ());
    testRuntimeSpan(std::span<std::string, 0>());

    testRuntimeSpan(std::span<int>(iArr2, 1));
    testRuntimeSpan(std::span<int>(iArr2, 2));
    testRuntimeSpan(std::span<int>(iArr2, 3));
    testRuntimeSpan(std::span<int>(iArr2, 4));
    testRuntimeSpan(std::span<int>(iArr2, 5));

    std::string s;
    testRuntimeSpan(std::span<std::string>(&s, static_cast<std::size_t>(0)));
    testRuntimeSpan(std::span<std::string>(&s, 1));

  return 0;
}
