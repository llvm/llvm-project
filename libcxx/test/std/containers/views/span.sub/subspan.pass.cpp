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

// template<size_t Offset, size_t Count = dynamic_extent>
//   constexpr span<element_type, see below> subspan() const;
//
// constexpr span<element_type, dynamic_extent> subspan(
//   index_type offset, index_type count = dynamic_extent) const;
//
//  Requires: (0 <= Offset && Offset <= size())
//      && (Count == dynamic_extent || Count >= 0 && Offset + Count <= size())

#include <span>
#include <cassert>
#include <algorithm>
#include <string>

#include "test_macros.h"

template <typename Span, size_t Offset, size_t Count>
constexpr bool testConstexprSpan(Span sp)
{
    LIBCPP_ASSERT((noexcept(sp.template subspan<Offset, Count>())));
    LIBCPP_ASSERT((noexcept(sp.subspan(Offset, Count))));
    auto s1 = sp.template subspan<Offset, Count>();
    auto s2 = sp.subspan(Offset, Count);
    using S1 = decltype(s1);
    using S2 = decltype(s2);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S1::value_type);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S2::value_type);
    static_assert(S1::extent == (Span::extent == std::dynamic_extent ? std::dynamic_extent : Count), "");
    static_assert(S2::extent == std::dynamic_extent, "");
    return
        s1.data() == s2.data()
     && s1.size() == s2.size()
     && std::equal(s1.begin(), s1.end(), sp.begin() + Offset);
}

template <typename Span, size_t Offset>
constexpr bool testConstexprSpan(Span sp)
{
    LIBCPP_ASSERT((noexcept(sp.template subspan<Offset>())));
    LIBCPP_ASSERT((noexcept(sp.subspan(Offset))));
    auto s1 = sp.template subspan<Offset>();
    auto s2 = sp.subspan(Offset);
    using S1 = decltype(s1);
    using S2 = decltype(s2);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S1::value_type);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S2::value_type);
    static_assert(S1::extent == (Span::extent == std::dynamic_extent ? std::dynamic_extent : Span::extent - Offset), "");
    static_assert(S2::extent == std::dynamic_extent, "");
    return
        s1.data() == s2.data()
     && s1.size() == s2.size()
     && std::equal(s1.begin(), s1.end(), sp.begin() + Offset, sp.end());
}


template <typename Span, size_t Offset, size_t Count>
void testRuntimeSpan(Span sp)
{
    LIBCPP_ASSERT((noexcept(sp.template subspan<Offset, Count>())));
    LIBCPP_ASSERT((noexcept(sp.subspan(Offset, Count))));
    auto s1 = sp.template subspan<Offset, Count>();
    auto s2 = sp.subspan(Offset, Count);
    using S1 = decltype(s1);
    using S2 = decltype(s2);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S1::value_type);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S2::value_type);
    static_assert(S1::extent == (Span::extent == std::dynamic_extent ? std::dynamic_extent : Count), "");
    static_assert(S2::extent == std::dynamic_extent, "");
    assert(s1.data() == s2.data());
    assert(s1.size() == s2.size());
    assert(std::equal(s1.begin(), s1.end(), sp.begin() + Offset));
}


template <typename Span, size_t Offset>
void testRuntimeSpan(Span sp)
{
    LIBCPP_ASSERT((noexcept(sp.template subspan<Offset>())));
    LIBCPP_ASSERT((noexcept(sp.subspan(Offset))));
    auto s1 = sp.template subspan<Offset>();
    auto s2 = sp.subspan(Offset);
    using S1 = decltype(s1);
    using S2 = decltype(s2);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S1::value_type);
    ASSERT_SAME_TYPE(typename Span::value_type, typename S2::value_type);
    static_assert(S1::extent == (Span::extent == std::dynamic_extent ? std::dynamic_extent : Span::extent - Offset), "");
    static_assert(S2::extent == std::dynamic_extent, "");
    assert(s1.data() == s2.data());
    assert(s1.size() == s2.size());
    assert(std::equal(s1.begin(), s1.end(), sp.begin() + Offset, sp.end()));
}


constexpr int carr1[] = {1,2,3,4};
          int  arr1[] = {5,6,7};

int main(int, char**)
{
    {
    using Sp = std::span<const int>;
    static_assert(testConstexprSpan<Sp, 0>(Sp{}), "");

    static_assert(testConstexprSpan<Sp, 0, 4>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 0>(Sp{carr1}), "");

    static_assert(testConstexprSpan<Sp, 1, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 2, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 3, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 4, 0>(Sp{carr1}), "");
    }

    {
    using Sp = std::span<const int, 4>;

    static_assert(testConstexprSpan<Sp, 0, 4>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 0, 0>(Sp{carr1}), "");

    static_assert(testConstexprSpan<Sp, 1, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 2, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 3, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 4, 0>(Sp{carr1}), "");
    }

    {
    using Sp = std::span<const int>;
    static_assert(testConstexprSpan<Sp, 0>(Sp{}), "");

    static_assert(testConstexprSpan<Sp, 0>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 4>(Sp{carr1}), "");
    }

    {
    using Sp = std::span<const int, 4>;

    static_assert(testConstexprSpan<Sp, 0>(Sp{carr1}), "");

    static_assert(testConstexprSpan<Sp, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 4>(Sp{carr1}), "");
    }

    {
    using Sp = std::span<int>;
    testRuntimeSpan<Sp, 0>(Sp{});

    testRuntimeSpan<Sp, 0, 3>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 0>(Sp{arr1});

    testRuntimeSpan<Sp, 1, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 2, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 3, 0>(Sp{arr1});
    }

    {
    using Sp = std::span<int, 3>;

    testRuntimeSpan<Sp, 0, 3>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 0, 0>(Sp{arr1});

    testRuntimeSpan<Sp, 1, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 2, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 3, 0>(Sp{arr1});
    }

    {
    using Sp = std::span<int>;
    testRuntimeSpan<Sp, 0>(Sp{});

    testRuntimeSpan<Sp, 0>(Sp{arr1});
    testRuntimeSpan<Sp, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 3>(Sp{arr1});
    }

    {
    using Sp = std::span<int, 3>;

    testRuntimeSpan<Sp, 0>(Sp{arr1});
    testRuntimeSpan<Sp, 1>(Sp{arr1});
    testRuntimeSpan<Sp, 2>(Sp{arr1});
    testRuntimeSpan<Sp, 3>(Sp{arr1});
    }

  return 0;
}
