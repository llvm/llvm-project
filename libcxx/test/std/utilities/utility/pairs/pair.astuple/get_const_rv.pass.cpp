//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template<size_t I, class T1, class T2>
//     const typename tuple_element<I, std::pair<T1, T2> >::type&&
//     get(const pair<T1, T2>&&);

// UNSUPPORTED: c++03 && !stdlib=libc++

#include <utility>
#include <memory>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
#if TEST_STD_VER >= 11
    typedef std::pair<std::unique_ptr<int>, short> P;
    const P p(std::unique_ptr<int>(new int(3)), static_cast<short>(4));
    static_assert(std::is_same<const std::unique_ptr<int>&&, decltype(std::get<0>(std::move(p)))>::value, "");
    static_assert(noexcept(std::get<0>(std::move(p))), "");
    const std::unique_ptr<int>&& ptr = std::get<0>(std::move(p));
    assert(*ptr == 3);
#endif
    }

    {
    int x = 42;
    int const y = 43;
    std::pair<int&, int const&> const p(x, y);
    static_assert(std::is_same<int&, decltype(std::get<0>(std::move(p)))>::value, "");
    static_assert(std::is_same<int const&, decltype(std::get<1>(std::move(p)))>::value, "");
#if TEST_STD_VER >= 11
    static_assert(noexcept(std::get<0>(std::move(p))), "");
    static_assert(noexcept(std::get<1>(std::move(p))), "");
#endif
    }

    {
#if TEST_STD_VER >= 11
    int x = 42;
    int const y = 43;
    std::pair<int&&, int const&&> const p(std::move(x), std::move(y));
    static_assert(std::is_same<int&&, decltype(std::get<0>(std::move(p)))>::value, "");
    static_assert(noexcept(std::get<0>(std::move(p))), "");
    static_assert(std::is_same<int const&&, decltype(std::get<1>(std::move(p)))>::value, "");
    static_assert(noexcept(std::get<1>(std::move(p))), "");
#endif
    }

#if TEST_STD_VER > 11
    {
    typedef std::pair<int, short> P;
    constexpr const P p1(3, static_cast<short>(4));
    static_assert(std::get<0>(std::move(p1)) == 3, "");
    static_assert(std::get<1>(std::move(p1)) == 4, "");
    }
#endif

  return 0;
}
