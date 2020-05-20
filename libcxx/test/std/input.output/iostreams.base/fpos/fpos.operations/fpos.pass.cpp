//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test_macros.h"

#include <string>
#include <cassert>
#include <type_traits>

// <string>

// template <class stateT>
// class fpos;

template<class T, class = void>
struct is_equality_comparable : std::false_type { };

template<class T>
struct is_equality_comparable
<T, typename std::enable_if<true, decltype(std::declval<T const&>() == std::declval<T const&>(),
                                           (void)0)>::type
> : std::true_type { };

template<class T>
void test_traits()
{
    static_assert(std::is_default_constructible <std::fpos<T> >::value, "");
    static_assert(std::is_copy_constructible    <std::fpos<T> >::value, "");
    static_assert(std::is_copy_assignable       <std::fpos<T> >::value, "");
    static_assert(std::is_destructible          <std::fpos<T> >::value, "");
    static_assert(is_equality_comparable        <std::fpos<T> >::value, "");

    static_assert(std::is_trivially_copy_constructible<T>::value ==
                  std::is_trivially_copy_constructible<std::fpos<T> >::value, "");
    static_assert(std::is_trivially_copy_assignable<T>::value ==
                  std::is_trivially_copy_assignable<std::fpos<T> >::value, "");
    static_assert(std::is_trivially_destructible<T>::value ==
                  std::is_trivially_destructible<std::fpos<T> >::value, "");
}

struct Foo { };

int main(int, char**)
{
    test_traits<std::mbstate_t>();
    test_traits<int>();
    test_traits<Foo>();

    // Position type requirements table 106 (in order):

    std::streampos p1(42);
    std::streamoff o1(p1);

    {
        assert(o1 == 42);
    }
    {
        std::streampos p2(42);
        std::streampos q1(43);
        std::streampos const p3(44);
        std::streampos const q2(45);
        assert(p2 != q1);
        assert(p3 != q2);
        assert(p2 != q2);
        assert(p3 != q1);
    }
    {
        std::streampos p2 = p1 + o1;
        assert(p2 == 84);
    }
    {
        std::streampos& p2 = p1 += o1;
        assert(p2 == 84);
        assert(p1 == 84);
    }
    {
        std::streampos p2 = p1 - o1;
        assert(p2 == 42);
    }
    {
        std::streampos& p2 = p1 -= o1;
        assert(p2 == 42);
        assert(p1 == 42);
    }
    {
        std::streampos p2 = o1 + p1;
        assert(p2 == 84);
    }
    {
        std::streampos q1(42);
        std::streamoff o2 = q1 - p1;
        assert(o2 == 0);
    }

    return 0;
}
