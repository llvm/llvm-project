//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// reference front();       // constexpr in C++17
// reference back();        // constexpr in C++17
// const_reference front(); // constexpr in C++14
// const_reference back();  // constexpr in C++14

#include <array>
#include <cassert>

#include "test_macros.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

#if TEST_STD_VER > 14
constexpr bool check_front( double val )
{
    std::array<double, 3> arr = {1, 2, 3.5};
    return arr.front() == val;
}

constexpr bool check_back( double val )
{
    std::array<double, 3> arr = {1, 2, 3.5};
    return arr.back() == val;
}
#endif

int main(int, char**)
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c = {1, 2, 3.5};

        C::reference r1 = c.front();
        assert(r1 == 1);
        r1 = 5.5;
        assert(c[0] == 5.5);

        C::reference r2 = c.back();
        assert(r2 == 3.5);
        r2 = 7.5;
        assert(c[2] == 7.5);
    }
    {
        typedef double T;
        typedef std::array<T, 3> C;
        const C c = {1, 2, 3.5};
        C::const_reference r1 = c.front();
        assert(r1 == 1);

        C::const_reference r2 = c.back();
        assert(r2 == 3.5);
    }
    {
      typedef double T;
      typedef std::array<T, 0> C;
      C c = {};
      C const& cc = c;
      ASSERT_SAME_TYPE(decltype( c.back()), typename C::reference);
      ASSERT_SAME_TYPE(decltype(cc.back()), typename C::const_reference);
      LIBCPP_ASSERT_NOEXCEPT(    c.back());
      LIBCPP_ASSERT_NOEXCEPT(   cc.back());
      ASSERT_SAME_TYPE(decltype( c.front()), typename C::reference);
      ASSERT_SAME_TYPE(decltype(cc.front()), typename C::const_reference);
      LIBCPP_ASSERT_NOEXCEPT(    c.front());
      LIBCPP_ASSERT_NOEXCEPT(   cc.front());
      if (c.size() > (0)) { // always false
        TEST_IGNORE_NODISCARD c.front();
        TEST_IGNORE_NODISCARD c.back();
        TEST_IGNORE_NODISCARD cc.front();
        TEST_IGNORE_NODISCARD cc.back();
      }
    }
    {
      typedef double T;
      typedef std::array<const T, 0> C;
      C c = {{}};
      C const& cc = c;
      ASSERT_SAME_TYPE(decltype( c.back()), typename C::reference);
      ASSERT_SAME_TYPE(decltype(cc.back()), typename C::const_reference);
      LIBCPP_ASSERT_NOEXCEPT(    c.back());
      LIBCPP_ASSERT_NOEXCEPT(   cc.back());
      ASSERT_SAME_TYPE(decltype( c.front()), typename C::reference);
      ASSERT_SAME_TYPE(decltype(cc.front()), typename C::const_reference);
      LIBCPP_ASSERT_NOEXCEPT(    c.front());
      LIBCPP_ASSERT_NOEXCEPT(   cc.front());
      if (c.size() > (0)) {
        TEST_IGNORE_NODISCARD c.front();
        TEST_IGNORE_NODISCARD c.back();
        TEST_IGNORE_NODISCARD cc.front();
        TEST_IGNORE_NODISCARD cc.back();
      }
    }
#if TEST_STD_VER > 11
    {
        typedef double T;
        typedef std::array<T, 3> C;
        constexpr C c = {1, 2, 3.5};

        constexpr T t1 = c.front();
        static_assert (t1 == 1, "");

        constexpr T t2 = c.back();
        static_assert (t2 == 3.5, "");
    }
#endif

#if TEST_STD_VER > 14
    {
        static_assert (check_front(1),   "");
        static_assert (check_back (3.5), "");
    }
#endif

  return 0;
}
