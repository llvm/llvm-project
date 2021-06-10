//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: gcc-10
// <optional>

// template <class U>
//   optional(optional<U>&& rhs);

#include <optional>
#include <type_traits>
#include <memory>
#include <cassert>

#include "test_macros.h"

using std::optional;

template <class T, class U>
TEST_CONSTEXPR_CXX20 void
test(optional<U>&& rhs, bool is_going_to_throw = false)
{
    bool rhs_engaged = static_cast<bool>(rhs);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        optional<T> lhs = std::move(rhs);
        assert(is_going_to_throw == false);
        assert(static_cast<bool>(lhs) == rhs_engaged);
    }
    catch (int i)
    {
        assert(i == 6);
    }
#else
    if (is_going_to_throw) return;
    optional<T> lhs = std::move(rhs);
    assert(static_cast<bool>(lhs) == rhs_engaged);
#endif
}

class X
{
    int i_;
public:
    TEST_CONSTEXPR_CXX20 X(int i) : i_(i) {}
    TEST_CONSTEXPR_CXX20 X(X&& x) : i_(std::exchange(x.i_, 0)) {}
    TEST_CONSTEXPR_CXX20 ~X() {i_ = 0;}
    friend constexpr bool operator==(const X& x, const X& y) {return x.i_ == y.i_;}
};

struct Z
{
    Z(int) { TEST_THROW(6); }
};

template<class T, class U>
TEST_CONSTEXPR_CXX20 bool test_all()
{
    {
        optional<T> rhs;
        test<U>(std::move(rhs));
    }
    {
        optional<T> rhs(short{3});
        test<U>(std::move(rhs));
    }
    return true;
}

int main(int, char**)
{
    test_all<short, int>();
    test_all<int, X>();
#if TEST_STD_VER > 17
    static_assert(test_all<short, int>());
    static_assert(test_all<int, X>());
#endif
    {
        optional<int> rhs;
        test<Z>(std::move(rhs));
    }
    {
        optional<int> rhs(3);
        test<Z>(std::move(rhs), true);
    }

    static_assert(!(std::is_constructible<optional<X>, optional<Z>>::value), "");

  return 0;
}
