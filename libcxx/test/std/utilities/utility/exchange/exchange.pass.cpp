//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <utility>

// exchange

// template<class T, class U=T>
//    constexpr T            // constexpr after C++17
//    exchange(T& obj, U&& new_value)
//      noexcept(is_nothrow_move_constructible<T>::value && is_nothrow_assignable<T&, U>::value);

#include <utility>
#include <cassert>
#include <string>

#include "test_macros.h"

#if TEST_STD_VER > 17
TEST_CONSTEXPR bool test_constexpr() {
    int v = 12;

    if (12 != std::exchange(v,23) || v != 23)
        return false;

    if (23 != std::exchange(v,static_cast<short>(67)) || v != 67)
        return false;

    if (67 != std::exchange<int, short>(v, {}) || v != 0)
        return false;
    return true;
    }
#endif

template<bool Move, bool Assign>
struct TestNoexcept {
    TestNoexcept() = default;
    TestNoexcept(const TestNoexcept&);
    TestNoexcept(TestNoexcept&&) noexcept(Move);
    TestNoexcept& operator=(const TestNoexcept&);
    TestNoexcept& operator=(TestNoexcept&&) noexcept(Assign);
};

constexpr bool test_noexcept() {
  {
    int x = 42;
    ASSERT_NOEXCEPT(std::exchange(x, 42));
  }
  {
    TestNoexcept<true, true> x;
    ASSERT_NOEXCEPT(std::exchange(x, std::move(x)));
    ASSERT_NOT_NOEXCEPT(std::exchange(x, x)); // copy-assignment is not noexcept
  }
  {
    TestNoexcept<true, false> x;
    ASSERT_NOT_NOEXCEPT(std::exchange(x, std::move(x)));
  }
  {
    TestNoexcept<false, true> x;
    ASSERT_NOT_NOEXCEPT(std::exchange(x, std::move(x)));
  }

  return true;
}

int main(int, char**)
{
    {
    int v = 12;
    assert ( std::exchange ( v, 23 ) == 12 );
    assert ( v == 23 );
    assert ( std::exchange ( v, static_cast<short>(67) ) == 23 );
    assert ( v == 67 );

    assert ((std::exchange<int, short> ( v, {} )) == 67 );
    assert ( v == 0 );

    }

    {
    bool b = false;
    assert ( !std::exchange ( b, true ));
    assert ( b );
    }

    {
    const std::string s1 ( "Hi Mom!" );
    const std::string s2 ( "Yo Dad!" );
    std::string s3 = s1; // Mom
    assert ( std::exchange ( s3, s2 ) == s1 );
    assert ( s3 == s2 );
    assert ( std::exchange ( s3, "Hi Mom!" ) == s2 );
    assert ( s3 == s1 );

    s3 = s2; // Dad
    assert ( std::exchange ( s3, {} ) == s2 );
    assert ( s3.size () == 0 );

    s3 = s2; // Dad
    assert ( std::exchange ( s3, "" ) == s2 );
    assert ( s3.size () == 0 );
    }

#if TEST_STD_VER > 17
    static_assert(test_constexpr());
#endif

    static_assert(test_noexcept(), "");

  return 0;
}
