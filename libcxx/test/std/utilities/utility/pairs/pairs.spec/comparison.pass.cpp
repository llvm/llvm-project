//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template <class T1, class T2, class U1, class U2> bool operator==(const pair<T1,T2>&, const pair<U1,U2>&);
// template <class T1, class T2, class U1, class U2> bool operator!=(const pair<T1,T2>&, const pair<U1,U2>&);
// template <class T1, class T2, class U1, class U2> bool operator< (const pair<T1,T2>&, const pair<U1,U2>&);
// template <class T1, class T2, class U1, class U2> bool operator> (const pair<T1,T2>&, const pair<U1,U2>&);
// template <class T1, class T2, class U1, class U2> bool operator>=(const pair<T1,T2>&, const pair<U1,U2>&);
// template <class T1, class T2, class U1, class U2> bool operator<=(const pair<T1,T2>&, const pair<U1,U2>&);

#include <utility>
#include <cassert>
#include <concepts>

#include "test_macros.h"

#if TEST_STD_VER >= 26

// Test SFINAE.

struct EqualityComparable {
  constexpr EqualityComparable(int value) : value_{value} {};

  friend constexpr bool operator==(const EqualityComparable&, const EqualityComparable&) noexcept = default;

  int value_;
};

static_assert(std::equality_comparable<EqualityComparable>);

static_assert(std::equality_comparable<std::pair<EqualityComparable, EqualityComparable>>);

struct NonComparable {};

static_assert(!std::equality_comparable<NonComparable>);

static_assert(!std::equality_comparable<std::pair<EqualityComparable, NonComparable>>);
static_assert(!std::equality_comparable<std::pair<NonComparable, EqualityComparable>>);

#endif // TEST_STD_VER >= 26

int main(int, char**)
{
    {
        typedef std::pair<int, short> P1;
        typedef std::pair<long, long> P2;
        P1 p1(3, static_cast<short>(4));
        P2 p2(3, 4);
        assert( (p1 == p2));
        assert(!(p1 != p2));
        assert(!(p1 <  p2));
        assert( (p1 <= p2));
        assert(!(p1 >  p2));
        assert( (p1 >= p2));
    }
    {
        typedef std::pair<int, short> P;
        P p1(3, static_cast<short>(4));
        P p2(3, static_cast<short>(4));
        assert( (p1 == p2));
        assert(!(p1 != p2));
        assert(!(p1 <  p2));
        assert( (p1 <= p2));
        assert(!(p1 >  p2));
        assert( (p1 >= p2));
    }
    {
        typedef std::pair<int, short> P;
        P p1(2, static_cast<short>(4));
        P p2(3, static_cast<short>(4));
        assert(!(p1 == p2));
        assert( (p1 != p2));
        assert( (p1 <  p2));
        assert( (p1 <= p2));
        assert(!(p1 >  p2));
        assert(!(p1 >= p2));
    }
    {
        typedef std::pair<int, short> P;
        P p1(3, static_cast<short>(2));
        P p2(3, static_cast<short>(4));
        assert(!(p1 == p2));
        assert( (p1 != p2));
        assert( (p1 <  p2));
        assert( (p1 <= p2));
        assert(!(p1 >  p2));
        assert(!(p1 >= p2));
    }
    {
        typedef std::pair<int, short> P;
        P p1(3, static_cast<short>(4));
        P p2(2, static_cast<short>(4));
        assert(!(p1 == p2));
        assert( (p1 != p2));
        assert(!(p1 <  p2));
        assert(!(p1 <= p2));
        assert( (p1 >  p2));
        assert( (p1 >= p2));
    }
    {
        typedef std::pair<int, short> P;
        P p1(3, static_cast<short>(4));
        P p2(3, static_cast<short>(2));
        assert(!(p1 == p2));
        assert( (p1 != p2));
        assert(!(p1 <  p2));
        assert(!(p1 <= p2));
        assert( (p1 >  p2));
        assert( (p1 >= p2));
    }

#if TEST_STD_VER > 11
    {
        typedef std::pair<int, short> P;
        constexpr P p1(3, static_cast<short>(4));
        constexpr P p2(3, static_cast<short>(2));
        static_assert(!(p1 == p2), "");
        static_assert( (p1 != p2), "");
        static_assert(!(p1 <  p2), "");
        static_assert(!(p1 <= p2), "");
        static_assert( (p1 >  p2), "");
        static_assert( (p1 >= p2), "");
    }
#endif

  return 0;
}
