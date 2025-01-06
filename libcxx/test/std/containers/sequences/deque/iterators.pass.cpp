//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// Test nested types and default template args:

// template <class T, class Allocator = allocator<T> >
// class deque;

// iterator, const_iterator

#include <deque>
#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef std::deque<int> C;
    C c;
    C::iterator i;
    i = c.begin();
    C::const_iterator j;
    j = c.cbegin();
    assert(i == j);
    }
#if TEST_STD_VER >= 11
    {
    typedef std::deque<int, min_allocator<int>> C;
    C c;
    C::iterator i;
    i = c.begin();
    C::const_iterator j;
    j = c.cbegin();

    assert(i == j);
    assert(!(i != j));

    assert(!(i < j));
    assert((i <= j));

    assert(!(i > j));
    assert((i >= j));

#  if TEST_STD_VER >= 20
    // P1614 + LWG3352
    // When the allocator does not have operator<=> then the iterator uses a
    // fallback to provide operator<=>.
    // Make sure to test with an allocator that does not have operator<=>.
    static_assert(!std::three_way_comparable<min_allocator<int>, std::strong_ordering>);
    static_assert(std::three_way_comparable<typename C::iterator, std::strong_ordering>);

    std::same_as<std::strong_ordering> decltype(auto) r1 = i <=> j;
    assert(r1 == std::strong_ordering::equal);
#  endif
    }
#endif
#if TEST_STD_VER > 11
    { // N3644 testing
        std::deque<int>::iterator ii1{}, ii2{};
        std::deque<int>::iterator ii4 = ii1;
        std::deque<int>::const_iterator cii{};
        assert ( ii1 == ii2 );
        assert ( ii1 == ii4 );

        assert (!(ii1 != ii2 ));

        assert ( (ii1 == cii ));
        assert ( (cii == ii1 ));
        assert (!(ii1 != cii ));
        assert (!(cii != ii1 ));
        assert (!(ii1 <  cii ));
        assert (!(cii <  ii1 ));
        assert ( (ii1 <= cii ));
        assert ( (cii <= ii1 ));
        assert (!(ii1 >  cii ));
        assert (!(cii >  ii1 ));
        assert ( (ii1 >= cii ));
        assert ( (cii >= ii1 ));
        assert (cii - ii1 == 0);
        assert (ii1 - cii == 0);

//         std::deque<int> c;
//         assert ( ii1 != c.cbegin());
//         assert ( cii != c.begin());
//         assert ( cii != c.cend());
//         assert ( ii1 != c.end());

#  if TEST_STD_VER >= 20
        // P1614 + LWG3352
        std::same_as<std::strong_ordering> decltype(auto) r1 = ii1 <=> ii2;
        assert(r1 == std::strong_ordering::equal);

        std::same_as<std::strong_ordering> decltype(auto) r2 = cii <=> ii2;
        assert(r2 == std::strong_ordering::equal);
#  endif // TEST_STD_VER > 20
    }
#endif

  return 0;
}
