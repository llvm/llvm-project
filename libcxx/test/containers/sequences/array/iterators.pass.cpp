//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <array>

// iterator, const_iterator

#include <array>
#include <iterator>
#include <cassert>

int main()
{
    {
    typedef std::array<int, 5> C;
    C c;
    C::iterator i;
    i = c.begin();
    C::const_iterator j;
    j = c.cbegin();
    assert(i == j);
    }
    {
    typedef std::array<int, 0> C;
    C c;
    C::iterator i;
    i = c.begin();
    C::const_iterator j;
    j = c.cbegin();
    assert(i == j);
    }

#if _LIBCPP_STD_VER > 11
    { // N3644 testing
        {
        typedef std::array<int, 5> C;
        C::iterator ii1{}, ii2{};
        C::iterator ii4 = ii1;
        C::const_iterator cii{};
        assert ( ii1 == ii2 );
        assert ( ii1 == ii4 );
        assert ( ii1 == cii );

        assert ( !(ii1 != ii2 ));
        assert ( !(ii1 != cii ));

//         C c;
//         assert ( ii1 != c.cbegin());
//         assert ( cii != c.begin());
//         assert ( cii != c.cend());
//         assert ( ii1 != c.end());
        }
        {
        typedef std::array<int, 0> C;
        C::iterator ii1{}, ii2{};
        C::iterator ii4 = ii1;
        C::const_iterator cii{};
        assert ( ii1 == ii2 );
        assert ( ii1 == ii4 );
        assert ( ii1 == cii );

        assert ( !(ii1 != ii2 ));
        assert ( !(ii1 != cii ));

//         C c;
//         assert ( ii1 != c.cbegin());
//         assert ( cii != c.begin());
//         assert ( cii != c.cend());
//         assert ( ii1 != c.end());
        }
    }
#endif
}
