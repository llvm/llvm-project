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

        C c;
        assert ( c.begin()   == std::begin(c));
        assert ( c.cbegin()  == std::cbegin(c));
        assert ( c.rbegin()  == std::rbegin(c));
        assert ( c.crbegin() == std::crbegin(c));
        assert ( c.end()     == std::end(c));
        assert ( c.cend()    == std::cend(c));
        assert ( c.rend()    == std::rend(c));
        assert ( c.crend()   == std::crend(c));

        assert ( std::begin(c)   != std::end(c));
        assert ( std::rbegin(c)  != std::rend(c));
        assert ( std::cbegin(c)  != std::cend(c));
        assert ( std::crbegin(c) != std::crend(c));
        }
        {
        typedef std::array<int, 0> C;
        C::iterator ii1{}, ii2{};
        C::iterator ii4 = ii1;
        C::const_iterator cii{};
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

        C c;
        assert ( c.begin()   == std::begin(c));
        assert ( c.cbegin()  == std::cbegin(c));
        assert ( c.rbegin()  == std::rbegin(c));
        assert ( c.crbegin() == std::crbegin(c));
        assert ( c.end()     == std::end(c));
        assert ( c.cend()    == std::cend(c));
        assert ( c.rend()    == std::rend(c));
        assert ( c.crend()   == std::crend(c));

        assert ( std::begin(c)   == std::end(c));
        assert ( std::rbegin(c)  == std::rend(c));
        assert ( std::cbegin(c)  == std::cend(c));
        assert ( std::crbegin(c) == std::crend(c));
        }
    }
#endif
}
