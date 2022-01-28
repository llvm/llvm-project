//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <string>

// iterator       begin();
// iterator       end();
// const_iterator begin()  const;
// const_iterator end()    const;
// const_iterator cbegin() const;
// const_iterator cend()   const;

#include <string>
#include <cassert>

#include "test_macros.h"

template<class C>
TEST_CONSTEXPR_CXX20 void test()
{
    { // N3644 testing
        typename C::iterator ii1{}, ii2{};
        typename C::iterator ii4 = ii1;
        typename C::const_iterator cii{};

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
    }
    {
        C a;
        typename C::iterator i1 = a.begin();
        typename C::iterator i2;
        i2 = i1;
        assert ( i1 == i2 );
    }
}

TEST_CONSTEXPR_CXX20 bool test() {
    test<std::string>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::wstring>();
#endif

#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    test<std::u8string>();
#endif

    test<std::u16string>();
    test<std::u32string>();

    return true;
}

int main(int, char**)
{
    test();
#if defined(__cpp_lib_constexpr_string) && __cpp_lib_constexpr_string >= 201907L
    static_assert(test());
#endif
    return 0;
}
