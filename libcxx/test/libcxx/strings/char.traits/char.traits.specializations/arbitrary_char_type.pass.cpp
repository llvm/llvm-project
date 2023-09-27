//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<T> for arbitrary T

// Non-standard but provided temporarily for users to migrate.

// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated

#include <string>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <class Char>
TEST_CONSTEXPR_CXX20 bool test() {
    static_assert(std::is_same<typename std::char_traits<Char>::char_type, Char>::value, "");
    static_assert(std::is_same<typename std::char_traits<Char>::int_type, int>::value, "");
    static_assert(std::is_same<typename std::char_traits<Char>::off_type, std::streamoff>::value, "");
    static_assert(std::is_same<typename std::char_traits<Char>::pos_type, std::streampos>::value, "");
    static_assert(std::is_same<typename std::char_traits<Char>::state_type, std::mbstate_t>::value, "");

    assert(std::char_traits<Char>::to_int_type(Char('a')) == Char('a'));
    assert(std::char_traits<Char>::to_int_type(Char('A')) == Char('A'));
    assert(std::char_traits<Char>::to_int_type(0) == 0);

    assert(std::char_traits<Char>::to_char_type(Char('a')) == Char('a'));
    assert(std::char_traits<Char>::to_char_type(Char('A')) == Char('A'));
    assert(std::char_traits<Char>::to_char_type(0) == 0);

    assert(std::char_traits<Char>::eof() == EOF);

    assert(std::char_traits<Char>::not_eof(Char('a')) == Char('a'));
    assert(std::char_traits<Char>::not_eof(Char('A')) == Char('A'));
    assert(std::char_traits<Char>::not_eof(0) == 0);
    assert(std::char_traits<Char>::not_eof(std::char_traits<Char>::eof()) !=
           std::char_traits<Char>::eof());

    assert(std::char_traits<Char>::lt(Char('\0'), Char('A')) == (Char('\0') < Char('A')));
    assert(std::char_traits<Char>::lt(Char('A'), Char('\0')) == (Char('A') < Char('\0')));
    assert(std::char_traits<Char>::lt(Char('a'), Char('a')) == (Char('a') < Char('a')));
    assert(std::char_traits<Char>::lt(Char('A'), Char('a')) == (Char('A') < Char('a')));
    assert(std::char_traits<Char>::lt(Char('a'), Char('A')) == (Char('a') < Char('A')));

    assert( std::char_traits<Char>::eq(Char('a'), Char('a')));
    assert(!std::char_traits<Char>::eq(Char('a'), Char('A')));

    assert( std::char_traits<Char>::eq_int_type(Char('a'), Char('a')));
    assert(!std::char_traits<Char>::eq_int_type(Char('a'), Char('A')));
    assert(!std::char_traits<Char>::eq_int_type(std::char_traits<Char>::eof(), Char('A')));
    assert( std::char_traits<Char>::eq_int_type(std::char_traits<Char>::eof(), std::char_traits<Char>::eof()));

    {
        Char s1[] = {1, 2, 3, 0};
        Char s2[] = {0};
        assert(std::char_traits<Char>::length(s1) == 3);
        assert(std::char_traits<Char>::length(s2) == 0);
    }

    {
        Char s1[] = {1, 2, 3};
        assert(std::char_traits<Char>::find(s1, 3, Char(1)) == s1);
        assert(std::char_traits<Char>::find(s1, 3, Char(2)) == s1+1);
        assert(std::char_traits<Char>::find(s1, 3, Char(3)) == s1+2);
        assert(std::char_traits<Char>::find(s1, 3, Char(4)) == 0);
        assert(std::char_traits<Char>::find(s1, 3, Char(0)) == 0);
        assert(std::char_traits<Char>::find(NULL, 0, Char(0)) == 0);
    }

    {
        Char s1[] = {1, 2, 3};
        Char s2[3] = {0};
        assert(std::char_traits<Char>::copy(s2, s1, 3) == s2);
        assert(s2[0] == Char(1));
        assert(s2[1] == Char(2));
        assert(s2[2] == Char(3));
        assert(std::char_traits<Char>::copy(NULL, s1, 0) == NULL);
        assert(std::char_traits<Char>::copy(s1, NULL, 0) == s1);
    }

    {
        Char s1[] = {1, 2, 3};
        assert(std::char_traits<Char>::move(s1, s1+1, 2) == s1);
        assert(s1[0] == Char(2));
        assert(s1[1] == Char(3));
        assert(s1[2] == Char(3));
        s1[2] = Char(0);
        assert(std::char_traits<Char>::move(s1+1, s1, 2) == s1+1);
        assert(s1[0] == Char(2));
        assert(s1[1] == Char(2));
        assert(s1[2] == Char(3));
        assert(std::char_traits<Char>::move(NULL, s1, 0) == NULL);
        assert(std::char_traits<Char>::move(s1, NULL, 0) == s1);
    }

    {
        Char s1[] = {0};
        assert(std::char_traits<Char>::compare(s1, s1, 0) == 0);
        assert(std::char_traits<Char>::compare(NULL, NULL, 0) == 0);

        Char s2[] = {1, 0};
        Char s3[] = {2, 0};
        assert(std::char_traits<Char>::compare(s2, s2, 1) == 0);
        assert(std::char_traits<Char>::compare(s2, s3, 1) < 0);
        assert(std::char_traits<Char>::compare(s3, s2, 1) > 0);
    }

    {
        Char s2[3] = {0};
        assert(std::char_traits<Char>::assign(s2, 3, Char(5)) == s2);
        assert(s2[0] == Char(5));
        assert(s2[1] == Char(5));
        assert(s2[2] == Char(5));
        assert(std::char_traits<Char>::assign(NULL, 0, Char(5)) == NULL);
    }

    {
        Char c = Char('\0');
        std::char_traits<Char>::assign(c, Char('a'));
        assert(c == Char('a'));
    }

    return true;
}

int main(int, char**) {
    test<unsigned char>();
    test<signed char>();
    test<unsigned long>();

#if TEST_STD_VER > 17
    static_assert(test<unsigned char>());
    static_assert(test<signed char>());
    static_assert(test<unsigned long>());
#endif

  return 0;
}
