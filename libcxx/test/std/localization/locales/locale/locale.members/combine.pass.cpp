//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: availability-char8_t_support-missing

// This test runs in C++20, but we have deprecated codecvt<char(16|32), char, mbstate_t> in C++20.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <locale>

// template <class Facet> locale combine(const locale& other) const;

#include <locale>
#include <stdexcept>
#include <cassert>

#include "count_new.h"

#include "test_macros.h"

template<class CharT>
void check_for(const std::locale& loc)
{
    assert(std::has_facet<std::collate<CharT> >(loc));

    assert(std::has_facet<std::ctype<CharT> >(loc));

    assert((std::has_facet<std::codecvt<CharT, char, std::mbstate_t> >(loc)));

    assert(std::has_facet<std::moneypunct<CharT> >(loc));
    assert(std::has_facet<std::money_get<CharT> >(loc));
    assert(std::has_facet<std::money_put<CharT> >(loc));

    assert(std::has_facet<std::numpunct<CharT> >(loc));
    assert(std::has_facet<std::num_get<CharT> >(loc));
    assert(std::has_facet<std::num_put<CharT> >(loc));

    assert(std::has_facet<std::time_get<CharT> >(loc));
    assert(std::has_facet<std::time_put<CharT> >(loc));

    assert(std::has_facet<std::messages<CharT> >(loc));
}

void check(const std::locale& loc)
{
    check_for<char>(loc);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    check_for<wchar_t>(loc);
#endif

    assert((std::has_facet<std::codecvt<char16_t, char, std::mbstate_t> >(loc)));
    assert((std::has_facet<std::codecvt<char32_t, char, std::mbstate_t> >(loc)));
#if TEST_STD_VER > 17
    assert((std::has_facet<std::codecvt<char16_t, char8_t, std::mbstate_t> >(loc)));
    assert((std::has_facet<std::codecvt<char32_t, char8_t, std::mbstate_t> >(loc)));
#endif
}

struct my_facet
    : public std::locale::facet
{
    int test() const {return 5;}

    static std::locale::id id;
};

std::locale::id my_facet::id;

int main(int, char**)
{
{
    globalMemCounter.reset();
    {
        std::locale loc;
        std::locale loc2(loc, new my_facet);
        std::locale loc3 = loc.combine<my_facet>(loc2);
        check(loc3);
        assert(loc3.name() == "*");
        assert((std::has_facet<my_facet>(loc3)));
        const my_facet& f = std::use_facet<my_facet>(loc3);
        assert(f.test() == 5);
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
}
#ifndef TEST_HAS_NO_EXCEPTIONS
{
    {
        std::locale loc;
        std::locale loc2;
        try
        {
            std::locale loc3 = loc.combine<my_facet>(loc2);
            assert(false);
        }
        catch (std::runtime_error&)
        {
        }
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
}
#endif

  return 0;
}
