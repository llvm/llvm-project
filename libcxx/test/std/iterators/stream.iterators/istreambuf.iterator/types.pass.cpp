//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class charT, class traits = char_traits<charT> >
// class istreambuf_iterator
//  : public iterator<input_iterator_tag, charT, traits::off_type, unspecified, charT> // until C++17
// {
// public:
//     typedef input_iterator_tag             iterator_category;
//     typedef charT                          value_type;
//     typedef traits::off_type               difference_type;
//     typedef unspecified                    pointer;
//     typedef charT                          reference;
//
//     typedef charT                          char_type;
//     typedef traits                         traits_type;
//     typedef traits::int_type               int_type;
//     typedef basic_streambuf<charT, traits> streambuf_type;
//     typedef basic_istream<charT, traits>   istream_type;
//     ...
//
// All specializations of istreambuf_iterator shall have a trivial copy constructor,
//    a constexpr default constructor and a trivial destructor.

#include <iterator>
#include <string>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef std::istreambuf_iterator<char> I1;
#if TEST_STD_VER <= 14
    typedef std::iterator<std::input_iterator_tag, char, std::char_traits<char>::off_type, char*, char> iterator_base;
    static_assert((std::is_base_of<iterator_base, I1>::value), "");
#endif
    static_assert((std::is_same<I1::iterator_category, std::input_iterator_tag>::value), "");
    static_assert((std::is_same<I1::value_type, char>::value), "");
    static_assert((std::is_same<I1::difference_type, std::char_traits<char>::off_type>::value), "");
    LIBCPP_STATIC_ASSERT((std::is_same<I1::pointer, char*>::value), "");
    static_assert((std::is_same<I1::reference, char>::value), "");
    static_assert((std::is_same<I1::char_type, char>::value), "");
    static_assert((std::is_same<I1::traits_type, std::char_traits<char> >::value), "");
    static_assert((std::is_same<I1::int_type, I1::traits_type::int_type>::value), "");
    static_assert((std::is_same<I1::streambuf_type, std::streambuf>::value), "");
    static_assert((std::is_same<I1::istream_type, std::istream>::value), "");
    static_assert((std::is_nothrow_default_constructible<I1>::value), "" );
    static_assert((std::is_trivially_copy_constructible<I1>::value), "" );
    static_assert((std::is_trivially_destructible<I1>::value), "" );
    }

    {
    typedef std::istreambuf_iterator<wchar_t> I2;
#if TEST_STD_VER <= 14
    typedef std::iterator<std::input_iterator_tag, wchar_t, std::char_traits<wchar_t>::off_type, wchar_t*, wchar_t> iterator_base;
    static_assert((std::is_base_of<iterator_base, I2>::value), "");
#endif
    static_assert((std::is_same<I2::iterator_category, std::input_iterator_tag>::value), "");
    static_assert((std::is_same<I2::value_type, wchar_t>::value), "");
    static_assert((std::is_same<I2::difference_type, std::char_traits<wchar_t>::off_type>::value), "");
    LIBCPP_STATIC_ASSERT((std::is_same<I2::pointer, wchar_t*>::value), "");
    static_assert((std::is_same<I2::reference, wchar_t>::value), "");
    static_assert((std::is_same<I2::char_type, wchar_t>::value), "");
    static_assert((std::is_same<I2::traits_type, std::char_traits<wchar_t> >::value), "");
    static_assert((std::is_same<I2::int_type, I2::traits_type::int_type>::value), "");
    static_assert((std::is_same<I2::streambuf_type, std::wstreambuf>::value), "");
    static_assert((std::is_same<I2::istream_type, std::wistream>::value), "");
    static_assert((std::is_nothrow_default_constructible<I2>::value), "" );
    static_assert((std::is_trivially_copy_constructible<I2>::value), "" );
    static_assert((std::is_trivially_destructible<I2>::value), "" );
    }

  return 0;
}
