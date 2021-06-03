//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <class T, class charT = char, class traits = char_traits<charT>,
//           class Distance = ptrdiff_t>
// class ostream_iterator
//  : public iterator<output_iterator_tag, void, void, void, void> // until C++17
// {
// public:
//     typedef output_iterator_tag          iterator_category;
//     typedef void                         value_type;
//     typedef void                         difference_type; // until C++20
//     typedef ptrdiff_t                    difference_type; // since C++20
//     typedef void                         pointer;
//     typedef void                         reference;
//
//     typedef charT                        char_type;
//     typedef traits                       traits_type;
//     typedef basic_ostream<charT, traits> ostream_type;
//     ...

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef std::ostream_iterator<double> I1;
#if TEST_STD_VER <= 14
    typedef std::iterator<std::output_iterator_tag, void, void, void, void> iterator_base;
    static_assert((std::is_base_of<iterator_base, I1>::value), "");
#endif
    static_assert((std::is_same<I1::iterator_category, std::output_iterator_tag>::value), "");
    static_assert((std::is_same<I1::value_type, void>::value), "");
#if TEST_STD_VER <= 17
    static_assert((std::is_same<I1::difference_type, void>::value), "");
#else
    static_assert((std::is_same<I1::difference_type, std::ptrdiff_t>::value), "");
#endif
    static_assert((std::is_same<I1::pointer, void>::value), "");
    static_assert((std::is_same<I1::reference, void>::value), "");
    static_assert((std::is_same<I1::char_type, char>::value), "");
    static_assert((std::is_same<I1::traits_type, std::char_traits<char> >::value), "");
    static_assert((std::is_same<I1::ostream_type, std::ostream>::value), "");
    }

    {
    typedef std::ostream_iterator<unsigned, wchar_t> I2;
#if TEST_STD_VER <= 14
    typedef std::iterator<std::output_iterator_tag, void, void, void, void> iterator_base;
    static_assert((std::is_base_of<iterator_base, I2>::value), "");
#endif
    static_assert((std::is_same<I2::iterator_category, std::output_iterator_tag>::value), "");
    static_assert((std::is_same<I2::value_type, void>::value), "");
#if TEST_STD_VER <= 17
    static_assert((std::is_same<I2::difference_type, void>::value), "");
#else
    static_assert((std::is_same<I2::difference_type, std::ptrdiff_t>::value), "");
#endif
    static_assert((std::is_same<I2::pointer, void>::value), "");
    static_assert((std::is_same<I2::reference, void>::value), "");
    static_assert((std::is_same<I2::char_type, wchar_t>::value), "");
    static_assert((std::is_same<I2::traits_type, std::char_traits<wchar_t> >::value), "");
    static_assert((std::is_same<I2::ostream_type, std::wostream>::value), "");
    }

  return 0;
}
