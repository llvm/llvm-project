//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <class T, class charT = char, class traits = char_traits<charT>,
//           class Distance = ptrdiff_t>
// class ostream_iterator
//     : public iterator<output_iterator_tag, void, void, void, void>
// {
// public:
//     typedef charT char_type;
//     typedef traits traits_type;
//     typedef basic_istream<charT,traits> istream_type;
//     ...

#include <iterator>
#include <type_traits>

int main()
{
    typedef std::ostream_iterator<double> I1;
    static_assert((std::is_convertible<I1,
        std::iterator<std::output_iterator_tag, void, void, void, void> >::value), "");
    static_assert((std::is_same<I1::char_type, char>::value), "");
    static_assert((std::is_same<I1::traits_type, std::char_traits<char> >::value), "");
    static_assert((std::is_same<I1::ostream_type, std::ostream>::value), "");
    typedef std::ostream_iterator<unsigned, wchar_t> I2;
    static_assert((std::is_convertible<I2,
        std::iterator<std::output_iterator_tag, void, void, void, void> >::value), "");
    static_assert((std::is_same<I2::char_type, wchar_t>::value), "");
    static_assert((std::is_same<I2::traits_type, std::char_traits<wchar_t> >::value), "");
    static_assert((std::is_same<I2::ostream_type, std::wostream>::value), "");
}
