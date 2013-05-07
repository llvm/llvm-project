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
// class istream_iterator
//     : public iterator<input_iterator_tag, T, Distance, const T*, const T&>
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
    typedef std::istream_iterator<double> I1;
    static_assert((std::is_convertible<I1,
        std::iterator<std::input_iterator_tag, double, std::ptrdiff_t,
        const double*, const double&> >::value), "");
    static_assert((std::is_same<I1::char_type, char>::value), "");
    static_assert((std::is_same<I1::traits_type, std::char_traits<char> >::value), "");
    static_assert((std::is_same<I1::istream_type, std::istream>::value), "");
    typedef std::istream_iterator<unsigned, wchar_t> I2;
    static_assert((std::is_convertible<I2,
        std::iterator<std::input_iterator_tag, unsigned, std::ptrdiff_t,
        const unsigned*, const unsigned&> >::value), "");
    static_assert((std::is_same<I2::char_type, wchar_t>::value), "");
    static_assert((std::is_same<I2::traits_type, std::char_traits<wchar_t> >::value), "");
    static_assert((std::is_same<I2::istream_type, std::wistream>::value), "");
}
