//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// Test fails due to use of is_trivially_* trait.
// XFAIL: gcc-4.9

// template<class charT, class traits = char_traits<charT> >
// class istreambuf_iterator
//     : public iterator<input_iterator_tag, charT,
//                       typename traits::off_type, unspecified,
//                       charT>
// {
// public:
//     typedef charT                         char_type;
//     typedef traits                        traits_type;
//     typedef typename traits::int_type     int_type;
//     typedef basic_streambuf<charT,traits> streambuf_type;
//     typedef basic_istream<charT,traits>   istream_type;
//     ...
//
// All specializations of istreambuf_iterator shall have a trivial copy constructor,
//    a constexpr default constructor and a trivial destructor.

#include <iterator>
#include <string>
#include <type_traits>

int main()
{
    typedef std::istreambuf_iterator<char> I1;
    static_assert((std::is_convertible<I1,
        std::iterator<std::input_iterator_tag, char, std::char_traits<char>::off_type,
        char*, char> >::value), "");
    static_assert((std::is_same<I1::char_type, char>::value), "");
    static_assert((std::is_same<I1::traits_type, std::char_traits<char> >::value), "");
    static_assert((std::is_same<I1::int_type, I1::traits_type::int_type>::value), "");
    static_assert((std::is_same<I1::streambuf_type, std::streambuf>::value), "");
    static_assert((std::is_same<I1::istream_type, std::istream>::value), "");
    static_assert((std::is_nothrow_default_constructible<I1>::value), "" );
    static_assert((std::is_trivially_copy_constructible<I1>::value), "" );
    static_assert((std::is_trivially_destructible<I1>::value), "" );

    typedef std::istreambuf_iterator<wchar_t> I2;
    static_assert((std::is_convertible<I2,
        std::iterator<std::input_iterator_tag, wchar_t, std::char_traits<wchar_t>::off_type,
        wchar_t*, wchar_t> >::value), "");
    static_assert((std::is_same<I2::char_type, wchar_t>::value), "");
    static_assert((std::is_same<I2::traits_type, std::char_traits<wchar_t> >::value), "");
    static_assert((std::is_same<I2::int_type, I2::traits_type::int_type>::value), "");
    static_assert((std::is_same<I2::streambuf_type, std::wstreambuf>::value), "");
    static_assert((std::is_same<I2::istream_type, std::wistream>::value), "");
    static_assert((std::is_nothrow_default_constructible<I2>::value), "" );
    static_assert((std::is_trivially_copy_constructible<I2>::value), "" );
    static_assert((std::is_trivially_destructible<I2>::value), "" );
}
