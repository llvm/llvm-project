//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// template <class T>
// struct hash
//     : public unary_function<T, size_t>
// {
//     size_t operator()(T val) const;
// };

// Not very portable

#include <string_view>
#include <cassert>
#include <type_traits>

using std::string_view;

template <class T>
void
test()
{
    typedef std::hash<T> H;
    static_assert((std::is_same<typename H::argument_type, T>::value), "" );
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "" );
    H h;
//     std::string g1 = "1234567890";
//     std::string g2 = "1234567891";
    typedef typename T::value_type char_type;
    char_type g1 [ 10 ];
    char_type g2 [ 10 ];
    for ( int i = 0; i < 10; ++i )
        g1[i] = g2[9-i] = '0' + i;
    T s1(g1, 10);
    T s2(g2, 10);
    assert(h(s1) != h(s2));
}

int main()
{
    test<std::string_view>();
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<std::u16string_view>();
    test<std::u32string_view>();
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
    test<std::wstring_view>();
}
