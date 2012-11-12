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

#include <string>
#include <cassert>
#include <type_traits>

template <class T>
void
test()
{
    typedef std::hash<T> H;
    static_assert((std::is_base_of<std::unary_function<T, std::size_t>,
                                   H>::value), "");
    H h;
    std::string g1 = "1234567890";
    std::string g2 = "1234567891";
    T s1(g1.begin(), g1.end());
    T s2(g2.begin(), g2.end());
    assert(h(s1) != h(s2));
}

int main()
{
    test<std::string>();
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<std::u16string>();
    test<std::u32string>();
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
    test<std::wstring>();
}
