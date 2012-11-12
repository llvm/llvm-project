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

#include <functional>
#include <cassert>
#include <type_traits>
#include <limits>

template <class T>
void
test()
{
    static_assert((std::is_base_of<std::unary_function<T, std::size_t>,
                                   std::hash<T> >::value), "");
    std::hash<T> h;
    for (int i = 0; i <= 5; ++i)
    {
        T t(i);
        if (sizeof(T) <= sizeof(std::size_t))
            assert(h(t) == t);
    }
}

int main()
{
    test<bool>();
    test<char>();
    test<signed char>();
    test<unsigned char>();
    test<char16_t>();
    test<char32_t>();
    test<wchar_t>();
    test<short>();
    test<unsigned short>();
    test<int>();
    test<unsigned int>();
    test<long>();
    test<unsigned long>();
    test<long long>();
    test<unsigned long long>();
}
