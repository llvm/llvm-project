//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// make_unsigned

#include <type_traits>

enum Enum {zero, one_};

enum BigEnum
{
    bzero,
    big = 0xFFFFFFFFFFFFFFFFULL
};

int main()
{
    static_assert((std::is_same<std::make_unsigned<signed char>::type, unsigned char>::value), "");
    static_assert((std::is_same<std::make_unsigned<unsigned char>::type, unsigned char>::value), "");
    static_assert((std::is_same<std::make_unsigned<char>::type, unsigned char>::value), "");
    static_assert((std::is_same<std::make_unsigned<short>::type, unsigned short>::value), "");
    static_assert((std::is_same<std::make_unsigned<unsigned short>::type, unsigned short>::value), "");
    static_assert((std::is_same<std::make_unsigned<int>::type, unsigned int>::value), "");
    static_assert((std::is_same<std::make_unsigned<unsigned int>::type, unsigned int>::value), "");
    static_assert((std::is_same<std::make_unsigned<long>::type, unsigned long>::value), "");
    static_assert((std::is_same<std::make_unsigned<unsigned long>::type, unsigned long>::value), "");
    static_assert((std::is_same<std::make_unsigned<long long>::type, unsigned long long>::value), "");
    static_assert((std::is_same<std::make_unsigned<unsigned long long>::type, unsigned long long>::value), "");
    static_assert((std::is_same<std::make_unsigned<wchar_t>::type, unsigned int>::value), "");
    static_assert((std::is_same<std::make_unsigned<const wchar_t>::type, const unsigned int>::value), "");
    static_assert((std::is_same<std::make_unsigned<const Enum>::type, const unsigned int>::value), "");
    static_assert((std::is_same<std::make_unsigned<BigEnum>::type,
                   std::conditional<sizeof(long) == 4, unsigned long long, unsigned long>::type>::value), "");
}
