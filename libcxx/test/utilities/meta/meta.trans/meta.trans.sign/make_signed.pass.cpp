//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// make_signed

#include <type_traits>

enum Enum {zero, one_};

enum BigEnum
{
    bzero,
    big = 0xFFFFFFFFFFFFFFFFULL
};

int main()
{
    static_assert((std::is_same<std::make_signed<signed char>::type, signed char>::value), "");
    static_assert((std::is_same<std::make_signed<unsigned char>::type, signed char>::value), "");
    static_assert((std::is_same<std::make_signed<char>::type, signed char>::value), "");
    static_assert((std::is_same<std::make_signed<short>::type, signed short>::value), "");
    static_assert((std::is_same<std::make_signed<unsigned short>::type, signed short>::value), "");
    static_assert((std::is_same<std::make_signed<int>::type, signed int>::value), "");
    static_assert((std::is_same<std::make_signed<unsigned int>::type, signed int>::value), "");
    static_assert((std::is_same<std::make_signed<long>::type, signed long>::value), "");
    static_assert((std::is_same<std::make_signed<unsigned long>::type, long>::value), "");
    static_assert((std::is_same<std::make_signed<long long>::type, signed long long>::value), "");
    static_assert((std::is_same<std::make_signed<unsigned long long>::type, signed long long>::value), "");
    static_assert((std::is_same<std::make_signed<wchar_t>::type, int>::value), "");
    static_assert((std::is_same<std::make_signed<const wchar_t>::type, const int>::value), "");
    static_assert((std::is_same<std::make_signed<const Enum>::type, const int>::value), "");
    static_assert((std::is_same<std::make_signed<BigEnum>::type,
                   std::conditional<sizeof(long) == 4, long long, long>::type>::value), "");
}
