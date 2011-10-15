//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// common_type

#include <type_traits>

int main()
{
    static_assert((std::is_same<std::common_type<int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<char>::type, char>::value), "");

    static_assert((std::is_same<std::common_type<double, char>::type, double>::value), "");
    static_assert((std::is_same<std::common_type<short, char>::type, int>::value), "");

    static_assert((std::is_same<std::common_type<double, char, long long>::type, double>::value), "");
    static_assert((std::is_same<std::common_type<unsigned, char, long long>::type, long long>::value), "");
}
