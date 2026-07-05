//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ostream>

// template <class charT, class traits = char_traits<charT> >
//   class basic_ostream;

// operator<<( int16_t val);
// operator<<(uint16_t val);
// operator<<( int32_t val);
// operator<<(uint32_t val);
// operator<<( int64_t val);
// operator<<(uint64_t val);

//  Testing to make sure that the max length values are correctly inserted

#include <sstream>
#include <ios>
#include <type_traits>
#include <cctype>
#include <cstdint>
#include <cassert>

#include "test_macros.h"

template <typename T>
void test_octal(const char *expected)
{
    std::stringstream ss;
    ss << std::oct << static_cast<T>(-1);
    assert(ss.str() == expected);
}

template <typename T>
void test_dec(const char *expected)
{
    std::stringstream ss;
    ss << std::dec << static_cast<T>(-1);
    assert(ss.str() == expected);
}

template <typename T>
void test_hex(const char *expected)
{
    std::stringstream ss;
    ss << std::hex << static_cast<T>(-1);

    std::string str = ss.str();
    for (std::size_t i = 0; i < str.size(); ++i )
        str[i] = static_cast<char>(std::toupper(str[i]));

    assert(str == expected);
}

int main(int, char**)
{

    test_octal<std::uint16_t>(                "177777");
    test_octal< std::int16_t>(                "177777");
    test_octal<std::uint32_t>(           "37777777777");
    test_octal< std::int32_t>(           "37777777777");
    test_octal<std::uint64_t>("1777777777777777777777");
    test_octal< std::int64_t>("1777777777777777777777");
    test_octal<std::uint64_t>("1777777777777777777777");

    const bool long_is_64 = std::integral_constant<bool, sizeof(long) == sizeof(std::int64_t)>::value; // avoid compiler warnings
    const bool long_long_is_64 = std::integral_constant<bool, sizeof(long long) == sizeof(std::int64_t)>::value; // avoid compiler warnings

    if (long_is_64) {
        test_octal< unsigned long>("1777777777777777777777");
        test_octal<          long>("1777777777777777777777");
    }
    if (long_long_is_64) {
        test_octal< unsigned long long>("1777777777777777777777");
        test_octal<          long long>("1777777777777777777777");
    }

    test_dec<std::uint16_t>(               "65535");
    test_dec< std::int16_t>(                  "-1");
    test_dec<std::uint32_t>(          "4294967295");
    test_dec< std::int32_t>(                  "-1");
    test_dec<std::uint64_t>("18446744073709551615");
    test_dec< std::int64_t>(                  "-1");
    if (long_is_64) {
        test_dec<unsigned long>("18446744073709551615");
        test_dec<         long>(                  "-1");
    }
    if (long_long_is_64) {
        test_dec<unsigned long long>("18446744073709551615");
        test_dec<         long long>(                  "-1");
    }

    test_hex<std::uint16_t>(            "FFFF");
    test_hex< std::int16_t>(            "FFFF");
    test_hex<std::uint32_t>(        "FFFFFFFF");
    test_hex< std::int32_t>(        "FFFFFFFF");
    test_hex<std::uint64_t>("FFFFFFFFFFFFFFFF");
    test_hex< std::int64_t>("FFFFFFFFFFFFFFFF");
    if (long_is_64) {
        test_hex<unsigned long>("FFFFFFFFFFFFFFFF");
        test_hex<         long>("FFFFFFFFFFFFFFFF");
    }
    if (long_long_is_64) {
        test_hex<unsigned long long>("FFFFFFFFFFFFFFFF");
        test_hex<         long long>("FFFFFFFFFFFFFFFF");
    }

  return 0;
}
