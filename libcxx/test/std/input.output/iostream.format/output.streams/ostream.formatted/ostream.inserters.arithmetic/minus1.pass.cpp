//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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
#include <cctype>
#include <cstdint>
#include <cassert>

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
    for (size_t i = 0; i < str.size(); ++i )
        str[i] = std::toupper(str[i]);

    assert(str == expected);
}

int main(int argc, char* argv[])
{

    test_octal<uint16_t>(                "177777");
    test_octal< int16_t>(                "177777");
    test_octal<uint32_t>(           "37777777777");
    test_octal< int32_t>(           "37777777777");
    test_octal<uint64_t>("1777777777777777777777");
    test_octal< int64_t>("1777777777777777777777");
    test_octal<uint64_t>("1777777777777777777777");
    if (sizeof(long) == sizeof(int64_t)) {
        test_octal< unsigned long>("1777777777777777777777");
        test_octal<          long>("1777777777777777777777");
    }
    if (sizeof(long long) == sizeof(int64_t)) {
        test_octal< unsigned long long>("1777777777777777777777");
        test_octal<          long long>("1777777777777777777777");
    }

    test_dec<uint16_t>(               "65535");
    test_dec< int16_t>(                  "-1");
    test_dec<uint32_t>(          "4294967295");
    test_dec< int32_t>(                  "-1");
    test_dec<uint64_t>("18446744073709551615");
    test_dec< int64_t>(                  "-1");
    if (sizeof(long) == sizeof(int64_t)) {
        test_dec<unsigned long>("18446744073709551615");
        test_dec<         long>(                  "-1");
    }
    if (sizeof(long long) == sizeof(int64_t)) {
        test_dec<unsigned long long>("18446744073709551615");
        test_dec<         long long>(                  "-1");
    }

    test_hex<uint16_t>(            "FFFF");
    test_hex< int16_t>(            "FFFF");
    test_hex<uint32_t>(        "FFFFFFFF");
    test_hex< int32_t>(        "FFFFFFFF");
    test_hex<uint64_t>("FFFFFFFFFFFFFFFF");
    test_hex< int64_t>("FFFFFFFFFFFFFFFF");
    if (sizeof(long) == sizeof(int64_t)) {
        test_hex<unsigned long>("FFFFFFFFFFFFFFFF");
        test_hex<         long>("FFFFFFFFFFFFFFFF");
    }
    if (sizeof(long long) == sizeof(int64_t)) {
        test_hex<unsigned long long>("FFFFFFFFFFFFFFFF");
        test_hex<         long long>("FFFFFFFFFFFFFFFF");
    }
}
