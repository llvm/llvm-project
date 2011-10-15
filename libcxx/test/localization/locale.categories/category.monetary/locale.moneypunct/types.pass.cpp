//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class _CharT, bool _International = false>
// class moneypunct
//     : public locale::facet,
//       public money_base
// {
// public:
//     typedef _CharT                  char_type;
//     typedef basic_string<char_type> string_type;

#include <locale>
#include <type_traits>

int main()
{
    static_assert((std::is_base_of<std::locale::facet, std::moneypunct<char> >::value), "");
    static_assert((std::is_base_of<std::locale::facet, std::moneypunct<wchar_t> >::value), "");
    static_assert((std::is_base_of<std::money_base, std::moneypunct<char> >::value), "");
    static_assert((std::is_base_of<std::money_base, std::moneypunct<wchar_t> >::value), "");
    static_assert((std::is_same<std::moneypunct<char>::char_type, char>::value), "");
    static_assert((std::is_same<std::moneypunct<wchar_t>::char_type, wchar_t>::value), "");
    static_assert((std::is_same<std::moneypunct<char>::string_type, std::string>::value), "");
    static_assert((std::is_same<std::moneypunct<wchar_t>::string_type, std::wstring>::value), "");
}
