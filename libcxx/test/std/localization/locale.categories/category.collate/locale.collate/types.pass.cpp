//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT>
// class collate
//     : public locale::facet {
// public:
//     typedef charT char_type;
//     typedef basic_string<charT>string_type;
//
//     static locale::id id;
// };

#include <locale>
#include <type_traits>
#include <cassert>

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        assert(std::has_facet<std::collate<char> >(l));
        const std::collate<char>& f = std::use_facet<std::collate<char> >(l);
        ((void)f); // Prevent unused warning
        {
            (void)std::collate<char>::id;
        }
        static_assert((std::is_same<std::collate<char>::char_type, char>::value), "");
        static_assert((std::is_same<std::collate<char>::string_type, std::string>::value), "");
        static_assert((std::is_base_of<std::locale::facet, std::collate<char> >::value), "");
    }
    {
        assert(std::has_facet<std::collate<wchar_t> >(l));
        const std::collate<wchar_t>& f = std::use_facet<std::collate<wchar_t> >(l);
        ((void)f); // Prevent unused warning
        {
            (void)std::collate<wchar_t>::id;
        }
        static_assert((std::is_same<std::collate<wchar_t>::char_type, wchar_t>::value), "");
        static_assert((std::is_same<std::collate<wchar_t>::string_type, std::wstring>::value), "");
        static_assert((std::is_base_of<std::locale::facet, std::collate<wchar_t> >::value), "");
    }

  return 0;
}
