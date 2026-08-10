//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class time_base
// {
// public:
//     enum dateorder {no_order, dmy, mdy, ymd, ydm};
// };
//
// template <class charT, class InputIterator = istreambuf_iterator<charT> >
// class time_get
//     : public locale::facet,
//       public time_base
// {
// public:
//     typedef charT         char_type;
//     typedef InputIterator iter_type;

#include <locale>
#include <iterator>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_base_of<std::locale::facet, std::time_get<char> >::value), "");
    static_assert((std::is_base_of<std::locale::facet, std::time_get<wchar_t> >::value), "");
    static_assert((std::is_base_of<std::time_base, std::time_get<char> >::value), "");
    static_assert((std::is_base_of<std::time_base, std::time_get<wchar_t> >::value), "");
    static_assert((std::is_same<std::time_get<char>::char_type, char>::value), "");
    static_assert((std::is_same<std::time_get<wchar_t>::char_type, wchar_t>::value), "");
    static_assert((std::is_same<std::time_get<char>::iter_type, std::istreambuf_iterator<char> >::value), "");
    static_assert((std::is_same<std::time_get<wchar_t>::iter_type, std::istreambuf_iterator<wchar_t> >::value), "");

  return 0;
}
