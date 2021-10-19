//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: libcpp-has-no-localization

// <experimental/iterator>
//
// template <class _Delim, class _CharT = char, class _Traits = char_traits<_CharT>>
//   class ostream_joiner;
//
//     ostream_joiner(ostream_type& __os, _Delim&& __d);
//     ostream_joiner(ostream_type& __os, const _Delim& __d);

#include <experimental/iterator>
#include <iostream>
#include <string>

#include "test_macros.h"

namespace exper = std::experimental;

int main(int, char**) {
    const char eight = '8';
    const std::string nine = "9";
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    const std::wstring ten = L"10";
#endif
    const int eleven = 11;

    // Narrow streams w/rvalues
    { exper::ostream_joiner<char>         oj(std::cout, '8'); }
    { exper::ostream_joiner<std::string>  oj(std::cout, std::string("9")); }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    { exper::ostream_joiner<std::wstring> oj(std::cout, std::wstring(L"10")); }
#endif
    { exper::ostream_joiner<int>          oj(std::cout, 11); }

    // Narrow streams w/lvalues
    { exper::ostream_joiner<char>         oj(std::cout, eight); }
    { exper::ostream_joiner<std::string>  oj(std::cout, nine); }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    { exper::ostream_joiner<std::wstring> oj(std::cout, ten); }
#endif
    { exper::ostream_joiner<int>          oj(std::cout, eleven); }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    // Wide streams w/rvalues
    { exper::ostream_joiner<char, wchar_t>         oj(std::wcout, '8'); }
    { exper::ostream_joiner<std::string, wchar_t>  oj(std::wcout, std::string("9")); }
    { exper::ostream_joiner<std::wstring, wchar_t> oj(std::wcout, std::wstring(L"10")); }
    { exper::ostream_joiner<int, wchar_t>          oj(std::wcout, 11); }

    // Wide streams w/lvalues
    { exper::ostream_joiner<char, wchar_t>         oj(std::wcout, eight); }
    { exper::ostream_joiner<std::string, wchar_t>  oj(std::wcout, nine); }
    { exper::ostream_joiner<std::wstring, wchar_t> oj(std::wcout, ten); }
    { exper::ostream_joiner<int, wchar_t>          oj(std::wcout, eleven); }
#endif

  return 0;
}
