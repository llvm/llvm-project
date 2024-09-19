//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_fstream

// basic_fstream();

#include <fstream>

#include "test_macros.h"
#include "operator_hijacker.h"

int main(int, char**)
{
    {
        std::fstream fs;
    }
    {
      std::basic_fstream<char, operator_hijacker_char_traits<char> > fs;
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wfstream fs;
    }
    {
      std::basic_fstream<wchar_t, operator_hijacker_char_traits<wchar_t> > fs;
    }
#endif

  return 0;
}
