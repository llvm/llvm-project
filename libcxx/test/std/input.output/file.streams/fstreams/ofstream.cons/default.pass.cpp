//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ofstream

// basic_ofstream();

#include <fstream>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::ofstream fs;
    }
    {
        std::wofstream fs;
    }

  return 0;
}
