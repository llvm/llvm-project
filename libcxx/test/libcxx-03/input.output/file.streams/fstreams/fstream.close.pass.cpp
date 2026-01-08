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

// close();

// Inspired by PR#38052 - std::fstream still good after closing and updating content

#include <fstream>
#include <cassert>
#include "test_macros.h"
#include "platform_support.h"

int main(int, char**)
{
    std::string temp = get_temp_file_name();

    std::fstream ofs(temp, std::ios::out | std::ios::trunc);
    ofs << "Hello, World!\n";
    assert( ofs.good());
    ofs.close();
    assert( ofs.good());
    ofs << "Hello, World!\n";
    assert(!ofs.good());

    std::remove(temp.c_str());

    return 0;
}
