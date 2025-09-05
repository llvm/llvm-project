//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This tests that swapping filebufs works correctly even when the small buffer
// optimization is in use (https://llvm.org/PR49938).

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_filebuf

// template <class charT, class traits>
// void
// swap(basic_filebuf<charT, traits>& x, basic_filebuf<charT, traits>& y);

#include <fstream>
#include <cassert>
#include "test_macros.h"
#include "platform_support.h"

int main(int, char**)
{
    std::string tmpA = get_temp_file_name();
    std::string tmpB = get_temp_file_name();

    {
        std::ofstream sa(tmpA), sb(tmpB);
        sa << "AAAA";
        sb << "BBBB";
    }

    std::filebuf f1;
    assert(f1.open(tmpA, std::ios_base::in) != 0);
    assert(f1.is_open());
    f1.pubsetbuf(0, 0);

    std::filebuf f2;
    assert(f2.open(tmpB, std::ios_base::in) != 0);
    assert(f2.is_open());
    f2.pubsetbuf(0, 0);

    assert(f1.sgetc() == 'A');
    assert(f2.sgetc() == 'B');

    swap(f1, f2);

    assert(f1.is_open());
    assert(f2.is_open());

    assert(f1.sgetc() == 'B');
    assert(f2.sgetc() == 'A');

    std::remove(tmpA.c_str());
    std::remove(tmpB.c_str());

    return 0;
}
