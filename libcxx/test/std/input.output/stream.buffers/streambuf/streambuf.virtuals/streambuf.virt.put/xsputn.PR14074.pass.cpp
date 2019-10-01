//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <streambuf>

// template <class charT, class traits = char_traits<charT> >
// class basic_streambuf;

// streamsize xsputn(const char_type* s, streamsize n);

// Test https://bugs.llvm.org/show_bug.cgi?id=14074. The bug is really inside
// basic_streambuf, but I can't seem to reproduce without going through one
// of its derived classes.

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include "test_macros.h"
#include "platform_support.h"


// Count the number of bytes in a file -- make sure to use only functionality
// provided by the C library to avoid relying on the C++ library, which we're
// trying to test.
static std::size_t count_bytes(char const* filename) {
    std::FILE* f = std::fopen(filename, "rb");
    std::size_t count = 0;
    while (std::fgetc(f) != EOF)
        ++count;
    std::fclose(f);
    return count;
}

int main(int, char**) {
    {
        // with basic_stringbuf
        std::basic_stringbuf<char> buf;
        std::streamsize sz = buf.sputn("\xFF", 1);
        assert(sz == 1);
        assert(buf.str().size() == 1);
    }
    {
        // with basic_filebuf
        std::string temp = get_temp_file_name();
        {
            std::basic_filebuf<char> buf;
            buf.open(temp.c_str(), std::ios_base::out);
            std::streamsize sz = buf.sputn("\xFF", 1);
            assert(sz == 1);
        }
        assert(count_bytes(temp.c_str()) == 1);
        std::remove(temp.c_str());
    }

    return 0;
}
