//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ifstream

// explicit basic_ifstream(const string& s, ios_base::openmode mode = ios_base::in);

#include <fstream>
#include <cassert>

int main(int, char**)
{
    {
        std::ifstream fs(std::string("test.dat"));
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    // std::ifstream(const std::string&, std::ios_base::openmode) is tested in
    // test/std/input.output/file.streams/fstreams/ofstream.cons/string.pass.cpp
    // which creates writable files.
    {
        std::wifstream fs(std::string("test.dat"));
        double x = 0;
        fs >> x;
        assert(x == 3.25);
    }
    // std::wifstream(const std::string&, std::ios_base::openmode) is tested in
    // test/std/input.output/file.streams/fstreams/ofstream.cons/string.pass.cpp
    // which creates writable files.

  return 0;
}
