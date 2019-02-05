//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringstream

// basic_stringstream(basic_stringstream&& rhs);

#include <sstream>
#include <vector>
#include <string>
#include <cassert>
#include <cstddef>

int main(int, char**)
{
    std::vector<std::istringstream> vecis;
    vecis.push_back(std::istringstream());
    vecis.back().str("hub started at [00 6b 8b 45 69]");
    vecis.push_back(std::istringstream());
    vecis.back().str("hub started at [00 6b 8b 45 69]");
    for (std::size_t n = 0; n < vecis.size(); n++)
    {
        assert(vecis[n].str().size() == 31);
        vecis[n].seekg(0, std::ios_base::beg);
        assert(vecis[n].str().size() == 31);
    }

  return 0;
}
