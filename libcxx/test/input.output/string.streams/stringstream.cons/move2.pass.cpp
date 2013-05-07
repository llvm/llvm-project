//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringstream

// basic_stringstream(basic_stringstream&& rhs);

#include <sstream>
#include <vector>
#include <string>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    std::vector<std::istringstream> vecis;
    vecis.push_back(std::istringstream());
    vecis.back().str("hub started at [00 6b 8b 45 69]");
    vecis.push_back(std::istringstream());
    vecis.back().str("hub started at [00 6b 8b 45 69]");
    for (int n = 0; n < vecis.size(); n++)
    {
        assert(vecis[n].str().size() == 31);
        vecis[n].seekg(0, std::ios_base::beg);
        assert(vecis[n].str().size() == 31);
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
