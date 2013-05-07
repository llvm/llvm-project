//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// map();

#include <map>

#if !__has_feature(cxx_noexcept)

struct X
{
    std::multimap<int, X> m;
};

#endif

int main()
{
}
