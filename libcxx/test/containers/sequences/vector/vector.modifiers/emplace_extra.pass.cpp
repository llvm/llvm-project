//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// template <class... Args> iterator emplace(const_iterator pos, Args&&... args);

#include <vector>
#include <cassert>
int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    {
        std::vector<int> v;
        v.reserve(3);
        v = { 1, 2, 3 };
        v.emplace(v.begin(), v.back());
        assert(v[0] == 3);
    }
    {
        std::vector<int> v;
        v.reserve(4);
        v = { 1, 2, 3 };
        v.emplace(v.begin(), v.back());
        assert(v[0] == 3);
    }
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
