//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class T>
//   T
//   max(initializer_list<T> t);

#include <algorithm>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    int i = std::max({2, 3, 1});
    assert(i == 3);
    i = std::max({2, 1, 3});
    assert(i == 3);
    i = std::max({3, 1, 2});
    assert(i == 3);
    i = std::max({3, 2, 1});
    assert(i == 3);
    i = std::max({1, 2, 3});
    assert(i == 3);
    i = std::max({1, 3, 2});
    assert(i == 3);
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
