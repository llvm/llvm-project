//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class T>
//   T
//   min(initializer_list<T> t);

#include <algorithm>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    int i = std::min({2, 3, 1});
    assert(i == 1);
    i = std::min({2, 1, 3});
    assert(i == 1);
    i = std::min({3, 1, 2});
    assert(i == 1);
    i = std::min({3, 2, 1});
    assert(i == 1);
    i = std::min({1, 2, 3});
    assert(i == 1);
    i = std::min({1, 3, 2});
    assert(i == 1);
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
