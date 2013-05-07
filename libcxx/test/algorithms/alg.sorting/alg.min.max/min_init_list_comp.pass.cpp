//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class T, class Compare>
//   T
//   min(initializer_list<T> t, Compare comp);

#include <algorithm>
#include <functional>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    int i = std::min({2, 3, 1}, std::greater<int>());
    assert(i == 3);
    i = std::min({2, 1, 3}, std::greater<int>());
    assert(i == 3);
    i = std::min({3, 1, 2}, std::greater<int>());
    assert(i == 3);
    i = std::min({3, 2, 1}, std::greater<int>());
    assert(i == 3);
    i = std::min({1, 2, 3}, std::greater<int>());
    assert(i == 3);
    i = std::min({1, 3, 2}, std::greater<int>());
    assert(i == 3);
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
