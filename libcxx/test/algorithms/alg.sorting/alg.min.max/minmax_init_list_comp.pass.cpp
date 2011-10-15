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
//   pair<T, T>
//   minmax(initializer_list<T> t, Compare comp);

#include <algorithm>
#include <functional>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    assert((std::minmax({1, 2, 3}, std::greater<int>()) == std::pair<int, int>(3, 1)));
    assert((std::minmax({1, 3, 2}, std::greater<int>()) == std::pair<int, int>(3, 1)));
    assert((std::minmax({2, 1, 3}, std::greater<int>()) == std::pair<int, int>(3, 1)));
    assert((std::minmax({2, 3, 1}, std::greater<int>()) == std::pair<int, int>(3, 1)));
    assert((std::minmax({3, 1, 2}, std::greater<int>()) == std::pair<int, int>(3, 1)));
    assert((std::minmax({3, 2, 1}, std::greater<int>()) == std::pair<int, int>(3, 1)));
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
