//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class T, StrictWeakOrder<auto, T> Compare>
//   requires !SameType<T, Compare> && CopyConstructible<Compare>
//   const T&
//   max(const T& a, const T& b, Compare comp);

#include <algorithm>
#include <functional>
#include <cassert>

template <class T, class C>
void
test(const T& a, const T& b, C c, const T& x)
{
    assert(&std::max(a, b, c) == &x);
}

int main()
{
    {
    int x = 0;
    int y = 0;
    test(x, y, std::greater<int>(), x);
    test(y, x, std::greater<int>(), y);
    }
    {
    int x = 0;
    int y = 1;
    test(x, y, std::greater<int>(), x);
    test(y, x, std::greater<int>(), x);
    }
    {
    int x = 1;
    int y = 0;
    test(x, y, std::greater<int>(), y);
    test(y, x, std::greater<int>(), y);
    }
}
