//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<LessThanComparable T>
//   const T&
//   max(const T& a, const T& b);

#include <algorithm>
#include <cassert>

template <class T>
void
test(const T& a, const T& b, const T& x)
{
    assert(&std::max(a, b) == &x);
}

int main()
{
    {
    int x = 0;
    int y = 0;
    test(x, y, x);
    test(y, x, y);
    }
    {
    int x = 0;
    int y = 1;
    test(x, y, y);
    test(y, x, y);
    }
    {
    int x = 1;
    int y = 0;
    test(x, y, x);
    test(y, x, x);
    }
}
