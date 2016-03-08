//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>
// XFAIL: c++03, c++11, c++14

// template<class T, class Compare>
//   const T&
//   clamp(const T& v, const T& lo, const T& hi, Compare comp);

#include <algorithm>
#include <functional>
#include <cassert>

template <class T, class C>
void
test(const T& v, const T& lo, const T& hi, C c, const T& x)
{
    assert(&std::clamp(v, lo, hi, c) == &x);
}

int main()
{
    {
    int x = 0;
    int y = 0;
    int z = 0;
    test(x, y, z, std::greater<int>(), x);
    test(y, x, z, std::greater<int>(), y);
    }
    {
    int x = 0;
    int y = 1;
    int z = -1;
    test(x, y, z, std::greater<int>(), x);
    test(y, x, z, std::greater<int>(), x);
    }
    {
    int x = 1;
    int y = 0;
    int z = 0;
    test(x, y, z, std::greater<int>(), y);
    test(y, x, z, std::greater<int>(), y);
    }
#if _LIBCPP_STD_VER > 11
    {
    typedef int T;
    constexpr T x = 1;
    constexpr T y = 0;
    constexpr T z = 0;
    static_assert(std::clamp(x, y, z, std::greater<T>()) == y, "" );
    static_assert(std::clamp(y, x, z, std::greater<T>()) == y, "" );
    }
#endif
}
