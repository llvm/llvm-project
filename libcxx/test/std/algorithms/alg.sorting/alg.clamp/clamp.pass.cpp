//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>
// XFAIL: c++98, c++03, c++11, c++14

// template<class T>
//   const T&
//   clamp(const T& v, const T& lo, const T& hi);

#include <algorithm>
#include <cassert>

template <class T>
void
test(const T& a, const T& lo, const T& hi, const T& x)
{
    assert(&std::clamp(a, lo, hi) == &x);
}

int main()
{
    {
    int x = 0;
    int y = 0;
    int z = 0;
    test(x, y, z, x);
    test(y, x, z, y);
    }
    {
    int x = 0;
    int y = 1;
    int z = 2;
    test(x, y, z, y);
    test(y, x, z, y);
    }
    {
    int x = 1;
    int y = 0;
    int z = 1;
    test(x, y, z, x);
    test(y, x, z, x);
    }
#if _LIBCPP_STD_VER > 11
    {
    typedef int T;
    constexpr T x = 1;
    constexpr T y = 0;
    constexpr T z = 1;
    static_assert(std::clamp(x, y, z) == x, "" );
    static_assert(std::clamp(y, x, z) == x, "" );
    }
#endif
}
