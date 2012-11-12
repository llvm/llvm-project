//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   bool
//   operator!=(const complex<T>& lhs, const T& rhs);

#include <complex>
#include <cassert>

template <class T>
void
test(const std::complex<T>& lhs, const T& rhs, bool x)
{
    assert((lhs != rhs) == x);
}

template <class T>
void
test()
{
    {
    std::complex<T> lhs(1.5,  2.5);
    T rhs(-2.5);
    test(lhs, rhs, true);
    }
    {
    std::complex<T> lhs(1.5,  0);
    T rhs(-2.5);
    test(lhs, rhs, true);
    }
    {
    std::complex<T> lhs(1.5, 2.5);
    T rhs(1.5);
    test(lhs, rhs, true);
    }
    {
    std::complex<T> lhs(1.5, 0);
    T rhs(1.5);
    test(lhs, rhs, false);
    }
}

int main()
{
    test<float>();
    test<double>();
    test<long double>();
}
