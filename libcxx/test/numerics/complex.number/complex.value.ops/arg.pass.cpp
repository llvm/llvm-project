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
//   T
//   arg(const complex<T>& x);

#include <complex>
#include <cassert>

#include "../cases.h"

template <class T>
void
test()
{
    std::complex<T> z(1, 0);
    assert(arg(z) == 0);
}

void test_edges()
{
    const double pi = std::atan2(+0., -0.);
    const unsigned N = sizeof(x) / sizeof(x[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        double r = arg(x[i]);
        if (std::isnan(x[i].real()) || std::isnan(x[i].imag()))
            assert(std::isnan(r));
        else
        {
            switch (classify(x[i]))
            {
            case zero:
                if (std::signbit(x[i].real()))
                {
                    if (std::signbit(x[i].imag()))
                        is_about(r, -pi);
                    else
                        is_about(r, pi);
                }
                else
                {
                    assert(std::signbit(x[i].imag()) == std::signbit(r));
                }
                break;
            case non_zero:
                if (x[i].real() == 0)
                {
                    if (x[i].imag() < 0)
                        is_about(r, -pi/2);
                    else
                        is_about(r, pi/2);
                }
                else if (x[i].imag() == 0)
                {
                    if (x[i].real() < 0)
                    {
                        if (std::signbit(x[i].imag()))
                            is_about(r, -pi);
                        else
                            is_about(r, pi);
                    }
                    else
                    {
                        assert(r == 0);
                        assert(std::signbit(x[i].imag()) == std::signbit(r));
                    }
                }
                else if (x[i].imag() > 0)
                    assert(r > 0);
                else
                    assert(r < 0);
                break;
            case inf:
                if (std::isinf(x[i].real()) && std::isinf(x[i].imag()))
                {
                    if (x[i].real() < 0)
                    {
                        if (x[i].imag() > 0)
                            is_about(r, 0.75 * pi);
                        else
                            is_about(r, -0.75 * pi);
                    }
                    else
                    {
                        if (x[i].imag() > 0)
                            is_about(r, 0.25 * pi);
                        else
                            is_about(r, -0.25 * pi);
                    }
                }
                else if (std::isinf(x[i].real()))
                {
                    if (x[i].real() < 0)
                    {
                        if (std::signbit(x[i].imag()))
                            is_about(r, -pi);
                        else
                            is_about(r, pi);
                    }
                    else
                    {
                        assert(r == 0);
                        assert(std::signbit(r) == std::signbit(x[i].imag()));
                    }
                }
                else
                {
                    if (x[i].imag() < 0)
                        is_about(r, -pi/2);
                    else
                        is_about(r, pi/2);
                }
                break;
            }
        }
    }
}

int main()
{
    test<float>();
    test<double>();
    test<long double>();
    test_edges();
}
