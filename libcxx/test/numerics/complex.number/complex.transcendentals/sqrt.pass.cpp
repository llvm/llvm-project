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
//   complex<T>
//   sqrt(const complex<T>& x);

#include <complex>
#include <cassert>

#include "../cases.h"

template <class T>
void
test(const std::complex<T>& c, std::complex<T> x)
{
    std::complex<T> a = sqrt(c);
    is_about(real(a), real(x));
    assert(std::abs(imag(c)) < 1.e-6);
}

template <class T>
void
test()
{
    test(std::complex<T>(64, 0), std::complex<T>(8, 0));
}

void test_edges()
{
    const unsigned N = sizeof(x) / sizeof(x[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        std::complex<double> r = sqrt(x[i]);
        if (x[i].real() == 0 && x[i].imag() == 0)
        {
            assert(!std::signbit(r.real()));
            assert(std::signbit(r.imag()) == std::signbit(x[i].imag()));
        }
        else if (std::isinf(x[i].imag()))
        {
            assert(std::isinf(r.real()));
            assert(r.real() > 0);
            assert(std::isinf(r.imag()));
            assert(std::signbit(r.imag()) == std::signbit(x[i].imag()));
        }
        else if (std::isfinite(x[i].real()) && std::isnan(x[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (std::isinf(x[i].real()) && x[i].real() < 0 && std::isfinite(x[i].imag()))
        {
            assert(r.real() == 0);
            assert(!std::signbit(r.real()));
            assert(std::isinf(r.imag()));
            assert(std::signbit(x[i].imag()) == std::signbit(r.imag()));
        }
        else if (std::isinf(x[i].real()) && x[i].real() > 0 && std::isfinite(x[i].imag()))
        {
            assert(std::isinf(r.real()));
            assert(r.real() > 0);
            assert(r.imag() == 0);
            assert(std::signbit(x[i].imag()) == std::signbit(r.imag()));
        }
        else if (std::isinf(x[i].real()) && x[i].real() < 0 && std::isnan(x[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isinf(r.imag()));
        }
        else if (std::isinf(x[i].real()) && x[i].real() > 0 && std::isnan(x[i].imag()))
        {
            assert(std::isinf(r.real()));
            assert(r.real() > 0);
            assert(std::isnan(r.imag()));
        }
        else if (std::isnan(x[i].real()) && (std::isfinite(x[i].imag()) || std::isnan(x[i].imag())))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (std::signbit(x[i].imag()))
        {
            assert(!std::signbit(r.real()));
            assert(std::signbit(r.imag()));
        }
        else
        {
            assert(!std::signbit(r.real()));
            assert(!std::signbit(r.imag()));
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
