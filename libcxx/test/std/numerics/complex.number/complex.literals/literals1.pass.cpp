//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <chrono>

#include <complex>
#include <type_traits>
#include <cassert>

int main(int, char**)
{
    using namespace std::literals;

    {
    std::complex<long double> c1 = 3.0il;
    assert ( c1 == std::complex<long double>(0, 3.0));
    auto c2 = 3il;
    assert ( c1 == c2 );
    }

    {
    std::complex<double> c1 = 3.0i;
    assert ( c1 == std::complex<double>(0, 3.0));
    auto c2 = 3i;
    assert ( c1 == c2 );
    }

    {
    std::complex<float> c1 = 3.0if;
    assert ( c1 == std::complex<float>(0, 3.0));
    auto c2 = 3if;
    assert ( c1 == c2 );
    }

  return 0;
}
