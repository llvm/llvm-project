//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template <class UIntType, UIntType a, UIntType c, UIntType m>
//   class linear_congruential_engine;

// result_type operator()();

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef unsigned long long T;

    // m might overflow, but the overflow is OK so it shouldn't use Schrage's algorithm
    typedef std::linear_congruential_engine<T, 25214903917ull, 1, (1ull << 48)> E1;
    E1 e1;
    // make sure the right algorithm was used
    assert(e1() == 25214903918ull);
    assert(e1() == 205774354444503ull);
    assert(e1() == 158051849450892ull);
    // make sure result is in bounds
    assert(e1() < (1ull << 48));
    assert(e1() < (1ull << 48));
    assert(e1() < (1ull << 48));
    assert(e1() < (1ull << 48));
    assert(e1() < (1ull << 48));

    // m might overflow. The overflow is not OK and result will be in bounds
    // so we should use Schrage's algorithm
    typedef std::linear_congruential_engine<T, (1ull << 32), 0, (1ull << 63) + 1ull> E2;
    E2 e2;
    // make sure Schrage's algorithm is used (it would be 0s after the first otherwise)
    assert(e2() == (1ull << 32));
    assert(e2() == (1ull << 63) - 1ull);
    assert(e2() == (1ull << 63) - 0x1ffffffffull);
    // make sure result is in bounds
    assert(e2() < (1ull << 63) + 1);
    assert(e2() < (1ull << 63) + 1);
    assert(e2() < (1ull << 63) + 1);
    assert(e2() < (1ull << 63) + 1);
    assert(e2() < (1ull << 63) + 1);

    // m might overflow. The overflow is not OK and result will be in bounds
    // so we should use Schrage's algorithm. m is even
    typedef std::linear_congruential_engine<T, 0x18000001ull, 0x12347ull, (3ull << 56)> E3;
    E3 e3;
    // make sure Schrage's algorithm is used
    assert(e3() == 0x18012348ull);
    assert(e3() == 0x2401b4ed802468full);
    assert(e3() == 0x18051ec400369d6ull);
    // make sure result is in bounds
    assert(e3() < (3ull << 56));
    assert(e3() < (3ull << 56));
    assert(e3() < (3ull << 56));
    assert(e3() < (3ull << 56));
    assert(e3() < (3ull << 56));

    // 32-bit case:
    // m might overflow. The overflow is not OK, result will be in bounds,
    // and Schrage's algorithm is incompatible here. Need to use 64 bit arithmetic.
    typedef std::linear_congruential_engine<unsigned, 0x10009u, 0u, 0x7fffffffu> E4;
    E4 e4;
    // make sure enough precision is used
    assert(e4() == 0x10009u);
    assert(e4() == 0x120053u);
    assert(e4() == 0xf5030fu);
    // make sure result is in bounds
    assert(e4() < 0x7fffffffu);
    assert(e4() < 0x7fffffffu);
    assert(e4() < 0x7fffffffu);
    assert(e4() < 0x7fffffffu);
    assert(e4() < 0x7fffffffu);

#ifndef _LIBCPP_HAS_NO_INT128
    // m might overflow. The overflow is not OK, result will be in bounds,
    // and Schrage's algorithm is incompatible here. Need to use 128 bit arithmetic.
    typedef std::linear_congruential_engine<T, 0x100000001ull, 0ull, (1ull << 61) - 1ull> E5;
    E5 e5;
    // make sure enough precision is used
    assert(e5() == 0x100000001ull);
    assert(e5() == 0x200000009ull);
    assert(e5() == 0xb00000019ull);
    // make sure result is in bounds
    assert(e5() < (1ull << 61) - 1ull);
    assert(e5() < (1ull << 61) - 1ull);
    assert(e5() < (1ull << 61) - 1ull);
    assert(e5() < (1ull << 61) - 1ull);
    assert(e5() < (1ull << 61) - 1ull);
#endif

    // m will not overflow so we should not use Schrage's algorithm
    typedef std::linear_congruential_engine<T, 1ull, 1, (1ull << 48)> E6;
    E6 e6;
    // make sure the correct algorithm was used
    assert(e6() == 2ull);
    assert(e6() == 3ull);
    assert(e6() == 4ull);
    // make sure result is in bounds
    assert(e6() < (1ull << 48));
    assert(e6() < (1ull << 48));
    assert(e6() < (1ull << 48));
    assert(e6() < (1ull << 48));
    assert(e6() < (1ull << 48));

    return 0;
}
