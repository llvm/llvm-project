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
    typedef std::linear_congruential_engine<T, (1ull << 32), 0, (1ull << 63) + 1> E2;
    E2 e2;
    // make sure Schrage's algorithm is used (it would be 0s after the first otherwise)
    assert(e2() == (1ull << 32));
    assert(e2() == (1ull << 63) - 1ull);
    assert(e2() == (1ull << 63) - (1ull << 33) + 1ull);
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
    assert(e3() == 402727752ull);
    assert(e3() == 162159612030764687ull);
    assert(e3() == 108176466184989142ull);
    // make sure result is in bounds
    assert(e3() < (3ull << 56));
    assert(e3() < (3ull << 56));
    assert(e3() < (3ull << 56));
    assert(e3() < (3ull << 56));
    assert(e3() < (3ull << 56));

    // m will not overflow so we should not use Schrage's algorithm
    typedef std::linear_congruential_engine<T, 1ull, 1, (1ull << 48)> E4;
    E4 e4;
    // make sure the correct algorithm was used
    assert(e4() == 2ull);
    assert(e4() == 3ull);
    assert(e4() == 4ull);
    // make sure result is in bounds
    assert(e4() < (1ull << 48));
    assert(e4() < (1ull << 48));
    assert(e4() < (1ull << 48));
    assert(e4() < (1ull << 48));
    assert(e4() < (1ull << 48));

    return 0;
}