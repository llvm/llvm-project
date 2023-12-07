//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: long_tests

// Not a portable test

// <__hash_table>

// size_t __next_prime(size_t n);

// If n == 0, return 0, else return the lowest prime greater than or equal to n

#include <__hash_table>
#include <cassert>

#include "test_macros.h"

bool
is_prime(std::size_t n)
{
    switch (n)
    {
    case 0:
    case 1:
        return false;
    }
    for (std::size_t i = 2; i*i <= n; ++i)
    {
        if (n % i == 0)
            return false;
    }
    return true;
}

int main(int, char**)
{
    assert(std::__next_prime(0) == 0);
    for (std::size_t n = 1; n <= 100000; ++n)
    {
        std::size_t p = std::__next_prime(n);
        assert(p >= n);
        for (std::size_t i = n; i < p; ++i)
            assert(!is_prime(i));
        assert(is_prime(p));
    }

  return 0;
}
