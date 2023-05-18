//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class seed_seq;

// template<class InputIterator>
//     seed_seq(InputIterator begin, InputIterator end);

#include <random>
#include <cassert>

#include "test_macros.h"

void test()
{
  {
    unsigned a[5] = {5, 4, 3, 2, 1};
    std::seed_seq s(a, a+5);
    assert(s.size() == 5);

    unsigned b[5] = {0};
    s.param(b);
    assert(b[0] == 5);
    assert(b[1] == 4);
    assert(b[2] == 3);
    assert(b[3] == 2);
    assert(b[4] == 1);
  }
  {
    // Test truncation to 32 bits
    unsigned long long a[4] = {
      0x1234000056780000uLL,
      0x0000001234567800uLL,
      0xFFFFFFFFFFFFFFFFuLL,
      0x0000000180000000uLL,
    };
    std::seed_seq s(a, a+4);
    assert(s.size() == 4);

    unsigned b[4] = {0};
    s.param(b);
    assert(b[0] == 0x56780000u);
    assert(b[1] == 0x34567800u);
    assert(b[2] == 0xFFFFFFFFu);
    assert(b[3] == 0x80000000u);
  }
#if TEST_STD_VER >= 11
  {
    // Test uniform initialization syntax (LWG 3422)
    unsigned a[3] = {1, 2, 3};
    std::seed_seq s{a, a+3};  // uniform initialization
    assert(s.size() == 3);

    unsigned b[3] = {0};
    s.param(b);
    assert(b[0] == 1);
    assert(b[1] == 2);
    assert(b[2] == 3);
  }
#endif // TEST_STD_VER >= 11
}

int main(int, char**)
{
  test();

  return 0;
}
