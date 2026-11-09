//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class UIntType, size_t w, size_t s, size_t r>
// class subtract_with_carry_engine;

// void seed(result_type s = default_seed);

#include <random>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "test_macros.h"

void
test1()
{
    for (int s = 0; s < 20; ++s)
    {
        typedef std::ranlux24_base E;
        E e1(s);
        E e2;
        e2.seed(s);
        assert(e1 == e2);
    }
}

void
test2()
{
    for (int s = 0; s < 20; ++s)
    {
        typedef std::ranlux48_base E;
        E e1(s);
        E e2;
        e2.seed(s);
        assert(e1 == e2);
    }
}

void test_lwg3809() {
  typedef std::subtract_with_carry_engine<std::uint16_t, 16, 5, 12> E;

  static_assert(std::is_same<decltype(E::default_seed), const std::uint_least32_t>::value, "");

  static_assert(E::default_seed == 19780503u, "");

  E e1;
  E e2(0);
  E e3;
  e3.seed();

  assert(e1 == e2);
  assert(e1 == e3);

  for (int i = 0; i < 100; ++i) {
    E::result_type value = e1();
    assert(value == e2());
    assert(value == e3());
  }

  E expected_engine;
  assert(expected_engine() == 20739);
}

void test_lwg4014() {
  typedef std::ranlux48_base E;

  const E::result_type modulus    = 2147483563ULL;
  const E::result_type large_seed = 4294967297ULL;

  E e1(large_seed);
  E e2(large_seed % modulus);
  E e3;
  e3.seed(large_seed);

  assert(e1 == e2);
  assert(e1 == e3);

  for (int i = 0; i < 100; ++i) {
    E::result_type value = e1();
    assert(value == e2());
    assert(value == e3());
  }

  E e4(12345);
  E e5(12345 + modulus);

  assert(e4 == e5);
}

int main(int, char**)
{
    test1();
    test2();
    test_lwg3809();
    test_lwg4014();

    return 0;
}
