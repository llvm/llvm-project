//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <experimental/simd>
//
// scalar access [simd.subscr]
// reference operator[](size_t);
// value_type operator[](size_t) const;

#include <experimental/simd>
#include <cassert>
#include <cstdint>

#include "test_macros.h"

namespace ex = std::experimental::parallelism_v2;

template <typename SimdType>
void test_access() {
  {
    SimdType a(42), b(4);
    static_assert(std::is_convertible<decltype(a[0]), int8_t>::value, "");

    assert(a[0] == 42);
    assert(!a[0] == !42);
    assert(~a[0] == ~42);
    assert(+a[0] == +42);
    assert(-a[0] == -42);
    assert(a[0] + b[0] == 42 + 4);
    assert(a[0] - b[0] == 42 - 4);
    assert(a[0] * b[0] == 42 * 4);
    assert(a[0] / b[0] == 42 / 4);
    assert(a[0] % b[0] == 42 % 4);
    assert(a[0] << b[0] == (42 << 4));
    assert(a[0] >> b[0] == (42 >> 4));
    assert(a[0] < b[0] == false);
    assert(a[0] <= b[0] == false);
    assert(a[0] > b[0] == true);
    assert(a[0] >= b[0] == true);
    assert(a[0] == b[0] == false);
    assert(a[0] != b[0] == true);
    assert((a[0] & b[0]) == (42 & 4));
    assert((a[0] | b[0]) == (42 | 4));
    assert((a[0] ^ b[0]) == (42 ^ 4));
    assert((a[0] && b[0]) == true);
    assert((a[0] || b[0]) == true);

    {
      auto c = a;
      ++c[0];
      assert(c[0] == 42 + 1);
      assert(c[1] == 42);
    }
    {
      auto c = a;
      auto ret = c[0]++;
      assert(ret == 42);
      assert(c[0] == 42 + 1);
      assert(c[1] == 42);
    }
    {
      auto c = a;
      --c[0];
      assert(c[0] == 42 - 1);
      assert(c[1] == 42);
    }
    {
      auto c = a;
      auto ret = c[0]--;
      assert(ret == 42);
      assert(c[0] == 42 - 1);
      assert(c[1] == 42);
    }

    {
      auto c = a;
      c[0] += b[0];
      assert(c[0] == 42 + 4);
      assert(c[1] == 42);
    }
    {
      auto c = a;
      c[0] -= b[0];
      assert(c[0] == 42 - 4);
      assert(c[1] == 42);
    }
    {
      auto c = a;
      c[0] *= b[0];
      assert(c[0] == 42 * 4);
      assert(c[1] == 42);
    }
    {
      auto c = a;
      c[0] /= b[0];
      assert(c[0] == 42 / 4);
      assert(c[1] == 42);
    }
    {
      auto c = a;
      c[0] %= b[0];
      assert(c[0] == 42 % 4);
      assert(c[1] == 42);
    }
    {
      auto c = a;
      c[0] >>= b[0];
      assert(c[0] == (42 >> 4));
      assert(c[1] == 42);
    }
    {
      auto c = a;
      c[0] <<= b[0];
      assert(c[0] == (42 << 4));
      assert(c[1] == 42);
    }
    {
      auto c = a;
      c[0] &= b[0];
      assert(c[0] == (42 & 4));
      assert(c[1] == 42);
    }
    {
      auto c = a;
      c[0] |= b[0];
      assert(c[0] == (42 | 4));
      assert(c[1] == 42);
    }
    {
      auto c = a;
      c[0] ^= b[0];
      assert(c[0] == (42 ^ 4));
      assert(c[1] == 42);
    }

    {
      auto c = a;
      (void)(a[0] + (c[0] += a[0]));
    }
    {
      auto c = a;
      (void)(a[0] + (c[0] -= a[0]));
    }
    {
      auto c = a;
      (void)(a[0] + (c[0] *= a[0]));
    }
    {
      auto c = a;
      (void)(a[0] + (c[0] /= a[0]));
    }
    {
      auto c = a;
      (void)(a[0] + (c[0] %= a[0]));
    }
    {
      auto c = a;
      (void)(a[0] + (c[0] >>= b[0]));
    }
    {
      auto c = a;
      (void)(a[0] + (c[0] <<= b[0]));
    }
    {
      auto c = a;
      (void)(a[0] + (c[0] &= a[0]));
    }
    {
      auto c = a;
      (void)(a[0] + (c[0] |= a[0]));
    }
    {
      auto c = a;
      (void)(a[0] + (c[0] ^= a[0]));
    }
  }
  {
    const SimdType a(42);
    const SimdType b(4);
    static_assert(std::is_same<decltype(a[0]), int>::value, "");

    assert(a[0] == 42);
    assert(!a[0] == !42);
    assert(~a[0] == ~42);
    assert(+a[0] == +42);
    assert(-a[0] == -42);
    assert(a[0] + b[0] == 42 + 4);
    assert(a[0] - b[0] == 42 - 4);
    assert(a[0] * b[0] == 42 * 4);
    assert(a[0] / b[0] == 42 / 4);
    assert(a[0] % b[0] == 42 % 4);
    assert(a[0] << b[0] == (42 << 4));
    assert(a[0] >> b[0] == (42 >> 4));
    assert(a[0] < b[0] == false);
    assert(a[0] <= b[0] == false);
    assert(a[0] > b[0] == true);
    assert(a[0] >= b[0] == true);
    assert(a[0] == b[0] == false);
    assert(a[0] != b[0] == true);
    assert((a[0] & b[0]) == (42 & 4));
    assert((a[0] | b[0]) == (42 | 4));
    assert((a[0] ^ b[0]) == (42 ^ 4));
    assert((a[0] && b[0]) == true);
    assert((a[0] || b[0]) == true);
  }
}

int main(int, char**) {
  test_access<ex::native_simd<int>>();
  test_access<ex::fixed_size_simd<int, 4>>();

  return 0;
}
