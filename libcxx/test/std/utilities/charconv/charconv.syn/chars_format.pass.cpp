//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <charconv>

// Bitmask type
// enum class chars_format {
//   scientific = unspecified,
//   fixed = unspecified,
//   hex = unspecified,
//   general = fixed | scientific
// };

#include <charconv>
#include <cassert>

#include "test_macros.h"

constexpr bool test() {
  using cf = std::chars_format;
  using ut = std::underlying_type<cf>::type;

  {
    cf x = cf::scientific;
    x |= cf::fixed;
    assert(x == cf::general);
  }
  {
    cf x = cf::general;
    x &= cf::fixed;
    assert(x == cf::fixed);
  }
  {
    cf x = cf::general;
    x ^= cf::fixed;
    assert(x == cf::scientific);
  }

  assert(static_cast<ut>(cf::scientific & (cf::fixed | cf::hex)) == 0);
  assert(static_cast<ut>(cf::fixed & (cf::scientific | cf::hex)) == 0);
  assert(static_cast<ut>(cf::hex & (cf::scientific | cf::fixed)) == 0);

  assert((cf::scientific | cf::fixed) == cf::general);

  assert(static_cast<ut>(cf::scientific & cf::fixed) == 0);

  assert((cf::general ^ cf::fixed) == cf::scientific);

  assert((~cf::hex & cf::general) == cf::general);

  return true;
}

std::chars_format x;
static_assert(std::is_same<std::chars_format, decltype(~x)>::value, "");
static_assert(std::is_same<std::chars_format, decltype(x & x)>::value, "");
static_assert(std::is_same<std::chars_format, decltype(x | x)>::value, "");
static_assert(std::is_same<std::chars_format, decltype(x ^ x)>::value, "");
static_assert(std::is_same<std::chars_format&, decltype(x &= x)>::value, "");
static_assert(std::is_same<std::chars_format&, decltype(x |= x)>::value, "");
static_assert(std::is_same<std::chars_format&, decltype(x ^= x)>::value, "");

int main(int, char**) {
  assert(test());
  static_assert(test(), "");

  return 0;
}
