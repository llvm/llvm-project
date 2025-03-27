//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class E2> friend constexpr bool operator==(const expected& x, const unexpected<E2>& e);

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"

struct Data {
  int i;
  constexpr Data(int ii) : i(ii) {}

  friend constexpr bool operator==(const Data& data, int ii) { return data.i == ii; }
};

constexpr bool test() {
  // x.has_value()
  {
    const std::expected<void, Data> e1;
    std::unexpected<int> un2(10);
    std::unexpected<int> un3(5);
    assert(e1 != un2);
    assert(e1 != un3);
  }

  // !x.has_value()
  {
    const std::expected<void, Data> e1(std::unexpect, 5);
    std::unexpected<int> un2(10);
    std::unexpected<int> un3(5);
    assert(e1 != un2);
    assert(e1 == un3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
