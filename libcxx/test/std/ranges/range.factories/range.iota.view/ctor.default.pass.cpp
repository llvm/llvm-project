//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// iota_view() requires default_initializable<W> = default;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "types.h"

constexpr bool test() {
  {
    std::ranges::iota_view<Int42<DefaultTo42>> io;
    assert((*io.begin()).value_ == 42);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  static_assert(!std::default_initializable<Int42<ValueCtor>>);
  static_assert( std::default_initializable<Int42<DefaultTo42>>);

  return 0;
}
