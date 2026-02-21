//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// template <> struct hash<text_encoding>

#include <cassert>
#include <cstdint>
#include <text_encoding>
#include <type_traits>

void test_te_hash() {
  using T = std::text_encoding;
  using H = std::hash<T>;

  {
    const T te(T::ASCII);
    const H h{};
    assert(h(te) == h(te));
    static_assert(std::is_same_v<decltype(h(te)), std::size_t>);
  }

  {
    const T te1(T::ASCII);
    const T te2(T::UTF8);
    const H h{};

    assert(h(te1) != h(te2));
  }

  {
    const T te1(T::unknown);
    const T te2(T::unknown);
    const H h{};
    assert(h(te1) == h(te2));
  }

  {
    const T te1(T::other);
    const T te2(T::other);
    const H h{};
    assert(h(te1) == h(te2));
  }
}

int main(int, char**) {
  test_te_hash();

  return 0;
}
