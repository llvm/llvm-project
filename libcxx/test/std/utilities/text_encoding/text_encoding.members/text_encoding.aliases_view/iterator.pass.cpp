//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// text_encoding::aliases_view::iterator (implementation-defined)

#include <cassert>
#include <compare>
#include <string_view>
#include <text_encoding>
#include <utility>

constexpr bool test() {
  // Test iterator operators.
  std::text_encoding te = std::text_encoding(std::text_encoding::ASCII); // 11 aliases

  auto i = te.aliases().begin();
  auto j = te.aliases().begin();
  auto k = te.aliases().end();

  static_assert(std::three_way_comparable<decltype(i), std::strong_ordering>);

  {
    assert(i == j);
    assert(i != k);
    assert((i <=> j) == std::strong_ordering::equal);
    assert(std::string_view(*i) == std::string_view(*j));
  }
  {
    assert(std::string_view(i[0]) == std::string_view(j[0]));
    assert(std::string_view(i[1]) != std::string_view(j[3]));
  }
  {
    i++;
    assert(i > j);
    assert((i <=> j) == std::strong_ordering::greater);
    assert(std::string_view(*i) != std::string_view(*j));
  }
  {
    i--;
    assert(i == te.aliases().begin());
    assert(i == j);
    assert(i != k);
    assert(std::string_view(*i) == std::string_view(*j));
  }
  {
    i++;
    j++;
    assert(i != te.aliases().begin());
    assert(i == j);
    assert(i != k);
    assert(std::string_view(*i) == std::string_view(*j));
  }
  {
    auto temp = i + 2;
    assert(i != temp);
    assert(std::string_view(*temp) != std::string_view(*j));
    auto temp2 = temp - 2;
    assert(std::string_view(*temp2) == std::string_view(*j));
  }
  {
    assert(i - j == 0);
    assert(k - i > 0);
  }
  {
    auto temp = ++i;
    assert(temp == i);
    auto temp2 = j++;
    assert(temp2 == j - 1);
    assert(i == j);
  }
  {
    i += 2;
    j += 3;

    auto tempi = i;
    auto tempj = j;
    assert(i != j);
    assert((i <=> j) == std::strong_ordering::less);
    i -= 2;
    j -= 3;
    assert(i == j);
    assert(i != tempi && (tempi - i) == 2);
    assert(j != tempj && (tempj - j) == 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
