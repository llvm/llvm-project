//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <ranges>
#include <cassert>
#include <vector>
#include <string>

constexpr bool test() {
  struct ConstexprStringStream {
    std::string str;

    constexpr ConstexprStringStream& operator<<(int x) {
      return *this << char(x + 48);
    }
    constexpr ConstexprStringStream& operator<<(char c) {
      str += c;
      return *this;
    }
  };

  const std::vector<int> v{0, 1, 2};
  ConstexprStringStream out;
  for (auto&& [a, b, c] : std::ranges::views::cartesian_product(v, v, v)) {
    out << a << ' ' << b << ' ' << c << '\n';
  }

  const std::string_view expected =
      "0 0 0\n"
      "0 0 1\n"
      "0 0 2\n"
      "0 1 0\n"
      "0 1 1\n"
      "0 1 2\n"
      "0 2 0\n"
      "0 2 1\n"
      "0 2 2\n"
      "1 0 0\n"
      "1 0 1\n"
      "1 0 2\n"
      "1 1 0\n"
      "1 1 1\n"
      "1 1 2\n"
      "1 2 0\n"
      "1 2 1\n"
      "1 2 2\n"
      "2 0 0\n"
      "2 0 1\n"
      "2 0 2\n"
      "2 1 0\n"
      "2 1 1\n"
      "2 1 2\n"
      "2 2 0\n"
      "2 2 1\n"
      "2 2 2\n";
  assert(out.str == expected);
  
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}