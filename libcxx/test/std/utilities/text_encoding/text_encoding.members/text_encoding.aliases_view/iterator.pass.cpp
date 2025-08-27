//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26

// struct text_encoding::aliases_view

#include "test_text_encoding.h"

constexpr bool iter_test() {
  constexpr int num_ascii_aliases = 11;

  constexpr auto te  = std::text_encoding("US-ASCII"); // 11 aliases
  constexpr auto te2 = std::text_encoding("ANSI_X3.4-1968");

  constexpr auto aliases = te.aliases(), aliases2 = te2.aliases();
  auto begin_ascii = aliases.begin(), begin_ascii2 = aliases2.begin();
  auto end_ascii = aliases.end(), end_ascii2 = aliases2.end();
  auto iter_ascii = aliases.begin(), iter_ascii2 = aliases2.begin();

  assert(*iter_ascii == *begin_ascii);
  assert(*iter_ascii == *begin_ascii2);
  assert(*iter_ascii2 == *begin_ascii);
  assert(*iter_ascii2 == *begin_ascii2);

  assert(*iter_ascii == iter_ascii[0]);
  assert(*iter_ascii == iter_ascii2[0]);
  assert(iter_ascii[0] == iter_ascii2[0]);
  assert(iter_ascii[0] == *begin_ascii);
  assert(iter_ascii[0] == *begin_ascii2);
  assert(iter_ascii2[0] == *begin_ascii);
  assert(iter_ascii2[0] == *begin_ascii2);
  assert(iter_ascii[1] == begin_ascii[1]);
  assert(iter_ascii[1] == begin_ascii2[1]);
  assert(begin_ascii[1] == begin_ascii2[1]);

  --iter_ascii;
  ++iter_ascii;
  assert(iter_ascii == begin_ascii);
  ++iter_ascii2;
  assert(iter_ascii != iter_ascii2);
  assert(iter_ascii2 != begin_ascii2);
  iter_ascii2--;
  assert(iter_ascii == iter_ascii2);
  assert(iter_ascii2 == begin_ascii2);

  auto iter3 = iter_ascii++;
  assert(iter3 == begin_ascii);
  assert(iter_ascii != begin_ascii);
  iter3 = iter_ascii--;
  assert(iter3 != begin_ascii);
  assert(iter_ascii == begin_ascii);

  iter_ascii++ ++; // Increments prvalue returned by operator++(int) instead of iter.
  assert(iter_ascii == iter3);
  iter_ascii-- --; // Decrements prvalue returned by operator++(int) instead of iter.
  assert(iter_ascii == begin_ascii);

  const auto d = std::ranges::distance(aliases);
  iter_ascii += d;
  iter_ascii2 += d;
  assert(iter_ascii == end_ascii);
  assert(iter_ascii == end_ascii2);
  assert(iter_ascii2 == end_ascii);
  assert(iter_ascii2 == end_ascii2);

  assert((iter_ascii - begin_ascii) == d);
  assert((begin_ascii + num_ascii_aliases) == iter_ascii);
  assert((num_ascii_aliases + begin_ascii) == iter_ascii);
  assert(iter_ascii[-1] == begin_ascii[num_ascii_aliases - 1]);

  iter_ascii -= d;
  iter_ascii2 -= d;
  assert(iter_ascii == begin_ascii);
  assert(iter_ascii == begin_ascii2);

  assert(iter_ascii2 == begin_ascii);
  assert(iter_ascii2 == begin_ascii2);

  assert(*(iter_ascii + 1) == iter_ascii[1]);
  assert((1 + iter_ascii - 1) == begin_ascii);
  assert((-1 + (iter_ascii - -2) + -1) == begin_ascii);

  std::ranges::iterator_t<std::text_encoding::aliases_view> singular{};
  assert((singular + 0) == singular);
  assert((singular - 0) == singular);

  return true;
}

int main() {
  static_assert(iter_test());
  assert(iter_test());
}
