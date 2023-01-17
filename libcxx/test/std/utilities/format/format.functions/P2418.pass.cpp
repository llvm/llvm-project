//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format
// TODO FMT Evaluate gcc-12 status
// UNSUPPORTED: gcc-12

// Tests whether a move only type can be formatted. This is required by
// P2418R2 "Add support for std::generator-like types to std::format"

// <format>

#include <format>
#include <cassert>

#include "MoveOnly.h"
#include "make_string.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <locale>
#endif

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
struct std::formatter<MoveOnly, CharT> : std::formatter<int, CharT> {
  auto format(const MoveOnly& m, auto& ctx) const -> decltype(ctx.out()) {
    return std::formatter<int, CharT>::format(m.get(), ctx);
  }
};

template <class CharT>
static void test() {
  MoveOnly m{10};
  CharT buffer[10];
#ifndef TEST_HAS_NO_LOCALIZATION
  std::locale loc;
#endif

  assert(std::format(SV("{}"), MoveOnly{}) == SV("1"));

  assert(std::format(SV("{}"), m) == SV("10"));
  assert(m.get() == 10);

  assert(std::format(SV("{}"), std::move(m)) == SV("10"));
  assert(m.get() == 10);

#ifndef TEST_HAS_NO_LOCALIZATION
  assert(std::format(loc, SV("{}"), MoveOnly{}) == SV("1"));

  assert(std::format(loc, SV("{}"), m) == SV("10"));
  assert(m.get() == 10);

  assert(std::format(loc, SV("{}"), std::move(m)) == SV("10"));
  assert(m.get() == 10);
#endif

  assert(std::format_to(buffer, SV("{}"), MoveOnly{}) == &buffer[1]);

  assert(std::format_to(buffer, SV("{}"), m) == &buffer[2]);
  assert(m.get() == 10);

  assert(std::format_to(buffer, SV("{}"), std::move(m)) == &buffer[2]);
  assert(m.get() == 10);

#ifndef TEST_HAS_NO_LOCALIZATION
  assert(std::format_to(buffer, loc, SV("{}"), MoveOnly{}) == &buffer[1]);

  assert(std::format_to(buffer, loc, SV("{}"), m) == &buffer[2]);
  assert(m.get() == 10);

  assert(std::format_to(buffer, loc, SV("{}"), std::move(m)) == &buffer[2]);
  assert(m.get() == 10);
#endif

  assert(std::format_to_n(buffer, 5, SV("{}"), MoveOnly{}).out == &buffer[1]);

  assert(std::format_to_n(buffer, 5, SV("{}"), m).out == &buffer[2]);
  assert(m.get() == 10);

  assert(std::format_to_n(buffer, 5, SV("{}"), std::move(m)).out == &buffer[2]);
  assert(m.get() == 10);

#ifndef TEST_HAS_NO_LOCALIZATION
  assert(std::format_to_n(buffer, 5, loc, SV("{}"), MoveOnly{}).out == &buffer[1]);

  assert(std::format_to_n(buffer, 5, loc, SV("{}"), m).out == &buffer[2]);
  assert(m.get() == 10);

  assert(std::format_to_n(buffer, 5, loc, SV("{}"), std::move(m)).out == &buffer[2]);
  assert(m.get() == 10);
#endif

  assert(std::formatted_size(SV("{}"), MoveOnly{}) == 1);

  assert(std::formatted_size(SV("{}"), m) == 2);
  assert(m.get() == 10);

  assert(std::formatted_size(SV("{}"), std::move(m)) == 2);
  assert(m.get() == 10);

#ifndef TEST_HAS_NO_LOCALIZATION
  assert(std::formatted_size(loc, SV("{}"), MoveOnly{}) == 1);

  assert(std::formatted_size(loc, SV("{}"), m) == 2);
  assert(m.get() == 10);

  assert(std::formatted_size(loc, SV("{}"), std::move(m)) == 2);
  assert(m.get() == 10);
#endif
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
