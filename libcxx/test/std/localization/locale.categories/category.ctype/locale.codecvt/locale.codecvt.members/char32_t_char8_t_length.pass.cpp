//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// XFAIL: availability-char8_t_support-missing

// <locale>

// template <> class codecvt<char32_t, char8_t, mbstate_t>

// int length(stateT& state, const externT* from, const externT* from_end, size_t max) const;

#include <cassert>
#include <locale>

int main(int, char**) {
  using F = std::codecvt<char32_t, char8_t, std::mbstate_t>;
  const F& f = std::use_facet<F>(std::locale::classic());
  std::mbstate_t mbs = {};
  const char8_t from[] = u8"some text";
  assert(f.length(mbs, from, from + 10, 0) == 0);
  assert(f.length(mbs, from, from + 10, 8) == 8);
  assert(f.length(mbs, from, from + 10, 9) == 9);
  assert(f.length(mbs, from, from + 10, 10) == 10);
  assert(f.length(mbs, from, from + 10, 100) == 10);
  return 0;
}
