//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class codecvt<char16_t, char8_t, mbstate_t>

// result unshift(stateT& state,
//                externT* to, externT* to_end, externT*& to_next) const;

// UNSUPPORTED: c++03, c++11, c++14, c++17

// C++20 codecvt specializations for char8_t are not yet implemented:
// UNSUPPORTED: libc++

#include <cassert>
#include <locale>

int main(int, char**) {
  using F = std::codecvt<char16_t, char8_t, std::mbstate_t>;
  const F& f = std::use_facet<F>(std::locale::classic());
  std::mbstate_t mbs = {};
  char8_t to[3];
  char8_t* to_next = nullptr;
  assert(f.unshift(mbs, to, to + 3, to_next) == F::noconv);
  assert(to_next == to);
  return 0;
}
