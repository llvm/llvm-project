//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class codecvt<char32_t, char8_t, mbstate_t>

// int encoding() const noexcept;

// UNSUPPORTED: c++03, c++11, c++14, c++17

// C++20 codecvt specializations for char8_t are not yet implemented:
// UNSUPPORTED: libc++

#include <cassert>
#include <locale>

int main(int, char**) {
  using F = std::codecvt<char32_t, char8_t, std::mbstate_t>;
  const F& f = std::use_facet<F>(std::locale::classic());
  assert(f.encoding() == 0);
  static_assert(noexcept(f.encoding()));
  return 0;
}
