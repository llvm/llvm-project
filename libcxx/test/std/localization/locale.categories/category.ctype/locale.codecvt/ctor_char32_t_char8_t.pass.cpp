//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// This test relies on https://wg21.link/P0482 being implemented, which isn't in
// older Apple dylibs
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0}}

// <locale>

// template <> class codecvt<char32_t, char8_t, mbstate_t>

// explicit codecvt(size_t refs = 0);

#include <cassert>
#include <locale>

using F = std::codecvt<char32_t, char8_t, std::mbstate_t>;

struct my_facet : F {
  static int count;

  explicit my_facet(std::size_t refs = 0) : F(refs) { ++count; }

  ~my_facet() { --count; }
};

int my_facet::count = 0;

int main(int, char**) {
  {
    std::locale l(std::locale::classic(), new my_facet);
    assert(my_facet::count == 1);
  }
  assert(my_facet::count == 0);
  {
    my_facet f(1);
    assert(my_facet::count == 1);
    {
      std::locale l(std::locale::classic(), &f);
      assert(my_facet::count == 1);
    }
    assert(my_facet::count == 1);
  }
  assert(my_facet::count == 0);

  return 0;
}
