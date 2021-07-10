//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// This test relies on P0482 being fixed, which isn't in
// older Apple dylibs
//
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}

// <locale>

// template <> class codecvt<char32_t, char8_t, mbstate_t>

// result out(stateT& state,
//            const internT* from, const internT* from_end, const internT*& from_next,
//            externT* to, externT* to_end, externT*& to_next) const;

#include <cassert>
#include <locale>

int main(int, char**) {
  using F = std::codecvt<char32_t, char8_t, std::mbstate_t>;
  const F& f = std::use_facet<F>(std::locale::classic());
  F::intern_type from[9] = {'s', 'o', 'm', 'e', ' ', 't', 'e', 'x', 't'};
  F::extern_type to[9] = {0};
  std::mbstate_t mbs = {};
  const F::intern_type* from_next = nullptr;
  F::extern_type* to_next = nullptr;
  F::result r = f.out(mbs, from, from + 9, from_next, to, to + 9, to_next);
  assert(r == F::ok);
  assert(from_next - from == 9);
  assert(to_next - to == 9);
  for (unsigned i = 0; i < 9; ++i)
    assert(to[i] == from[i]);
  return 0;
}
