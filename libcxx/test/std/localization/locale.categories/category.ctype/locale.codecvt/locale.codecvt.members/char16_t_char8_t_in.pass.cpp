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
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.15
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.14
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.13
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.12
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.11
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.10
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.9

// <locale>

// template <> class codecvt<char16_t, char8_t, mbstate_t>

// result in(stateT& state,
//           const externT* from, const externT* from_end, const externT*& from_next,
//           internT* to, internT* to_end, internT*& to_next) const;

#include <cassert>
#include <locale>

int main(int, char**) {
  using F = std::codecvt<char16_t, char8_t, std::mbstate_t>;
  const F::extern_type from[] = u8"some text";
  F::intern_type to[9];
  const F& f = std::use_facet<F>(std::locale::classic());
  std::mbstate_t mbs = {};
  const F::extern_type* from_next = nullptr;
  F::intern_type* to_next = nullptr;
  assert(f.in(mbs, from, from + 9, from_next, to, to + 9, to_next) == F::ok);
  assert(from_next - from == 9);
  assert(to_next - to == 9);
  for (unsigned i = 0; i < 9; ++i)
    assert(to[i] == from[i]);
  return 0;
}
