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

// template <>
// class codecvt<char16_t, char8_t, mbstate_t>
//     : public locale::facet,
//       public codecvt_base
// {
// public:
//     typedef char16_t  intern_type;
//     typedef char8_t   extern_type;
//     typedef mbstate_t state_type;
//     ...
// };

#include <cassert>
#include <locale>
#include <type_traits>

int main(int, char**) {
  using F = std::codecvt<char16_t, char8_t, std::mbstate_t>;
  static_assert(std::is_base_of_v<std::locale::facet, F>);
  static_assert(std::is_base_of_v<std::codecvt_base, F>);
  static_assert(std::is_same_v<F::intern_type, char16_t>);
  static_assert(std::is_same_v<F::extern_type, char8_t>);
  static_assert(std::is_same_v<F::state_type, std::mbstate_t>);
  std::locale l = std::locale::classic();
  assert(std::has_facet<F>(l));
  const F& f = std::use_facet<F>(l);
  (void)F::id;
  (void)f;
  return 0;
}
