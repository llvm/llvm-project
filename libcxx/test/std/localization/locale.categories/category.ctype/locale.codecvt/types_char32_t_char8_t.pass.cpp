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

// template <>
// class codecvt<char32_t, char8_t, mbstate_t>
//     : public locale::facet,
//       public codecvt_base
// {
// public:
//     typedef char32_t  intern_type;
//     typedef char8_t   extern_type;
//     typedef mbstate_t state_type;
//     ...
// };

#include <cassert>
#include <locale>
#include <type_traits>

int main(int, char**) {
  using F = std::codecvt<char32_t, char8_t, std::mbstate_t>;
  static_assert(std::is_base_of_v<std::locale::facet, F>);
  static_assert(std::is_base_of_v<std::codecvt_base, F>);
  static_assert(std::is_same_v<F::intern_type, char32_t>);
  static_assert(std::is_same_v<F::extern_type, char8_t>);
  static_assert(std::is_same_v<F::state_type, std::mbstate_t>);
  std::locale l = std::locale::classic();
  assert(std::has_facet<F>(l));
  const F& f = std::use_facet<F>(l);
  (void)F::id;
  (void)f;
  return 0;
}
