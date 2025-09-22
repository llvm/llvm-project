//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string_view>

#include "constexpr_char_traits.h"
#include "test_macros.h"
#include "type_algorithms.h"

template <typename CharT, typename TraitsT, typename AllocT>
constexpr void test() {
#if TEST_STD_VER >= 26
  std::basic_string_view<CharT, TraitsT> sv;

  sv.subview(); // expected-warning {{ignoring return value of function}}
#endif
}

class Test {
public:
  template <typename CharT>
  constexpr void operator()() const {
    test<CharT, std::char_traits<CharT>>();
  }
};

void test() { types::for_each(types::character_types(), Test{}); }
