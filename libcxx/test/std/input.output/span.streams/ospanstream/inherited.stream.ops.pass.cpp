//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <spanstream>

//   template<class charT, class traits = char_traits<charT>>
//   class basic_spanstream
//     : public basic_iostream<charT, traits> {

//   Test stream operations inherited from `basic_istream` and `basic_ostream`

#include <cassert>
#include <span>
#include <spanstream>
#include <string>
#include <string_view>

#include "constexpr_char_traits.h"
#include "test_macros.h"

#include "../helper_functions.h"
#include "../helper_macros.h"
#include "../helper_types.h"

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpStream = std::basic_ospanstream<CharT, TraitsT>;

  constexpr std::basic_string_view<CharT, TraitsT> sv{SV("zmt 94 hkt 82 pir 43vr")};
  constexpr auto arrSize{30UZ};
  assert(sv.size() < arrSize);

  // Create a std::span test value
  CharT arr[arrSize]{};
  initialize_array_from_string_view(arr, sv);

  std::span<CharT> sp{arr};

  // `std::span` + Mode: default (`out`)
  {
    SpStream spSt(sp);

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());
    assert(spSt.span().size() == 0);

    spSt.clear();

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
