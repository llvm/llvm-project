//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

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
#include <vector>

#include <print>

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
  std::vector<CharT> vec(arr, arr + arrSize);

  // `std::span` + Mode: default (`out`)
  {
    std::span<CharT> sp{arr};
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
    assert(spSt.span().size() == 0);
  }
  {
    std::span<CharT> sp{arr};
    SpStream spSt(sp);

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());
    assert(spSt.span().size() == 0);

    spSt << 10;

    assert(spSt.span().size() == 2);

    spSt.clear();

    // assert(spSt);
    // assert(!spSt.bad());
    // assert(!spSt.fail());
    // assert(spSt.good());
    // assert(spSt.span().size() == 0);

    spSt << SV("gh");
    assert(spSt.span().size() == 4);
    std::println(stderr, "{}", spSt.span()[0]);
    std::println(stderr, "{}", arr[0]);
    assert(spSt.span()[0] == '1');
    assert(arr[0] == '1');
    assert(spSt.span()[1] == '0');
    assert(arr[1] == '0');
    assert(spSt.span()[2] == 'g');
    assert(arr[2] == 'g');

    CharT output_buffer[30];
    std::basic_ospanstream<CharT, TraitsT> os{std::span<CharT>{output_buffer}};

    assert(os.good());
    assert(!os.fail());
    assert(!os.bad());
    os << 10 << 20 << 30;
    os << SV("GH");
    assert(os.good());
    assert(!os.fail());
    assert(!os.bad());
    std::println(stderr, "{}", os.span()[0]);
    std::println(stderr, "{}", output_buffer[0]);
    std::println(stderr, "{}", os.span().size());
    std::println(stderr, "{}", os.span()[6]);
    std::println(stderr, "{}", output_buffer[7]);
    assert(false);
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // test<wchar_t>();
#endif

  return 0;
}
