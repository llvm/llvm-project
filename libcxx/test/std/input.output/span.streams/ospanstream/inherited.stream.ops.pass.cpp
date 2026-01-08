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

//   Test stream operations inherited from `basic_ostream`

#include <algorithm>
#include <cassert>
#include <span>
#include <spanstream>
#include <string>
#include <string_view>

#include "constexpr_char_traits.h"
#include "test_macros.h"

#include "../helper_macros.h"
#include "../helper_types.h"

#include <print>    // REMOVE ME
#include <iostream> // REMOVE ME

template <typename CharT, typename TraitsT>
void test() {
  using SpStream = std::basic_ospanstream<CharT, TraitsT>;

  constexpr std::basic_string_view<CharT, TraitsT> sv{SV("zmt 94 hkt 82 pir 43vr")};
  constexpr auto arrSize{30UZ};
  assert(sv.size() < arrSize);

  constexpr std::basic_string_view<CharT, TraitsT> sv2{SV("This string should overflow! This string should overflow!")};
  assert(sv2.size() >= arrSize);

  // Create a std::span test value
  CharT arr[arrSize]{};
  std::ranges::copy(sv, arr);
  std::span<CharT> sp{arr};

  // `std::span` + Mode: default (`out`)
  {
    // Construct stream
    SpStream spSt(sp);

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());
    assert(spSt.span().size() == 0);

    assert(arr[0] == CH('z')); // Check underlying buffer
    assert(arr[1] == CH('m'));
    assert(arr[2] == CH('t'));

    // Write to stream
    spSt << SV("snt");

    assert(spSt.span().size() == 3);

    assert(spSt.span()[0] == CH('s'));
    assert(spSt.span()[1] == CH('n'));
    assert(spSt.span()[2] == CH('t'));

    assert(arr[0] == CH('s')); // Check underlying buffer
    assert(arr[1] == CH('n'));
    assert(arr[2] == CH('t'));

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());

    spSt << 71;

    assert(spSt.span().size() == 5);

    assert(spSt.span()[3] == CH('7'));
    assert(spSt.span()[4] == CH('1'));

    assert(arr[3] == CH('7')); // Check underlying buffer
    assert(arr[4] == CH('1'));

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());

    spSt.put(CH('!'));

    assert(spSt.span().size() == 6);

    assert(spSt.span()[5] == CH('!'));

    assert(arr[5] == CH('!')); // Check underlying buffer

    spSt.write(CS("?#?"), 1);

    assert(spSt.span().size() == 7);

    assert(spSt.span()[6] == CH('?'));

    assert(arr[6] == CH('?')); // Check underlying buffer

    assert(spSt.tellp() == 7);

    // Write to stream with overflow
    spSt << sv2;

    assert(spSt.span().size() == 30);

    assert(!spSt);
    assert(spSt.bad());
    assert(spSt.fail());
    assert(!spSt.good());

    // Test error state
    assert(spSt.tellp() == -1);

    // Clear stream
    spSt.clear();
    spSt.seekp(0);

    assert(spSt.span().size() == 0);

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());
  }
}

int main(int, char**) {
  test<char, std::char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // test<wchar_t, std::char_traits<wchar_t>>();
#endif

  return 0;
}
