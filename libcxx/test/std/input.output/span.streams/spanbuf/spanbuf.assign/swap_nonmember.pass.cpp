//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <spanstream>

//   template<class charT, class traits>
//     void swap(basic_spanbuf<charT, traits>& x, basic_spanbuf<charT, traits>& y);

#include <cassert>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "test_convertible.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  CharT arr[4];

  std::span<CharT> sp{arr};
  assert(sp.data() == arr);
  assert(!sp.empty());
  assert(sp.size() == 4);

  // Mode: default (`in` | `out`)
  {
    SpBuf rhsSpBuf{sp};
    assert(rhsSpBuf.span().data() == arr);
    // Mode `out` counts read characters
    assert(rhsSpBuf.span().empty());
    assert(rhsSpBuf.span().size() == 0);

    SpBuf spBuf;
    assert(spBuf.span().data() == nullptr);
    // Mode `out` counts read characters
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);

    std::swap(spBuf, rhsSpBuf);
    assert(spBuf.span().data() == arr);
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);
    assert(rhsSpBuf.span().data() == nullptr);
    assert(rhsSpBuf.span().empty());
    assert(rhsSpBuf.span().size() == 0);
  }
  // Mode: `ios_base::in`
  {
    SpBuf rhsSpBuf{sp, std::ios_base::in};
    assert(rhsSpBuf.span().data() == arr);
    assert(!rhsSpBuf.span().empty());
    assert(rhsSpBuf.span().size() == 4);

    SpBuf spBuf(std::span<CharT>{});
    assert(spBuf.span().data() == nullptr);
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);

    std::swap(spBuf, rhsSpBuf);
    assert(spBuf.span().data() == arr);
    assert(!spBuf.span().empty());
    assert(spBuf.span().size() == 4);
    assert(rhsSpBuf.span().data() == nullptr);
    assert(rhsSpBuf.span().empty());
    assert(rhsSpBuf.span().size() == 0);
  }
  // Mode `ios_base::out`
  {
    SpBuf rhsSpBuf{sp, std::ios_base::out};
    assert(rhsSpBuf.span().data() == arr);
    // Mode `out` counts read characters
    assert(rhsSpBuf.span().empty());
    assert(rhsSpBuf.span().size() == 0);

    SpBuf spBuf;
    assert(spBuf.span().data() == nullptr);
    // Mode `out` counts read characters
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);

    std::swap(spBuf, rhsSpBuf);
    assert(spBuf.span().data() == arr);
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);
    assert(rhsSpBuf.span().data() == nullptr);
    assert(rhsSpBuf.span().empty());
    assert(rhsSpBuf.span().size() == 0);
  }
  // Mode: multiple
  {
    SpBuf rhsSpBuf{sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary};
    assert(rhsSpBuf.span().data() == arr);
    // Mode `out` counts read characters
    assert(rhsSpBuf.span().empty());
    assert(rhsSpBuf.span().size() == 0);

    SpBuf spBuf;
    assert(spBuf.span().data() == nullptr);
    // Mode `out` counts read characters
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);

    std::swap(spBuf, rhsSpBuf);
    assert(spBuf.span().data() == arr);
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);
    assert(rhsSpBuf.span().data() == nullptr);
    assert(rhsSpBuf.span().empty());
    assert(rhsSpBuf.span().size() == 0);
  }
  // Mode: `ios_base::ate`
  {
    SpBuf rhsSpBuf{sp, std::ios_base::out | std::ios_base::ate};
    assert(rhsSpBuf.span().data() == arr);
    assert(!rhsSpBuf.span().empty());
    assert(rhsSpBuf.span().size() == 4);

    SpBuf spBuf(std::span<CharT>{});
    assert(spBuf.span().data() == nullptr);
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);

    std::swap(spBuf, rhsSpBuf);
    assert(spBuf.span().data() == arr);
    assert(!spBuf.span().empty());
    assert(spBuf.span().size() == 4);
    assert(rhsSpBuf.span().data() == nullptr);
    assert(rhsSpBuf.span().empty());
    assert(rhsSpBuf.span().size() == 0);
  }
}

int main(int, char**) {
  test<char>();
  test<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
  test<wchar_t, constexpr_char_traits<wchar_t>>();
#endif

  return 0;
}
