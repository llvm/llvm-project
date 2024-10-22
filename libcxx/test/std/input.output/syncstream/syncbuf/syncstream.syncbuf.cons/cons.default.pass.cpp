//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-experimental-syncstream

// <syncstream>

// template <class charT, class traits, class Allocator>
// class basic_syncbuf;

// basic_syncbuf();

#include <cassert>
#include <concepts>
#include <syncstream>

#include "test_macros.h"
#include "constexpr_char_traits.h"
#include "test_allocator.h"

template <class CharT>
std::basic_syncbuf<CharT> lwg3253_default_constructor_is_not_explicit() {
  return {};
}

template <class CharT>
void test() {
  lwg3253_default_constructor_is_not_explicit<CharT>();

  {
    using Buf = std::basic_syncbuf<CharT>;
    static_assert(std::default_initializable<Buf>);
    Buf buf;
    assert(buf.get_wrapped() == nullptr);
    assert(buf.get_allocator() == std::allocator<CharT>());
  }
  {
    using Buf = std::basic_syncbuf<CharT, constexpr_char_traits<CharT>>;
    static_assert(std::default_initializable<Buf>);
    Buf buf;
    assert(buf.get_wrapped() == nullptr);
    assert(buf.get_allocator() == std::allocator<CharT>());
  }
  {
    using Buf = std::basic_syncbuf<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>;
    static_assert(std::default_initializable<Buf>);
    Buf buf;
    assert(buf.get_wrapped() == nullptr);
    assert(buf.get_allocator() == test_allocator<CharT>());
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
