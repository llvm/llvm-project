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
// class basic_osyncstream;

// basic_osyncstream& operator=(basic_osyncstream&& rhs);

#include <syncstream>
#include <sstream>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

template <class CharT, bool propagate>
static void test() {
  using Traits    = std::char_traits<CharT>;
  using Allocator = std::conditional_t<propagate, other_allocator<CharT>, test_allocator<CharT>>;
  static_assert(std::allocator_traits<Allocator>::propagate_on_container_move_assignment::value == propagate);

  using OS = std::basic_osyncstream<CharT, Traits, Allocator>;

  std::basic_stringbuf<CharT, Traits, Allocator> base1;
  std::basic_stringbuf<CharT, Traits, Allocator> base2;

  {
    OS out1{&base1, Allocator{42}};
    assert(out1.get_wrapped() == &base1);

    typename OS::syncbuf_type* sb1 = out1.rdbuf();
    assert(sb1->get_wrapped() == &base1);
    assert(sb1->get_allocator().get_data() == 42);

    out1 << CharT('A');

    static_assert(!noexcept(out1.operator=(std::move(out1)))); // LWG-3867
    OS out2{&base2, Allocator{99}};

    out2 << CharT('Z');

    // Validate the data is still in the syncbuf and not in the stringbuf.
    assert(base1.str().empty());
    assert(base2.str().empty());

    out2 = std::move(out1);

    // Since sb2 is overwritten by the move its data should be in its stringbuf.
    assert(base1.str().empty());
    assert(base2.str().size() == 1);
    assert(base2.str()[0] == CharT('Z'));

    assert(out2.get_wrapped() == &base1);

    typename OS::syncbuf_type* sb2 = out2.rdbuf();
    assert(sb2->get_wrapped() == &base1);
    if constexpr (std::allocator_traits<Allocator>::propagate_on_container_move_assignment::value)
      assert(sb2->get_allocator().get_data() == 42);
    else
      assert(sb2->get_allocator().get_data() == 99);

    assert(out1.get_wrapped() == nullptr);
    assert(sb1->get_wrapped() == nullptr);

    // The data written to 2 will be stored in sb1. The write happens after the destruction.
    out2 << CharT('B');
    assert(base1.str().empty());
  }

  assert(base1.str().size() == 2);
  assert(base1.str()[0] == CharT('A'));
  assert(base1.str()[1] == CharT('B'));
  assert(base2.str().size() == 1);
  assert(base2.str()[0] == CharT('Z'));
}

template <class CharT>
static void test() {
  test<CharT, true>();
  test<CharT, false>();
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
