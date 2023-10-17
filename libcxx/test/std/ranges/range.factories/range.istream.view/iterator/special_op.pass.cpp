//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14, c++17

//    iterator(const iterator&) = delete;
//    iterator(iterator&&) = default;
//    iterator& operator=(const iterator&) = delete;
//    iterator& operator=(iterator&&) = default;

#include <cassert>
#include <ranges>
#include <sstream>
#include <type_traits>

#include "test_macros.h"
#include "../utils.h"

template <class CharT>
using Iter = std::ranges::iterator_t<std::ranges::basic_istream_view<int, CharT>>;
static_assert(!std::copy_constructible<Iter<char>>);
static_assert(!std::is_copy_assignable_v<Iter<char>>);
static_assert(std::move_constructible<Iter<char>>);
static_assert(std::is_move_assignable_v<Iter<char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::copy_constructible<Iter<wchar_t>>);
static_assert(!std::is_copy_assignable_v<Iter<wchar_t>>);
static_assert(std::move_constructible<Iter<wchar_t>>);
static_assert(std::is_move_assignable_v<Iter<wchar_t>>);
#endif

template <class CharT>
void test() {
  // test move constructor
  {
    auto iss = make_string_stream<CharT>("12   3");
    std::ranges::basic_istream_view<int, CharT> isv{iss};
    auto it  = isv.begin();
    auto it2 = std::move(it);
    assert(*it2 == 12);
  }

  // test move assignment
  {
    auto iss1 = make_string_stream<CharT>("12   3");
    std::ranges::basic_istream_view<int, CharT> isv1{iss1};
    auto iss2 = make_string_stream<CharT>("45   6");
    std::ranges::basic_istream_view<int, CharT> isv2{iss2};

    auto it1 = isv1.begin();
    assert(*it1 == 12);
    it1 = isv2.begin();
    assert(*it1 == 45);
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
