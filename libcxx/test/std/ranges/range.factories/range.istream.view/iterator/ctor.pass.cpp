//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr explicit iterator(basic_istream_view& parent) noexcept;

#include <cassert>
#include <ranges>
#include <sstream>

#include "test_macros.h"
#include "../utils.h"

// test that the constructor is explicit
template <class CharT>
using IstreamView = std::ranges::basic_istream_view<int, CharT>;
template <class CharT>
using Iter = std::ranges::iterator_t<IstreamView<CharT>>;

static_assert(std::constructible_from<Iter<char>, IstreamView<char>&>);
static_assert(!std::convertible_to<IstreamView<char>&, Iter<char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::constructible_from<Iter<wchar_t>, IstreamView<wchar_t>&>);
static_assert(!std::convertible_to<IstreamView<wchar_t>&, Iter<wchar_t>>);
#endif

// test that the constructor is noexcept
static_assert(std::is_nothrow_constructible_v<Iter<char>, IstreamView<char>&>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_nothrow_constructible_v<Iter<wchar_t>, IstreamView<wchar_t>&>);
#endif

template <class CharT>
void test() {
  auto iss = make_string_stream<CharT>("123");
  std::ranges::basic_istream_view<int, CharT> isv{iss};
  Iter<CharT> it{isv};
  ++it;
  assert(*it == 123);
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
