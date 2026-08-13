//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr explicit basic_istream_view(basic_istream<CharT, Traits>& stream);

#include <cassert>
#include <ranges>
#include <sstream>

#include "test_macros.h"
#include "utils.h"

// test that the constructor is explicit
static_assert(std::constructible_from<std::ranges::istream_view<int>, std::istream&>);
static_assert(!std::convertible_to<std::istream&, std::ranges::istream_view<int>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::constructible_from<std::ranges::wistream_view<int>, std::wistream&>);
static_assert(!std::convertible_to<std::wistream&, std::ranges::wistream_view<int>>);
#endif

struct NoopExtraction {
  int n_; // intentionally doesn't have default member initializer
};

template <class CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits>& is, const NoopExtraction&) {
  return is; // intentionally doesn't modify the NoopExtraction object
}

template <class CharT>
void test() {
  // test constructor init the stream pointer to the passed one
  {
    auto iss = make_string_stream<CharT>("123");
    std::ranges::basic_istream_view<int, CharT> isv{iss};
    auto it = isv.begin();
    assert(*it == 123);
  }

  // LWG 3568. value_ must be initialized for basic_istream_view to be constexpr-constructible
  {
    static auto fn_local_iss            = make_string_stream<CharT>("");
    [[maybe_unused]] constexpr auto isv = std::views::istream<int>(fn_local_iss);
  }

  // LWG 3568. basic_istream_view needs to initialize value_
  {
    auto iss = make_string_stream<CharT>("123");
    std::ranges::basic_istream_view<NoopExtraction, CharT> isv{iss};
    auto it = isv.begin();
    assert((*it).n_ == 0);
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
