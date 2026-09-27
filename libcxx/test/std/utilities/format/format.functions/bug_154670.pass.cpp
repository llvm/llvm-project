//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// XFAIL: availability-fp_to_chars-missing

// https://llvm.org/PR154670
//
// Formatting into a back_insert_iterator of a string, vector or deque goes
// through a fixed-size buffer. Tests that the code unit written after an
// argument or a fill that ends at a multiple of the buffer's size is not
// written past the end of that buffer.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <format>
#include <iterator>
#include <string>
#include <vector>

#include "make_string.h"
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT, class Container>
void test(std::size_t size) {
  const std::basic_string<CharT> arg(size, CharT('a'));
  const std::basic_string<CharT> expected = arg + CharT('!');

  { // The argument is copied.
    Container out;
    std::format_to(std::back_inserter(out), SV("{}!"), arg);
    assert(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
  }
  if (size != 0) { // The argument's padding is filled.
    Container out;
    std::format_to(std::back_inserter(out), SV("{:a<{}}!"), SV(""), size);
    assert(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
  }
  for (std::size_t n : {size, size + 1, size + 2}) {
    Container out;
    auto result = std::format_to_n(std::back_inserter(out), n, SV("{}!"), arg);
    assert(result.size == static_cast<std::ptrdiff_t>(expected.size()));
    std::size_t written = std::min(n, expected.size());
    assert(std::equal(out.begin(), out.end(), expected.begin(), expected.begin() + written));
  }
}

template <class CharT>
void test() {
  for (std::size_t size : {0, 1, 255, 256, 257, 511, 512, 513, 1024}) {
    test<CharT, std::basic_string<CharT>>(size);
    test<CharT, std::vector<CharT>>(size);
    test<CharT, std::deque<CharT>>(size);
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
