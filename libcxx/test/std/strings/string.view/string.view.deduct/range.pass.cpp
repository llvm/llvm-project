//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <string_view>

//  template<class Range>
//    basic_string_view(Range&&) -> basic_string_view<ranges::range_value_t<Range>>; // C++23

#include <string_view>
#include <cassert>

#include "make_string.h"
#include "test_iterators.h"
#include "test_macros.h"

template<class CharT>
void test() {
  auto val = MAKE_STRING(CharT, "test");
  auto sv = std::basic_string_view(val);
  ASSERT_SAME_TYPE(decltype(sv), std::basic_string_view<CharT>);
  assert(sv.size() == val.size());
  assert(sv.data() == val.data());
}

void test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  test<char8_t>();
  test<char16_t>();
  test<char32_t>();
  test<char>();

  struct Widget {
    const char16_t *data_ = u"foo";
    contiguous_iterator<const char16_t*> begin() const { return contiguous_iterator<const char16_t*>(data_); }
    contiguous_iterator<const char16_t*> end() const { return contiguous_iterator<const char16_t*>(data_ + 3); }
  };
  std::basic_string_view bsv = Widget();
  ASSERT_SAME_TYPE(decltype(bsv), std::basic_string_view<char16_t>);
}

int main(int, char**) {
  test();

  return 0;
}

