//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_INPUTOUTPUT_STRINGSTREAMS_TYPES_H
#define TEST_STD_INPUTOUTPUT_STRINGSTREAMS_TYPES_H

#include <string_view>
#include <concepts>

#include "test_macros.h"

template <typename CharT, class Traits = std::char_traits<CharT>>
class CustomStringView {
public:
  explicit CustomStringView(const CharT* cs) : cs_{cs} {}

  template <std::same_as<std::basic_string_view<CharT, Traits>> T>
  operator T() const {
    return std::basic_string_view<CharT, Traits>(cs_);
  }

private:
  const CharT* cs_;
};

static_assert(std::constructible_from<std::basic_string_view<char>, CustomStringView<char>>);
static_assert(std::convertible_to<CustomStringView<char>, std::basic_string_view<char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::convertible_to<CustomStringView<wchar_t>, std::basic_string_view<wchar_t>>);
#endif

struct SomeObject {};

struct NonMode {};

struct NonAllocator {};

#endif // TEST_STD_INPUTOUTPUT_STRINGSTREAMS_TYPES_H
