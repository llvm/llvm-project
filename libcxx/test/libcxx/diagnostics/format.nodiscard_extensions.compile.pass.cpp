//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that format functions aren't marked [[nodiscard]] when
// _LIBCPP_DISBALE_NODISCARD_EXT is defined

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// UNSUPPORTED: c++03, c++11, c++14 ,c++17

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_NODISCARD_EXT

#include <format>

#include "test_macros.h"

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <locale>
#endif

void test() {
  std::format("");
  std::vformat("", std::make_format_args());
  std::formatted_size("");
  std::make_format_args();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::format(L"");
  std::vformat(L"", std::make_wformat_args());
  std::formatted_size(L"");
  std::make_wformat_args();
#endif // TEST_HAS_NO_WIDE_CHARACTERS

#ifndef TEST_HAS_NO_LOCALIZATION
  std::format(std::locale::classic(), "");
  std::vformat(std::locale::classic(), "", std::make_format_args());
  std::formatted_size(std::locale::classic(), "");
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::format(std::locale::classic(), L"");
  std::vformat(std::locale::classic(), L"", std::make_wformat_args());
  std::formatted_size(std::locale::classic(), L"");
#  endif // TEST_HAS_NO_WIDE_CHARACTERS
#endif   // TEST_HAS_NO_LOCALIZATION
}
