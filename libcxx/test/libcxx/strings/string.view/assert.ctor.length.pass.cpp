//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers

// UNSUPPORTED: c++03, c++11
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// Construct a string_view from an invalid length
// constexpr basic_string_view( const _CharT* s, size_type len )

#include <string_view>

#include "check_assertion.h"

int main(int, char**) {
  char c = 0;
  TEST_LIBCPP_ASSERT_FAILURE(
      std::string_view(&c, -1), "string_view::string_view(_CharT *, size_t): length does not fit in difference_type");
  return 0;
}
