//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: c++11 && gcc
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// Construct a string_view from an invalid length
// constexpr basic_string_view( const _CharT* s, size_type len )

#include <string_view>

#include "check_assertion.h"
#include "test_macros.h"

// We're testing for assertions here, so let's not diagnose the misuses at compile time
// FIXME: This should really be in ADDITIONAL_COMPILE_FLAGS, but it that doesn't work due to a Clang bug
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wnonnull")

int main(int, char**) {
  char c = 0;
  TEST_LIBCPP_ASSERT_FAILURE(
      std::string_view(&c, -1), "string_view::string_view(_CharT *, size_t): length does not fit in difference_type");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::string_view(nullptr, 1), "string_view::string_view(_CharT *, size_t): received nullptr");
  return 0;
}
