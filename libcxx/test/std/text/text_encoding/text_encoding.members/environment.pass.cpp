//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// REQUIRES: locale.en_US.UTF-8

// UNSUPPORTED: no-localization
// UNSUPPORTED: availability-te-environment-missing
// UNSUPPORTED: LLVM-LIBC-FIXME

// <text_encoding>

// text_encoding text_encoding::environment();

#include <cassert>
#include <clocale>
#include <format>
#include <iostream>
#include <text_encoding>

#include "platform_support.h"
#include "test_macros.h"

int main(int, char**) {
  auto check_env = []() {
#if defined(__ANDROID__)
    constexpr std::text_encoding::id expected_id = std::text_encoding::UTF8;
#elif defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    constexpr std::text_encoding::id expected_id = std::text_encoding::ASCII;
#elif defined(_WIN32)
    constexpr std::text_encoding::id expected_id = std::text_encoding::windows1252;
#elif defined(_AIX)
    constexpr std::text_encoding::id expected_id = std::text_encoding::ISOLatin1;
#else
    constexpr std::text_encoding::id expected_id = std::text_encoding::unknown;
#endif

    std::same_as<std::text_encoding> decltype(auto) te = std::text_encoding::environment();

    bool fail = false;
    if (te != expected_id) {
      std::cerr << std::format(
          "Environment mismatch: Expected ID {}, received: {{{},{}}}\n", int(expected_id), int(te.mib()), te.name());
      fail = true;
    }
    std::same_as<bool> decltype(auto) env_is_expected = std::text_encoding::environment_is<expected_id>();
    if (!env_is_expected) {
      fail = true;
    }

    return !fail;
  };

  {
    // 1. Depending on the platform's default, verify that environment() returns the corresponding text encoding.
    assert(check_env());
  }

  auto te = std::text_encoding::environment();
  // 2. text_encoding::environment()'s return value isn't altered by changes to locale.
  {
    std::setlocale(LC_ALL, LOCALE_en_US_UTF_8);

    auto te2 = std::text_encoding::environment();
    assert(te == te2);
  }

  {
    std::setlocale(LC_CTYPE, LOCALE_en_US_UTF_8);

    auto te2 = std::text_encoding::environment();
    assert(te == te2);
  }
  return 0;
}
