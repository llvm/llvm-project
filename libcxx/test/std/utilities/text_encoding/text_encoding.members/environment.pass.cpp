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

// <text_encoding>

// text_encoding text_encoding::environment();

#include <cassert>
#include <clocale>
#include <format>
#include <iostream>
#include <text_encoding>

#include "platform_support.h"

int main(int, char**) {
#if !defined(__ANDROID__) || (defined(__ANDROID__) && __ANDROID_API__ >= 26)
  auto check_env = []() {
#  if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    constexpr std::text_encoding::id expected_id = std::text_encoding::ASCII;
#  elif defined(__ANDROID__)
    constexpr std::text_encoding::id expected_id = std::text_encoding::UTF8;
#  elif defined(_WIN32)
    constexpr std::text_encoding::id expected_id = std::text_encoding::windows1252;
#  elif defined(_AIX)
    constexpr std::text_encoding::id expected_id = std::text_encoding::ISOLatin1;
#  else
    constexpr std::text_encoding::id expected_id = std::text_encoding::unknown;
#  endif

    std::text_encoding te = std::text_encoding::environment();
    bool fail             = false;
    if (te != expected_id) {
      std::cerr << std::format(
          "Environment mismatch: Expected ID {}, received: {{{},{}}}\n", int(expected_id), int(te.mib()), te.name());
      fail = true;
    }

    if (!std::text_encoding::environment_is<expected_id>()) {
      fail = true;
    }

    return !fail;
  };

  {
    // 1. Depending on the platform's default, verify that environment() returns the corresponding text encoding.
    assert(check_env());
  }

  std::text_encoding te = std::text_encoding::environment();
  // 2. text_encoding::environment() still returns the default locale encoding when the locale is set to "en_US.UTF-8".
  {
    std::setlocale(LC_ALL, LOCALE_en_US_UTF_8);

    std::text_encoding te2 = std::text_encoding::environment();
    assert(te2 != std::text_encoding::UTF8);
    assert(te == te2);
  }

  {
    std::setlocale(LC_CTYPE, LOCALE_en_US_UTF_8);

    std::text_encoding te2 = std::text_encoding::environment();
    assert(te2 != std::text_encoding::UTF8);
    assert(te == te2);
  }
#endif
  return 0;
}
