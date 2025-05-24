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
// UNSUPPORTED: android
// UNSUPPORTED: availability-te-environment-missing

// <text_encoding>

// text_encoding text_encoding::environment();

#include <cassert>
#include <clocale>
#include <text_encoding>

#include "../test_text_encoding.h"
#include "platform_support.h"

int main(int, char**) {
  std::text_encoding te = std::text_encoding::environment();
  // 1. Depending on the platform's default, verify that environment() returns the corresponding text encoding.
  {
#if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    assert(te.mib() == std::text_encoding::ASCII);
    assert(std::text_encoding::environment_is<std::text_encoding::ASCII>());
#elif defined(_WIN32)
    assert(te.mib() == std::text_encoding::windows1252);
    assert(std::text_encoding::environment_is<std::text_encoding::windows1252>());
#endif
  }

  // 2. text_encoding::environment() still returns the default locale encoding when the locale is set to "en_US.UTF-8".
  {
    std::setlocale(LC_ALL, LOCALE_en_US_UTF_8);

    std::text_encoding te2 = std::text_encoding::environment();
    assert(te2 != std::text_encoding::UTF8);
    assert(te == te2);
  }

  return 0;
}
