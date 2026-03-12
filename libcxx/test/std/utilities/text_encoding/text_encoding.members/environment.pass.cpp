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
#include <text_encoding>

#include "platform_support.h"

int main(int, char**) {
#if !defined(__ANDROID__) || (defined(__ANDROID__) && __ANDROID_API__ >= 26)
  std::text_encoding te = std::text_encoding::environment();

  // 1. text_encoding::environment() still returns the default system encoding when the "LC_ALL" locale is changed.
  {
    std::setlocale(LC_ALL, LOCALE_en_US_UTF_8);

    std::text_encoding te2 = std::text_encoding::environment();
    assert(te2 != std::text_encoding::UTF8);
    assert(te == te2);
  }

  // 2. text_encoding::environment() still returns the default system encoding when the "LC_CTYPE" locale is changed.
  {
    std::setlocale(LC_CTYPE, LOCALE_en_US_UTF_8);

    std::text_encoding te2 = std::text_encoding::environment();
    assert(te2 != std::text_encoding::UTF8);
    assert(te == te2);
  }
#endif
  return 0;
}
