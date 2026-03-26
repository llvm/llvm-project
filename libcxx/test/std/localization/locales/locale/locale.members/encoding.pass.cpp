//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26
// REQUIRES: locale.en_US.UTF-8
// UNSUPPORTED: no-localization

// class locale

// text_encoding locale::encoding() const

#include <cassert>
#include <locale>
#include <text_encoding>

#include "platform_support.h"

using id = std::text_encoding::id;

int main(int, char**) {
  {
    // 1. Default locale returns a text_encoding representing "ASCII", or "UTF-8 on Android.
    const std::locale loc{};

    std::text_encoding te = loc.encoding();

#if !defined(__ANDROID__)
    std::text_encoding classic_te = std::text_encoding(id::ASCII);
    assert(te == id::ASCII);
    assert(te == classic_te);
#else
    auto utf8_te = std::text_encoding(id::UTF8);
    assert(te == id::UTF8);
    assert(te == utf8_te);
#endif
  }

  {
    // 2. Locale built with en_US.UTF-8 returns text_encoding representing "UTF-8"
    const std::locale utf8_locale(LOCALE_en_US_UTF_8);

    std::text_encoding te      = utf8_locale.encoding();
    std::text_encoding utf8_te = std::text_encoding(id::UTF8);
    assert(te == id::UTF8);
    assert(te == utf8_te);
  }
  return 0;
}
