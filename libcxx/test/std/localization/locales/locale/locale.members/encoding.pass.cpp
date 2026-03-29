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
#include <format>
#include <iostream>
#include <locale>
#include <text_encoding>

#include "platform_support.h"

int main(int, char**) {
  {
    // 1. Locale built with en_US.UTF-8 returns text_encoding representing "UTF-8"
    const std::locale utf8_locale(LOCALE_en_US_UTF_8);
    std::text_encoding te      = utf8_locale.encoding();
    std::text_encoding utf8_te = std::text_encoding(std::text_encoding::id::UTF8);

    if (te != std::text_encoding::id::UTF8) {
      std::cerr << std::format("Expected UTF-8, received {{ {}, \"{}\" }}", int(te.mib()), te.name());
      assert(false);
    }
    assert(te == utf8_te);
  }
  return 0;
}
