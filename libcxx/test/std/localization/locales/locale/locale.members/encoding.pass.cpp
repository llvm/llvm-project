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
// UNSUPPORTED: availability-te-environment-missing

// class locale

// text_encoding locale::encoding() const

#include <cassert>
#include <concepts>
#include <format>
#include <iostream>
#include <locale>
#include <text_encoding>

#include "platform_support.h"

int main(int, char**) {
  {
    // 1. Locale built with en_US.UTF-8 returns text_encoding representing "UTF-8"
    const std::locale utf8_locale(LOCALE_en_US_UTF_8);
    std::same_as<std::text_encoding> decltype(auto) te = utf8_locale.encoding();
    auto utf8_te                                       = std::text_encoding{std::text_encoding::UTF8};

    if (te != std::text_encoding::UTF8) {
      std::cerr << std::format("Expected UTF-8, received {{ {}, \"{}\" }}", int(te.mib()), te.name());
      assert(false);
    }
    assert(te == utf8_te);
  }
#if defined(_WIN32)
  {
    // BCP-47 locale name
    const std::locale loc("en-US");
    std::same_as<std::text_encoding> decltype(auto) te = loc.encoding();
    auto w1252                                         = std::text_encoding{std::text_encoding::windows1252};
    assert(te == std::text_encoding::windows1252);
    assert(te == w1252);
  }
#endif
  return 0;
}
