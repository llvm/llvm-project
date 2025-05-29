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
// UNSUPPORTED: windows

// class text_encoding

// text_encoding text_encoding::environment();

// Concerns:
// 1. text_encoding::environment() returns the encoding for the environment's default locale.
// 2. text_encoding::environment() still returns the default locale encoding when the locale is set to "en_US.UTF-8".
// 3. text_encoding::environment() is affected by changes to the "LANG" environment variable.

// The current implementation of text_encoding::environment() while conformant,
// is unfortunately affected by changes to the "LANG" environment variable.

#include <cassert>
#include <clocale>
#include <cstdlib>
#include <string>
#include <string_view>
#include <text_encoding>

#include "platform_support.h"
#include "test_macros.h"
#include "test_text_encoding.h"

std::string extractEncodingFromLocale(std::string locale_str) {
  auto dot_pos = locale_str.find('.'), at_pos = locale_str.find('@');

  if (dot_pos == std::string::npos) {
    return "ANSI_X3.4-1968"; // default is ASCII
  }

  if (at_pos == std::string::npos) {
    return locale_str.substr(dot_pos + 1);
  }

  return locale_str.substr(dot_pos + 1, at_pos - 1 - dot_pos);
}

int main() {
  auto default_locale   = std::setlocale(LC_ALL, nullptr);
  auto default_encoding = extractEncodingFromLocale(std::string(default_locale));
  auto default_te       = std::text_encoding(default_encoding);

  { // 1
    auto env_te = std::text_encoding::environment();
    assert(env_te == std::text_encoding::environment());
    assert(checkTextEncoding(env_te, default_te));
  }

  { // 2
    std::setlocale(LC_ALL, LOCALE_en_US_UTF_8);

    auto env_te = std::text_encoding::environment();

    assert(checkTextEncoding(env_te, default_te));
  }

  { // 3
    setenv("LANG", LOCALE_en_US_UTF_8, 1);

    auto te = std::text_encoding::environment();

    assert(te == std::text_encoding::environment());
    assert(te.mib() == std::text_encoding::id::UTF8);
    assert(std::string_view(te.name()) == "UTF-8");
    assert(checkTextEncoding(te, std::text_encoding("UTF-8")));

    assert(std::text_encoding::environment_is<std::text_encoding::id::UTF8>());
  }

  return 0;
}
