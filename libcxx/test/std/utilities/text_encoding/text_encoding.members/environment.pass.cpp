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

// class text_encoding

// text_encoding text_encoding::environment();

// Concerns:
// 1. Depending on the environment text_encoding mib, verify that environment_is returns true for that mib.
// 2. text_encoding::environment() still returns the default locale encoding when the locale is set to "en_US.UTF-8".
// 3. text_encoding::environment() is affected by changes to the "LANG" environment variable, except for Windows.

// The current implementation of text_encoding::environment() while conformant,
// is unfortunately affected by changes to the "LANG" environment variable.

#include "test_text_encoding.h"

using id = std::text_encoding::id;
int main() {
  auto env_te = std::text_encoding::environment();
  // 1
  {
    auto mib = env_te.mib();

    if (mib == std::text_encoding::ASCII) {
      assert(std::text_encoding::environment_is<std::text_encoding::ASCII>());
    }
    if (mib == std::text_encoding::UTF8) {
      assert(std::text_encoding::environment_is<std::text_encoding::UTF8>());
    }
    if (mib == std::text_encoding::ISOLatin1) {
      assert(std::text_encoding::environment_is<std::text_encoding::ISOLatin1>());
    }
    if (mib == std::text_encoding::windows1252) {
      assert(std::text_encoding::environment_is<std::text_encoding::windows1252>());
    }
  }

  { // 2
    std::setlocale(LC_ALL, LOCALE_en_US_UTF_8);

    auto env_te2 = std::text_encoding::environment();

    assert(checkTextEncoding(env_te, env_te2));
  }

#if !defined(_WIN32)
  { // 3
    setenv("LANG", LOCALE_en_US_UTF_8, 1);

    auto te = std::text_encoding::environment();

    assert(te == std::text_encoding::environment());
    assert(te.mib() == std::text_encoding::id::UTF8);
    assert(std::string_view(te.name()) == "UTF-8");
    assert(checkTextEncoding(te, std::text_encoding("UTF-8")));

    assert(std::text_encoding::environment_is<std::text_encoding::id::UTF8>());
  }
#endif
  return 0;
}
