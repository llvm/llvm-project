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
// UNSUPPORTED: android

// class locale

// text_encoding encoding() const

#include <cassert>
#include <locale>
#include <text_encoding>

#include "test_macros.h"
#include "platform_support.h"

using id = std::text_encoding::id;

int main(int, char**) {
// FIXME: enable once locale::encoding() is implemented
#if false
  {
    // 1. Default locale returns a text_encoding representing "ASCII"
    std::locale loc;

    auto te        = loc.encoding();
    auto classicTE = std::text_encoding(id::ASCII);
    assert(te == id::ASCII);
    assert(te == classicTE);
  }

  {
    // 2. Locale built with en_US.UTF-8 returns text_encoding representing "UTF-8"
    std::locale utf8Locale(LOCALE_en_US_UTF_8);

    auto te     = utf8Locale.encoding();
    auto utf8TE = std::text_encoding(id::UTF8);
    assert(te == id::UTF8);
    assert(te == utf8TE);
  }
#endif
  return 0;
}
