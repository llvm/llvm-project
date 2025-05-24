
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// libc++ not built with C++26 yet
// XFAIL: * 
// REQUIRES: std-at-least-c++26
// REQUIRES: locale.en_US.UTF-8
// UNSUPPORTED: no-localization

// class locale

// text_encoding encoding() const

// Concerns:
// 1. Default locale returns a text_encoding representing "ASCII"
// 2. Locale built with en_US.UTF-8 returns text_encoding representing "UTF-8"

#include <cassert>
#include <locale>
#include <text_encoding>

#include "test_macros.h"
#include "platform_support.h"

using id = std::text_encoding::id;

int main() {

  {
    std::locale loc;

    auto te = loc.encoding(); 
    auto classicTE = std::text_encoding(id::ASCII);
    assert(te == id::ASCII);
    assert(te == classicTE);
  }

  {
    std::locale utf8Locale(LOCALE_en_US_UTF_8);

    auto te = utf8Locale.encoding();
    auto utf8TE = std::text_encoding(id::UTF8);
    assert(te == id::UTF8);
    assert(te == utf8TE);
  }

  return 0; 
}
