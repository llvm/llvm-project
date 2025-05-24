//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26
// REQUIRES: locale.fr_CA.ISO8859-1

// UNSUPPORTED: no-localization
// UNSUPPORTED: android
// UNSUPPORTED: windows
// UNSUPPORTED: availability-te-environment-missing

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <text_encoding>

#include "platform_support.h" // locale name macros

int main(int, char**) {
  // text_encoding::environment() is (unfortunately) affected by changes to the "LANG" environment variable on POSIX systems.
  {
    setenv("LANG", LOCALE_fr_CA_ISO8859_1, 1);

    auto te = std::text_encoding::environment();

    assert(std::text_encoding::environment_is<std::text_encoding::id::ISOLatin1>());
    assert(te == std::text_encoding::environment());
    assert(te.mib() == std::text_encoding::id::ISOLatin1);
    assert(std::ranges::contains(te.aliases(), std::string_view("ISO_8859-1")));
  }
}
