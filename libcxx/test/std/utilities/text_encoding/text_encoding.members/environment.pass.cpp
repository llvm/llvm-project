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

// libc++ is not built with C++26, and the implementation for this function is in a source file.
// XFAIL: * 

// class text_encoding

// text_encoding text_encoding::environment(); 

// Concerns:
// 1. text_encoding::environment() returns the encoding for the "C" locale, which should be the default for any C++ program.
// 2. text_encoding::environment() still returns the "C" locale encoding when the locale is set to "en_US.UTF-8".
// 3. text_encoding::environment() is affected by changes to the "LANG" environment variable. 

// The current implementation of text_encoding::environment() while conformant, 
// is unfortunately affected by changes to the "LANG" environment variable.

#include <cassert>
#include <clocale>
#include <cstdlib>
#include <string_view>
#include <text_encoding>

#include "platform_support.h" 
#include "test_macros.h"
#include "test_text_encoding.h"

int main() {

  { // 1
    auto te = std::text_encoding::environment(); 

    assert(te == std::text_encoding::environment());
    assert(te.mib() == std::text_encoding::id::ASCII);
    assert(te == std::text_encoding::id::ASCII);
    assert(std::string_view(te.name()) == "ANSI_X3.4-1968");
    assert(te == std::text_encoding("ANSI_X3.4-1968"));

    assert(std::text_encoding::environment_is<std::text_encoding::id::ASCII>());
  }

  { // 2
    std::setlocale(LC_ALL, "en_US.UTF-8");

    auto te = std::text_encoding::environment();

    assert(te == std::text_encoding::environment());
    assert(te.mib() == std::text_encoding::id::ASCII);
    assert(std::string_view(te.name()) == "ANSI_X3.4-1968");
    assert(te == std::text_encoding("ANSI_X3.4-1968"));

    assert(std::text_encoding::environment_is<std::text_encoding::id::ASCII>());
  }

  { // 3
    setenv("LANG", LOCALE_en_US_UTF_8, 1);
    
    auto te = std::text_encoding::environment();

    assert(te == std::text_encoding::environment());
    assert(te.mib() == std::text_encoding::id::UTF8);
    assert(std::string_view(te.name()) == "UTF-8");
    assert(te == std::text_encoding("UTF-8"));

    assert(std::text_encoding::environment_is<std::text_encoding::id::UTF8>());
  }
  
  return 0;
}
