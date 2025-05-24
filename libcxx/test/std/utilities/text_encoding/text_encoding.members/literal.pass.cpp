
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26

// UNSUPPORTED: no-localization

// class text_encoding

// text_encoding text_encoding::literal() noexcept;

// Concerns:
// 1. text_encoding::literal() returns the proper encoding depending on the compiler, else unknown.

#include <cassert>
#include <text_encoding>
#include <type_traits>
#include <string_view>

#include "test_macros.h"
#include "test_text_encoding.h"

int main() {
#if __CHAR_BIT__ == 8
 
  {
    auto te = std::text_encoding::literal();
#  ifdef __GNUC_EXECUTION_CHARSET_NAME
    assert(std::string_view(te.name()) == std::string_view(__GNUC_EXECUTION_CHARSET_NAME));
#  elif defined(__clang_literal_encoding__)
    assert(std::string_view(te.name()) == std::string_view(__clang_literal_encoding__));
#  elif defined(__clang__)
    assert(std::string_view(te.name()) == "UTF-8");
    assert(te.mib() == std::text_encoding::id::UTF8);
#  else
    assert(te.mib() = std::text_encoding::id::unknown);
#  endif
  }

#endif // if __CHAR_BIT__ == 8
}
