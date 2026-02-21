//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// text_encoding text_encoding::literal() noexcept;

#include <cassert>
#include <text_encoding>

constexpr bool test() {
  std::text_encoding te = std::text_encoding::literal();
#ifdef __GNUC_EXECUTION_CHARSET_NAME
  assert(std::string_view(te.name()) == std::string_view(__GNUC_EXECUTION_CHARSET_NAME));
#elif defined(__clang_literal_encoding__)
  assert(std::string_view(te.name()) == std::string_view(__clang_literal_encoding__));
#else
  assert(te.mib() = std::text_encoding::id::unknown);
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
