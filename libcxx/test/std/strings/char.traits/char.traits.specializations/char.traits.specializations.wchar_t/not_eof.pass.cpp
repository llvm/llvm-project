//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<wchar_t>

// static constexpr int_type not_eof(int_type c);

// UNSUPPORTED: no-wide-characters

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  assert(std::char_traits<wchar_t>::not_eof(L'a') == L'a');
  assert(std::char_traits<wchar_t>::not_eof(L'A') == L'A');
  assert(std::char_traits<wchar_t>::not_eof(0) == 0);
  assert(std::char_traits<wchar_t>::not_eof(std::char_traits<wchar_t>::eof()) != std::char_traits<wchar_t>::eof());

  return 0;
}
