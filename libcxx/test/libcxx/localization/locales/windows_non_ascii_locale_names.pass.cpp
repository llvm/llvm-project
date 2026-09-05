//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// REQUIRES: windows

// The C RunTime library on Windows supports locale strings with
// characters outside the ASCII range. This poses challenges for
// code that temporarily set a custom thread locale.
//
// https://github.com/llvm/llvm-project/issues/160478

#include <locale>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <cstdlib>
#include <cassert>
#include <clocale>

#include "test_macros.h"

void locale_name_replace_codepage(std::string& locale_name, const std::string& codepage) {
  auto dot_position = locale_name.rfind('.');
  LIBCPP_ASSERT(dot_position != std::string::npos);

  locale_name = locale_name.substr(0, dot_position) + codepage;
}

int main(int, char**) {
  _configthreadlocale(_ENABLE_PER_THREAD_LOCALE);

  std::string locale_name = std::setlocale(LC_ALL, "norwegian-bokmal");

  const auto& not_ascii = [](char c) { return (c & 0x80) != 0; };
  LIBCPP_ASSERT(std::any_of(locale_name.begin(), locale_name.end(), not_ascii));

  locale_name_replace_codepage(locale_name, ".437");
  LIBCPP_ASSERT(std::setlocale(LC_ALL, locale_name.c_str()));

  std::cerr.imbue(std::locale::classic());
  std::cerr << std::setprecision(2) << 0.1 << std::endl;

  return EXIT_SUCCESS;
}
