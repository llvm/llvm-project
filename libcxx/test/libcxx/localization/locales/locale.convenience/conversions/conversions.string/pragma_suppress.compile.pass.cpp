//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// Ensure that users can suppress the wstring_convert deprecation warning
// using #pragma clang diagnostic ignored around the usage site.

// REQUIRES: c++17 || c++20 || c++23
// UNSUPPORTED: no-wide-characters

#include <codecvt>
#include <locale>
#include <string>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
void test() {
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
  (void)converter.to_bytes(std::basic_string<char16_t>(u"hello"));
  (void)converter.from_bytes(std::string("hello"));
}
#pragma clang diagnostic pop
