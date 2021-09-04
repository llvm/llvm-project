//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-format
// UNSUPPORTED: libcpp-has-no-wide-characters

// Validate it works regardless of the signedness of `char`.
// RUN: %{cxx} %{flags} %{compile_flags} -fsigned-char -fsyntax-only %s
// RUN: %{cxx} %{flags} %{compile_flags} -funsigned-char -fsyntax-only %s

// <format>

// [format.arg]/5.2
// - otherwise, if T is char and char_type is wchar_t, initializes value with static_cast<wchar_t>(v);

#include <format>
#include <string>

void test() {
  std::make_format_args<std::basic_format_context<
      std::back_insert_iterator<std::basic_string<wchar_t>>, wchar_t>>('c');
}
