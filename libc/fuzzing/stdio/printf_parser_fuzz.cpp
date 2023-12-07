//===-- printf_parser_fuzz.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc qsort implementation.
///
//===----------------------------------------------------------------------===//

#ifndef LIBC_COPT_MOCK_ARG_LIST
#error The printf Parser Fuzzer must be compiled with LIBC_COPT_MOCK_ARG_LIST, and the parser itself must also be compiled with that option when it's linked against the fuzzer.
#endif

#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/parser.h"

#include <stdarg.h>
#include <stdint.h>

using namespace __llvm_libc;

// The design for the printf parser fuzzer is fairly simple. The parser uses a
// mock arg list that will never fail, and is passed a randomized string. The
// format sections it outputs are checked against a count of the number of '%'
// signs are in the original string. This is a fairly basic test, and the main
// intent is to run this under sanitizers, which will check for buffer overruns.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  char *in_str = new char[size + 1];

  for (size_t i = 0; i < size; ++i)
    in_str[i] = data[i];

  in_str[size] = '\0';

  auto mock_arg_list = internal::MockArgList();

  auto parser = printf_core::Parser(in_str, mock_arg_list);

  int str_percent_count = 0;

  for (size_t i = 0; i < size && in_str[i] != '\0'; ++i) {
    if (in_str[i] == '%') {
      ++str_percent_count;
    }
  }

  int section_percent_count = 0;

  for (printf_core::FormatSection cur_section = parser.get_next_section();
       !cur_section.raw_string.empty();
       cur_section = parser.get_next_section()) {
    if (cur_section.has_conv) {
      ++section_percent_count;
      if (cur_section.conv_name == '%') {
        ++section_percent_count;
      }
    } else if (cur_section.raw_string[0] == '%') {
      // If the conversion would be undefined, it's instead raw, but it still
      // starts with a %.
      ++section_percent_count;
    }
  }

  if (str_percent_count != section_percent_count) {
    __builtin_trap();
  }

  delete[] in_str;
  return 0;
}
