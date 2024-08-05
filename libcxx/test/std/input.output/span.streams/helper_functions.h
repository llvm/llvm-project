//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_INPUTOUTPUT_SPANSTREAMS_FUNCTIONS_H
#define TEST_STD_INPUTOUTPUT_SPANSTREAMS_FUNCTIONS_H

#include <string_view>

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void initialize_array_from_string_view(CharT* arr, std::basic_string_view<CharT, TraitsT> sv) {
  for (std::size_t i = 0; i != sv.size(); ++i) {
    arr[i] = sv[i];
  }
}

#endif // TEST_STD_INPUTOUTPUT_SPANSTREAMS_FUNCTIONS_H
