//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-exceptions

// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing

#include <cstdint>
#include <format>
#include <string_view>

#include "fuzz.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  try {
    [[maybe_unused]] auto result = std::vformat(std::string_view{(const char*)(data), size}, std::make_format_args());
  } catch (std::format_error const&) {
    // If the fuzzing input isn't a valid thing we can format and we detect it, it's okay. We are looking for crashes.
    return 0;
  }
  return 0;
}
