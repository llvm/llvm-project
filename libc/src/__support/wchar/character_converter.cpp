//===-- Implementation of a class for conversion --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/char32_t.h"
#include "hdr/types/char8_t.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/utf_ret.h"

#include "character_converter.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

CharacterConverter::CharacterConverter(mbstate *mbstate) { state = mbstate; }

bool CharacterConverter::isComplete() {
  return state->bytes_processed == state->total_bytes;
}

int CharacterConverter::push(char8_t utf8_byte) {
  // Checking the first byte if first push
  if (state->bytes_processed == 0 && state->total_bytes == 0) {
    state->partial = static_cast<char32_t>(0);
    // 1 byte total
    if (cpp::countl_one(utf8_byte) == 0) {
      state->total_bytes = 1;
    }
    // 2 bytes total
    else if (cpp::countl_one(utf8_byte) == 2) {
      state->total_bytes = 2;
      utf8_byte &= 0x1F;
    }
    // 3 bytes total
    else if (cpp::countl_one(utf8_byte) == 3) {
      state->total_bytes = 3;
      utf8_byte &= 0x0F;
    }
    // 4 bytes total
    else if (cpp::countl_one(utf8_byte) == 4) {
      state->total_bytes = 4;
      utf8_byte &= 0x07;
    }
    // Invalid byte -> reset mbstate
    else {
      state->partial = static_cast<char32_t>(0);
      state->bytes_processed = 0;
      state->total_bytes = 0;
      return -1;
    }
    state->partial = static_cast<char32_t>(utf8_byte);
    state->bytes_processed++;
    return 0;
  }
  // Any subsequent push
  if (cpp::countl_one(utf8_byte) == 1 && !isComplete()) {
    char32_t byte = utf8_byte & 0x3F;
    state->partial = state->partial << 6;
    state->partial |= byte;
    state->bytes_processed++;
    return 0;
  }
  // Invalid byte -> reset if we didn't get successful complete read
  if (!isComplete()) {
    state->partial = static_cast<char32_t>(0);
    state->bytes_processed = 0;
    state->total_bytes = 0;
  }
  return -1;
}

utf_ret<char32_t> CharacterConverter::pop_utf32() {
  utf_ret<char32_t> utf32;
  utf32.error = 0;
  utf32.out = state->partial;
  if (!isComplete())
    utf32.error = -1;
  state->bytes_processed = 0;
  state->total_bytes = 0;
  state->partial = static_cast<char32_t>(0);
  return utf32;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
