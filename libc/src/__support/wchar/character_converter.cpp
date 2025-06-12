//===-- Implementation of a class for conversion --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/char32_t.h"
#include "hdr/types/char8_t.h"
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
    // 1 byte total
    if ((utf8_byte & 128) == 0) {
      state->total_bytes = 1;
      state->bytes_processed = 1;
      state->partial = static_cast<char32_t>(utf8_byte);
      return 0;
    }
    // 2 bytes total
    else if ((utf8_byte & 0xE0) == 0xC0) {
      state->total_bytes = 2;
      state->bytes_processed = 1;
      utf8_byte &= 0x1F;
      state->partial = static_cast<char32_t>(utf8_byte);
      return 0;
    }
    // 3 bytes total
    else if ((utf8_byte & 0xF0) == 0xE0) {
      state->total_bytes = 3;
      state->bytes_processed = 1;
      utf8_byte &= 0x0F;
      state->partial = static_cast<char32_t>(utf8_byte);
      return 0;
    }
    // 4 bytes total
    else if ((utf8_byte & 0xF8) == 0xF0) {
      state->total_bytes = 4;
      state->bytes_processed = 1;
      utf8_byte &= 0x07;
      state->partial = static_cast<char32_t>(utf8_byte);
      return 0;
    }
    // Invalid
    else {
      state->bytes_processed++;
      return -1;
    }
  }
  // Any subsequent push
  if ((utf8_byte & 0xC0) == 0x80) {
    state->partial = state->partial << 6;
    char32_t byte = utf8_byte & 0x3F;
    state->partial |= byte;
    state->bytes_processed++;
    return 0;
  }
  state->bytes_processed++;
  return -1;
}

utf_ret<char32_t> CharacterConverter::pop_utf32() {
  utf_ret<char32_t> utf32;
  utf32.error = 0;
  utf32.out = state->partial;
  return utf32;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
