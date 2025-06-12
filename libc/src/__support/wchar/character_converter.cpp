//===-- Implementation of a class for conversion --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/char32_t.h"
#include "hdr/types/char8_t.h"
#include "src/__support/common.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/utf_ret.h"

#include "character_converter.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

CharacterConverter::CharacterConverter(mbstate *mbstate) { state = mbstate; }

bool CharacterConverter::isComplete() {
  return state->bytes_processed == state->total_bytes;
}

int CharacterConverter::push(char32_t utf32) {
  state->partial = utf32;
  state->bytes_processed = 0;
  state->total_bytes = 0;

  // determine number of utf-8 bytes needed to represent this utf32 value
  char32_t ranges[] = {0x7f, 0x7ff, 0xffff, 0x10ffff};
  const int num_ranges = 4;
  for (uint8_t i = 0; i < num_ranges; i++) {
    if (state->partial <= ranges[i]) {
      state->total_bytes = i + 1;
      break;
    }
  }
  if (state->total_bytes == 0)
    return -1;

  return 0;
}

utf_ret<char8_t> CharacterConverter::pop_utf8() {
  if (state->bytes_processed >= state->total_bytes)
    return {.out = 0, .error = -1};

  char8_t first_byte_headers[] = {0, 0xC0, 0xE0, 0xF0};
  char32_t utf32 = state->partial;
  char32_t tb = state->total_bytes;
  char32_t bp = state->bytes_processed;
  char32_t output;
  if (state->bytes_processed == 0) {
    /*
      Choose the correct set of most significant bits to encode the length
      of the utf8 sequence. The remaining bits contain the most significant
      bits of the unicode value of the character.
    */
    output = first_byte_headers[tb - 1] | (utf32 >> ((tb - 1) * 6));
  } else {
    // Get the next 6 bits and format it like so: 10xxxxxx
    const char32_t shift_amount = (tb - bp - 1) * 6;
    output = 0x80 | ((utf32 >> shift_amount) & 0x3f);
  }

  state->bytes_processed++;
  return {.out = (char8_t)output, .error = 0};
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
