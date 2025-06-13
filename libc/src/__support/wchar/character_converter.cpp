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
#include "src/__support/error_or.h"
#include "src/__support/wchar/mbstate.h"

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

ErrorOr<char8_t> CharacterConverter::pop_utf8() {
  if (state->bytes_processed >= state->total_bytes)
    return Error(-1);

  const char8_t first_byte_headers[] = {0, 0xC0, 0xE0, 0xF0};
  const char32_t utf32 = state->partial;
  const char32_t tot_bytes = state->total_bytes;
  const char32_t bytes_proc = state->bytes_processed;

  char32_t output;
  // Shift to get the next 6 bits from the utf32 encoding
  const char32_t shift_amount = (tot_bytes - bytes_proc - 1) * 6;
  if (state->bytes_processed == 0) {
    /*
      Choose the correct set of most significant bits to encode the length
      of the utf8 sequence. The remaining bits contain the most significant
      bits of the unicode value of the character.
    */
    output = first_byte_headers[tot_bytes - 1] | (utf32 >> shift_amount);
  } else {
    // Get the next 6 bits and format it like so: 10xxxxxx
    output = 0x80 | ((utf32 >> shift_amount) & 0x3f);
  }

  state->bytes_processed++;
  return (char8_t)output;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
