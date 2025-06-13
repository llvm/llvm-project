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
#include "src/__support/math_extras.h"
#include "src/__support/wchar/mbstate.h"

#include "character_converter.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

CharacterConverter::CharacterConverter(mbstate *mbstate) { state = mbstate; }

void CharacterConverter::clear() {
  state->partial = 0;
  state->bytes_processed = 0;
  state->total_bytes = 0;
}

bool CharacterConverter::isComplete() {
  return state->bytes_processed == state->total_bytes;
}

int CharacterConverter::push(char32_t utf32) {
  state->partial = utf32;
  state->bytes_processed = 0;
  state->total_bytes = 0;

  // determine number of utf-8 bytes needed to represent this utf32 value
  constexpr char32_t ranges[] = {0x7f, 0x7ff, 0xffff, 0x10ffff};
  constexpr int num_ranges = 4;
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

  constexpr char8_t FIRST_BYTE_HEADERS[] = {0, 0xC0, 0xE0, 0xF0};
  constexpr char8_t CONTINUING_BYTE_HEADER = 0x80;

  // the number of bits per utf-8 byte that actually encode character
  // information not metadata (# of bits excluding the byte headers)
  constexpr size_t ENCODED_BITS_PER_UTF8 = 6;
  constexpr int MASK_ENCODED_BITS =
      mask_trailing_ones<unsigned int, ENCODED_BITS_PER_UTF8>();

  char32_t output;

  // Shift to get the next 6 bits from the utf32 encoding
  const char32_t shift_amount =
      (state->total_bytes - state->bytes_processed - 1) * ENCODED_BITS_PER_UTF8;
  if (state->bytes_processed == 0) {
    /*
      Choose the correct set of most significant bits to encode the length
      of the utf8 sequence. The remaining bits contain the most significant
      bits of the unicode value of the character.
    */
    output = FIRST_BYTE_HEADERS[state->total_bytes - 1] |
             (state->partial >> shift_amount);
  } else {
    // Get the next 6 bits and format it like so: 10xxxxxx
    output = CONTINUING_BYTE_HEADER |
             ((state->partial >> shift_amount) & MASK_ENCODED_BITS);
  }

  state->bytes_processed++;
  return (char8_t)output;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
