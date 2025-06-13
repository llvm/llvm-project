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
#include "src/__support/error_or.h"
#include "src/__support/math_extras.h"
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
    uint8_t numOnes = static_cast<uint8_t>(cpp::countl_one(utf8_byte));
    // 1 byte total
    if (numOnes == 0) {
      state->total_bytes = 1;
    }
    // 2 through 4 bytes total
    else if (numOnes >= 2 && numOnes <= 4) {
      /* Since the format is 110xxxxx, 1110xxxx, and 11110xxx for 2, 3, and 4,
      we will make the base mask with 7 ones and right shift it as necessary. */
      constexpr size_t significant_bits = 7;
      state->total_bytes = numOnes;
      utf8_byte &=
          (mask_trailing_ones<uint32_t, significant_bits>() >> numOnes);
    }
    // Invalid first byte
    else {
      return -1;
    }
    state->partial = static_cast<char32_t>(utf8_byte);
    state->bytes_processed++;
    return 0;
  }
  // Any subsequent push
  // Adding 6 more bits so need to left shift
  constexpr size_t ENCODED_BITS_PER_UTF8 = 6;
  if (cpp::countl_one(utf8_byte) == 1 && !isComplete()) {
    char32_t byte =
        utf8_byte & mask_trailing_ones<uint32_t, ENCODED_BITS_PER_UTF8>();
    state->partial = state->partial << ENCODED_BITS_PER_UTF8;
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

ErrorOr<char32_t> CharacterConverter::pop_utf32() {
  // if pop is called too early
  if (!isComplete())
    return Error(-1);

  char32_t utf32 = state->partial;

  // reset if successful pop
  state->bytes_processed = 0;
  state->total_bytes = 0;
  state->partial = static_cast<char32_t>(0);
  return utf32;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
