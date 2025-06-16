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

void CharacterConverter::clear() {
  state->partial = 0;
  state->bytes_processed = 0;
  state->total_bytes = 0;
}

bool CharacterConverter::isComplete() {
  return state->bytes_processed == state->total_bytes;
}

int CharacterConverter::push(char8_t utf8_byte) {
  uint8_t num_ones = static_cast<uint8_t>(cpp::countl_one(utf8_byte));
  // Checking the first byte if first push
  if (state->bytes_processed == 0) {
    // UTF-8 char has 1 byte total
    if (num_ones == 0) {
      state->total_bytes = 1;
    }
    // UTF-8 char has 2 through 4 bytes total
    else if (num_ones >= 2 && num_ones <= 4) {
      /* Since the format is 110xxxxx, 1110xxxx, and 11110xxx for 2, 3, and 4,
      we will make the base mask with 7 ones and right shift it as necessary. */
      constexpr size_t SIGNIFICANT_BITS = 7;
      uint32_t base_mask = mask_trailing_ones<uint32_t, SIGNIFICANT_BITS>();
      state->total_bytes = num_ones;
      utf8_byte &= (base_mask >> num_ones);
    }
    // Invalid first byte
    else {
      // bytes_processed and total_bytes will always be 0 here
      state->partial = static_cast<char32_t>(0);
      return -1;
    }
    state->partial = static_cast<char32_t>(utf8_byte);
    state->bytes_processed++;
    return 0;
  }
  // Any subsequent push
  // Adding 6 more bits so need to left shift
  constexpr size_t ENCODED_BITS_PER_UTF8 = 6;
  if (num_ones == 1 && !isComplete()) {
    char32_t byte =
        utf8_byte & mask_trailing_ones<uint32_t, ENCODED_BITS_PER_UTF8>();
    state->partial = state->partial << ENCODED_BITS_PER_UTF8;
    state->partial |= byte;
    state->bytes_processed++;
    return 0;
  }
  // Invalid byte -> reset the state
  clear();
  return -1;
}

ErrorOr<char32_t> CharacterConverter::pop_utf32() {
  // If pop is called too early, do not reset the state, use error to determine
  // whether enough bytes have been pushed
  if (!isComplete() || state->bytes_processed == 0)
    return Error(-1);
  char32_t utf32 = state->partial;
  // reset if successful pop
  clear();
  return utf32;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
