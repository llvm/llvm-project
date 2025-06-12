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
    int numOnes = cpp::countl_one(utf8_byte);
    switch (numOnes) {
    // 1 byte total
    case 0:
      state->total_bytes = 1;
      break;
    // 2 bytes total
    case 2:
      state->total_bytes = 2;
      utf8_byte &= 0x1F;
      break;
    // 3 bytes total
    case 3:
      state->total_bytes = 3;
      utf8_byte &= 0x0F;
      break;
    // 4 bytes total
    case 4:
      state->total_bytes = 4;
      utf8_byte &= 0x07;
      break;
    // Invalid first byte
    default:
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

ErrorOr<char32_t> CharacterConverter::pop_utf32() {
  char32_t utf32;
  utf32 = state->partial;
  // if pop is called too early
  if (!isComplete()) {
    return Error(-1);
  }
  // reset if successful pop
  state->bytes_processed = 0;
  state->total_bytes = 0;
  state->partial = static_cast<char32_t>(0);
  return utf32;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
