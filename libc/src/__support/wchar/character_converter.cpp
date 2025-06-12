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

int CharacterConverter::push(char8_t utf8_byte) { return utf8_byte; }

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
  if (state->total_bytes == 0) {
    return -1;
  }

  return 0;
}

utf_ret<char8_t> CharacterConverter::pop_utf8_seqlength1() {
  utf_ret<char8_t> result;
  result.error = 0;

  // 0xxxxxxx
  switch (state->bytes_processed) {
  case 0:
    result.out = (char8_t)(state->partial);
    break;
  default:
    result.error = -1;
    return result;
  }

  state->bytes_processed++;
  return result;
}

utf_ret<char8_t> CharacterConverter::pop_utf8_seqlength2() {
  utf_ret<char8_t> result;
  result.error = 0;

  // 110xxxxx 10xxxxxx
  char32_t utf32 = state->partial;
  switch (state->bytes_processed) {
  case 0:
    result.out = (char8_t)(0xC0 | (utf32 >> 6));
    break;
  case 1:
    result.out = (char8_t)(0x80 | (utf32 & 0x3f));
    break;
  default:
    result.error = -1;
    return result;
  }

  state->bytes_processed++;
  return result;
}

utf_ret<char8_t> CharacterConverter::pop_utf8_seqlength3() {
  utf_ret<char8_t> result;
  result.error = 0;

  // 1110xxxx 10xxxxxx 10xxxxxx
  char32_t utf32 = state->partial;
  switch (state->bytes_processed) {
  case 0:
    result.out = (char8_t)(0xE0 | (utf32 >> 12));
    break;
  case 1:
    result.out = (char8_t)(0x80 | ((utf32 >> 6) & 0x3f));
    break;
  case 2:
    result.out = (char8_t)(0x80 | (utf32 & 0x3f));
    break;
  default:
    result.error = -1;
    return result;
  }

  state->bytes_processed++;
  return result;
}

utf_ret<char8_t> CharacterConverter::pop_utf8_seqlength4() {
  utf_ret<char8_t> result;
  result.error = 0;

  // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
  char32_t utf32 = state->partial;
  switch (state->bytes_processed) {
  case 0:
    result.out = (char8_t)(0xF0 | (utf32 >> 18));
    break;
  case 1:
    result.out = (char8_t)(0x80 | ((utf32 >> 12) & 0x3f));
    break;
  case 2:
    result.out = (char8_t)(0x80 | ((utf32 >> 6) & 0x3f));
    break;
  case 3:
    result.out = (char8_t)(0x80 | (utf32 & 0x3f));
    break;
  default:
    result.error = -1;
    return result;
  }

  state->bytes_processed++;
  return result;
}

utf_ret<char8_t> CharacterConverter::pop_utf8() {
  switch (state->total_bytes) {
  case 1:
    return pop_utf8_seqlength1();
  case 2:
    return pop_utf8_seqlength2();
  case 3:
    return pop_utf8_seqlength3();
  case 4:
    return pop_utf8_seqlength4();
  }

  return {.out = 0, .error = -1};
}

utf_ret<char32_t> CharacterConverter::pop_utf32() { return {0, -1}; }

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
