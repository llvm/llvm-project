//===-- Definition of a class for mbstate_t and conversion -----*-- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CHARACTER_CONVERTER_H
#define LLVM_LIBC_SRC___SUPPORT_CHARACTER_CONVERTER_H

#include "hdr/errno_macros.h"
#include "hdr/types/char16_t.h"
#include "hdr/types/char32_t.h"
#include "hdr/types/char8_t.h"
#include "hdr/types/size_t.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/math_extras.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

class CharacterConverter {
private:
  mbstate *state;

  // This is for utf-8 bytes other than the first byte
  static constexpr size_t ENCODED_BITS_PER_UTF8 = 6;

  // The number of bits per utf-8 byte that actually encode character
  // Information not metadata (# of bits excluding the byte headers)
  static constexpr uint32_t MASK_ENCODED_BITS =
      mask_trailing_ones<uint32_t, ENCODED_BITS_PER_UTF8>();

  // Maximum value for utf-32 for a utf-8 sequence of a given length
  static constexpr char32_t MAX_VALUE_PER_UTF8_LEN[] = {0x7f, 0x7ff, 0xffff,
                                                        0x10ffff};
  static constexpr int MAX_UTF8_LENGTH = 4;

public:
  explicit LIBC_INLINE CharacterConverter(mbstate *state_ptr)
      : state(state_ptr) {}

  LIBC_INLINE void clear() {
    state->partial = 0;
    state->bytes_stored = 0;
    state->total_bytes = 0;
  }
  LIBC_INLINE bool isFull() {
    return state->bytes_stored == state->total_bytes && state->total_bytes != 0;
  }
  LIBC_INLINE bool isEmpty() { return state->bytes_stored == 0; }
  bool isValidState();

  /*
   * When converting between UTF-16 and UTF-8, the converter might be in a state
   * of either partially-pushing or partially-popping. For example consider the
   * following sequence:
   * *   push((char8_t)0xF0);
   * *   push((char8_t)0xA4);
   * *   [A](partial=0x24,bytes_stored=2,total_bytes=4)
   * *   push((char8_t)0xAD);
   * *   push((char8_t)0xA2);
   * *   (partial=0x24B62,bytes_stored=4,total_bytes=4)
   * *   pop_utf16(); -> 0xD852
   * *   [B](partial=0x24B62,bytes_stored=2,total_bytes=4)
   * *   pop_utf16(); -> 0xDF62
   * At point [A] and point [B], the stored bytes would both be 2. But at [A] we
   * must continue reading input, while at [B] we must write the output low
   * surrogate and clear the state.
   * Thus we must have some means to distinguish between pushing and popping.
   * Notice that: When pushing, we do left-shifts on state->partial before each
   * push. When popping, we keep state->partial at the complete code point.
   * Since we disallow overlong encodings, the complete UTF-8 code point is
   * always greater than
   * *   MAX_VALUE_PER_UTF8_LEN[state->total_bytes - 2]
   * but at most
   * *   MAX_VALUE_PER_UTF8_LEN[state->total_bytes - 1]
   * If we are partially through pushing, state->partial is at most
   * *   char32 >> 6
   * thus at most
   * *   MAX_VALUE_PER_UTF8_LEN[state->total_bytes - 1] >> 6
   * Observe that, an item in MAX_VALUE_PER_UTF8_LEN right-shifted by 6 is
   * always less than the preceding item. Based on these facts, we can determine
   * if we are partially through pushing or popping, by comparing state->partial
   * with MAX_VALUE_PER_UTF8_LEN[state->total_bytes - 2]
   * Also when converting UTF-16 -> UTF-8, in a partial state,
   * *   bytes_stored=2
   * *   total_bytes=4
   * so the threshold 0xFFFF is also correct.
   */
  LIBC_INLINE bool isPartiallyPopping() {
    if (isEmpty() || isFull())
      return false;
    // When neither empty nor full, state->total_bytes is at least 2
    char32_t threshold = MAX_VALUE_PER_UTF8_LEN[state->total_bytes - 2];
    return state->partial > threshold;
  }

  template <typename CharType> size_t sizeAs();

  int push(char8_t utf8_byte);
  int push(char16_t utf16);
  int push(char32_t utf32);

  ErrorOr<char8_t> pop_utf8();
  ErrorOr<char16_t> pop_utf16();
  ErrorOr<char32_t> pop_utf32();

#if defined(LIBC_TYPES_WCHAR_T_IS_UTF32) || defined(LIBC_TYPES_WCHAR_T_IS_UTF16)
  int push(wchar_t wchar);
  ErrorOr<wchar_t> pop_wchar();
#endif // LIBC_TYPES_WCHAR_T_IS_UTF32 || LIBC_TYPES_WCHAR_T_IS_UTF16

  template <typename CharType> ErrorOr<CharType> pop();
};

LIBC_INLINE bool CharacterConverter::isValidState() {
  if (state->total_bytes > MAX_UTF8_LENGTH)
    return false;

  const char32_t max_utf32_value =
      state->total_bytes == 0 ? 0
                              : MAX_VALUE_PER_UTF8_LEN[state->total_bytes - 1];
  return state->bytes_stored <= state->total_bytes &&
         state->partial <= max_utf32_value;
}

LIBC_INLINE int CharacterConverter::push(char8_t utf8_byte) {
  uint8_t num_ones = static_cast<uint8_t>(cpp::countl_one(utf8_byte));
  // Checking the first byte if first push
  if (isEmpty()) {
    // UTF-8 char has 1 byte total
    if (num_ones == 0) {
      state->total_bytes = 1;
    }
    // UTF-8 char has 2 through 4 bytes total
    else if (num_ones >= 2 && num_ones <= 4) {
      /* Since the format is 110xxxxx, 1110xxxx, and 11110xxx for 2, 3, and 4,
      we will make the base mask with 7 ones and right shift it as necessary. */
      constexpr size_t SIGNIFICANT_BITS = 7;
      char8_t base_mask =
          static_cast<char8_t>(mask_trailing_ones<uint8_t, SIGNIFICANT_BITS>());
      state->total_bytes = num_ones;
      utf8_byte &= (base_mask >> num_ones);
    }
    // Invalid first byte
    else {
      // bytes_stored and total_bytes will always be 0 here
      state->partial = static_cast<char32_t>(0);
      return EILSEQ;
    }
    state->partial = static_cast<char32_t>(utf8_byte);
    state->bytes_stored++;
    return 0;
  }
  // Pushing when partially popping is a misuse in the library
  if (isPartiallyPopping()) {
    clear();
    return -1;
  }
  // Any subsequent push
  // Adding 6 more bits so need to left shift
  if (num_ones == 1 && !isFull()) {
    char32_t byte = utf8_byte & MASK_ENCODED_BITS;
    state->partial = state->partial << ENCODED_BITS_PER_UTF8;
    state->partial |= byte;
    state->bytes_stored++;
    if (isFull()) {
      // Detect and reject overlong encodings
      // Needed for isPartiallyPopping to work
      if (state->partial <= MAX_VALUE_PER_UTF8_LEN[state->total_bytes - 2]) {
        clear();
        return EILSEQ;
      }
      // Also reject out-of-range code points
      if (state->partial > MAX_VALUE_PER_UTF8_LEN[state->total_bytes - 1]) {
        clear();
        return EILSEQ;
      }
    }
    return 0;
  }

  // Invalid byte -> reset the state
  clear();
  return EILSEQ;
}

LIBC_INLINE int CharacterConverter::push(char16_t utf16) {
  // Checking for high surrogate if first push
  if (isEmpty()) {
    bool is_surrogate = (utf16 & 0xF800) == 0xD800;
    if (is_surrogate) {
      bool is_high_surrogate = (utf16 & 0x400) == 0;
      if (!is_high_surrogate) {
        // unpaired surrogate -> reset the state
        // bytes_stored and total_bytes will always be 0 here
        state->partial = static_cast<char32_t>(0);
        return EILSEQ;
      }
      // high surrogate
      // code point between 0x10000 and 0x10FFFF
      state->total_bytes = 4;
      state->bytes_stored = 2;
      // 10 bits each from the high surrogate and the low surrogate, and the
      // full code point is from 0x10000, so we add 0x40 to partial, which after
      // shifting becomes 0x10000
      state->partial = static_cast<char32_t>(utf16 & 0x3FF) + 0x40;
    } else {
      state->partial = static_cast<char32_t>(utf16);
      // determine number of utf-8 bytes needed to represent this utf16 value
      uint8_t i;
      for (i = 0; i < 2; i++) {
        if (state->partial <= MAX_VALUE_PER_UTF8_LEN[i])
          break;
      }
      state->total_bytes = i + 1;
      state->bytes_stored = i + 1;
    }
    return 0;
  }
  // Pushing when partially popping is a misuse in the library
  if (isFull() || isPartiallyPopping()) {
    clear();
    return -1;
  }
  bool is_low_surrogate = (utf16 & 0xFC00) == 0xDC00;
  if (!is_low_surrogate) {
    // unpaired surrogate -> reset the state
    clear();
    return EILSEQ;
  }
  char32_t low = utf16 & 0x3FF;
  state->partial <<= 10;
  state->partial |= low;
  state->bytes_stored += 2;
  return 0;
}

LIBC_INLINE int CharacterConverter::push(char32_t utf32) {
  // we can't be partially through a conversion when pushing a utf32 value
  if (!isEmpty())
    return -1;

  state->partial = utf32;

  // determine number of utf-8 bytes needed to represent this utf32 value
  for (uint8_t i = 0; i < MAX_UTF8_LENGTH; i++) {
    if (state->partial <= MAX_VALUE_PER_UTF8_LEN[i]) {
      state->total_bytes = i + 1;
      state->bytes_stored = i + 1;
      return 0;
    }
  }

  // `utf32` contains a value that is too large to actually represent a valid
  // unicode character
  clear();
  return EILSEQ;
}

LIBC_INLINE ErrorOr<char32_t> CharacterConverter::pop_utf32() {
  // If pop is called too early, do not reset the state, use error to determine
  // whether enough bytes have been pushed
  if (!isFull())
    return Error(-1);
  char32_t utf32 = state->partial;
  // reset if successful pop
  clear();
  return utf32;
}

LIBC_INLINE ErrorOr<char16_t> CharacterConverter::pop_utf16() {
  if (!isFull() && !isPartiallyPopping())
    return Error(-1);
  char32_t output;
  if (state->total_bytes <= 3) {
    // no surrogates needed
    output = state->partial;
    // disallow using UTF-8 or UTF-32 to encode unpaired surrogates
    bool is_surrogate = (output & 0xF800) == 0xD800;
    if (is_surrogate) {
      clear();
      return Error(EILSEQ);
    }
    clear();
  } else {
    if (isFull()) {
      // high surrogate first
      output = ((state->partial - 0x10000) >> 10) | 0xD800;
      state->bytes_stored = 2;
    } else {
      // low surrogate
      output = ((state->partial - 0x10000) & 0x3FF) | 0xDC00;
      clear();
    }
  }
  return static_cast<char16_t>(output);
}

LIBC_INLINE ErrorOr<char8_t> CharacterConverter::pop_utf8() {
  if (!isFull() && !isPartiallyPopping())
    return Error(-1);

  constexpr char8_t FIRST_BYTE_HEADERS[] = {0, 0xC0, 0xE0, 0xF0};
  constexpr char8_t CONTINUING_BYTE_HEADER = 0x80;

  char32_t output;

  // Shift to get the next 6 bits from the utf32 encoding
  const size_t shift_amount = (state->bytes_stored - 1) * ENCODED_BITS_PER_UTF8;
  if (isFull()) {
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

  state->bytes_stored--;
  if (state->bytes_stored == 0)
    clear();

  return static_cast<char8_t>(output);
}

template <> LIBC_INLINE ErrorOr<char8_t> CharacterConverter::pop() {
  return pop_utf8();
}

template <> LIBC_INLINE ErrorOr<char16_t> CharacterConverter::pop() {
  return pop_utf16();
}

template <> LIBC_INLINE ErrorOr<char32_t> CharacterConverter::pop() {
  return pop_utf32();
}

template <> LIBC_INLINE size_t CharacterConverter::sizeAs<char8_t>() {
  return state->total_bytes;
}

template <> LIBC_INLINE size_t CharacterConverter::sizeAs<char16_t>() {
  return state->total_bytes > 3 ? 2 : 1;
}

template <> LIBC_INLINE size_t CharacterConverter::sizeAs<char32_t>() {
  return 1;
}

#if defined(LIBC_TYPES_WCHAR_T_IS_UTF32)

LIBC_INLINE int CharacterConverter::push(wchar_t wchar) {
  return push(static_cast<char32_t>(wchar));
}

LIBC_INLINE ErrorOr<wchar_t> CharacterConverter::pop_wchar() {
  ErrorOr<char32_t> Result = pop_utf32();
  if (!Result)
    return Error(Result.error());
  return static_cast<wchar_t>(*Result);
}

template <> LIBC_INLINE ErrorOr<wchar_t> CharacterConverter::pop() {
  return pop_wchar();
}

template <> LIBC_INLINE size_t CharacterConverter::sizeAs<wchar_t>() {
  return sizeAs<char32_t>();
}

#elif defined(LIBC_TYPES_WCHAR_T_IS_UTF16)

LIBC_INLINE int CharacterConverter::push(wchar_t wchar) {
  return push(static_cast<char16_t>(wchar));
}

LIBC_INLINE ErrorOr<wchar_t> CharacterConverter::pop_wchar() {
  ErrorOr<char16_t> Result = pop_utf16();
  if (!Result)
    return Error(Result.error());
  return static_cast<wchar_t>(*Result);
}

template <> LIBC_INLINE ErrorOr<wchar_t> CharacterConverter::pop() {
  return pop_wchar();
}

template <> LIBC_INLINE size_t CharacterConverter::sizeAs<wchar_t>() {
  return sizeAs<char16_t>();
}

#endif // LIBC_TYPES_WCHAR_T_IS_UTF32 || LIBC_TYPES_WCHAR_T_IS_UTF16

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_CHARACTER_CONVERTER_H
