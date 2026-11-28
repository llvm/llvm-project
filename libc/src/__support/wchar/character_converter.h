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

  static constexpr char16_t UTF16_SURROGATE_START = 0xD800;
  static constexpr char16_t UTF16_SURROGATE_END = 0xDFFF;
  static constexpr char16_t UTF16_HIGH_SURROGATE_START = 0xD800;
  static constexpr char16_t UTF16_HIGH_SURROGATE_END = 0xDBFF;
  static constexpr char16_t UTF16_LOW_SURROGATE_START = 0xDC00;
  static constexpr char16_t UTF16_LOW_SURROGATE_END = 0xDFFF;
  static constexpr char32_t UTF16_SURROGATE_ENCODED_BASE = 0x10000;
  static constexpr size_t ENCODED_BITS_PER_UTF16_SURROGATE = 10;
  static constexpr uint16_t MASK_ENCODED_BITS_UTF16_SURROGATE =
      mask_trailing_ones<uint16_t, ENCODED_BITS_PER_UTF16_SURROGATE>();

public:
  explicit LIBC_INLINE CharacterConverter(mbstate *state_ptr)
      : state(state_ptr) {}

  LIBC_INLINE void clear() {
    state->partial = 0;
    state->conversion_state = MBConversionState::EMPTY;
    state->bytes_stored = 0;
    state->total_bytes = 0;
  }
  LIBC_INLINE bool isFull() {
    return state->conversion_state == MBConversionState::FULL;
  }
  LIBC_INLINE bool isEmpty() {
    return state->conversion_state == MBConversionState::EMPTY;
  }
  bool isValidState();
  LIBC_INLINE bool isPartiallyPopping() {
    return state->conversion_state == MBConversionState::POPPING;
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

  // mbrtowc, mbrtoc16, mbrtoc32
  // mbrtoc16 may return a surrogate pair in 2 consecutive calls, and specify -3
  // as a return value (for the second call). mbrtowc doesn't specify -3 as a
  // return value. Set allowPartialPop to true for mbrtoc16.
  template <typename T, bool allowPartialPop>
  ErrorOr<size_t> mbrto_generic(T *__restrict dest_ptr,
                                const char *__restrict src_ptr,
                                size_t max_src_bytes);
  // wcrtomb, c16rtomb, c32rtomb
  // c16rtomb may consume a surrogate pair in 2 consecutive calls, and return 0
  // on the first call. wcrtomb doesn't use that behavior. Set allowPartialPush
  // to true for c16rtomb.
  template <typename T, bool allowPartialPush>
  ErrorOr<size_t> rtomb_generic(char *__restrict dest_ptr, T src);
};

LIBC_INLINE bool CharacterConverter::isValidState() {
  if (state->total_bytes > MAX_UTF8_LENGTH)
    return false;

  const char32_t max_utf32_value =
      state->total_bytes == 0 ? 0
                              : MAX_VALUE_PER_UTF8_LEN[state->total_bytes - 1];
  if (state->bytes_stored > state->total_bytes ||
      state->partial > max_utf32_value)
    return false;

  switch (state->conversion_state) {
  case MBConversionState::EMPTY:
    return state->bytes_stored == 0;
  case MBConversionState::FULL:
    return state->total_bytes != 0 && state->bytes_stored == state->total_bytes;
  case MBConversionState::PUSHING:
  case MBConversionState::POPPING:
    return state->bytes_stored != 0 && state->bytes_stored < state->total_bytes;
  }
  return false;
}

LIBC_INLINE int CharacterConverter::push(char8_t utf8_byte) {
  const uint8_t num_ones = static_cast<uint8_t>(cpp::countl_one(utf8_byte));
  // Checking the first byte if first push
  if (isEmpty()) {
    // UTF-8 char has 1 byte total
    if (num_ones == 0) {
      state->conversion_state = MBConversionState::FULL;
      state->total_bytes = 1;
    }
    // UTF-8 char has 2 through 4 bytes total
    else if (num_ones >= 2 && num_ones <= 4) {
      state->conversion_state = MBConversionState::PUSHING;
      /* Since the format is 110xxxxx, 1110xxxx, and 11110xxx for 2, 3, and 4,
      we will make the base mask with 7 ones and right shift it as necessary. */
      constexpr size_t SIGNIFICANT_BITS = 7;
      constexpr char8_t base_mask =
          static_cast<char8_t>(mask_trailing_ones<uint8_t, SIGNIFICANT_BITS>());
      state->total_bytes = num_ones;
      utf8_byte &= (base_mask >> num_ones);
    }
    // Invalid first byte
    else {
      // state is already empty here
      return EILSEQ;
    }
    state->partial = static_cast<char32_t>(utf8_byte);
    state->bytes_stored++;
    return 0;
  }
  // Any subsequent push
  // Adding 6 more bits so need to left shift
  if (num_ones == 1 && !isFull()) {
    LIBC_ASSERT(state->conversion_state == MBConversionState::PUSHING);
    const char32_t byte = utf8_byte & MASK_ENCODED_BITS;
    state->partial = state->partial << ENCODED_BITS_PER_UTF8;
    state->partial |= byte;
    state->bytes_stored++;
    if (state->bytes_stored == state->total_bytes) {
      state->conversion_state = MBConversionState::FULL;
    }
    return 0;
  }

  // Invalid byte -> reset the state
  clear();
  return EILSEQ;
}

LIBC_INLINE int CharacterConverter::push(char16_t utf16) {
  const bool is_high_surrogate =
      utf16 >= UTF16_HIGH_SURROGATE_START && utf16 <= UTF16_HIGH_SURROGATE_END;
  const bool is_low_surrogate =
      utf16 >= UTF16_LOW_SURROGATE_START && utf16 <= UTF16_LOW_SURROGATE_END;
  // Checking for high surrogate if first push
  if (isEmpty()) {
    if (is_low_surrogate) {
      // unpaired surrogate -> reset the state
      // bytes_stored and total_bytes will always be 0 here
      state->partial = static_cast<char32_t>(0);
      return EILSEQ;
    }
    if (is_high_surrogate) {
      // high surrogate
      // code point between 0x10000 and 0x10FFFF
      state->conversion_state = MBConversionState::PUSHING;
      state->total_bytes = 4;
      state->bytes_stored = 2;
      constexpr char32_t ENCODED_BASE_PRESHIFT =
          UTF16_SURROGATE_ENCODED_BASE >> ENCODED_BITS_PER_UTF16_SURROGATE;
      state->partial =
          static_cast<char32_t>(utf16 & MASK_ENCODED_BITS_UTF16_SURROGATE) +
          ENCODED_BASE_PRESHIFT;
    } else {
      state->partial = static_cast<char32_t>(utf16);
      state->conversion_state = MBConversionState::FULL;
      // determine number of utf-8 bytes needed to represent this utf16 value
      uint8_t i;
      for (i = 0; i < 2 && MAX_VALUE_PER_UTF8_LEN[i] < state->partial; i++)
        ;
      state->total_bytes = i + 1;
      state->bytes_stored = i + 1;
    }
    return 0;
  }
  LIBC_ASSERT(state->conversion_state == MBConversionState::PUSHING);
  if (!is_low_surrogate) {
    // unpaired surrogate -> reset the state
    clear();
    return EILSEQ;
  }
  const char32_t low = utf16 & MASK_ENCODED_BITS_UTF16_SURROGATE;
  state->partial <<= 10;
  state->partial |= low;
  state->conversion_state = MBConversionState::FULL;
  state->bytes_stored = 4;
  return 0;
}

LIBC_INLINE int CharacterConverter::push(char32_t utf32) {
  // we can't be partially through a conversion when pushing a utf32 value
  if (!isEmpty())
    return -1;

  state->partial = utf32;
  state->conversion_state = MBConversionState::FULL;

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
  const char32_t utf32 = state->partial;
  // reset if successful pop
  clear();
  return utf32;
}

LIBC_INLINE ErrorOr<char16_t> CharacterConverter::pop_utf16() {
  if (isEmpty())
    return Error(-1);
  LIBC_ASSERT(state->conversion_state != MBConversionState::PUSHING);

  char32_t output;
  // state->total_bytes refers to UTF-8 bytes
  // 3 byte UTF-8 has max code point U+FFFF and can fit in 1 UTF-16 code unit.
  if (state->total_bytes <= 3) {
    // no surrogates needed
    output = state->partial;
    // disallow using UTF-8 or UTF-32 to encode unpaired surrogates
    const bool is_surrogate =
        output >= static_cast<char32_t>(UTF16_SURROGATE_START) &&
        output <= static_cast<char32_t>(UTF16_SURROGATE_END);
    if (is_surrogate) {
      clear();
      return Error(EILSEQ);
    }
    clear();
  } else {
    if (isFull()) {
      // high surrogate first
      output = ((state->partial - UTF16_SURROGATE_ENCODED_BASE) >>
                ENCODED_BITS_PER_UTF16_SURROGATE) |
               UTF16_HIGH_SURROGATE_START;
      state->conversion_state = MBConversionState::POPPING;
      state->bytes_stored = 2;
    } else {
      // low surrogate
      output = ((state->partial - UTF16_SURROGATE_ENCODED_BASE) &
                MASK_ENCODED_BITS_UTF16_SURROGATE) |
               UTF16_LOW_SURROGATE_START;
      clear();
    }
  }
  return static_cast<char16_t>(output);
}

LIBC_INLINE ErrorOr<char8_t> CharacterConverter::pop_utf8() {
  if (isEmpty())
    return Error(-1);
  LIBC_ASSERT(state->conversion_state != MBConversionState::PUSHING);

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
  else
    state->conversion_state = MBConversionState::POPPING;

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

template <typename T, bool allowPartialPop>
LIBC_INLINE ErrorOr<size_t>
CharacterConverter::mbrto_generic(T *__restrict dest_ptr,
                                  const char *__restrict src_ptr,
                                  size_t max_src_bytes) {
  if (!isValidState())
    return Error(EINVAL);
  if (!allowPartialPop && isPartiallyPopping()) {
    // don't allow a complete char32 spanning over 2 consecutive calls of char16
    clear();
    return Error(EILSEQ);
  }

  char empty_src = '\0';
  if (src_ptr == nullptr) {
    dest_ptr = nullptr;
    src_ptr = &empty_src;
    max_src_bytes = 1;
  }

  size_t i = 0;
  // If we still have the lower part to write, don't read yet (UTF-16)
  if (!isPartiallyPopping()) {
    // Reading in bytes until we have a complete char32 or error
    for (; i < max_src_bytes && !isFull(); ++i) {
      int err = push(static_cast<char8_t>(src_ptr[i]));
      // Encoding error
      if (err != 0)
        return Error(err);
    }
  }
  if (!isFull() && !isPartiallyPopping()) {
    // We haven't read in a full char32 yet.
    // Incomplete but potentially valid
    return (size_t)-2;
  }
  auto result = pop<T>();
  if (!result.has_value()) {
    return Error(result.error());
  }
  if (!allowPartialPop && !isEmpty()) {
    // don't allow a complete char32 spanning over 2 consecutive calls of char16
    clear();
    return Error(EILSEQ);
  }
  if (dest_ptr != nullptr)
    *dest_ptr = result.value();
  if (i == 0) {
    // lower part written, no input processed
    return (size_t)-3;
  }
  // null terminator -> return 0
  if (result.value() == T{})
    return 0;
  return i;
}

template <typename T, bool allowPartialPush>
LIBC_INLINE ErrorOr<size_t>
CharacterConverter::rtomb_generic(char *__restrict dest_ptr, T src) {
  if (dest_ptr == nullptr)
    src = {};

  if (!isValidState())
    return Error(EINVAL);
  if (!allowPartialPush && !isEmpty()) {
    // don't allow a complete char32 spanning over 2 consecutive calls of char16
    clear();
    return Error(EILSEQ);
  }

  int status = push(static_cast<T>(src));
  if (status != 0)
    return Error(status);

  if (!isFull()) {
    if (!allowPartialPush) {
      // don't allow a complete char32 spanning over 2 consecutive calls of
      // char16
      clear();
      return Error(EILSEQ);
    }
    return 0;
  }

  size_t count = 0;
  for (; !isEmpty(); ++count) {
    auto utf8 = pop_utf8(); // can never fail as long as the push succeeded
    LIBC_ASSERT(utf8.has_value());

    if (dest_ptr != nullptr) {
      *dest_ptr = static_cast<char>(utf8.value());
      dest_ptr++;
    }
  }
  return count;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_CHARACTER_CONVERTER_H
