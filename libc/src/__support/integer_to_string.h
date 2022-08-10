//===-- Utilities to convert integral values to string ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_INTEGER_TO_STRING_H
#define LLVM_LIBC_SRC_SUPPORT_INTEGER_TO_STRING_H

#include "src/__support/CPP/ArrayRef.h"
#include "src/__support/CPP/StringView.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/type_traits.h"

namespace __llvm_libc {

template <typename T, uint8_t BASE = 10,
          cpp::enable_if_t<2 <= BASE && BASE <= 36, int> = 0>
class IntegerToString {
public:
  static constexpr inline size_t floor_log_2(size_t num) {
    size_t i = 0;
    for (; num > 1; num /= 2) {
      ++i;
    }
    return i;
  }
  // We size the string buffer for base 10 using an approximation algorithm:
  //
  //   size = ceil(sizeof(T) * 5 / 2)
  //
  // If sizeof(T) is 1, then size is 3 (actually 3)
  // If sizeof(T) is 2, then size is 5 (actually need 5)
  // If sizeof(T) is 4, then size is 10 (actually need 10)
  // If sizeof(T) is 8, then size is 20 (actually need 20)
  // If sizeof(T) is 16, then size is 40 (actually need 39)
  //
  // NOTE: The ceil operation is actually implemented as
  //     floor(((sizeof(T) * 5) + 1)/2)
  // where floor operation is just integer division.
  //
  // This estimation grows slightly faster than the actual value, but the
  // overhead is small enough to tolerate. In the actual formula below, we
  // add an additional byte to accommodate the '-' sign in case of signed
  // integers.
  // For other bases, we approximate by rounding down to the nearest power of
  // two base, since the space needed is easy to calculate and it won't
  // overestimate by too much.

  static constexpr size_t bufsize() {
    constexpr size_t BITS_PER_DIGIT = floor_log_2(BASE);
    constexpr size_t BUFSIZE_COMMON =
        ((sizeof(T) * 8 + (BITS_PER_DIGIT - 1)) / BITS_PER_DIGIT);
    constexpr size_t BUFSIZE_BASE10 = (sizeof(T) * 5 + 1) / 2;
    return (cpp::is_signed<T>() ? 1 : 0) +
           (BASE == 10 ? BUFSIZE_BASE10 : BUFSIZE_COMMON);
  }

  static constexpr size_t BUFSIZE = bufsize();

private:
  static_assert(cpp::is_integral_v<T>,
                "IntegerToString can only be used with integral types.");

  using UnsignedType = cpp::make_unsigned_t<T>;

  char strbuf[BUFSIZE] = {'\0'};
  cpp::StringView str_view;

  static inline constexpr cpp::StringView
  convert_alpha_numeric(T val, cpp::MutableArrayRef<char> &buffer,
                        bool lowercase, const uint8_t conv_base) {
    UnsignedType uval = val < 0 ? UnsignedType(-val) : UnsignedType(val);

    const char a = lowercase ? 'a' : 'A';

    size_t len = 0;

    size_t buffptr = buffer.size();
    if (uval == 0) {
      buffer[buffptr - 1] = '0';
      --buffptr;
    } else {
      for (; uval > 0; --buffptr, uval /= conv_base) {
        UnsignedType digit = (uval % conv_base);
        buffer[buffptr - 1] = digit < 10 ? digit + '0' : digit + a - 10;
      }
    }
    len = buffer.size() - buffptr;

    if (val < 0) {
      // This branch will be taken only for negative signed values.
      ++len;
      buffer[buffer.size() - len] = '-';
    }
    cpp::StringView buff_str(buffer.data() + buffer.size() - len, len);
    return buff_str;
  }

  // This function exists to check at compile time that the base is valid, as
  // well as to convert the templated call into a non-templated call. This
  // allows the compiler to decide to do strength reduction and constant folding
  // on the base or not, depending on if size or performance is required.
  static inline constexpr cpp::StringView
  convert_internal(T val, cpp::MutableArrayRef<char> &buffer, bool lowercase) {
    return convert_alpha_numeric(val, buffer, lowercase, BASE);
  }

public:
  static inline cpp::optional<cpp::StringView>
  convert(T val, cpp::MutableArrayRef<char> &buffer, bool lowercase) {
    // If This function can actually be a constexpr, then the below "if" will be
    // optimized out.
    if (buffer.size() < bufsize())
      return cpp::optional<cpp::StringView>();
    return cpp::optional<cpp::StringView>(
        convert_internal(val, buffer, lowercase));
  }

  constexpr explicit IntegerToString(T val) {
    cpp::MutableArrayRef<char> bufref(strbuf, BUFSIZE);
    str_view = convert_internal(val, bufref, true);
  }

  constexpr explicit IntegerToString(T val, bool lowercase) {
    cpp::MutableArrayRef<char> bufref(strbuf, BUFSIZE);
    str_view = convert_internal(val, bufref, lowercase);
  }

  cpp::StringView str() const { return str_view; }

  operator cpp::StringView() const { return str(); }
};

template <typename T> IntegerToString<T> integer_to_string(T val) {
  return IntegerToString<T>(val);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_INTEGER_TO_STRING_H
