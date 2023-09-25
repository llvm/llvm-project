//===-- Utilities to convert integral values to string ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts an integer to a string.
//
// By default, the string is written as decimal to an internal buffer and
// accessed via the 'view' method.
//
//   IntegerToString<int> buffer(42);
//   cpp::string_view view = buffer.view();
//
// The buffer is allocated on the stack and its size is so that the conversion
// always succeeds.
//
// It is also possible to write the data to a preallocated buffer, but this may
// fail.
//
//   char buffer[8];
//   if (auto maybe_view = IntegerToString<int>::write_to_span(buffer, 42)) {
//     cpp::string_view view = *maybe_view;
//   }
//
// The first template parameter is the type of the integer.
// The second template parameter defines how the integer is formatted.
// Available default are 'radix::Bin', 'radix::Oct', 'radix::Dec' and
// 'radix::Hex'.
//
// For 'radix::Bin', 'radix::Oct' and 'radix::Hex' the value is always
// interpreted as a positive type but 'radix::Dec' will honor negative values.
// e.g.,
//
//   IntegerToString<int8_t>(-1)             // "-1"
//   IntegerToString<int8_t, radix::Dec>(-1) // "-1"
//   IntegerToString<int8_t, radix::Bin>(-1) // "11111111"
//   IntegerToString<int8_t, radix::Oct>(-1) // "377"
//   IntegerToString<int8_t, radix::Hex>(-1) // "ff"
//
// Additionnally, the format can be changed by navigating the subtypes:
//  - WithPrefix    : Adds "0b", "0", "0x" for binary, octal and hexadecimal
//  - WithWidth<XX> : Pad string to XX characters filling leading digits with 0
//  - Uppercase     : Use uppercase letters (only for HexString)
//  - WithSign      : Prepend '+' for positive values (only for DecString)
//
// Examples
// --------
//   IntegerToString<int8_t, radix::Dec::WithWidth<2>::WithSign>(0)     : "+00"
//   IntegerToString<int8_t, radix::Dec::WithWidth<2>::WithSign>(-1)    : "-01"
//   IntegerToString<uint8_t, radix::Hex::WithPrefix::Uppercase>(255)   : "0xFF"
//   IntegerToString<uint8_t, radix::Hex::WithWidth<4>::Uppercase>(255) : "00FF"
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_INTEGER_TO_STRING_H
#define LLVM_LIBC_SRC___SUPPORT_INTEGER_TO_STRING_H

#include <stdint.h>

#include "src/__support/CPP/algorithm.h" // max
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"

namespace __llvm_libc {

namespace details {

template <uint8_t base, bool prefix = false, bool force_sign = false,
          bool is_uppercase = false, size_t min_digits = 1>
struct Fmt {
  static constexpr uint8_t BASE = base;
  static constexpr size_t MIN_DIGITS = min_digits;
  static constexpr bool IS_UPPERCASE = is_uppercase;
  static constexpr bool PREFIX = prefix;
  static constexpr char FORCE_SIGN = force_sign;

  using WithPrefix = Fmt<BASE, true, FORCE_SIGN, IS_UPPERCASE, MIN_DIGITS>;
  using WithSign = Fmt<BASE, PREFIX, true, IS_UPPERCASE, MIN_DIGITS>;
  using Uppercase = Fmt<BASE, PREFIX, FORCE_SIGN, true, MIN_DIGITS>;
  template <size_t value>
  using WithWidth = Fmt<BASE, PREFIX, FORCE_SIGN, IS_UPPERCASE, value>;

  // Invariants
  static constexpr uint8_t NUMERICAL_DIGITS = 10;
  static constexpr uint8_t ALPHA_DIGITS = 26;
  static constexpr uint8_t MAX_DIGIT = NUMERICAL_DIGITS + ALPHA_DIGITS;
  static_assert(BASE > 1 && BASE <= MAX_DIGIT);
  static_assert(!IS_UPPERCASE || BASE > 10, "Uppercase is only for radix > 10");
  static_assert(!FORCE_SIGN || BASE == 10, "WithSign is only for radix == 10");
  static_assert(!PREFIX || (BASE == 2 || BASE == 8 || BASE == 16),
                "WithPrefix is only for radix == 2, 8 or 16");
};

// Move this to a separate header since it might be useful elsewhere.
template <bool forward> class StringBufferWriterImpl {
  cpp::span<char> buffer;
  size_t index = 0;
  bool out_of_range = false;

  LIBC_INLINE size_t location() const {
    return forward ? index : buffer.size() - 1 - index;
  }

public:
  StringBufferWriterImpl(const StringBufferWriterImpl &) = delete;
  StringBufferWriterImpl(cpp::span<char> buffer) : buffer(buffer) {}

  LIBC_INLINE size_t size() const { return index; }
  LIBC_INLINE size_t remainder_size() const { return buffer.size() - size(); }
  LIBC_INLINE bool empty() const { return size() == 0; }
  LIBC_INLINE bool full() const { return size() == buffer.size(); }
  LIBC_INLINE bool ok() const { return !out_of_range; }

  LIBC_INLINE StringBufferWriterImpl &push(char c) {
    if (ok()) {
      if (!full()) {
        buffer[location()] = c;
        ++index;
      } else {
        out_of_range = true;
      }
    }
    return *this;
  }

  LIBC_INLINE cpp::span<char> remainder_span() const {
    return forward ? buffer.last(remainder_size())
                   : buffer.first(remainder_size());
  }

  LIBC_INLINE cpp::span<char> buffer_span() const {
    return forward ? buffer.first(size()) : buffer.last(size());
  }

  LIBC_INLINE cpp::string_view buffer_view() const {
    const auto s = buffer_span();
    return {s.data(), s.size()};
  }
};

using StringBufferWriter = StringBufferWriterImpl<true>;
using BackwardStringBufferWriter = StringBufferWriterImpl<false>;

} // namespace details

namespace radix {

using Bin = details::Fmt<2>;
using Oct = details::Fmt<8>;
using Dec = details::Fmt<10>;
using Hex = details::Fmt<16>;
template <size_t radix> using Custom = details::Fmt<radix>;

} // namespace radix

// See file header for documentation.
template <typename T, typename Fmt = radix::Dec> class IntegerToString {
  static_assert(cpp::is_integral_v<T>);

  LIBC_INLINE static constexpr size_t compute_buffer_size() {
    constexpr auto max_digits = []() -> size_t {
      // We size the string buffer for base 10 using an approximation algorithm:
      //
      //   size = ceil(sizeof(T) * 5 / 2)
      //
      // If sizeof(T) is 1, then size is 3 (actually need 3)
      // If sizeof(T) is 2, then size is 5 (actually need 5)
      // If sizeof(T) is 4, then size is 10 (actually need 10)
      // If sizeof(T) is 8, then size is 20 (actually need 20)
      // If sizeof(T) is 16, then size is 40 (actually need 39)
      //
      // NOTE: The ceil operation is actually implemented as
      //     floor(((sizeof(T) * 5) + 1) / 2)
      // where floor operation is just integer division.
      //
      // This estimation grows slightly faster than the actual value, but the
      // overhead is small enough to tolerate.
      if constexpr (Fmt::BASE == 10)
        return ((sizeof(T) * 5) + 1) / 2;
      // For other bases, we approximate by rounding down to the nearest power
      // of two base, since the space needed is easy to calculate and it won't
      // overestimate by too much.
      constexpr auto floor_log_2 = [](size_t num) -> size_t {
        size_t i = 0;
        for (; num > 1; num /= 2)
          ++i;
        return i;
      };
      constexpr size_t BITS_PER_DIGIT = floor_log_2(Fmt::BASE);
      return ((sizeof(T) * 8 + (BITS_PER_DIGIT - 1)) / BITS_PER_DIGIT);
    };
    constexpr size_t digit_size = cpp::max(max_digits(), Fmt::MIN_DIGITS);
    constexpr size_t sign_size = Fmt::BASE == 10 ? 1 : 0;
    constexpr size_t prefix_size = Fmt::PREFIX ? 2 : 0;
    return digit_size + sign_size + prefix_size;
  }

  static constexpr size_t BUFFER_SIZE = compute_buffer_size();
  static_assert(BUFFER_SIZE > 0);

  // An internal stateless structure that handles the number formatting logic.
  struct IntegerWriter {
    static_assert(cpp::is_integral_v<T>);
    using UNSIGNED_T = cpp::make_unsigned_t<T>;

    LIBC_INLINE static char digit_char(uint8_t digit) {
      if (digit < 10)
        return '0' + digit;
      return (Fmt::IS_UPPERCASE ? 'A' : 'a') + (digit - 10);
    }

    LIBC_INLINE static void
    write_unsigned_number(UNSIGNED_T value,
                          details::BackwardStringBufferWriter &sink) {
      for (; sink.ok() && value != 0; value /= Fmt::BASE) {
        const uint8_t digit(value % Fmt::BASE);
        sink.push(digit_char(digit));
      }
    }

    // Returns the absolute value of 'value' as 'UNSIGNED_T'.
    LIBC_INLINE static UNSIGNED_T abs(T value) {
      if (cpp::is_unsigned_v<T> || value >= 0)
        return value; // already of the right sign.

      // Signed integers are asymmetric (e.g., int8_t ∈ [-128, 127]).
      // Thus negating the type's minimum value would overflow.
      // From C++20 on, signed types are guaranteed to be represented as 2's
      // complement. We take advantage of this representation and negate the
      // value by using the exact same bit representation, e.g.,
      // binary : 0b1000'0000
      // int8_t : -128
      // uint8_t:  128

      // Note: the compiler can completely optimize out the two branches and
      // replace them by a simple negate instruction.
      // https://godbolt.org/z/hE7zahT9W
      if (value == cpp::numeric_limits<T>::min()) {
        return cpp::bit_cast<UNSIGNED_T>(value);
      } else {
        return -value; // legal and representable both as T and UNSIGNED_T.`
      }
    }

    LIBC_INLINE static void write(T value,
                                  details::BackwardStringBufferWriter &sink) {
      if constexpr (Fmt::BASE == 10) {
        write_unsigned_number(abs(value), sink);
      } else {
        write_unsigned_number(cpp::bit_cast<UNSIGNED_T>(value), sink);
      }
      // width
      while (sink.ok() && sink.size() < Fmt::MIN_DIGITS)
        sink.push('0');
      // sign
      if constexpr (Fmt::BASE == 10) {
        if (value < 0)
          sink.push('-');
        else if (Fmt::FORCE_SIGN)
          sink.push('+');
      }
      // prefix
      if constexpr (Fmt::PREFIX) {
        if constexpr (Fmt::BASE == 2) {
          sink.push('b');
          sink.push('0');
        }
        if constexpr (Fmt::BASE == 16) {
          sink.push('x');
          sink.push('0');
        }
        if constexpr (Fmt::BASE == 8) {
          const cpp::string_view written = sink.buffer_view();
          if (written.empty() || written.front() != '0')
            sink.push('0');
        }
      }
    }
  };

  cpp::array<char, BUFFER_SIZE> array;
  size_t written = 0;

public:
  IntegerToString(const IntegerToString &) = delete;
  IntegerToString(T value) {
    details::BackwardStringBufferWriter writer(array);
    IntegerWriter::write(value, writer);
    written = writer.size();
  }

  [[nodiscard]] LIBC_INLINE static cpp::optional<cpp::string_view>
  format_to(cpp::span<char> buffer, T value) {
    details::BackwardStringBufferWriter writer(buffer);
    IntegerWriter::write(value, writer);
    if (writer.ok())
      return cpp::string_view(buffer.data() + buffer.size() - writer.size(),
                              writer.size());
    return cpp::nullopt;
  }

  LIBC_INLINE static constexpr size_t buffer_size() { return BUFFER_SIZE; }

  LIBC_INLINE size_t size() const { return written; }
  LIBC_INLINE cpp::string_view view() && = delete;
  LIBC_INLINE cpp::string_view view() const & {
    return cpp::string_view(array.data() + array.size() - size(), size());
  }
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_INTEGER_TO_STRING_H
