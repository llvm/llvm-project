//===-- lib/runtime/edit-input.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "edit-input.h"
#include "flang-rt/runtime/namelist.h"
#include "flang-rt/runtime/utf.h"
#include "flang/Common/optional.h"
#include "flang/Common/real.h"
#include "flang/Common/uint128.h"
#include "flang/Runtime/freestanding-tools.h"
#include <algorithm>
#include <cfenv>

namespace Fortran::runtime::io {
RT_OFFLOAD_API_GROUP_BEGIN

static inline RT_API_ATTRS bool IsCharValueSeparator(
    const DataEdit &edit, char32_t ch) {
  return ch == ' ' || ch == '\t' || ch == '/' ||
      ch == edit.modes.GetSeparatorChar() ||
      (edit.IsNamelist() && (ch == '&' || ch == '$'));
}

// Checks that a list-directed input value has been entirely consumed and
// doesn't contain unparsed characters before the next value separator.
static RT_API_ATTRS bool CheckCompleteListDirectedField(
    IoStatementState &io, const DataEdit &edit) {
  if (edit.IsListDirected()) {
    std::size_t byteCount;
    if (auto ch{io.GetCurrentChar(byteCount)}) {
      if (!IsCharValueSeparator(edit, *ch)) {
        const auto &connection{io.GetConnectionState()};
        io.GetIoErrorHandler().SignalError(IostatBadListDirectedInputSeparator,
            "invalid character (0x%x) after list-directed input value, "
            "at column %d in record %d",
            static_cast<unsigned>(*ch),
            static_cast<int>(connection.positionInRecord + 1),
            static_cast<int>(connection.currentRecordNumber));
        return false;
      }
    }
  }
  return true;
}

template <int LOG2_BASE>
static RT_API_ATTRS bool EditBOZInput(
    IoStatementState &io, const DataEdit &edit, void *n, std::size_t bytes) {
  // Skip leading white space & zeroes
  common::optional<int> remaining{io.CueUpInput(edit)};
  auto start{io.GetConnectionState().positionInRecord};
  common::optional<char32_t> next{io.NextInField(remaining, edit)};
  if (next.value_or('?') == '0') {
    do {
      start = io.GetConnectionState().positionInRecord;
      next = io.NextInField(remaining, edit);
    } while (next && *next == '0');
  }
  // Count significant digits after any leading white space & zeroes
  int digits{0};
  int significantBits{0};
  char32_t comma{edit.modes.GetSeparatorChar()};
  for (; next; next = io.NextInField(remaining, edit)) {
    char32_t ch{*next};
    if (ch == ' ' || ch == '\t') {
      if (edit.modes.editingFlags & blankZero) {
        ch = '0'; // BZ mode - treat blank as if it were zero
      } else {
        continue;
      }
    }
    if (ch >= '0' && ch <= '1') {
    } else if (LOG2_BASE >= 3 && ch >= '2' && ch <= '7') {
    } else if (LOG2_BASE >= 4 && ch >= '8' && ch <= '9') {
    } else if (LOG2_BASE >= 4 && ch >= 'A' && ch <= 'F') {
    } else if (LOG2_BASE >= 4 && ch >= 'a' && ch <= 'f') {
    } else if (ch == comma) {
      break; // end non-list-directed field early
    } else {
      io.GetIoErrorHandler().SignalError(
          "Bad character '%lc' in B/O/Z input field", ch);
      return false;
    }
    if (digits++ == 0) {
      if (ch >= '0' && ch <= '1') {
        significantBits = 1;
      } else if (ch >= '2' && ch <= '3') {
        significantBits = 2;
      } else if (ch >= '4' && ch <= '7') {
        significantBits = 3;
      } else {
        significantBits = 4;
      }
    } else {
      significantBits += LOG2_BASE;
    }
  }
  auto significantBytes{static_cast<std::size_t>(significantBits + 7) / 8};
  if (significantBytes > bytes) {
    io.GetIoErrorHandler().SignalError(IostatBOZInputOverflow,
        "B/O/Z input of %d digits overflows %zd-byte variable", digits, bytes);
    return false;
  }
  // Reset to start of significant digits
  io.HandleAbsolutePosition(start);
  remaining.reset();
  // Make a second pass now that the digit count is known
  runtime::memset(n, 0, bytes);
  int increment{isHostLittleEndian ? -1 : 1};
  auto *data{reinterpret_cast<unsigned char *>(n) +
      (isHostLittleEndian ? significantBytes - 1 : bytes - significantBytes)};
  int bitsAfterFirstDigit{(digits - 1) * LOG2_BASE};
  int shift{bitsAfterFirstDigit & 7};
  if (shift + (significantBits - bitsAfterFirstDigit) > 8) {
    shift = shift - 8; // misaligned octal
  }
  while (digits > 0) {
    char32_t ch{io.NextInField(remaining, edit).value_or(' ')};
    int digit{0};
    if (ch == ' ' || ch == '\t') {
      if (edit.modes.editingFlags & blankZero) {
        ch = '0'; // BZ mode - treat blank as if it were zero
      } else {
        continue;
      }
    }
    --digits;
    if (ch >= '0' && ch <= '9') {
      digit = ch - '0';
    } else if (ch >= 'A' && ch <= 'F') {
      digit = ch + 10 - 'A';
    } else if (ch >= 'a' && ch <= 'f') {
      digit = ch + 10 - 'a';
    } else {
      continue;
    }
    if (shift < 0) {
      if (shift + LOG2_BASE > 0) { // misaligned octal
        *data |= digit >> -shift;
      }
      shift += 8;
      data += increment;
    }
    *data |= digit << shift;
    shift -= LOG2_BASE;
  }
  return CheckCompleteListDirectedField(io, edit);
}

// Prepares input from a field, and returns the sign, if any, else '\0'.
static RT_API_ATTRS char ScanNumericPrefix(IoStatementState &io,
    const DataEdit &edit, common::optional<char32_t> &next,
    common::optional<int> &remaining,
    IoStatementState::FastAsciiField *fastField = nullptr) {
  remaining = io.CueUpInput(edit, fastField);
  next = io.NextInField(remaining, edit, fastField);
  char sign{'\0'};
  if (next) {
    if (*next == '-' || *next == '+') {
      sign = *next;
      if (!edit.IsListDirected()) {
        io.SkipSpaces(remaining, fastField);
      }
      next = io.NextInField(remaining, edit, fastField);
    }
  }
  return sign;
}

RT_API_ATTRS bool EditIntegerInput(IoStatementState &io, const DataEdit &edit,
    void *n, int kind, bool isSigned) {
  auto &handler{io.GetIoErrorHandler()};
  RUNTIME_CHECK(handler, kind >= 1 && !(kind & (kind - 1)));
  if (!n) {
    handler.Crash("Null address for integer input item");
  }
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
    if (IsNamelistNameOrSlash(io)) {
      return false;
    }
    break;
  case 'G':
  case 'I':
    break;
  case 'B':
    return EditBOZInput<1>(io, edit, n, kind);
  case 'O':
    return EditBOZInput<3>(io, edit, n, kind);
  case 'Z':
    return EditBOZInput<4>(io, edit, n, kind);
  case 'A': // legacy extension
    return EditCharacterInput(io, edit, reinterpret_cast<char *>(n), kind);
  default:
    handler.SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used with an INTEGER data item",
        edit.descriptor);
    return false;
  }
  common::optional<int> remaining;
  common::optional<char32_t> next;
  auto fastField{io.GetUpcomingFastAsciiField()};
  char sign{ScanNumericPrefix(io, edit, next, remaining, &fastField)};
  if (sign == '-' && !isSigned) {
    handler.SignalError("Negative sign in UNSIGNED input field");
    return false;
  }
  common::uint128_t value{0};
  bool any{!!sign};
  bool overflow{false};
  char32_t comma{edit.modes.GetSeparatorChar()};
  static constexpr auto maxu128{~common::uint128_t{0}};
  for (; next; next = io.NextInField(remaining, edit, &fastField)) {
    char32_t ch{*next};
    if (ch == ' ' || ch == '\t') {
      if (edit.modes.editingFlags & blankZero) {
        ch = '0'; // BZ mode - treat blank as if it were zero
      } else {
        continue;
      }
    }
    int digit{0};
    if (ch >= '0' && ch <= '9') {
      digit = ch - '0';
    } else if (ch == comma) {
      break; // end non-list-directed field early
    } else {
      if (edit.modes.inNamelist && ch == edit.modes.GetRadixPointChar()) {
        // Ignore any fractional part that might appear in NAMELIST integer
        // input, like a few other Fortran compilers do.
        // TODO: also process exponents?  Some compilers do, but they obviously
        // can't just be ignored.
        while ((next = io.NextInField(remaining, edit, &fastField))) {
          if (*next < '0' || *next > '9') {
            break;
          }
        }
        if (!next || *next == comma) {
          break;
        }
      }
      handler.SignalError("Bad character '%lc' in INTEGER input field", ch);
      return false;
    }
    static constexpr auto maxu128OverTen{maxu128 / 10};
    static constexpr int maxLastDigit{
        static_cast<int>(maxu128 - (maxu128OverTen * 10))};
    overflow |= value >= maxu128OverTen &&
        (value > maxu128OverTen || digit > maxLastDigit);
    value *= 10;
    value += digit;
    any = true;
  }
  if (!any && !remaining) {
    handler.SignalError(
        "Integer value absent from NAMELIST or list-directed input");
    return false;
  }
  if (isSigned) {
    auto maxForKind{common::uint128_t{1} << ((8 * kind) - 1)};
    overflow |= value >= maxForKind && (value > maxForKind || sign != '-');
  } else {
    auto maxForKind{maxu128 >> (((16 - kind) * 8) + (isSigned ? 1 : 0))};
    overflow |= value >= maxForKind;
  }
  if (overflow) {
    handler.SignalError(IostatIntegerInputOverflow,
        "Decimal input overflows INTEGER(%d) variable", kind);
    return false;
  }
  if (sign == '-') {
    value = -value;
  }
  if (any || !handler.InError()) {
    // The value is stored in the lower order bits on big endian platform.
    // For memcpy, shift the value to the highest order bits.
#if USING_NATIVE_INT128_T
    auto shft{static_cast<int>(sizeof value - kind)};
    if (!isHostLittleEndian && shft >= 0) {
      auto shifted{value << (8 * shft)};
      runtime::memcpy(n, &shifted, kind);
    } else {
      runtime::memcpy(n, &value, kind); // a blank field means zero
    }
#else
    auto shft{static_cast<int>(sizeof(value.low())) - kind};
    // For kind==8 (i.e. shft==0), the value is stored in low_ in big endian.
    if (!isHostLittleEndian && shft >= 0) {
      auto l{value.low() << (8 * shft)};
      runtime::memcpy(n, &l, kind);
    } else {
      runtime::memcpy(n, &value, kind); // a blank field means zero
    }
#endif
    io.GotChar(fastField.got());
    return true;
  } else {
    return false;
  }
}

// Parses a REAL input number from the input source as a normalized
// fraction into a supplied buffer -- there's an optional '-', a
// decimal point when the input is not hexadecimal, and at least one
// digit.  Replaces blanks with zeroes where appropriate.
struct ScannedRealInput {
  // Number of characters that (should) have been written to the
  // buffer -- this can be larger than the buffer size, which
  // indicates buffer overflow.  Zero indicates an error.
  int got{0};
  int exponent{0}; // adjusted as necessary; binary if isHexadecimal
  bool isHexadecimal{false}; // 0X...
};
static RT_API_ATTRS ScannedRealInput ScanRealInput(
    char *buffer, int bufferSize, IoStatementState &io, const DataEdit &edit) {
  common::optional<int> remaining;
  common::optional<char32_t> next;
  int got{0};
  common::optional<int> radixPointOffset;
  // The following lambda definition violates the conding style,
  // but cuda-11.8 nvcc hits an internal error with the brace initialization.
  auto Put = [&](char ch) -> void {
    if (got < bufferSize) {
      buffer[got] = ch;
    }
    ++got;
  };
  char sign{ScanNumericPrefix(io, edit, next, remaining)};
  if (sign == '-') {
    Put('-');
  }
  bool bzMode{(edit.modes.editingFlags & blankZero) != 0};
  int exponent{0};
  char32_t comma{edit.modes.GetSeparatorChar()};
  if (!next || (!bzMode && *next == ' ') || *next == comma) {
    if (!edit.IsListDirected() && !io.GetConnectionState().IsAtEOF()) {
      // An empty/blank field means zero when not list-directed.
      // A fixed-width field containing only a sign is also zero;
      // this behavior isn't standard-conforming in F'2023 but it is
      // required to pass FCVS.
      Put('0');
    }
    return {got, exponent, false};
  }
  char32_t radixPointChar{edit.modes.GetRadixPointChar()};
  char32_t first{*next >= 'a' && *next <= 'z' ? *next + 'A' - 'a' : *next};
  bool isHexadecimal{false};
  if (first == 'N' || first == 'I') {
    // NaN or infinity - convert to upper case
    // Subtle: a blank field of digits could be followed by 'E' or 'D',
    for (; next &&
         ((*next >= 'a' && *next <= 'z') || (*next >= 'A' && *next <= 'Z'));
         next = io.NextInField(remaining, edit)) {
      if (*next >= 'a' && *next <= 'z') {
        Put(*next - 'a' + 'A');
      } else {
        Put(*next);
      }
    }
    if (first == 'N' && (!next || *next == '(') &&
        remaining.value_or(1) > 0) { // NaN(...)?
      std::size_t byteCount{0};
      if (!next) { // NextInField won't return '(' for list-directed
        next = io.GetCurrentChar(byteCount);
      }
      if (next && *next == '(') {
        int depth{1};
        while (true) {
          if (*next >= 'a' && *next <= 'z') {
            *next = *next - 'a' + 'A';
          }
          Put(*next);
          io.HandleRelativePosition(byteCount);
          io.GotChar(byteCount);
          if (remaining) {
            *remaining -= byteCount;
          }
          if (depth == 0) {
            break; // done
          }
          next = io.GetCurrentChar(byteCount);
          if (!next || remaining.value_or(1) < 1) {
            return {}; // error
          } else if (*next == '(') {
            ++depth;
          } else if (*next == ')') {
            --depth;
          }
        }
        next = io.NextInField(remaining, edit);
      }
    }
  } else if (first == radixPointChar || (first >= '0' && first <= '9') ||
      (bzMode && (first == ' ' || first == '\t')) ||
      (remaining.has_value() &&
          (first == 'D' || first == 'E' || first == 'Q'))) {
    if (first == '0') {
      next = io.NextInField(remaining, edit);
      if (next && (*next == 'x' || *next == 'X')) { // 0X...
        isHexadecimal = true;
        next = io.NextInField(remaining, edit);
      } else {
        Put('0');
      }
    }
    // input field is normalized to a fraction
    if (!isHexadecimal) {
      Put('.');
    }
    auto start{got};
    for (; next; next = io.NextInField(remaining, edit)) {
      char32_t ch{*next};
      if (ch == ' ' || ch == '\t') {
        if (isHexadecimal) {
          return {}; // error
        } else if (bzMode) {
          ch = '0'; // BZ mode - treat blank as if it were zero
        } else {
          continue; // ignore blank in fixed field
        }
      }
      if (ch == '0' && got == start && !radixPointOffset) {
        // omit leading zeroes before the radix point
      } else if (ch >= '0' && ch <= '9') {
        Put(ch);
      } else if (ch == radixPointChar && !radixPointOffset) {
        // The radix point character is *not* copied to the buffer.
        radixPointOffset = got - start; // # of digits before the radix point
      } else if (isHexadecimal && ch >= 'A' && ch <= 'F') {
        Put(ch);
      } else if (isHexadecimal && ch >= 'a' && ch <= 'f') {
        Put(ch - 'a' + 'A'); // normalize to capitals
      } else {
        break;
      }
    }
    if (got == start) {
      // Nothing but zeroes and maybe a radix point.  F'2018 requires
      // at least one digit, but F'77 did not, and a bare "." shows up in
      // the FCVS suite.
      Put('0'); // emit at least one digit
    }
    // In list-directed input, a bad exponent is not consumed.
    auto nextBeforeExponent{next};
    auto startExponent{io.GetConnectionState().positionInRecord};
    bool hasGoodExponent{false};
    if (next) {
      if (isHexadecimal) {
        if (*next == 'p' || *next == 'P') {
          next = io.NextInField(remaining, edit);
        } else {
          // The binary exponent is not optional in the standard.
          return {}; // error
        }
      } else if (*next == 'e' || *next == 'E' || *next == 'd' || *next == 'D' ||
          *next == 'q' || *next == 'Q') {
        // Optional exponent letter.  Blanks are allowed between the
        // optional exponent letter and the exponent value.
        io.SkipSpaces(remaining);
        next = io.NextInField(remaining, edit);
        if (!next) {
          if (remaining.has_value()) {
            // bare exponent letter accepted in fixed-width field
            hasGoodExponent = true;
          } else {
            return {}; // error
          }
        }
      }
    }
    if (next &&
        (*next == '-' || *next == '+' || (*next >= '0' && *next <= '9') ||
            *next == ' ' || *next == '\t')) {
      bool negExpo{*next == '-'};
      if (negExpo || *next == '+') {
        next = io.NextInField(remaining, edit);
      }
      for (; next; next = io.NextInField(remaining, edit)) {
        if (*next >= '0' && *next <= '9') {
          hasGoodExponent = true;
          if (exponent < 10000) {
            exponent = 10 * exponent + *next - '0';
          }
        } else if (*next == ' ' || *next == '\t') {
          if (isHexadecimal) {
            break;
          } else if (bzMode) {
            hasGoodExponent = true;
            exponent = 10 * exponent;
          }
        } else {
          break;
        }
      }
      if (negExpo) {
        exponent = -exponent;
      }
    }
    if (!hasGoodExponent) {
      if (isHexadecimal) {
        return {}; // error
      }
      // There isn't a good exponent; do not consume it.
      next = nextBeforeExponent;
      io.HandleAbsolutePosition(startExponent);
      // The default exponent is -kP, but the scale factor doesn't affect
      // an explicit exponent.
      exponent = -edit.modes.scale;
    }
    // Adjust exponent by number of digits before the radix point.
    if (isHexadecimal) {
      // Exponents for hexadecimal input are binary.
      exponent += radixPointOffset.value_or(got - start) * 4;
    } else if (radixPointOffset) {
      exponent += *radixPointOffset;
    } else {
      // When no radix point (or comma) appears in the value, the 'd'
      // part of the edit descriptor must be interpreted as the number of
      // digits in the value to be interpreted as being to the *right* of
      // the assumed radix point (13.7.2.3.2)
      exponent += got - start - edit.digits.value_or(0);
    }
  }
  // Consume the trailing ')' of a list-directed or NAMELIST complex
  // input value.
  if (edit.descriptor == DataEdit::ListDirectedImaginaryPart) {
    if (!next || *next == ' ' || *next == '\t') {
      io.SkipSpaces(remaining);
      next = io.NextInField(remaining, edit);
    }
    if (!next || *next == ')') { // NextInField fails on separators like ')'
      std::size_t byteCount{1};
      if (!next) {
        next = io.GetCurrentChar(byteCount);
      }
      if (next && *next == ')') {
        io.HandleRelativePosition(byteCount);
      }
    }
  } else if (remaining) {
    while (next && (*next == ' ' || *next == '\t')) {
      next = io.NextInField(remaining, edit);
    }
    if (next && *next != comma) {
      return {}; // error: unused nonblank character in fixed-width field
    }
  }
  return {got, exponent, isHexadecimal};
}

static RT_API_ATTRS void RaiseFPExceptions(
    decimal::ConversionResultFlags flags) {
#undef RAISE
#if defined(RT_DEVICE_COMPILATION)
  Terminator terminator(__FILE__, __LINE__);
#define RAISE(e) \
  terminator.Crash( \
      "not implemented yet: raising FP exception in device code: %s", #e);
#else // !defined(RT_DEVICE_COMPILATION)
#ifdef feraisexcept // a macro in some environments; omit std::
#define RAISE feraiseexcept
#else
#define RAISE std::feraiseexcept
#endif
#endif // !defined(RT_DEVICE_COMPILATION)

// Some environment (e.g. emscripten, musl) don't define FE_OVERFLOW as allowed
// by c99 (but not c++11) :-/
#if defined(FE_OVERFLOW) || defined(RT_DEVICE_COMPILATION)
  if (flags & decimal::ConversionResultFlags::Overflow) {
    RAISE(FE_OVERFLOW);
  }
#endif
#if defined(FE_UNDERFLOW) || defined(RT_DEVICE_COMPILATION)
  if (flags & decimal::ConversionResultFlags::Underflow) {
    RAISE(FE_UNDERFLOW);
  }
#endif
#if defined(FE_INEXACT) || defined(RT_DEVICE_COMPILATION)
  if (flags & decimal::ConversionResultFlags::Inexact) {
    RAISE(FE_INEXACT);
  }
#endif
#if defined(FE_INVALID) || defined(RT_DEVICE_COMPILATION)
  if (flags & decimal::ConversionResultFlags::Invalid) {
    RAISE(FE_INVALID);
  }
#endif
#undef RAISE
}

// If no special modes are in effect and the form of the input value
// that's present in the input stream is acceptable to the decimal->binary
// converter without modification, this fast path for real input
// saves time by avoiding memory copies and reformatting of the exponent.
template <int PRECISION>
static RT_API_ATTRS bool TryFastPathRealDecimalInput(
    IoStatementState &io, const DataEdit &edit, void *n) {
  if (edit.modes.editingFlags & (blankZero | decimalComma)) {
    return false;
  }
  if (edit.modes.scale != 0) {
    return false;
  }
  const ConnectionState &connection{io.GetConnectionState()};
  if (connection.internalIoCharKind > 1) {
    return false; // reading non-default character
  }
  const char *str{nullptr};
  std::size_t got{io.GetNextInputBytes(str)};
  if (got == 0 || str == nullptr || !connection.recordLength.has_value()) {
    return false; // could not access reliably-terminated input stream
  }
  const char *p{str};
  std::int64_t maxConsume{
      std::min<std::int64_t>(got, edit.width.value_or(got))};
  const char *limit{str + maxConsume};
  decimal::ConversionToBinaryResult<PRECISION> converted{
      decimal::ConvertToBinary<PRECISION>(p, edit.modes.round, limit)};
  if (converted.flags & (decimal::Invalid | decimal::Overflow)) {
    return false;
  }
  if (edit.digits.value_or(0) != 0) {
    // Edit descriptor is Fw.d (or other) with d != 0, which
    // implies scaling
    const char *q{str};
    for (; q < limit; ++q) {
      if (*q == '.' || *q == 'n' || *q == 'N') {
        break;
      }
    }
    if (q == limit) {
      // No explicit decimal point, and not NaN/Inf.
      return false;
    }
  }
  if (edit.descriptor == DataEdit::ListDirectedImaginaryPart) {
    // Need to consume a trailing ')', possibly with leading spaces
    for (; p < limit && (*p == ' ' || *p == '\t'); ++p) {
    }
    if (p < limit && *p == ')') {
      ++p;
    } else {
      return false;
    }
  } else if (edit.IsListDirected()) {
    if (p < limit && !IsCharValueSeparator(edit, *p)) {
      return false;
    }
  } else {
    for (; p < limit && (*p == ' ' || *p == '\t'); ++p) {
    }
    if (edit.width && p < str + *edit.width) {
      return false; // unconverted characters remain in fixed width field
    }
  }
  // Success on the fast path!
  *reinterpret_cast<decimal::BinaryFloatingPointNumber<PRECISION> *>(n) =
      converted.binary;
  io.HandleRelativePosition(p - str);
  io.GotChar(p - str);
  // Set FP exception flags
  if (converted.flags != decimal::ConversionResultFlags::Exact) {
    RaiseFPExceptions(converted.flags);
  }
  return true;
}

template <int binaryPrecision>
RT_API_ATTRS decimal::ConversionToBinaryResult<binaryPrecision>
ConvertHexadecimal(
    const char *&p, enum decimal::FortranRounding rounding, int expo) {
  using RealType = decimal::BinaryFloatingPointNumber<binaryPrecision>;
  using RawType = typename RealType::RawType;
  bool isNegative{*p == '-'};
  constexpr RawType one{1};
  RawType signBit{0};
  if (isNegative) {
    ++p;
    signBit = one << (RealType::bits - 1);
  }
  RawType fraction{0};
  // Adjust the incoming binary P+/- exponent to shift the radix point
  // to below the LSB and add in the bias.
  expo += binaryPrecision - 1 + RealType::exponentBias;
  // Input the fraction.
  int roundingBit{0};
  int guardBit{0};
  for (; *p; ++p) {
    fraction <<= 4;
    expo -= 4;
    if (*p >= '0' && *p <= '9') {
      fraction |= *p - '0';
    } else if (*p >= 'A' && *p <= 'F') {
      fraction |= *p - 'A' + 10; // data were normalized to capitals
    } else {
      break;
    }
    if (fraction >> binaryPrecision) {
      while (fraction >> binaryPrecision) {
        guardBit |= roundingBit;
        roundingBit = (int)fraction & 1;
        fraction >>= 1;
        ++expo;
      }
      // Consume excess digits
      while (*++p) {
        if (*p == '0') {
        } else if ((*p >= '1' && *p <= '9') || (*p >= 'A' && *p <= 'F')) {
          guardBit = 1;
        } else {
          break;
        }
      }
      break;
    }
  }
  if (fraction) {
    // Boost biased expo if too small
    while (expo < 1) {
      guardBit |= roundingBit;
      roundingBit = (int)fraction & 1;
      fraction >>= 1;
      ++expo;
    }
    // Normalize
    while (expo > 1 && !(fraction >> (binaryPrecision - 1))) {
      fraction <<= 1;
      --expo;
      guardBit = roundingBit = 0;
    }
  }
  // Rounding
  bool increase{false};
  switch (rounding) {
  case decimal::RoundNearest: // RN & RP
    increase = roundingBit && (guardBit | ((int)fraction & 1));
    break;
  case decimal::RoundUp: // RU
    increase = !isNegative && (roundingBit | guardBit);
    break;
  case decimal::RoundDown: // RD
    increase = isNegative && (roundingBit | guardBit);
    break;
  case decimal::RoundToZero: // RZ
    break;
  case decimal::RoundCompatible: // RC
    increase = roundingBit != 0;
    break;
  }
  if (increase) {
    ++fraction;
    if (fraction >> binaryPrecision) {
      fraction >>= 1;
      ++expo;
    }
  }
  // Package & return result
  constexpr RawType significandMask{(one << RealType::significandBits) - 1};
  int flags{(roundingBit | guardBit) ? decimal::Inexact : decimal::Exact};
  if (!fraction) {
    expo = 0;
  } else if (expo == 1 && !(fraction >> (binaryPrecision - 1))) {
    expo = 0; // subnormal
    flags |= decimal::Underflow;
  } else if (expo >= RealType::maxExponent) {
    if (rounding == decimal::RoundToZero ||
        (rounding == decimal::RoundDown && !isNegative) ||
        (rounding == decimal::RoundUp && isNegative)) {
      expo = RealType::maxExponent - 1; // +/-HUGE()
      fraction = significandMask;
    } else {
      expo = RealType::maxExponent; // +/-Inf
      fraction = 0;
      flags |= decimal::Overflow;
    }
  } else {
    fraction &= significandMask; // remove explicit normalization unless x87
  }
  return decimal::ConversionToBinaryResult<binaryPrecision>{
      RealType{static_cast<RawType>(signBit |
          static_cast<RawType>(expo) << RealType::significandBits | fraction)},
      static_cast<decimal::ConversionResultFlags>(flags)};
}

template <int KIND>
RT_API_ATTRS bool EditCommonRealInput(
    IoStatementState &io, const DataEdit &edit, void *n) {
  constexpr int binaryPrecision{common::PrecisionOfRealKind(KIND)};
  if (TryFastPathRealDecimalInput<binaryPrecision>(io, edit, n)) {
    return CheckCompleteListDirectedField(io, edit);
  }
  // Fast path wasn't available or didn't work; go the more general route
  static constexpr int maxDigits{
      common::MaxDecimalConversionDigits(binaryPrecision)};
  static constexpr int bufferSize{maxDigits + 18};
  char buffer[bufferSize];
  auto scanned{ScanRealInput(buffer, maxDigits + 2, io, edit)};
  int got{scanned.got};
  if (got >= maxDigits + 2) {
    io.GetIoErrorHandler().Crash("EditCommonRealInput: buffer was too small");
    return false;
  }
  if (got == 0) {
    const auto &connection{io.GetConnectionState()};
    io.GetIoErrorHandler().SignalError(IostatBadRealInput,
        "Bad real input data at column %d of record %d",
        static_cast<int>(connection.positionInRecord + 1),
        static_cast<int>(connection.currentRecordNumber));
    return false;
  }
  decimal::ConversionToBinaryResult<binaryPrecision> converted;
  const char *p{buffer};
  if (scanned.isHexadecimal) {
    buffer[got] = '\0';
    converted = ConvertHexadecimal<binaryPrecision>(
        p, edit.modes.round, scanned.exponent);
  } else {
    bool hadExtra{got > maxDigits};
    int exponent{scanned.exponent};
    if (exponent != 0) {
      buffer[got++] = 'e';
      if (exponent < 0) {
        buffer[got++] = '-';
        exponent = -exponent;
      }
      if (exponent > 9999) {
        exponent = 9999; // will convert to +/-Inf
      }
      if (exponent > 999) {
        int dig{exponent / 1000};
        buffer[got++] = '0' + dig;
        int rest{exponent - 1000 * dig};
        dig = rest / 100;
        buffer[got++] = '0' + dig;
        rest -= 100 * dig;
        dig = rest / 10;
        buffer[got++] = '0' + dig;
        buffer[got++] = '0' + (rest - 10 * dig);
      } else if (exponent > 99) {
        int dig{exponent / 100};
        buffer[got++] = '0' + dig;
        int rest{exponent - 100 * dig};
        dig = rest / 10;
        buffer[got++] = '0' + dig;
        buffer[got++] = '0' + (rest - 10 * dig);
      } else if (exponent > 9) {
        int dig{exponent / 10};
        buffer[got++] = '0' + dig;
        buffer[got++] = '0' + (exponent - 10 * dig);
      } else {
        buffer[got++] = '0' + exponent;
      }
    }
    buffer[got] = '\0';
    converted = decimal::ConvertToBinary<binaryPrecision>(p, edit.modes.round);
    if (hadExtra) {
      converted.flags = static_cast<enum decimal::ConversionResultFlags>(
          converted.flags | decimal::Inexact);
    }
  }
  if (*p) { // unprocessed junk after value
    const auto &connection{io.GetConnectionState()};
    io.GetIoErrorHandler().SignalError(IostatBadRealInput,
        "Trailing characters after real input data at column %d of record %d",
        static_cast<int>(connection.positionInRecord + 1),
        static_cast<int>(connection.currentRecordNumber));
    return false;
  }
  *reinterpret_cast<decimal::BinaryFloatingPointNumber<binaryPrecision> *>(n) =
      converted.binary;
  // Set FP exception flags
  if (converted.flags != decimal::ConversionResultFlags::Exact) {
    if (converted.flags & decimal::ConversionResultFlags::Overflow) {
      io.GetIoErrorHandler().SignalError(IostatRealInputOverflow);
      return false;
    }
    RaiseFPExceptions(converted.flags);
  }
  return CheckCompleteListDirectedField(io, edit);
}

template <int KIND>
RT_API_ATTRS bool EditRealInput(
    IoStatementState &io, const DataEdit &edit, void *n) {
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
    if (IsNamelistNameOrSlash(io)) {
      return false;
    }
    return EditCommonRealInput<KIND>(io, edit, n);
  case DataEdit::ListDirectedRealPart:
  case DataEdit::ListDirectedImaginaryPart:
  case 'F':
  case 'E': // incl. EN, ES, & EX
  case 'D':
  case 'G':
    return EditCommonRealInput<KIND>(io, edit, n);
  case 'B':
    return EditBOZInput<1>(io, edit, n,
        common::BitsForBinaryPrecision(common::PrecisionOfRealKind(KIND)) >> 3);
  case 'O':
    return EditBOZInput<3>(io, edit, n,
        common::BitsForBinaryPrecision(common::PrecisionOfRealKind(KIND)) >> 3);
  case 'Z':
    return EditBOZInput<4>(io, edit, n,
        common::BitsForBinaryPrecision(common::PrecisionOfRealKind(KIND)) >> 3);
  case 'A': // legacy extension
    return EditCharacterInput(io, edit, reinterpret_cast<char *>(n), KIND);
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used for REAL input",
        edit.descriptor);
    return false;
  }
}

// 13.7.3 in Fortran 2018
RT_API_ATTRS bool EditLogicalInput(
    IoStatementState &io, const DataEdit &edit, bool &x) {
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
    if (IsNamelistNameOrSlash(io)) {
      return false;
    }
    break;
  case 'L':
  case 'G':
    break;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used for LOGICAL input",
        edit.descriptor);
    return false;
  }
  common::optional<int> remaining{io.CueUpInput(edit)};
  common::optional<char32_t> next{io.NextInField(remaining, edit)};
  if (next && *next == '.') { // skip optional period
    next = io.NextInField(remaining, edit);
  }
  if (!next) {
    io.GetIoErrorHandler().SignalError("Empty LOGICAL input field");
    return false;
  }
  switch (*next) {
  case 'T':
  case 't':
    x = true;
    break;
  case 'F':
  case 'f':
    x = false;
    break;
  default:
    io.GetIoErrorHandler().SignalError(
        "Bad character '%lc' in LOGICAL input field", *next);
    return false;
  }
  if (remaining || edit.descriptor == DataEdit::ListDirected) {
    // Ignore the rest of the input field; stop after separator when
    // not list-directed.
    char32_t comma{edit.modes.GetSeparatorChar()};
    while (next && *next != comma) {
      next = io.NextInField(remaining, edit);
    }
  }
  return CheckCompleteListDirectedField(io, edit);
}

// See 13.10.3.1 paragraphs 7-9 in Fortran 2018
template <typename CHAR>
static RT_API_ATTRS bool EditDelimitedCharacterInput(
    IoStatementState &io, CHAR *x, std::size_t length, char32_t delimiter) {
  bool result{true};
  while (true) {
    std::size_t byteCount{0};
    auto ch{io.GetCurrentChar(byteCount)};
    if (!ch) {
      if (io.AdvanceRecord()) {
        continue;
      } else {
        result = false; // EOF in character value
        break;
      }
    }
    io.HandleRelativePosition(byteCount);
    if (*ch == delimiter) {
      auto next{io.GetCurrentChar(byteCount)};
      if (next && *next == delimiter) {
        // Repeated delimiter: use as character value
        io.HandleRelativePosition(byteCount);
      } else {
        break; // closing delimiter
      }
    }
    if (length > 0) {
      *x++ = static_cast<CHAR>(*ch);
      --length;
    }
  }
  Fortran::runtime::fill_n(x, length, ' ');
  return result;
}

template <typename CHAR>
static RT_API_ATTRS bool EditListDirectedCharacterInput(
    IoStatementState &io, CHAR *x, std::size_t length, const DataEdit &edit) {
  std::size_t byteCount{0};
  auto ch{io.GetCurrentChar(byteCount)};
  if (ch && (*ch == '\'' || *ch == '"')) {
    io.HandleRelativePosition(byteCount);
    return EditDelimitedCharacterInput(io, x, length, *ch);
  }
  if (IsNamelistNameOrSlash(io) || io.GetConnectionState().IsAtEOF()) {
    return false;
  }
  // Undelimited list-directed character input: stop at a value separator
  // or the end of the current record.
  while (auto ch{io.GetCurrentChar(byteCount)}) {
    if (IsCharValueSeparator(edit, *ch)) {
      break;
    }
    if (length > 0) {
      *x++ = static_cast<CHAR>(*ch);
      --length;
    } else if (edit.IsNamelist()) {
      // GNU compatibility
      break;
    }
    io.HandleRelativePosition(byteCount);
    io.GotChar(byteCount);
  }
  Fortran::runtime::fill_n(x, length, ' ');
  return true;
}

template <typename CHAR>
RT_API_ATTRS bool EditCharacterInput(IoStatementState &io, const DataEdit &edit,
    CHAR *x, std::size_t lengthChars) {
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
    return EditListDirectedCharacterInput(io, x, lengthChars, edit);
  case 'A':
  case 'G':
    break;
  case 'B':
    return EditBOZInput<1>(io, edit, x, lengthChars * sizeof *x);
  case 'O':
    return EditBOZInput<3>(io, edit, x, lengthChars * sizeof *x);
  case 'Z':
    return EditBOZInput<4>(io, edit, x, lengthChars * sizeof *x);
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used with a CHARACTER data item",
        edit.descriptor);
    return false;
  }
  const ConnectionState &connection{io.GetConnectionState()};
  std::size_t remainingChars{lengthChars};
  // Skip leading characters.
  // Their bytes don't count towards INQUIRE(IOLENGTH=).
  std::size_t skipChars{0};
  if (edit.width && *edit.width > 0) {
    remainingChars = *edit.width;
    if (remainingChars > lengthChars) {
      skipChars = remainingChars - lengthChars;
    }
  }
  // When the field is wider than the variable, we drop the leading
  // characters.  When the variable is wider than the field, there can be
  // trailing padding or an EOR condition.
  const char *input{nullptr};
  std::size_t readyBytes{0};
  // Transfer payload bytes; these do count.
  while (remainingChars > 0) {
    if (readyBytes == 0) {
      readyBytes = io.GetNextInputBytes(input);
      if (readyBytes == 0 ||
          (readyBytes < remainingChars && edit.modes.nonAdvancing)) {
        if (io.CheckForEndOfRecord(readyBytes, connection)) {
          if (readyBytes == 0) {
            // PAD='YES' and no more data
            Fortran::runtime::fill_n(x, lengthChars, ' ');
            return !io.GetIoErrorHandler().InError();
          } else {
            // Do partial read(s) then pad on last iteration
          }
        } else {
          return !io.GetIoErrorHandler().InError();
        }
      }
    }
    std::size_t chunkBytes;
    std::size_t chunkChars{1};
    bool skipping{skipChars > 0};
    if (connection.isUTF8) {
      chunkBytes = MeasureUTF8Bytes(*input);
      if (skipping) {
        --skipChars;
      } else if (auto ucs{DecodeUTF8(input)}) {
        if ((sizeof *x == 1 && *ucs > 0xff) ||
            (sizeof *x == 2 && *ucs > 0xffff)) {
          *x++ = '?';
        } else {
          *x++ = static_cast<CHAR>(*ucs);
        }
        --lengthChars;
      } else if (chunkBytes == 0) {
        // error recovery: skip bad encoding
        chunkBytes = 1;
      }
    } else if (connection.internalIoCharKind > 1) {
      // Reading from non-default character internal unit
      chunkBytes = connection.internalIoCharKind;
      if (skipping) {
        --skipChars;
      } else {
        char32_t buffer{0};
        runtime::memcpy(&buffer, input, chunkBytes);
        if ((sizeof *x == 1 && buffer > 0xff) ||
            (sizeof *x == 2 && buffer > 0xffff)) {
          *x++ = '?';
        } else {
          *x++ = static_cast<CHAR>(buffer);
        }
        --lengthChars;
      }
    } else if constexpr (sizeof *x > 1) {
      // Read single byte with expansion into multi-byte CHARACTER
      chunkBytes = 1;
      if (skipping) {
        --skipChars;
      } else {
        *x++ = static_cast<unsigned char>(*input);
        --lengthChars;
      }
    } else { // single bytes -> default CHARACTER
      if (skipping) {
        chunkBytes = std::min<std::size_t>(skipChars, readyBytes);
        chunkChars = chunkBytes;
        skipChars -= chunkChars;
      } else {
        chunkBytes = std::min<std::size_t>(remainingChars, readyBytes);
        chunkBytes = std::min<std::size_t>(lengthChars, chunkBytes);
        chunkChars = chunkBytes;
        runtime::memcpy(x, input, chunkBytes);
        x += chunkBytes;
        lengthChars -= chunkChars;
      }
    }
    input += chunkBytes;
    remainingChars -= chunkChars;
    if (!skipping) {
      io.GotChar(chunkBytes);
    }
    io.HandleRelativePosition(chunkBytes);
    readyBytes -= chunkBytes;
  }
  // Pad the remainder of the input variable, if any.
  Fortran::runtime::fill_n(x, lengthChars, ' ');
  return CheckCompleteListDirectedField(io, edit);
}

template RT_API_ATTRS bool EditRealInput<2>(
    IoStatementState &, const DataEdit &, void *);
template RT_API_ATTRS bool EditRealInput<3>(
    IoStatementState &, const DataEdit &, void *);
template RT_API_ATTRS bool EditRealInput<4>(
    IoStatementState &, const DataEdit &, void *);
template RT_API_ATTRS bool EditRealInput<8>(
    IoStatementState &, const DataEdit &, void *);
template RT_API_ATTRS bool EditRealInput<10>(
    IoStatementState &, const DataEdit &, void *);
// TODO: double/double
template RT_API_ATTRS bool EditRealInput<16>(
    IoStatementState &, const DataEdit &, void *);

template RT_API_ATTRS bool EditCharacterInput(
    IoStatementState &, const DataEdit &, char *, std::size_t);
template RT_API_ATTRS bool EditCharacterInput(
    IoStatementState &, const DataEdit &, char16_t *, std::size_t);
template RT_API_ATTRS bool EditCharacterInput(
    IoStatementState &, const DataEdit &, char32_t *, std::size_t);

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime::io
