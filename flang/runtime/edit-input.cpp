//===-- runtime/edit-input.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "edit-input.h"
#include "namelist.h"
#include "flang/Common/real.h"
#include "flang/Common/uint128.h"
#include <algorithm>

namespace Fortran::runtime::io {

static bool EditBOZInput(IoStatementState &io, const DataEdit &edit, void *n,
    int base, int totalBitSize) {
  std::optional<int> remaining;
  std::optional<char32_t> next{io.PrepareInput(edit, remaining)};
  common::UnsignedInt128 value{0};
  for (; next; next = io.NextInField(remaining)) {
    char32_t ch{*next};
    if (ch == ' ' || ch == '\t') {
      continue;
    }
    int digit{0};
    if (ch >= '0' && ch <= '1') {
      digit = ch - '0';
    } else if (base >= 8 && ch >= '2' && ch <= '7') {
      digit = ch - '0';
    } else if (base >= 10 && ch >= '8' && ch <= '9') {
      digit = ch - '0';
    } else if (base == 16 && ch >= 'A' && ch <= 'Z') {
      digit = ch + 10 - 'A';
    } else if (base == 16 && ch >= 'a' && ch <= 'z') {
      digit = ch + 10 - 'a';
    } else {
      io.GetIoErrorHandler().SignalError(
          "Bad character '%lc' in B/O/Z input field", ch);
      return false;
    }
    value *= base;
    value += digit;
  }
  // TODO: check for overflow
  std::memcpy(n, &value, totalBitSize >> 3);
  return true;
}

// Prepares input from a field, and consumes the sign, if any.
// Returns true if there's a '-' sign.
static bool ScanNumericPrefix(IoStatementState &io, const DataEdit &edit,
    std::optional<char32_t> &next, std::optional<int> &remaining) {
  next = io.PrepareInput(edit, remaining);
  bool negative{false};
  if (next) {
    negative = *next == '-';
    if (negative || *next == '+') {
      io.GotChar();
      io.SkipSpaces(remaining);
      next = io.NextInField(remaining);
    }
  }
  return negative;
}

bool EditIntegerInput(
    IoStatementState &io, const DataEdit &edit, void *n, int kind) {
  RUNTIME_CHECK(io.GetIoErrorHandler(), kind >= 1 && !(kind & (kind - 1)));
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
    if (IsNamelistName(io)) {
      return false;
    }
    break;
  case 'G':
  case 'I':
    break;
  case 'B':
    return EditBOZInput(io, edit, n, 2, kind << 3);
  case 'O':
    return EditBOZInput(io, edit, n, 8, kind << 3);
  case 'Z':
    return EditBOZInput(io, edit, n, 16, kind << 3);
  case 'A': // legacy extension
    return EditDefaultCharacterInput(
        io, edit, reinterpret_cast<char *>(n), kind);
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used with an INTEGER data item",
        edit.descriptor);
    return false;
  }
  std::optional<int> remaining;
  std::optional<char32_t> next;
  bool negate{ScanNumericPrefix(io, edit, next, remaining)};
  common::UnsignedInt128 value;
  for (; next; next = io.NextInField(remaining)) {
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
    } else {
      io.GetIoErrorHandler().SignalError(
          "Bad character '%lc' in INTEGER input field", ch);
      return false;
    }
    value *= 10;
    value += digit;
  }
  if (negate) {
    value = -value;
  }
  std::memcpy(n, &value, kind);
  return true;
}

// Parses a REAL input number from the input source as a normalized
// fraction into a supplied buffer -- there's an optional '-', a
// decimal point, and at least one digit.  The adjusted exponent value
// is returned in a reference argument.  The returned value is the number
// of characters that (should) have been written to the buffer -- this can
// be larger than the buffer size and can indicate overflow.  Replaces
// blanks with zeroes if appropriate.
static int ScanRealInput(char *buffer, int bufferSize, IoStatementState &io,
    const DataEdit &edit, int &exponent) {
  std::optional<int> remaining;
  std::optional<char32_t> next;
  int got{0};
  std::optional<int> decimalPoint;
  auto Put{[&](char ch) -> void {
    if (got < bufferSize) {
      buffer[got] = ch;
    }
    ++got;
  }};
  if (ScanNumericPrefix(io, edit, next, remaining)) {
    Put('-');
  }
  if (next.value_or(' ') == ' ') { // empty/blank field means zero
    remaining.reset();
    Put('0');
    return got;
  }
  char32_t decimal = edit.modes.editingFlags & decimalComma ? ',' : '.';
  char32_t first{*next >= 'a' && *next <= 'z' ? *next + 'A' - 'a' : *next};
  if (first == 'N' || first == 'I') {
    // NaN or infinity - convert to upper case
    // Subtle: a blank field of digits could be followed by 'E' or 'D',
    for (; next &&
         ((*next >= 'a' && *next <= 'z') || (*next >= 'A' && *next <= 'Z'));
         next = io.NextInField(remaining)) {
      if (*next >= 'a' && *next <= 'z') {
        Put(*next - 'a' + 'A');
      } else {
        Put(*next);
      }
    }
    if (next && *next == '(') { // NaN(...)
      while (next && *next != ')') {
        next = io.NextInField(remaining);
      }
    }
    exponent = 0;
  } else if (first == decimal || (first >= '0' && first <= '9') ||
      first == 'E' || first == 'D' || first == 'Q') {
    Put('.'); // input field is normalized to a fraction
    auto start{got};
    bool bzMode{(edit.modes.editingFlags & blankZero) != 0};
    for (; next; next = io.NextInField(remaining)) {
      char32_t ch{*next};
      if (ch == ' ' || ch == '\t') {
        if (bzMode) {
          ch = '0'; // BZ mode - treat blank as if it were zero
        } else {
          continue;
        }
      }
      if (ch == '0' && got == start && !decimalPoint) {
        // omit leading zeroes before the decimal
      } else if (ch >= '0' && ch <= '9') {
        Put(ch);
      } else if (ch == decimal && !decimalPoint) {
        // the decimal point is *not* copied to the buffer
        decimalPoint = got - start; // # of digits before the decimal point
      } else {
        break;
      }
    }
    if (got == start) {
      Put('0'); // emit at least one digit
    }
    if (next &&
        (*next == 'e' || *next == 'E' || *next == 'd' || *next == 'D' ||
            *next == 'q' || *next == 'Q')) {
      // Optional exponent letter.  Blanks are allowed between the
      // optional exponent letter and the exponent value.
      io.SkipSpaces(remaining);
      next = io.NextInField(remaining);
    }
    // The default exponent is -kP, but the scale factor doesn't affect
    // an explicit exponent.
    exponent = -edit.modes.scale;
    if (next &&
        (*next == '-' || *next == '+' || (*next >= '0' && *next <= '9') ||
            (bzMode && (*next == ' ' || *next == '\t')))) {
      bool negExpo{*next == '-'};
      if (negExpo || *next == '+') {
        next = io.NextInField(remaining);
      }
      for (exponent = 0; next; next = io.NextInField(remaining)) {
        if (*next >= '0' && *next <= '9') {
          exponent = 10 * exponent + *next - '0';
        } else if (bzMode && (*next == ' ' || *next == '\t')) {
          exponent = 10 * exponent;
        } else {
          break;
        }
      }
      if (negExpo) {
        exponent = -exponent;
      }
    }
    if (decimalPoint) {
      exponent += *decimalPoint;
    } else {
      // When no decimal point (or comma) appears in the value, the 'd'
      // part of the edit descriptor must be interpreted as the number of
      // digits in the value to be interpreted as being to the *right* of
      // the assumed decimal point (13.7.2.3.2)
      exponent += got - start - edit.digits.value_or(0);
    }
  } else {
    // TODO: hex FP input
    exponent = 0;
    return 0;
  }
  // Consume the trailing ')' of a list-directed or NAMELIST complex
  // input value.
  if (edit.descriptor == DataEdit::ListDirectedImaginaryPart) {
    if (next && (*next == ' ' || *next == '\t')) {
      next = io.NextInField(remaining);
    }
    if (!next) { // NextInField fails on separators like ')'
      next = io.GetCurrentChar();
      if (next && *next == ')') {
        io.HandleRelativePosition(1);
      }
    }
  } else if (remaining) {
    while (next && (*next == ' ' || *next == '\t')) {
      next = io.NextInField(remaining);
    }
    if (next) {
      return 0; // error: unused nonblank character in fixed-width field
    }
  }
  return got;
}

template <int KIND>
bool EditCommonRealInput(IoStatementState &io, const DataEdit &edit, void *n) {
  constexpr int binaryPrecision{common::PrecisionOfRealKind(KIND)};
  static constexpr int maxDigits{
      common::MaxDecimalConversionDigits(binaryPrecision)};
  static constexpr int bufferSize{maxDigits + 18};
  char buffer[bufferSize];
  int exponent{0};
  int got{ScanRealInput(buffer, maxDigits + 2, io, edit, exponent)};
  if (got >= maxDigits + 2) {
    io.GetIoErrorHandler().Crash("EditCommonRealInput: buffer was too small");
    return false;
  }
  if (got == 0) {
    io.GetIoErrorHandler().SignalError("Bad REAL input value");
    return false;
  }
  bool hadExtra{got > maxDigits};
  if (exponent != 0) {
    got += std::snprintf(&buffer[got], bufferSize - got, "e%d", exponent);
  }
  buffer[got] = '\0';
  const char *p{buffer};
  decimal::ConversionToBinaryResult<binaryPrecision> converted{
      decimal::ConvertToBinary<binaryPrecision>(p, edit.modes.round)};
  if (hadExtra) {
    converted.flags = static_cast<enum decimal::ConversionResultFlags>(
        converted.flags | decimal::Inexact);
  }
  // TODO: raise converted.flags as exceptions?
  *reinterpret_cast<decimal::BinaryFloatingPointNumber<binaryPrecision> *>(n) =
      converted.binary;
  return true;
}

template <int KIND>
bool EditRealInput(IoStatementState &io, const DataEdit &edit, void *n) {
  constexpr int binaryPrecision{common::PrecisionOfRealKind(KIND)};
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
    if (IsNamelistName(io)) {
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
    return EditBOZInput(
        io, edit, n, 2, common::BitsForBinaryPrecision(binaryPrecision));
  case 'O':
    return EditBOZInput(
        io, edit, n, 8, common::BitsForBinaryPrecision(binaryPrecision));
  case 'Z':
    return EditBOZInput(
        io, edit, n, 16, common::BitsForBinaryPrecision(binaryPrecision));
  case 'A': // legacy extension
    return EditDefaultCharacterInput(
        io, edit, reinterpret_cast<char *>(n), KIND);
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used for REAL input",
        edit.descriptor);
    return false;
  }
}

// 13.7.3 in Fortran 2018
bool EditLogicalInput(IoStatementState &io, const DataEdit &edit, bool &x) {
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
    if (IsNamelistName(io)) {
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
  std::optional<int> remaining;
  std::optional<char32_t> next{io.PrepareInput(edit, remaining)};
  if (next && *next == '.') { // skip optional period
    next = io.NextInField(remaining);
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
  if (remaining) { // ignore the rest of the field
    io.HandleRelativePosition(*remaining);
  } else if (edit.descriptor == DataEdit::ListDirected) {
    while (io.NextInField(remaining)) { // discard rest of field
    }
  }
  return true;
}

// See 13.10.3.1 paragraphs 7-9 in Fortran 2018
static bool EditDelimitedCharacterInput(
    IoStatementState &io, char *x, std::size_t length, char32_t delimiter) {
  bool result{true};
  while (true) {
    auto ch{io.GetCurrentChar()};
    if (!ch) {
      if (io.AdvanceRecord()) {
        continue;
      } else {
        result = false; // EOF in character value
        break;
      }
    }
    io.HandleRelativePosition(1);
    if (*ch == delimiter) {
      auto next{io.GetCurrentChar()};
      if (next && *next == delimiter) {
        // Repeated delimiter: use as character value
        io.HandleRelativePosition(1);
      } else {
        break; // closing delimiter
      }
    }
    if (length > 0) {
      *x++ = *ch;
      --length;
    }
  }
  std::fill_n(x, length, ' ');
  return result;
}

static bool EditListDirectedDefaultCharacterInput(
    IoStatementState &io, char *x, std::size_t length) {
  auto ch{io.GetCurrentChar()};
  if (ch && (*ch == '\'' || *ch == '"')) {
    io.HandleRelativePosition(1);
    return EditDelimitedCharacterInput(io, x, length, *ch);
  }
  if (IsNamelistName(io)) {
    return false;
  }
  // Undelimited list-directed character input: stop at a value separator
  // or the end of the current record.
  std::optional<int> remaining{length};
  for (std::optional<char32_t> next{io.NextInField(remaining)}; next;
       next = io.NextInField(remaining)) {
    switch (*next) {
    case ' ':
    case '\t':
    case ',':
    case ';':
    case '/':
      remaining = 0; // value separator: stop
      break;
    default:
      *x++ = *next;
      --length;
    }
  }
  std::fill_n(x, length, ' ');
  return true;
}

bool EditDefaultCharacterInput(
    IoStatementState &io, const DataEdit &edit, char *x, std::size_t length) {
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
    return EditListDirectedDefaultCharacterInput(io, x, length);
  case 'A':
  case 'G':
    break;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used with a CHARACTER data item",
        edit.descriptor);
    return false;
  }
  std::optional<int> remaining{length};
  if (edit.width && *edit.width > 0) {
    remaining = *edit.width;
  }
  // When the field is wider than the variable, we drop the leading
  // characters.  When the variable is wider than the field, there's
  // trailing padding.
  std::int64_t skip{*remaining - static_cast<std::int64_t>(length)};
  for (std::optional<char32_t> next{io.NextInField(remaining)}; next;
       next = io.NextInField(remaining)) {
    if (skip > 0) {
      --skip;
      io.GotChar(-1);
    } else {
      *x++ = *next;
      --length;
    }
  }
  std::fill_n(x, length, ' ');
  return true;
}

template bool EditRealInput<2>(IoStatementState &, const DataEdit &, void *);
template bool EditRealInput<3>(IoStatementState &, const DataEdit &, void *);
template bool EditRealInput<4>(IoStatementState &, const DataEdit &, void *);
template bool EditRealInput<8>(IoStatementState &, const DataEdit &, void *);
template bool EditRealInput<10>(IoStatementState &, const DataEdit &, void *);
// TODO: double/double
template bool EditRealInput<16>(IoStatementState &, const DataEdit &, void *);
} // namespace Fortran::runtime::io
