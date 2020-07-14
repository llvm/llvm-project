//===-- runtime/edit-input.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "edit-input.h"
#include "flang/Common/real.h"
#include "flang/Common/uint128.h"
#include <algorithm>

namespace Fortran::runtime::io {

static std::optional<char32_t> PrepareInput(
    IoStatementState &io, const DataEdit &edit, std::optional<int> &remaining) {
  remaining.reset();
  if (edit.descriptor == DataEdit::ListDirected) {
    io.GetNextNonBlank();
  } else {
    if (edit.width.value_or(0) > 0) {
      remaining = *edit.width;
    }
    io.SkipSpaces(remaining);
  }
  return io.NextInField(remaining);
}

static bool EditBOZInput(IoStatementState &io, const DataEdit &edit, void *n,
    int base, int totalBitSize) {
  std::optional<int> remaining;
  std::optional<char32_t> next{PrepareInput(io, edit, remaining)};
  common::UnsignedInt128 value{0};
  for (; next; next = io.NextInField(remaining)) {
    char32_t ch{*next};
    if (ch == ' ') {
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

// Returns false if there's a '-' sign
static bool ScanNumericPrefix(IoStatementState &io, const DataEdit &edit,
    std::optional<char32_t> &next, std::optional<int> &remaining) {
  next = PrepareInput(io, edit, remaining);
  bool negative{false};
  if (next) {
    negative = *next == '-';
    if (negative || *next == '+') {
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
  case 'G':
  case 'I':
    break;
  case 'B':
    return EditBOZInput(io, edit, n, 2, kind << 3);
  case 'O':
    return EditBOZInput(io, edit, n, 8, kind << 3);
  case 'Z':
    return EditBOZInput(io, edit, n, 16, kind << 3);
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
    if (ch == ' ') {
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

static int ScanRealInput(char *buffer, int bufferSize, IoStatementState &io,
    const DataEdit &edit, int &exponent) {
  std::optional<int> remaining;
  std::optional<char32_t> next;
  int got{0};
  std::optional<int> decimalPoint;
  if (ScanNumericPrefix(io, edit, next, remaining) && next) {
    if (got < bufferSize) {
      buffer[got++] = '-';
    }
  }
  if (!next) { // empty field means zero
    if (got < bufferSize) {
      buffer[got++] = '0';
    }
    return got;
  }
  if (got < bufferSize) {
    buffer[got++] = '.'; // input field is normalized to a fraction
  }
  char32_t decimal = edit.modes.editingFlags & decimalComma ? ',' : '.';
  auto start{got};
  if ((*next >= 'a' && *next <= 'z') || (*next >= 'A' && *next <= 'Z')) {
    // NaN or infinity - convert to upper case
    for (; next &&
         ((*next >= 'a' && *next <= 'z') || (*next >= 'A' && *next <= 'Z'));
         next = io.NextInField(remaining)) {
      if (got < bufferSize) {
        if (*next >= 'a' && *next <= 'z') {
          buffer[got++] = *next - 'a' + 'A';
        } else {
          buffer[got++] = *next;
        }
      }
    }
    if (next && *next == '(') { // NaN(...)
      while (next && *next != ')') {
        next = io.NextInField(remaining);
      }
    }
    exponent = 0;
  } else if (*next == decimal || (*next >= '0' && *next <= '9')) {
    for (; next; next = io.NextInField(remaining)) {
      char32_t ch{*next};
      if (ch == ' ') {
        if (edit.modes.editingFlags & blankZero) {
          ch = '0'; // BZ mode - treat blank as if it were zero
        } else {
          continue;
        }
      }
      if (ch == '0' && got == start) {
        // omit leading zeroes
      } else if (ch >= '0' && ch <= '9') {
        if (got < bufferSize) {
          buffer[got++] = ch;
        }
      } else if (ch == decimal && !decimalPoint) {
        // the decimal point is *not* copied to the buffer
        decimalPoint = got - start; // # of digits before the decimal point
      } else {
        break;
      }
    }
    if (got == start && got < bufferSize) {
      buffer[got++] = '0'; // all digits were zeroes
    }
    if (next &&
        (*next == 'e' || *next == 'E' || *next == 'd' || *next == 'D' ||
            *next == 'q' || *next == 'Q')) {
      io.SkipSpaces(remaining);
      next = io.NextInField(remaining);
    }
    exponent = -edit.modes.scale; // default exponent is -kP
    if (next &&
        (*next == '-' || *next == '+' || (*next >= '0' && *next <= '9'))) {
      bool negExpo{*next == '-'};
      if (negExpo || *next == '+') {
        next = io.NextInField(remaining);
      }
      for (exponent = 0; next && (*next >= '0' && *next <= '9');
           next = io.NextInField(remaining)) {
        exponent = 10 * exponent + *next - '0';
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
  if (remaining) {
    while (next && *next == ' ') {
      next = io.NextInField(remaining);
    }
    if (next) {
      return 0; // error: unused nonblank character in fixed-width field
    }
  }
  return got;
}

template <int binaryPrecision>
bool EditCommonRealInput(IoStatementState &io, const DataEdit &edit, void *n) {
  static constexpr int maxDigits{
      common::MaxDecimalConversionDigits(binaryPrecision)};
  static constexpr int bufferSize{maxDigits + 18};
  char buffer[bufferSize];
  int exponent{0};
  int got{ScanRealInput(buffer, maxDigits + 2, io, edit, exponent)};
  if (got >= maxDigits + 2) {
    io.GetIoErrorHandler().Crash("EditRealInput: buffer was too small");
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

template <int binaryPrecision>
bool EditRealInput(IoStatementState &io, const DataEdit &edit, void *n) {
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
  case 'F':
  case 'E': // incl. EN, ES, & EX
  case 'D':
  case 'G':
    return EditCommonRealInput<binaryPrecision>(io, edit, n);
  case 'B':
    return EditBOZInput(
        io, edit, n, 2, common::BitsForBinaryPrecision(binaryPrecision));
  case 'O':
    return EditBOZInput(
        io, edit, n, 8, common::BitsForBinaryPrecision(binaryPrecision));
  case 'Z':
    return EditBOZInput(
        io, edit, n, 16, common::BitsForBinaryPrecision(binaryPrecision));
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
  std::optional<char32_t> next{PrepareInput(io, edit, remaining)};
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
  while (true) {
    if (auto ch{io.GetCurrentChar()}) {
      io.HandleRelativePosition(1);
      if (*ch == delimiter) {
        ch = io.GetCurrentChar();
        if (ch && *ch == delimiter) {
          // Repeated delimiter: use as character value.  Can't straddle a
          // record boundary.
          io.HandleRelativePosition(1);
        } else {
          std::fill_n(x, length, ' ');
          return true;
        }
      }
      if (length > 0) {
        *x++ = *ch;
        --length;
      }
    } else if (!io.AdvanceRecord()) { // EOF
      std::fill_n(x, length, ' ');
      return false;
    }
  }
}

static bool EditListDirectedDefaultCharacterInput(
    IoStatementState &io, char *x, std::size_t length) {
  auto ch{io.GetCurrentChar()};
  if (ch && (*ch == '\'' || *ch == '"')) {
    io.HandleRelativePosition(1);
    return EditDelimitedCharacterInput(io, x, length, *ch);
  }
  // Undelimited list-directed character input: stop at a value separator
  // or the end of the current record.
  std::optional<int> remaining{length};
  for (std::optional<char32_t> next{io.NextInField(remaining)}; next;
       next = io.NextInField(remaining)) {
    switch (*next) {
    case ' ':
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
    } else {
      *x++ = *next;
      --length;
    }
  }
  std::fill_n(x, length, ' ');
  return true;
}

template bool EditRealInput<8>(IoStatementState &, const DataEdit &, void *);
template bool EditRealInput<11>(IoStatementState &, const DataEdit &, void *);
template bool EditRealInput<24>(IoStatementState &, const DataEdit &, void *);
template bool EditRealInput<53>(IoStatementState &, const DataEdit &, void *);
template bool EditRealInput<64>(IoStatementState &, const DataEdit &, void *);
template bool EditRealInput<113>(IoStatementState &, const DataEdit &, void *);
} // namespace Fortran::runtime::io
