//===-- runtime/edit-input.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "edit-input.h"
#include "namelist.h"
#include "utf.h"
#include "flang/Common/real.h"
#include "flang/Common/uint128.h"
#include <algorithm>
#include <cfenv>

namespace Fortran::runtime::io {

template <int LOG2_BASE>
static bool EditBOZInput(
    IoStatementState &io, const DataEdit &edit, void *n, std::size_t bytes) {
  std::optional<int> remaining;
  std::optional<char32_t> next{io.PrepareInput(edit, remaining)};
  if (next.value_or('?') == '0') {
    do {
      next = io.NextInField(remaining, edit);
    } while (next && *next == '0');
  }
  // Count significant digits after any leading white space & zeroes
  int digits{0};
  int chars{0};
  for (; next; next = io.NextInField(remaining, edit)) {
    ++chars;
    char32_t ch{*next};
    if (ch == ' ' || ch == '\t') {
      continue;
    }
    if (ch >= '0' && ch <= '1') {
    } else if (LOG2_BASE >= 3 && ch >= '2' && ch <= '7') {
    } else if (LOG2_BASE >= 4 && ch >= '8' && ch <= '9') {
    } else if (LOG2_BASE >= 4 && ch >= 'A' && ch <= 'F') {
    } else if (LOG2_BASE >= 4 && ch >= 'a' && ch <= 'f') {
    } else {
      io.GetIoErrorHandler().SignalError(
          "Bad character '%lc' in B/O/Z input field", ch);
      return false;
    }
    ++digits;
  }
  auto significantBytes{static_cast<std::size_t>(digits * LOG2_BASE + 7) / 8};
  if (significantBytes > bytes) {
    io.GetIoErrorHandler().SignalError(IostatBOZInputOverflow,
        "B/O/Z input of %d digits overflows %zd-byte variable", digits, bytes);
    return false;
  }
  // Reset to start of significant digits
  io.HandleRelativePosition(-chars);
  remaining.reset();
  // Make a second pass now that the digit count is known
  std::memset(n, 0, bytes);
  int increment{isHostLittleEndian ? -1 : 1};
  auto *data{reinterpret_cast<unsigned char *>(n) +
      (isHostLittleEndian ? significantBytes - 1 : 0)};
  int shift{((digits - 1) * LOG2_BASE) & 7};
  if (shift + LOG2_BASE > 8) {
    shift -= 8; // misaligned octal
  }
  while (digits > 0) {
    char32_t ch{*io.NextInField(remaining, edit)};
    int digit{0};
    if (ch >= '0' && ch <= '9') {
      digit = ch - '0';
    } else if (ch >= 'A' && ch <= 'F') {
      digit = ch + 10 - 'A';
    } else if (ch >= 'a' && ch <= 'f') {
      digit = ch + 10 - 'a';
    } else {
      continue;
    }
    --digits;
    if (shift < 0) {
      shift += 8;
      if (shift + LOG2_BASE > 8) { // misaligned octal
        *data |= digit >> (8 - shift);
      }
      data += increment;
    }
    *data |= digit << shift;
    shift -= LOG2_BASE;
  }
  return true;
}

static inline char32_t GetDecimalPoint(const DataEdit &edit) {
  return edit.modes.editingFlags & decimalComma ? char32_t{','} : char32_t{'.'};
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
      io.SkipSpaces(remaining);
      next = io.NextInField(remaining, edit);
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
    return EditBOZInput<1>(io, edit, n, kind);
  case 'O':
    return EditBOZInput<3>(io, edit, n, kind);
  case 'Z':
    return EditBOZInput<4>(io, edit, n, kind);
  case 'A': // legacy extension
    return EditCharacterInput(io, edit, reinterpret_cast<char *>(n), kind);
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used with an INTEGER data item",
        edit.descriptor);
    return false;
  }
  std::optional<int> remaining;
  std::optional<char32_t> next;
  bool negate{ScanNumericPrefix(io, edit, next, remaining)};
  common::UnsignedInt128 value{0};
  bool any{negate};
  bool overflow{false};
  for (; next; next = io.NextInField(remaining, edit)) {
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
    static constexpr auto maxu128{~common::UnsignedInt128{0}};
    static constexpr auto maxu128OverTen{maxu128 / 10};
    static constexpr int maxLastDigit{
        static_cast<int>(maxu128 - (maxu128OverTen * 10))};
    overflow |= value >= maxu128OverTen &&
        (value > maxu128OverTen || digit > maxLastDigit);
    value *= 10;
    value += digit;
    any = true;
  }
  auto maxForKind{common::UnsignedInt128{1} << ((8 * kind) - 1)};
  overflow |= value >= maxForKind && (value > maxForKind || !negate);
  if (overflow) {
    io.GetIoErrorHandler().SignalError(IostatIntegerInputOverflow,
        "Decimal input overflows INTEGER(%d) variable", kind);
    return false;
  }
  if (negate) {
    value = -value;
  }
  if (any || !io.GetConnectionState().IsAtEOF()) {
    std::memcpy(n, &value, kind); // a blank field means zero
  }
  return any;
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
  bool bzMode{(edit.modes.editingFlags & blankZero) != 0};
  if (!next || (!bzMode && *next == ' ')) { // empty/blank field means zero
    remaining.reset();
    if (!io.GetConnectionState().IsAtEOF()) {
      Put('0');
    }
    return got;
  }
  char32_t decimal{GetDecimalPoint(edit)};
  char32_t first{*next >= 'a' && *next <= 'z' ? *next + 'A' - 'a' : *next};
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
    if (next && *next == '(') { // NaN(...)
      Put('(');
      int depth{1};
      while (true) {
        next = io.NextInField(remaining, edit);
        if (depth == 0) {
          break;
        } else if (!next) {
          return 0; // error
        } else if (*next == '(') {
          ++depth;
        } else if (*next == ')') {
          --depth;
        }
        Put(*next);
      }
    }
    exponent = 0;
  } else if (first == decimal || (first >= '0' && first <= '9') ||
      (bzMode && (first == ' ' || first == '\t')) || first == 'E' ||
      first == 'D' || first == 'Q') {
    Put('.'); // input field is normalized to a fraction
    auto start{got};
    for (; next; next = io.NextInField(remaining, edit)) {
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
      // Nothing but zeroes and maybe a decimal point.  F'2018 requires
      // at least one digit, but F'77 did not, and a bare "." shows up in
      // the FCVS suite.
      Put('0'); // emit at least one digit
    }
    if (next &&
        (*next == 'e' || *next == 'E' || *next == 'd' || *next == 'D' ||
            *next == 'q' || *next == 'Q')) {
      // Optional exponent letter.  Blanks are allowed between the
      // optional exponent letter and the exponent value.
      io.SkipSpaces(remaining);
      next = io.NextInField(remaining, edit);
    }
    // The default exponent is -kP, but the scale factor doesn't affect
    // an explicit exponent.
    exponent = -edit.modes.scale;
    if (next &&
        (*next == '-' || *next == '+' || (*next >= '0' && *next <= '9') ||
            *next == ' ' || *next == '\t')) {
      bool negExpo{*next == '-'};
      if (negExpo || *next == '+') {
        next = io.NextInField(remaining, edit);
      }
      for (exponent = 0; next; next = io.NextInField(remaining, edit)) {
        if (*next >= '0' && *next <= '9') {
          if (exponent < 10000) {
            exponent = 10 * exponent + *next - '0';
          }
        } else if (*next == ' ' || *next == '\t') {
          if (bzMode) {
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
      next = io.NextInField(remaining, edit);
    }
    if (!next) { // NextInField fails on separators like ')'
      std::size_t byteCount{0};
      next = io.GetCurrentChar(byteCount);
      if (next && *next == ')') {
        io.HandleRelativePosition(byteCount);
      }
    }
  } else if (remaining) {
    while (next && (*next == ' ' || *next == '\t')) {
      next = io.NextInField(remaining, edit);
    }
    if (next) {
      return 0; // error: unused nonblank character in fixed-width field
    }
  }
  return got;
}

static void RaiseFPExceptions(decimal::ConversionResultFlags flags) {
#undef RAISE
#ifdef feraisexcept // a macro in some environments; omit std::
#define RAISE feraiseexcept
#else
#define RAISE std::feraiseexcept
#endif
  if (flags & decimal::ConversionResultFlags::Overflow) {
    RAISE(FE_OVERFLOW);
  }
  if (flags & decimal::ConversionResultFlags::Inexact) {
    RAISE(FE_INEXACT);
  }
  if (flags & decimal::ConversionResultFlags::Invalid) {
    RAISE(FE_INVALID);
  }
#undef RAISE
}

// If no special modes are in effect and the form of the input value
// that's present in the input stream is acceptable to the decimal->binary
// converter without modification, this fast path for real input
// saves time by avoiding memory copies and reformatting of the exponent.
template <int PRECISION>
static bool TryFastPathRealInput(
    IoStatementState &io, const DataEdit &edit, void *n) {
  if (edit.modes.editingFlags & (blankZero | decimalComma)) {
    return false;
  }
  if (edit.modes.scale != 0) {
    return false;
  }
  const char *str{nullptr};
  std::size_t got{io.GetNextInputBytes(str)};
  if (got == 0 || str == nullptr ||
      !io.GetConnectionState().recordLength.has_value()) {
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
  for (; p < limit && (*p == ' ' || *p == '\t'); ++p) {
  }
  if (edit.descriptor == DataEdit::ListDirectedImaginaryPart) {
    // Need to consume a trailing ')' and any white space after
    if (p >= limit || *p != ')') {
      return false;
    }
    for (++p; p < limit && (*p == ' ' || *p == '\t'); ++p) {
    }
  }
  if (edit.width && p < str + *edit.width) {
    return false; // unconverted characters remain in fixed width field
  }
  // Success on the fast path!
  *reinterpret_cast<decimal::BinaryFloatingPointNumber<PRECISION> *>(n) =
      converted.binary;
  io.HandleRelativePosition(p - str);
  // Set FP exception flags
  if (converted.flags != decimal::ConversionResultFlags::Exact) {
    RaiseFPExceptions(converted.flags);
  }
  return true;
}

template <int KIND>
bool EditCommonRealInput(IoStatementState &io, const DataEdit &edit, void *n) {
  constexpr int binaryPrecision{common::PrecisionOfRealKind(KIND)};
  if (TryFastPathRealInput<binaryPrecision>(io, edit, n)) {
    return true;
  }
  // Fast path wasn't available or didn't work; go the more general route
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
    io.GetIoErrorHandler().SignalError(IostatBadRealInput);
    return false;
  }
  bool hadExtra{got > maxDigits};
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
  const char *p{buffer};
  decimal::ConversionToBinaryResult<binaryPrecision> converted{
      decimal::ConvertToBinary<binaryPrecision>(p, edit.modes.round)};
  if (hadExtra) {
    converted.flags = static_cast<enum decimal::ConversionResultFlags>(
        converted.flags | decimal::Inexact);
  }
  if (*p) { // unprocessed junk after value
    io.GetIoErrorHandler().SignalError(IostatBadRealInput);
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
  return true;
}

template <int KIND>
bool EditRealInput(IoStatementState &io, const DataEdit &edit, void *n) {
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
  if (remaining) { // ignore the rest of the field
    io.HandleRelativePosition(*remaining);
  } else if (edit.descriptor == DataEdit::ListDirected) {
    while (io.NextInField(remaining, edit)) { // discard rest of field
    }
  }
  return true;
}

// See 13.10.3.1 paragraphs 7-9 in Fortran 2018
template <typename CHAR>
static bool EditDelimitedCharacterInput(
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
      *x++ = *ch;
      --length;
    }
  }
  std::fill_n(x, length, ' ');
  return result;
}

template <typename CHAR>
static bool EditListDirectedCharacterInput(
    IoStatementState &io, CHAR *x, std::size_t length, const DataEdit &edit) {
  std::size_t byteCount{0};
  auto ch{io.GetCurrentChar(byteCount)};
  if (ch && (*ch == '\'' || *ch == '"')) {
    io.HandleRelativePosition(byteCount);
    return EditDelimitedCharacterInput(io, x, length, *ch);
  }
  if (IsNamelistName(io) || io.GetConnectionState().IsAtEOF()) {
    return false;
  }
  // Undelimited list-directed character input: stop at a value separator
  // or the end of the current record.  Subtlety: the "remaining" count
  // here is a dummy that's used to avoid the interpretation of separators
  // in NextInField.
  std::optional<int> remaining{length > 0 ? maxUTF8Bytes : 0};
  while (std::optional<char32_t> next{io.NextInField(remaining, edit)}) {
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
      remaining = --length > 0 ? maxUTF8Bytes : 0;
    }
  }
  std::fill_n(x, length, ' ');
  return true;
}

template <typename CHAR>
bool EditCharacterInput(
    IoStatementState &io, const DataEdit &edit, CHAR *x, std::size_t length) {
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
    return EditListDirectedCharacterInput(io, x, length, edit);
  case 'A':
  case 'G':
    break;
  case 'B':
    return EditBOZInput<1>(io, edit, x, length * sizeof *x);
  case 'O':
    return EditBOZInput<3>(io, edit, x, length * sizeof *x);
  case 'Z':
    return EditBOZInput<4>(io, edit, x, length * sizeof *x);
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used with a CHARACTER data item",
        edit.descriptor);
    return false;
  }
  const ConnectionState &connection{io.GetConnectionState()};
  std::size_t remaining{length};
  if (edit.width && *edit.width > 0) {
    remaining = *edit.width;
  }
  // When the field is wider than the variable, we drop the leading
  // characters.  When the variable is wider than the field, there can be
  // trailing padding.
  const char *input{nullptr};
  std::size_t ready{0};
  // Skip leading bytes.
  // These bytes don't count towards INQUIRE(IOLENGTH=).
  std::size_t skip{remaining > length ? remaining - length : 0};
  // Transfer payload bytes; these do count.
  while (remaining > 0) {
    if (ready == 0) {
      ready = io.GetNextInputBytes(input);
      if (ready == 0) {
        if (io.CheckForEndOfRecord()) {
          std::fill_n(x, length, ' '); // PAD='YES'
        }
        return !io.GetIoErrorHandler().InError();
      }
    }
    std::size_t chunk;
    bool skipping{skip > 0};
    if (connection.isUTF8) {
      chunk = MeasureUTF8Bytes(*input);
      if (skipping) {
        --skip;
      } else if (auto ucs{DecodeUTF8(input)}) {
        *x++ = *ucs;
        --length;
      } else if (chunk == 0) {
        // error recovery: skip bad encoding
        chunk = 1;
      }
      --remaining;
    } else if constexpr (sizeof *x > 1) {
      // Read single byte with expansion into multi-byte CHARACTER
      chunk = 1;
      if (skipping) {
        --skip;
      } else {
        *x++ = static_cast<unsigned char>(*input);
        --length;
      }
      --remaining;
    } else { // single bytes -> default CHARACTER
      if (skipping) {
        chunk = std::min<std::size_t>(skip, ready);
        skip -= chunk;
      } else {
        chunk = std::min<std::size_t>(remaining, ready);
        std::memcpy(x, input, chunk);
        x += chunk;
        length -= chunk;
      }
      remaining -= chunk;
    }
    input += chunk;
    if (!skipping) {
      io.GotChar(chunk);
    }
    io.HandleRelativePosition(chunk);
    ready -= chunk;
  }
  // Pad the remainder of the input variable, if any.
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

template bool EditCharacterInput(
    IoStatementState &, const DataEdit &, char *, std::size_t);
template bool EditCharacterInput(
    IoStatementState &, const DataEdit &, char16_t *, std::size_t);
template bool EditCharacterInput(
    IoStatementState &, const DataEdit &, char32_t *, std::size_t);

} // namespace Fortran::runtime::io
