//===-- runtime/edit-output.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "edit-output.h"
#include "flang/Common/uint128.h"
#include <algorithm>

namespace Fortran::runtime::io {

template <typename INT, typename UINT>
bool EditIntegerOutput(IoStatementState &io, const DataEdit &edit, INT n) {
  char buffer[130], *end = &buffer[sizeof buffer], *p = end;
  bool isNegative{false};
  if constexpr (std::is_same_v<INT, UINT>) {
    isNegative = (n >> (8 * sizeof(INT) - 1)) != 0;
  } else {
    isNegative = n < 0;
  }
  UINT un{static_cast<UINT>(isNegative ? -n : n)};
  int signChars{0};
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
  case 'G':
  case 'I':
    if (isNegative || (edit.modes.editingFlags & signPlus)) {
      signChars = 1; // '-' or '+'
    }
    while (un > 0) {
      auto quotient{un / 10u};
      *--p = '0' + static_cast<int>(un - UINT{10} * quotient);
      un = quotient;
    }
    break;
  case 'B':
    for (; un > 0; un >>= 1) {
      *--p = '0' + (static_cast<int>(un) & 1);
    }
    break;
  case 'O':
    for (; un > 0; un >>= 3) {
      *--p = '0' + (static_cast<int>(un) & 7);
    }
    break;
  case 'Z':
    for (; un > 0; un >>= 4) {
      int digit = static_cast<int>(un) & 0xf;
      *--p = digit >= 10 ? 'A' + (digit - 10) : '0' + digit;
    }
    break;
  default:
    io.GetIoErrorHandler().Crash(
        "Data edit descriptor '%c' may not be used with an INTEGER data item",
        edit.descriptor);
    return false;
  }

  int digits = end - p;
  int leadingZeroes{0};
  int editWidth{edit.width.value_or(0)};
  if (edit.digits && digits <= *edit.digits) { // Iw.m
    if (*edit.digits == 0 && n == 0) {
      // Iw.0 with zero value: output field must be blank.  For I0.0
      // and a zero value, emit one blank character.
      signChars = 0; // in case of SP
      editWidth = std::max(1, editWidth);
    } else {
      leadingZeroes = *edit.digits - digits;
    }
  } else if (n == 0) {
    leadingZeroes = 1;
  }
  int total{signChars + leadingZeroes + digits};
  if (editWidth > 0 && total > editWidth) {
    return io.EmitRepeated('*', editWidth);
  }
  int leadingSpaces{std::max(0, editWidth - total)};
  if (edit.IsListDirected()) {
    if (static_cast<std::size_t>(total) >
            io.GetConnectionState().RemainingSpaceInRecord() &&
        !io.AdvanceRecord()) {
      return false;
    }
    leadingSpaces = 1;
  }
  return io.EmitRepeated(' ', leadingSpaces) &&
      io.Emit(n < 0 ? "-" : "+", signChars) &&
      io.EmitRepeated('0', leadingZeroes) && io.Emit(p, digits);
}

// Formats the exponent (see table 13.1 for all the cases)
const char *RealOutputEditingBase::FormatExponent(
    int expo, const DataEdit &edit, int &length) {
  char *eEnd{&exponent_[sizeof exponent_]};
  char *exponent{eEnd};
  for (unsigned e{static_cast<unsigned>(std::abs(expo))}; e > 0;) {
    unsigned quotient{e / 10u};
    *--exponent = '0' + e - 10 * quotient;
    e = quotient;
  }
  if (edit.expoDigits) {
    if (int ed{*edit.expoDigits}) { // Ew.dEe with e > 0
      while (exponent > exponent_ + 2 /*E+*/ && exponent + ed > eEnd) {
        *--exponent = '0';
      }
    } else if (exponent == eEnd) {
      *--exponent = '0'; // Ew.dE0 with zero-valued exponent
    }
  } else { // ensure at least two exponent digits
    while (exponent + 2 > eEnd) {
      *--exponent = '0';
    }
  }
  *--exponent = expo < 0 ? '-' : '+';
  if (edit.expoDigits || edit.IsListDirected() || exponent + 3 == eEnd) {
    *--exponent = edit.descriptor == 'D' ? 'D' : 'E'; // not 'G'
  }
  length = eEnd - exponent;
  return exponent;
}

bool RealOutputEditingBase::EmitPrefix(
    const DataEdit &edit, std::size_t length, std::size_t width) {
  if (edit.IsListDirected()) {
    int prefixLength{edit.descriptor == DataEdit::ListDirectedRealPart ? 2
            : edit.descriptor == DataEdit::ListDirectedImaginaryPart   ? 0
                                                                       : 1};
    int suffixLength{edit.descriptor == DataEdit::ListDirectedRealPart ||
                edit.descriptor == DataEdit::ListDirectedImaginaryPart
            ? 1
            : 0};
    length += prefixLength + suffixLength;
    ConnectionState &connection{io_.GetConnectionState()};
    return (connection.positionInRecord == 0 ||
               length <= connection.RemainingSpaceInRecord() ||
               io_.AdvanceRecord()) &&
        io_.Emit(" (", prefixLength);
  } else if (width > length) {
    return io_.EmitRepeated(' ', width - length);
  } else {
    return true;
  }
}

bool RealOutputEditingBase::EmitSuffix(const DataEdit &edit) {
  if (edit.descriptor == DataEdit::ListDirectedRealPart) {
    return io_.Emit(edit.modes.editingFlags & decimalComma ? ";" : ",", 1);
  } else if (edit.descriptor == DataEdit::ListDirectedImaginaryPart) {
    return io_.Emit(")", 1);
  } else {
    return true;
  }
}

template <int binaryPrecision>
decimal::ConversionToDecimalResult RealOutputEditing<binaryPrecision>::Convert(
    int significantDigits, const DataEdit &edit, int flags) {
  if (edit.modes.editingFlags & signPlus) {
    flags |= decimal::AlwaysSign;
  }
  auto converted{decimal::ConvertToDecimal<binaryPrecision>(buffer_,
      sizeof buffer_, static_cast<enum decimal::DecimalConversionFlags>(flags),
      significantDigits, edit.modes.round, x_)};
  if (!converted.str) { // overflow
    io_.GetIoErrorHandler().Crash(
        "RealOutputEditing::Convert : buffer size %zd was insufficient",
        sizeof buffer_);
  }
  return converted;
}

// 13.7.2.3.3 in F'2018
template <int binaryPrecision>
bool RealOutputEditing<binaryPrecision>::EditEorDOutput(const DataEdit &edit) {
  int editDigits{edit.digits.value_or(0)}; // 'd' field
  int editWidth{edit.width.value_or(0)}; // 'w' field
  int significantDigits{editDigits};
  int flags{0};
  if (editWidth == 0) { // "the processor selects the field width"
    if (edit.digits.has_value()) { // E0.d
      editWidth = editDigits + 6; // -.666E+ee
    } else { // E0
      flags |= decimal::Minimize;
      significantDigits =
          sizeof buffer_ - 5; // sign, NUL, + 3 extra for EN scaling
    }
  }
  bool isEN{edit.variation == 'N'};
  bool isES{edit.variation == 'S'};
  int scale{isEN || isES ? 1 : edit.modes.scale}; // 'kP' value
  int zeroesAfterPoint{0};
  if (scale < 0) {
    zeroesAfterPoint = -scale;
    significantDigits = std::max(0, significantDigits - zeroesAfterPoint);
  } else if (scale > 0) {
    ++significantDigits;
    scale = std::min(scale, significantDigits + 1);
  }
  // In EN editing, multiple attempts may be necessary, so it's in a loop.
  while (true) {
    decimal::ConversionToDecimalResult converted{
        Convert(significantDigits, edit, flags)};
    if (IsInfOrNaN(converted)) {
      return EmitPrefix(edit, converted.length, editWidth) &&
          io_.Emit(converted.str, converted.length) && EmitSuffix(edit);
    }
    if (!IsZero()) {
      converted.decimalExponent -= scale;
    }
    if (isEN && scale < 3 && (converted.decimalExponent % 3) != 0) {
      // EN mode: boost the scale and significant digits, try again; need
      // an effective exponent field that's a multiple of three.
      ++scale;
      ++significantDigits;
      continue;
    }
    // Format the exponent (see table 13.1 for all the cases)
    int expoLength{0};
    const char *exponent{
        FormatExponent(converted.decimalExponent, edit, expoLength)};
    int signLength{*converted.str == '-' || *converted.str == '+' ? 1 : 0};
    int convertedDigits{static_cast<int>(converted.length) - signLength};
    int zeroesBeforePoint{std::max(0, scale - convertedDigits)};
    int digitsBeforePoint{std::max(0, scale - zeroesBeforePoint)};
    int digitsAfterPoint{convertedDigits - digitsBeforePoint};
    int trailingZeroes{flags & decimal::Minimize
            ? 0
            : std::max(0,
                  significantDigits - (convertedDigits + zeroesBeforePoint))};
    int totalLength{signLength + digitsBeforePoint + zeroesBeforePoint +
        1 /*'.'*/ + zeroesAfterPoint + digitsAfterPoint + trailingZeroes +
        expoLength};
    int width{editWidth > 0 ? editWidth : totalLength};
    if (totalLength > width) {
      return io_.EmitRepeated('*', width);
    }
    if (totalLength < width && digitsBeforePoint == 0 &&
        zeroesBeforePoint == 0) {
      zeroesBeforePoint = 1;
      ++totalLength;
    }
    return EmitPrefix(edit, totalLength, width) &&
        io_.Emit(converted.str, signLength + digitsBeforePoint) &&
        io_.EmitRepeated('0', zeroesBeforePoint) &&
        io_.Emit(edit.modes.editingFlags & decimalComma ? "," : ".", 1) &&
        io_.EmitRepeated('0', zeroesAfterPoint) &&
        io_.Emit(
            converted.str + signLength + digitsBeforePoint, digitsAfterPoint) &&
        io_.EmitRepeated('0', trailingZeroes) &&
        io_.Emit(exponent, expoLength) && EmitSuffix(edit);
  }
}

// 13.7.2.3.2 in F'2018
template <int binaryPrecision>
bool RealOutputEditing<binaryPrecision>::EditFOutput(const DataEdit &edit) {
  int fracDigits{edit.digits.value_or(0)}; // 'd' field
  const int editWidth{edit.width.value_or(0)}; // 'w' field
  int flags{0};
  if (editWidth == 0) { // "the processor selects the field width"
    if (!edit.digits.has_value()) { // F0
      flags |= decimal::Minimize;
      fracDigits = sizeof buffer_ - 2; // sign & NUL
    }
  }
  // Multiple conversions may be needed to get the right number of
  // effective rounded fractional digits.
  int extraDigits{0};
  while (true) {
    decimal::ConversionToDecimalResult converted{
        Convert(extraDigits + fracDigits, edit, flags)};
    if (IsInfOrNaN(converted)) {
      return EmitPrefix(edit, converted.length, editWidth) &&
          io_.Emit(converted.str, converted.length) && EmitSuffix(edit);
    }
    int scale{IsZero() ? 1 : edit.modes.scale}; // kP
    int expo{converted.decimalExponent + scale};
    if (expo > extraDigits && extraDigits >= 0) {
      extraDigits = expo;
      if (!edit.digits.has_value()) { // F0
        fracDigits = sizeof buffer_ - extraDigits - 2; // sign & NUL
      }
      continue;
    } else if (expo < extraDigits && extraDigits > -fracDigits) {
      extraDigits = std::max(expo, -fracDigits);
      continue;
    }
    int signLength{*converted.str == '-' || *converted.str == '+' ? 1 : 0};
    int convertedDigits{static_cast<int>(converted.length) - signLength};
    int digitsBeforePoint{std::max(0, std::min(expo, convertedDigits))};
    int zeroesBeforePoint{std::max(0, expo - digitsBeforePoint)};
    int zeroesAfterPoint{std::min(fracDigits, std::max(0, -expo))};
    int digitsAfterPoint{convertedDigits - digitsBeforePoint};
    int trailingZeroes{flags & decimal::Minimize
            ? 0
            : std::max(0, fracDigits - (zeroesAfterPoint + digitsAfterPoint))};
    if (digitsBeforePoint + zeroesBeforePoint + zeroesAfterPoint +
            digitsAfterPoint + trailingZeroes ==
        0) {
      zeroesBeforePoint = 1; // "." -> "0."
    }
    int totalLength{signLength + digitsBeforePoint + zeroesBeforePoint +
        1 /*'.'*/ + zeroesAfterPoint + digitsAfterPoint + trailingZeroes};
    int width{editWidth > 0 ? editWidth : totalLength};
    if (totalLength > width) {
      return io_.EmitRepeated('*', width);
    }
    if (totalLength < width && digitsBeforePoint + zeroesBeforePoint == 0) {
      zeroesBeforePoint = 1;
      ++totalLength;
    }
    return EmitPrefix(edit, totalLength, width) &&
        io_.Emit(converted.str, signLength + digitsBeforePoint) &&
        io_.EmitRepeated('0', zeroesBeforePoint) &&
        io_.Emit(edit.modes.editingFlags & decimalComma ? "," : ".", 1) &&
        io_.EmitRepeated('0', zeroesAfterPoint) &&
        io_.Emit(
            converted.str + signLength + digitsBeforePoint, digitsAfterPoint) &&
        io_.EmitRepeated('0', trailingZeroes) &&
        io_.EmitRepeated(' ', trailingBlanks_) && EmitSuffix(edit);
  }
}

// 13.7.5.2.3 in F'2018
template <int binaryPrecision>
DataEdit RealOutputEditing<binaryPrecision>::EditForGOutput(DataEdit edit) {
  edit.descriptor = 'E';
  int significantDigits{
      edit.digits.value_or(BinaryFloatingPoint::decimalPrecision)}; // 'd'
  if (!edit.width.has_value() || (*edit.width > 0 && significantDigits == 0)) {
    return edit; // Gw.0 -> Ew.0 for w > 0
  }
  decimal::ConversionToDecimalResult converted{
      Convert(significantDigits, edit)};
  if (IsInfOrNaN(converted)) {
    return edit;
  }
  int expo{IsZero() ? 1 : converted.decimalExponent}; // 's'
  if (expo < 0 || expo > significantDigits) {
    return edit; // Ew.d
  }
  edit.descriptor = 'F';
  edit.modes.scale = 0; // kP is ignored for G when no exponent field
  trailingBlanks_ = 0;
  int editWidth{edit.width.value_or(0)};
  if (editWidth > 0) {
    int expoDigits{edit.expoDigits.value_or(0)};
    trailingBlanks_ = expoDigits > 0 ? expoDigits + 2 : 4; // 'n'
    *edit.width = std::max(0, editWidth - trailingBlanks_);
  }
  if (edit.digits.has_value()) {
    *edit.digits = std::max(0, *edit.digits - expo);
  }
  return edit;
}

// 13.10.4 in F'2018
template <int binaryPrecision>
bool RealOutputEditing<binaryPrecision>::EditListDirectedOutput(
    const DataEdit &edit) {
  decimal::ConversionToDecimalResult converted{Convert(1, edit)};
  if (IsInfOrNaN(converted)) {
    return EditEorDOutput(edit);
  }
  int expo{converted.decimalExponent};
  if (expo < 0 || expo > BinaryFloatingPoint::decimalPrecision) {
    DataEdit copy{edit};
    copy.modes.scale = 1; // 1P
    return EditEorDOutput(copy);
  }
  return EditFOutput(edit);
}

// 13.7.5.2.6 in F'2018
template <int binaryPrecision>
bool RealOutputEditing<binaryPrecision>::EditEXOutput(const DataEdit &) {
  io_.GetIoErrorHandler().Crash(
      "EX output editing is not yet implemented"); // TODO
}

template <int binaryPrecision>
bool RealOutputEditing<binaryPrecision>::Edit(const DataEdit &edit) {
  switch (edit.descriptor) {
  case 'D':
    return EditEorDOutput(edit);
  case 'E':
    if (edit.variation == 'X') {
      return EditEXOutput(edit);
    } else {
      return EditEorDOutput(edit);
    }
  case 'F':
    return EditFOutput(edit);
  case 'B':
  case 'O':
  case 'Z':
    return EditIntegerOutput(io_, edit,
        decimal::BinaryFloatingPointNumber<binaryPrecision>{x_}.raw());
  case 'G':
    return Edit(EditForGOutput(edit));
  default:
    if (edit.IsListDirected()) {
      return EditListDirectedOutput(edit);
    }
    io_.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used with a REAL data item",
        edit.descriptor);
    return false;
  }
  return false;
}

bool ListDirectedLogicalOutput(IoStatementState &io,
    ListDirectedStatementState<Direction::Output> &list, bool truth) {
  return list.EmitLeadingSpaceOrAdvance(io, 1) && io.Emit(truth ? "T" : "F", 1);
}

bool EditLogicalOutput(IoStatementState &io, const DataEdit &edit, bool truth) {
  switch (edit.descriptor) {
  case 'L':
  case 'G':
    return io.EmitRepeated(' ', std::max(0, edit.width.value_or(1) - 1)) &&
        io.Emit(truth ? "T" : "F", 1);
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used with a LOGICAL data item",
        edit.descriptor);
    return false;
  }
}

bool ListDirectedDefaultCharacterOutput(IoStatementState &io,
    ListDirectedStatementState<Direction::Output> &list, const char *x,
    std::size_t length) {
  bool ok{list.EmitLeadingSpaceOrAdvance(io, length, true)};
  MutableModes &modes{io.mutableModes()};
  ConnectionState &connection{io.GetConnectionState()};
  if (modes.delim) {
    // Value is delimited with ' or " marks, and interior
    // instances of that character are doubled.  When split
    // over multiple lines, delimit each lines' part.
    ok &= io.Emit(&modes.delim, 1);
    for (std::size_t j{0}; j < length; ++j) {
      if (list.NeedAdvance(connection, 2)) {
        ok &= io.Emit(&modes.delim, 1) && io.AdvanceRecord() &&
            io.Emit(&modes.delim, 1);
      }
      if (x[j] == modes.delim) {
        ok &= io.EmitRepeated(modes.delim, 2);
      } else {
        ok &= io.Emit(&x[j], 1);
      }
    }
    ok &= io.Emit(&modes.delim, 1);
  } else {
    // Undelimited list-directed output
    std::size_t put{0};
    while (put < length) {
      auto chunk{std::min(length - put, connection.RemainingSpaceInRecord())};
      ok &= io.Emit(x + put, chunk);
      put += chunk;
      if (put < length) {
        ok &= io.AdvanceRecord() && io.Emit(" ", 1);
      }
    }
    list.lastWasUndelimitedCharacter = true;
  }
  return ok;
}

bool EditDefaultCharacterOutput(IoStatementState &io, const DataEdit &edit,
    const char *x, std::size_t length) {
  switch (edit.descriptor) {
  case 'A':
  case 'G':
    break;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used with a CHARACTER data item",
        edit.descriptor);
    return false;
  }
  int len{static_cast<int>(length)};
  int width{edit.width.value_or(len)};
  return io.EmitRepeated(' ', std::max(0, width - len)) &&
      io.Emit(x, std::min(width, len));
}

template bool EditIntegerOutput<std::int64_t, std::uint64_t>(
    IoStatementState &, const DataEdit &, std::int64_t);
template bool EditIntegerOutput<common::uint128_t, common::uint128_t>(
    IoStatementState &, const DataEdit &, common::uint128_t);

template class RealOutputEditing<2>;
template class RealOutputEditing<3>;
template class RealOutputEditing<4>;
template class RealOutputEditing<8>;
template class RealOutputEditing<10>;
// TODO: double/double
template class RealOutputEditing<16>;
} // namespace Fortran::runtime::io
