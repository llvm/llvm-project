//===-- runtime/edit-output.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "edit-output.h"
#include "utf.h"
#include "flang/Common/uint128.h"
#include <algorithm>

namespace Fortran::runtime::io {

// B/O/Z output of arbitrarily sized data emits a binary/octal/hexadecimal
// representation of what is interpreted to be a single unsigned integer value.
// When used with character data, endianness is exposed.
template <int LOG2_BASE>
static bool EditBOZOutput(IoStatementState &io, const DataEdit &edit,
    const unsigned char *data0, std::size_t bytes) {
  int digits{static_cast<int>((bytes * 8) / LOG2_BASE)};
  int get{static_cast<int>(bytes * 8) - digits * LOG2_BASE};
  if (get > 0) {
    ++digits;
  } else {
    get = LOG2_BASE;
  }
  int shift{7};
  int increment{isHostLittleEndian ? -1 : 1};
  const unsigned char *data{data0 + (isHostLittleEndian ? bytes - 1 : 0)};
  int skippedZeroes{0};
  int digit{0};
  // The same algorithm is used to generate digits for real (below)
  // as well as for generating them only to skip leading zeroes (here).
  // Bits are copied one at a time from the source data.
  // TODO: Multiple bit copies for hexadecimal, where misalignment
  // is not possible; or for octal when all 3 bits come from the
  // same byte.
  while (bytes > 0) {
    if (get == 0) {
      if (digit != 0) {
        break; // first nonzero leading digit
      }
      ++skippedZeroes;
      get = LOG2_BASE;
    } else if (shift < 0) {
      data += increment;
      --bytes;
      shift = 7;
    } else {
      digit = 2 * digit + ((*data >> shift--) & 1);
      --get;
    }
  }
  // Emit leading spaces and zeroes; detect field overflow
  int leadingZeroes{0};
  int editWidth{edit.width.value_or(0)};
  int significant{digits - skippedZeroes};
  if (edit.digits && significant <= *edit.digits) { // Bw.m, Ow.m, Zw.m
    if (*edit.digits == 0 && bytes == 0) {
      editWidth = std::max(1, editWidth);
    } else {
      leadingZeroes = *edit.digits - significant;
    }
  } else if (bytes == 0) {
    leadingZeroes = 1;
  }
  int subTotal{leadingZeroes + significant};
  int leadingSpaces{std::max(0, editWidth - subTotal)};
  if (editWidth > 0 && leadingSpaces + subTotal > editWidth) {
    return io.EmitRepeated('*', editWidth);
  }
  if (!(io.EmitRepeated(' ', leadingSpaces) &&
          io.EmitRepeated('0', leadingZeroes))) {
    return false;
  }
  // Emit remaining digits
  while (bytes > 0) {
    if (get == 0) {
      char ch{static_cast<char>(digit >= 10 ? 'A' + digit - 10 : '0' + digit)};
      if (!io.Emit(&ch, 1)) {
        return false;
      }
      get = LOG2_BASE;
      digit = 0;
    } else if (shift < 0) {
      data += increment;
      --bytes;
      shift = 7;
    } else {
      digit = 2 * digit + ((*data >> shift--) & 1);
      --get;
    }
  }
  return true;
}

template <int KIND>
bool EditIntegerOutput(IoStatementState &io, const DataEdit &edit,
    common::HostSignedIntType<8 * KIND> n) {
  char buffer[130], *end{&buffer[sizeof buffer]}, *p{end};
  bool isNegative{n < 0};
  using Unsigned = common::HostUnsignedIntType<8 * KIND>;
  Unsigned un{static_cast<Unsigned>(n)};
  int signChars{0};
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
  case 'G':
  case 'I':
    if (isNegative) {
      un = -un;
    }
    if (isNegative || (edit.modes.editingFlags & signPlus)) {
      signChars = 1; // '-' or '+'
    }
    while (un > 0) {
      auto quotient{un / 10u};
      *--p = '0' + static_cast<int>(un - Unsigned{10} * quotient);
      un = quotient;
    }
    break;
  case 'B':
    return EditBOZOutput<1>(
        io, edit, reinterpret_cast<const unsigned char *>(&n), KIND);
  case 'O':
    return EditBOZOutput<3>(
        io, edit, reinterpret_cast<const unsigned char *>(&n), KIND);
  case 'Z':
    return EditBOZOutput<4>(
        io, edit, reinterpret_cast<const unsigned char *>(&n), KIND);
  case 'A': // legacy extension
    return EditCharacterOutput(
        io, edit, reinterpret_cast<char *>(&n), sizeof n);
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
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
  int subTotal{signChars + leadingZeroes + digits};
  int leadingSpaces{std::max(0, editWidth - subTotal)};
  if (editWidth > 0 && leadingSpaces + subTotal > editWidth) {
    return io.EmitRepeated('*', editWidth);
  }
  if (edit.IsListDirected()) {
    int total{std::max(leadingSpaces, 1) + subTotal};
    if (io.GetConnectionState().NeedAdvance(static_cast<std::size_t>(total)) &&
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
    return (!connection.NeedAdvance(length) || io_.AdvanceRecord()) &&
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
    int significantDigits, enum decimal::FortranRounding rounding, int flags) {
  auto converted{decimal::ConvertToDecimal<binaryPrecision>(buffer_,
      sizeof buffer_, static_cast<enum decimal::DecimalConversionFlags>(flags),
      significantDigits, rounding, x_)};
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
  if (edit.modes.editingFlags & signPlus) {
    flags |= decimal::AlwaysSign;
  }
  if (editWidth == 0) { // "the processor selects the field width"
    if (edit.digits.has_value()) { // E0.d
      if (editDigits == 0) { // E0.0
        editWidth = 7; // -.0E+ee
      } else {
        editWidth = editDigits + 6; // -.666E+ee
      }
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
    if (scale <= -editDigits) {
      io_.GetIoErrorHandler().SignalError(IostatBadScaleFactor,
          "Scale factor (kP) %d cannot be less than -d (%d)", scale,
          -editDigits);
      return false;
    }
    zeroesAfterPoint = -scale;
    significantDigits = std::max(0, significantDigits - zeroesAfterPoint);
  } else if (scale > 0) {
    if (scale >= editDigits + 2) {
      io_.GetIoErrorHandler().SignalError(IostatBadScaleFactor,
          "Scale factor (kP) %d cannot be greater than d+2 (%d)", scale,
          editDigits + 2);
      return false;
    }
    ++significantDigits;
    scale = std::min(scale, significantDigits + 1);
  }
  // In EN editing, multiple attempts may be necessary, so it's in a loop.
  while (true) {
    decimal::ConversionToDecimalResult converted{
        Convert(significantDigits, edit.modes.round, flags)};
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
  enum decimal::FortranRounding rounding{edit.modes.round};
  int flags{0};
  if (edit.modes.editingFlags & signPlus) {
    flags |= decimal::AlwaysSign;
  }
  if (editWidth == 0) { // "the processor selects the field width"
    if (!edit.digits.has_value()) { // F0
      flags |= decimal::Minimize;
      fracDigits = sizeof buffer_ - 2; // sign & NUL
    }
  }
  // Multiple conversions may be needed to get the right number of
  // effective rounded fractional digits.
  int extraDigits{0};
  bool canIncrease{true};
  while (true) {
    decimal::ConversionToDecimalResult converted{
        Convert(extraDigits + fracDigits, rounding, flags)};
    if (IsInfOrNaN(converted)) {
      return EmitPrefix(edit, converted.length, editWidth) &&
          io_.Emit(converted.str, converted.length) && EmitSuffix(edit);
    }
    int expo{converted.decimalExponent + edit.modes.scale /*kP*/};
    int signLength{*converted.str == '-' || *converted.str == '+' ? 1 : 0};
    int convertedDigits{static_cast<int>(converted.length) - signLength};
    if (IsZero()) { // don't treat converted "0" as significant digit
      expo = 0;
      convertedDigits = 0;
    }
    int trailingOnes{0};
    if (expo > extraDigits && extraDigits >= 0 && canIncrease) {
      extraDigits = expo;
      if (!edit.digits.has_value()) { // F0
        fracDigits = sizeof buffer_ - extraDigits - 2; // sign & NUL
      }
      canIncrease = false; // only once
      continue;
    } else if (expo == -fracDigits && convertedDigits > 0) {
      if (rounding != decimal::FortranRounding::RoundToZero) {
        // Convert again without rounding so that we can round here
        rounding = decimal::FortranRounding::RoundToZero;
        continue;
      } else if (converted.str[signLength] >= '5') {
        // Value rounds up to a scaled 1 (e.g., 0.06 for F5.1 -> 0.1)
        ++expo;
        convertedDigits = 0;
        trailingOnes = 1;
      } else {
        // Value rounds down to zero
        expo = 0;
        convertedDigits = 0;
      }
    } else if (expo < extraDigits && extraDigits > -fracDigits) {
      extraDigits = std::max(expo, -fracDigits);
      continue;
    }
    int digitsBeforePoint{std::max(0, std::min(expo, convertedDigits))};
    int zeroesBeforePoint{std::max(0, expo - digitsBeforePoint)};
    int zeroesAfterPoint{std::min(fracDigits, std::max(0, -expo))};
    int digitsAfterPoint{convertedDigits - digitsBeforePoint};
    int trailingZeroes{flags & decimal::Minimize
            ? 0
            : std::max(0,
                  fracDigits -
                      (zeroesAfterPoint + digitsAfterPoint + trailingOnes))};
    if (digitsBeforePoint + zeroesBeforePoint + zeroesAfterPoint +
            digitsAfterPoint + trailingOnes + trailingZeroes ==
        0) {
      zeroesBeforePoint = 1; // "." -> "0."
    }
    int totalLength{signLength + digitsBeforePoint + zeroesBeforePoint +
        1 /*'.'*/ + zeroesAfterPoint + digitsAfterPoint + trailingOnes +
        trailingZeroes};
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
        io_.EmitRepeated('1', trailingOnes) &&
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
  int flags{0};
  if (edit.modes.editingFlags & signPlus) {
    flags |= decimal::AlwaysSign;
  }
  decimal::ConversionToDecimalResult converted{
      Convert(significantDigits, edit.modes.round, flags)};
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
  decimal::ConversionToDecimalResult converted{Convert(1, edit.modes.round)};
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
      "not yet implemented: EX output editing"); // TODO
}

template <int KIND> bool RealOutputEditing<KIND>::Edit(const DataEdit &edit) {
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
    return EditBOZOutput<1>(io_, edit,
        reinterpret_cast<const unsigned char *>(&x_),
        common::BitsForBinaryPrecision(common::PrecisionOfRealKind(KIND)) >> 3);
  case 'O':
    return EditBOZOutput<3>(io_, edit,
        reinterpret_cast<const unsigned char *>(&x_),
        common::BitsForBinaryPrecision(common::PrecisionOfRealKind(KIND)) >> 3);
  case 'Z':
    return EditBOZOutput<4>(io_, edit,
        reinterpret_cast<const unsigned char *>(&x_),
        common::BitsForBinaryPrecision(common::PrecisionOfRealKind(KIND)) >> 3);
  case 'G':
    return Edit(EditForGOutput(edit));
  case 'A': // legacy extension
    return EditCharacterOutput(
        io_, edit, reinterpret_cast<char *>(&x_), sizeof x_);
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
  return list.EmitLeadingSpaceOrAdvance(io) && io.Emit(truth ? "T" : "F", 1);
}

bool EditLogicalOutput(IoStatementState &io, const DataEdit &edit, bool truth) {
  switch (edit.descriptor) {
  case 'L':
  case 'G':
    return io.EmitRepeated(' ', std::max(0, edit.width.value_or(1) - 1)) &&
        io.Emit(truth ? "T" : "F", 1);
  case 'B':
    return EditBOZOutput<1>(io, edit,
        reinterpret_cast<const unsigned char *>(&truth), sizeof truth);
  case 'O':
    return EditBOZOutput<3>(io, edit,
        reinterpret_cast<const unsigned char *>(&truth), sizeof truth);
  case 'Z':
    return EditBOZOutput<4>(io, edit,
        reinterpret_cast<const unsigned char *>(&truth), sizeof truth);
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used with a LOGICAL data item",
        edit.descriptor);
    return false;
  }
}

template <typename CHAR>
bool ListDirectedCharacterOutput(IoStatementState &io,
    ListDirectedStatementState<Direction::Output> &list, const CHAR *x,
    std::size_t length) {
  bool ok{true};
  MutableModes &modes{io.mutableModes()};
  ConnectionState &connection{io.GetConnectionState()};
  if (modes.delim) {
    ok = ok && list.EmitLeadingSpaceOrAdvance(io);
    // Value is delimited with ' or " marks, and interior
    // instances of that character are doubled.
    auto EmitOne{[&](CHAR ch) {
      if (connection.NeedAdvance(1)) {
        ok = ok && io.AdvanceRecord();
      }
      ok = ok && io.EmitEncoded(&ch, 1);
    }};
    EmitOne(modes.delim);
    for (std::size_t j{0}; j < length; ++j) {
      // Doubled delimiters must be put on the same record
      // in order to be acceptable as list-directed or NAMELIST
      // input; however, this requirement is not always possible
      // when the records have a fixed length, as is the case with
      // internal output.  The standard is silent on what should
      // happen, and no two extant Fortran implementations do
      // the same thing when tested with this case.
      // This runtime splits the doubled delimiters across
      // two records for lack of a better alternative.
      if (x[j] == static_cast<CHAR>(modes.delim)) {
        EmitOne(x[j]);
      }
      EmitOne(x[j]);
    }
    EmitOne(modes.delim);
  } else {
    // Undelimited list-directed output
    ok = ok && list.EmitLeadingSpaceOrAdvance(io, length > 0 ? 1 : 0, true);
    std::size_t put{0};
    std::size_t oneIfUTF8{connection.useUTF8<CHAR>() ? 1 : length};
    while (ok && put < length) {
      if (std::size_t chunk{std::min<std::size_t>(
              std::min<std::size_t>(length - put, oneIfUTF8),
              connection.RemainingSpaceInRecord())}) {
        ok = io.EmitEncoded(x + put, chunk);
        put += chunk;
      } else {
        ok = io.AdvanceRecord() && io.Emit(" ", 1);
      }
    }
    list.set_lastWasUndelimitedCharacter(true);
  }
  return ok;
}

template <typename CHAR>
bool EditCharacterOutput(IoStatementState &io, const DataEdit &edit,
    const CHAR *x, std::size_t length) {
  int len{static_cast<int>(length)};
  int width{edit.width.value_or(len)};
  switch (edit.descriptor) {
  case 'A':
    break;
  case 'G':
    if (width == 0) {
      width = len;
    }
    break;
  case 'B':
    return EditBOZOutput<1>(io, edit,
        reinterpret_cast<const unsigned char *>(x), sizeof(CHAR) * length);
  case 'O':
    return EditBOZOutput<3>(io, edit,
        reinterpret_cast<const unsigned char *>(x), sizeof(CHAR) * length);
  case 'Z':
    return EditBOZOutput<4>(io, edit,
        reinterpret_cast<const unsigned char *>(x), sizeof(CHAR) * length);
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInFormat,
        "Data edit descriptor '%c' may not be used with a CHARACTER data item",
        edit.descriptor);
    return false;
  }
  return io.EmitRepeated(' ', std::max(0, width - len)) &&
      io.EmitEncoded(x, std::min(width, len));
}

template bool EditIntegerOutput<1>(
    IoStatementState &, const DataEdit &, std::int8_t);
template bool EditIntegerOutput<2>(
    IoStatementState &, const DataEdit &, std::int16_t);
template bool EditIntegerOutput<4>(
    IoStatementState &, const DataEdit &, std::int32_t);
template bool EditIntegerOutput<8>(
    IoStatementState &, const DataEdit &, std::int64_t);
template bool EditIntegerOutput<16>(
    IoStatementState &, const DataEdit &, common::int128_t);

template class RealOutputEditing<2>;
template class RealOutputEditing<3>;
template class RealOutputEditing<4>;
template class RealOutputEditing<8>;
template class RealOutputEditing<10>;
// TODO: double/double
template class RealOutputEditing<16>;

template bool ListDirectedCharacterOutput(IoStatementState &,
    ListDirectedStatementState<Direction::Output> &, const char *,
    std::size_t chars);
template bool ListDirectedCharacterOutput(IoStatementState &,
    ListDirectedStatementState<Direction::Output> &, const char16_t *,
    std::size_t chars);
template bool ListDirectedCharacterOutput(IoStatementState &,
    ListDirectedStatementState<Direction::Output> &, const char32_t *,
    std::size_t chars);

template bool EditCharacterOutput(
    IoStatementState &, const DataEdit &, const char *, std::size_t chars);
template bool EditCharacterOutput(
    IoStatementState &, const DataEdit &, const char16_t *, std::size_t chars);
template bool EditCharacterOutput(
    IoStatementState &, const DataEdit &, const char32_t *, std::size_t chars);

} // namespace Fortran::runtime::io
