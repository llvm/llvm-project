//===-- runtime/numeric-output.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_NUMERIC_OUTPUT_H_
#define FORTRAN_RUNTIME_NUMERIC_OUTPUT_H_

// Output data editing templates implementing the FORMAT data editing
// descriptors E, EN, ES, EX, D, F, and G for REAL data (and COMPLEX
// components, I and G for INTEGER, and B/O/Z for both.
// See subclauses in 13.7.2.3 of Fortran 2018 for the
// detailed specifications of these descriptors.
// List-directed output (13.10.4) for numeric types is also done here.
// Drives the same fast binary-to-decimal formatting templates used
// in the f18 front-end.

#include "format.h"
#include "io-stmt.h"
#include "flang/Decimal/decimal.h"

namespace Fortran::runtime::io {

class IoStatementState;

// I, B, O, Z, and G output editing for INTEGER.
// edit is const here (and elsewhere in this header) so that one
// edit descriptor with a repeat factor may safely serve to edit
// multiple elements of an array.
bool EditIntegerOutput(IoStatementState &, const DataEdit &, std::int64_t);

// Encapsulates the state of a REAL output conversion.
class RealOutputEditingBase {
protected:
  explicit RealOutputEditingBase(IoStatementState &io) : io_{io} {}

  static bool IsDecimalNumber(const char *p) {
    if (!p) {
      return false;
    }
    if (*p == '-' || *p == '+') {
      ++p;
    }
    return *p >= '0' && *p <= '9';
  }

  const char *FormatExponent(int, const DataEdit &edit, int &length);
  bool EmitPrefix(const DataEdit &, std::size_t length, std::size_t width);
  bool EmitSuffix(const DataEdit &);

  IoStatementState &io_;
  int trailingBlanks_{0};  // created when Gw editing maps to Fw
  char exponent_[16];
};

template<int binaryPrecision = 53>
class RealOutputEditing : public RealOutputEditingBase {
public:
  template<typename A>
  RealOutputEditing(IoStatementState &io, A x)
    : RealOutputEditingBase{io}, x_{x} {}
  bool Edit(const DataEdit &);

private:
  using BinaryFloatingPoint =
      decimal::BinaryFloatingPointNumber<binaryPrecision>;

  // The DataEdit arguments here are const references or copies so that
  // the original DataEdit can safely serve multiple array elements when
  // it has a repeat count.
  bool EditEorDOutput(const DataEdit &);
  bool EditFOutput(const DataEdit &);
  DataEdit EditForGOutput(DataEdit);  // returns an E or F edit
  bool EditEXOutput(const DataEdit &);
  bool EditListDirectedOutput(const DataEdit &);

  bool IsZero() const { return x_.IsZero(); }

  decimal::ConversionToDecimalResult Convert(
      int significantDigits, const DataEdit &, int flags = 0);

  BinaryFloatingPoint x_;
  char buffer_[BinaryFloatingPoint::maxDecimalConversionDigits +
      EXTRA_DECIMAL_CONVERSION_SPACE];
};

template<int binaryPrecision>
decimal::ConversionToDecimalResult RealOutputEditing<binaryPrecision>::Convert(
    int significantDigits, const DataEdit &edit, int flags) {
  if (edit.modes.editingFlags & signPlus) {
    flags |= decimal::AlwaysSign;
  }
  auto converted{decimal::ConvertToDecimal<binaryPrecision>(buffer_,
      sizeof buffer_, static_cast<enum decimal::DecimalConversionFlags>(flags),
      significantDigits, edit.modes.round, x_)};
  if (!converted.str) {  // overflow
    io_.GetIoErrorHandler().Crash(
        "RealOutputEditing::Convert : buffer size %zd was insufficient",
        sizeof buffer_);
  }
  return converted;
}

// 13.7.2.3.3 in F'2018
template<int binaryPrecision>
bool RealOutputEditing<binaryPrecision>::EditEorDOutput(const DataEdit &edit) {
  int editDigits{edit.digits.value_or(0)};  // 'd' field
  int editWidth{edit.width.value_or(0)};  // 'w' field
  int significantDigits{editDigits};
  int flags{0};
  if (editWidth == 0) {  // "the processor selects the field width"
    if (edit.digits.has_value()) {  // E0.d
      editWidth = editDigits + 6;  // -.666E+ee
    } else {  // E0
      flags |= decimal::Minimize;
      significantDigits =
          sizeof buffer_ - 5;  // sign, NUL, + 3 extra for EN scaling
    }
  }
  bool isEN{edit.variation == 'N'};
  bool isES{edit.variation == 'S'};
  int scale{isEN || isES ? 1 : edit.modes.scale};  // 'kP' value
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
    if (converted.length > 0 && !IsDecimalNumber(converted.str)) {  // Inf, NaN
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
template<int binaryPrecision>
bool RealOutputEditing<binaryPrecision>::EditFOutput(const DataEdit &edit) {
  int fracDigits{edit.digits.value_or(0)};  // 'd' field
  int extraDigits{0};
  int editWidth{edit.width.value_or(0)};  // 'w' field
  int flags{0};
  if (editWidth == 0) {  // "the processor selects the field width"
    if (!edit.digits.has_value()) {  // F0
      flags |= decimal::Minimize;
      fracDigits = sizeof buffer_ - 2;  // sign & NUL
    }
  }
  // Multiple conversions may be needed to get the right number of
  // effective rounded fractional digits.
  while (true) {
    decimal::ConversionToDecimalResult converted{
        Convert(extraDigits + fracDigits, edit, flags)};
    if (converted.length > 0 && !IsDecimalNumber(converted.str)) {  // Inf, NaN
      return EmitPrefix(edit, converted.length, editWidth) &&
          io_.Emit(converted.str, converted.length) && EmitSuffix(edit);
    }
    int scale{IsZero() ? -1 : edit.modes.scale};
    int expo{converted.decimalExponent - scale};
    if (expo > extraDigits) {
      extraDigits = expo;
      if (flags & decimal::Minimize) {
        fracDigits = sizeof buffer_ - extraDigits - 2;  // sign & NUL
      }
      continue;  // try again
    }
    int signLength{*converted.str == '-' || *converted.str == '+' ? 1 : 0};
    int convertedDigits{static_cast<int>(converted.length) - signLength};
    int digitsBeforePoint{std::max(0, std::min(expo, convertedDigits))};
    int zeroesBeforePoint{std::max(0, expo - digitsBeforePoint)};
    int zeroesAfterPoint{std::max(0, -expo)};
    int digitsAfterPoint{convertedDigits - digitsBeforePoint};
    int trailingZeroes{flags & decimal::Minimize
            ? 0
            : std::max(0, fracDigits - (zeroesAfterPoint + digitsAfterPoint))};
    if (digitsBeforePoint + zeroesBeforePoint + zeroesAfterPoint +
            digitsAfterPoint + trailingZeroes ==
        0) {
      ++zeroesBeforePoint;  // "." -> "0."
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
template<int binaryPrecision>
DataEdit RealOutputEditing<binaryPrecision>::EditForGOutput(DataEdit edit) {
  edit.descriptor = 'E';
  if (!edit.width.has_value() ||
      (*edit.width > 0 && edit.digits.value_or(-1) == 0)) {
    return edit;  // Gw.0 -> Ew.0 for w > 0
  }
  decimal::ConversionToDecimalResult converted{Convert(1, edit)};
  if (!IsDecimalNumber(converted.str)) {  // Inf, NaN
    return edit;
  }
  int expo{IsZero() ? 1 : converted.decimalExponent};  // 's'
  int significantDigits{
      edit.digits.value_or(BinaryFloatingPoint::decimalPrecision)};  // 'd'
  if (expo < 0 || expo > significantDigits) {
    return edit;  // Ew.d
  }
  edit.descriptor = 'F';
  edit.modes.scale = 0;  // kP is ignored for G when no exponent field
  trailingBlanks_ = 0;
  int editWidth{edit.width.value_or(0)};
  if (editWidth > 0) {
    int expoDigits{edit.expoDigits.value_or(0)};
    trailingBlanks_ = expoDigits > 0 ? expoDigits + 2 : 4;  // 'n'
    *edit.width = std::max(0, editWidth - trailingBlanks_);
  }
  if (edit.digits.has_value()) {
    *edit.digits = std::max(0, *edit.digits - expo);
  }
  return edit;
}

// 13.10.4 in F'2018
template<int binaryPrecision>
bool RealOutputEditing<binaryPrecision>::EditListDirectedOutput(
    const DataEdit &edit) {
  decimal::ConversionToDecimalResult converted{Convert(1, edit)};
  if (!IsDecimalNumber(converted.str)) {  // Inf, NaN
    return EditEorDOutput(edit);
  }
  int expo{converted.decimalExponent};
  if (expo < 0 || expo > BinaryFloatingPoint::decimalPrecision) {
    DataEdit copy{edit};
    copy.modes.scale = 1;  // 1P
    return EditEorDOutput(copy);
  }
  return EditFOutput(edit);
}

// 13.7.5.2.6 in F'2018
template<int binaryPrecision>
bool RealOutputEditing<binaryPrecision>::EditEXOutput(const DataEdit &) {
  io_.GetIoErrorHandler().Crash(
      "EX output editing is not yet implemented");  // TODO
}

template<int binaryPrecision>
bool RealOutputEditing<binaryPrecision>::Edit(const DataEdit &edit) {
  switch (edit.descriptor) {
  case 'D': return EditEorDOutput(edit);
  case 'E':
    if (edit.variation == 'X') {
      return EditEXOutput(edit);
    } else {
      return EditEorDOutput(edit);
    }
  case 'F': return EditFOutput(edit);
  case 'B':
  case 'O':
  case 'Z':
    return EditIntegerOutput(
        io_, edit, decimal::BinaryFloatingPointNumber<binaryPrecision>{x_}.raw);
  case 'G': return Edit(EditForGOutput(edit));
  default:
    if (edit.IsListDirected()) {
      return EditListDirectedOutput(edit);
    }
    io_.GetIoErrorHandler().Crash(
        "Data edit descriptor '%c' may not be used with a REAL data item",
        edit.descriptor);
    return false;
  }
  return false;
}

}
#endif  // FORTRAN_RUNTIME_NUMERIC_OUTPUT_H_
