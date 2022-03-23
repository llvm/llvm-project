//===-- runtime/edit-output.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_EDIT_OUTPUT_H_
#define FORTRAN_RUNTIME_EDIT_OUTPUT_H_

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
#include "flang/Common/uint128.h"
#include "flang/Decimal/decimal.h"

namespace Fortran::runtime::io {

// I, B, O, Z, and G output editing for INTEGER.
// The DataEdit reference is const here (and elsewhere in this header) so that
// one edit descriptor with a repeat factor may safely serve to edit
// multiple elements of an array.
template <int KIND>
bool EditIntegerOutput(
    IoStatementState &, const DataEdit &, common::HostSignedIntType<8 * KIND>);

// Encapsulates the state of a REAL output conversion.
class RealOutputEditingBase {
protected:
  explicit RealOutputEditingBase(IoStatementState &io) : io_{io} {}

  static bool IsInfOrNaN(const decimal::ConversionToDecimalResult &res) {
    const char *p{res.str};
    if (!p || res.length < 1) {
      return false;
    }
    if (*p == '-' || *p == '+') {
      if (res.length == 1) {
        return false;
      }
      ++p;
    }
    return *p < '0' || *p > '9';
  }

  const char *FormatExponent(int, const DataEdit &edit, int &length);
  bool EmitPrefix(const DataEdit &, std::size_t length, std::size_t width);
  bool EmitSuffix(const DataEdit &);

  IoStatementState &io_;
  int trailingBlanks_{0}; // created when Gw editing maps to Fw
  char exponent_[16];
};

template <int KIND> class RealOutputEditing : public RealOutputEditingBase {
public:
  static constexpr int binaryPrecision{common::PrecisionOfRealKind(KIND)};
  using BinaryFloatingPoint =
      decimal::BinaryFloatingPointNumber<binaryPrecision>;
  template <typename A>
  RealOutputEditing(IoStatementState &io, A x)
      : RealOutputEditingBase{io}, x_{x} {}
  bool Edit(const DataEdit &);

private:
  // The DataEdit arguments here are const references or copies so that
  // the original DataEdit can safely serve multiple array elements when
  // it has a repeat count.
  bool EditEorDOutput(const DataEdit &);
  bool EditFOutput(const DataEdit &);
  DataEdit EditForGOutput(DataEdit); // returns an E or F edit
  bool EditEXOutput(const DataEdit &);
  bool EditListDirectedOutput(const DataEdit &);

  bool IsZero() const { return x_.IsZero(); }

  decimal::ConversionToDecimalResult Convert(
      int significantDigits, enum decimal::FortranRounding, int flags = 0);

  BinaryFloatingPoint x_;
  char buffer_[BinaryFloatingPoint::maxDecimalConversionDigits +
      EXTRA_DECIMAL_CONVERSION_SPACE];
};

bool ListDirectedLogicalOutput(
    IoStatementState &, ListDirectedStatementState<Direction::Output> &, bool);
bool EditLogicalOutput(IoStatementState &, const DataEdit &, bool);

template <typename CHAR>
bool ListDirectedCharacterOutput(IoStatementState &,
    ListDirectedStatementState<Direction::Output> &, const CHAR *,
    std::size_t chars);
extern template bool ListDirectedCharacterOutput(IoStatementState &,
    ListDirectedStatementState<Direction::Output> &, const char *,
    std::size_t chars);
extern template bool ListDirectedCharacterOutput(IoStatementState &,
    ListDirectedStatementState<Direction::Output> &, const char16_t *,
    std::size_t chars);
extern template bool ListDirectedCharacterOutput(IoStatementState &,
    ListDirectedStatementState<Direction::Output> &, const char32_t *,
    std::size_t chars);

template <typename CHAR>
bool EditCharacterOutput(
    IoStatementState &, const DataEdit &, const CHAR *, std::size_t chars);
extern template bool EditCharacterOutput(
    IoStatementState &, const DataEdit &, const char *, std::size_t chars);
extern template bool EditCharacterOutput(
    IoStatementState &, const DataEdit &, const char16_t *, std::size_t chars);
extern template bool EditCharacterOutput(
    IoStatementState &, const DataEdit &, const char32_t *, std::size_t chars);

extern template bool EditIntegerOutput<1>(
    IoStatementState &, const DataEdit &, std::int8_t);
extern template bool EditIntegerOutput<2>(
    IoStatementState &, const DataEdit &, std::int16_t);
extern template bool EditIntegerOutput<4>(
    IoStatementState &, const DataEdit &, std::int32_t);
extern template bool EditIntegerOutput<8>(
    IoStatementState &, const DataEdit &, std::int64_t);
extern template bool EditIntegerOutput<16>(
    IoStatementState &, const DataEdit &, common::int128_t);

extern template class RealOutputEditing<2>;
extern template class RealOutputEditing<3>;
extern template class RealOutputEditing<4>;
extern template class RealOutputEditing<8>;
extern template class RealOutputEditing<10>;
// TODO: double/double
extern template class RealOutputEditing<16>;

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_EDIT_OUTPUT_H_
