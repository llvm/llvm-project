//===-- runtime/numeric-output.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "numeric-output.h"
#include "flang/Common/unsigned-const-division.h"

namespace Fortran::runtime::io {

bool EditIntegerOutput(
    IoStatementState &io, const DataEdit &edit, std::int64_t n) {
  char buffer[66], *end = &buffer[sizeof buffer], *p = end;
  std::uint64_t un{static_cast<std::uint64_t>(n < 0 ? -n : n)};
  int signChars{0};
  switch (edit.descriptor) {
  case DataEdit::ListDirected:
  case 'G':
  case 'I':
    if (n < 0 || (edit.modes.editingFlags & signPlus)) {
      signChars = 1;  // '-' or '+'
    }
    while (un > 0) {
      auto quotient{common::DivideUnsignedBy<std::uint64_t, 10>(un)};
      *--p = '0' + un - 10 * quotient;
      un = quotient;
    }
    break;
  case 'B':
    for (; un > 0; un >>= 1) {
      *--p = '0' + (un & 1);
    }
    break;
  case 'O':
    for (; un > 0; un >>= 3) {
      *--p = '0' + (un & 7);
    }
    break;
  case 'Z':
    for (; un > 0; un >>= 4) {
      int digit = un & 0xf;
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
  if (edit.digits && digits <= *edit.digits) {  // Iw.m
    if (*edit.digits == 0 && n == 0) {
      // Iw.0 with zero value: output field must be blank.  For I0.0
      // and a zero value, emit one blank character.
      signChars = 0;  // in case of SP
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
    unsigned quotient{common::DivideUnsignedBy<unsigned, 10>(e)};
    *--exponent = '0' + e - 10 * quotient;
    e = quotient;
  }
  if (edit.expoDigits) {
    if (int ed{*edit.expoDigits}) {  // Ew.dEe with e > 0
      while (exponent > exponent_ + 2 /*E+*/ && exponent + ed > eEnd) {
        *--exponent = '0';
      }
    } else if (exponent == eEnd) {
      *--exponent = '0';  // Ew.dE0 with zero-valued exponent
    }
  } else {  // ensure at least two exponent digits
    while (exponent + 2 > eEnd) {
      *--exponent = '0';
    }
  }
  *--exponent = expo < 0 ? '-' : '+';
  if (edit.expoDigits || exponent + 3 == eEnd) {
    *--exponent = edit.descriptor == 'D' ? 'D' : 'E';  // not 'G'
  }
  length = eEnd - exponent;
  return exponent;
}

bool RealOutputEditingBase::EmitPrefix(
    const DataEdit &edit, std::size_t length, std::size_t width) {
  if (edit.IsListDirected()) {
    int prefixLength{edit.descriptor == DataEdit::ListDirectedRealPart
            ? 2
            : edit.descriptor == DataEdit::ListDirectedImaginaryPart ? 0 : 1};
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

}
