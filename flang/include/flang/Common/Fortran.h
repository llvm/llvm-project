//===-- include/flang/Common/Fortran.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_FORTRAN_H_
#define FORTRAN_COMMON_FORTRAN_H_

// Fortran language concepts that are used in many phases are defined
// once here to avoid redundancy and needless translation.

#include "idioms.h"
#include <cinttypes>
#include <vector>

namespace Fortran::common {

// Fortran has five kinds of intrinsic data types, plus the derived types.
ENUM_CLASS(TypeCategory, Integer, Real, Complex, Character, Logical, Derived)

constexpr bool IsNumericTypeCategory(TypeCategory category) {
  return category == TypeCategory::Integer || category == TypeCategory::Real ||
      category == TypeCategory::Complex;
}

// Kinds of IMPORT statements. Default means IMPORT or IMPORT :: names.
ENUM_CLASS(ImportKind, Default, Only, None, All)

// The attribute on a type parameter can be KIND or LEN.
ENUM_CLASS(TypeParamAttr, Kind, Len)

ENUM_CLASS(NumericOperator, Power, Multiply, Divide, Add, Subtract)
const char *AsFortran(NumericOperator);

ENUM_CLASS(LogicalOperator, And, Or, Eqv, Neqv, Not)
const char *AsFortran(LogicalOperator);

ENUM_CLASS(RelationalOperator, LT, LE, EQ, NE, GE, GT)
const char *AsFortran(RelationalOperator);

ENUM_CLASS(Intent, Default, In, Out, InOut)

ENUM_CLASS(IoStmtKind, None, Backspace, Close, Endfile, Flush, Inquire, Open,
    Print, Read, Rewind, Wait, Write)

// Union of specifiers for all I/O statements.
ENUM_CLASS(IoSpecKind, Access, Action, Advance, Asynchronous, Blank, Decimal,
    Delim, Direct, Encoding, End, Eor, Err, Exist, File, Fmt, Form, Formatted,
    Id, Iomsg, Iostat, Name, Named, Newunit, Nextrec, Nml, Number, Opened, Pad,
    Pending, Pos, Position, Read, Readwrite, Rec, Recl, Round, Sequential, Sign,
    Size, Status, Stream, Unformatted, Unit, Write,
    Carriagecontrol, // nonstandard
    Convert, // nonstandard
    Dispose, // nonstandard
)

// Floating-point rounding modes; these are packed into a byte to save
// room in the runtime's format processing context structure.
enum class RoundingMode : std::uint8_t {
  TiesToEven, // ROUND=NEAREST, RN - default IEEE rounding
  ToZero, // ROUND=ZERO, RZ - truncation
  Down, // ROUND=DOWN, RD
  Up, // ROUND=UP, RU
  TiesAwayFromZero, // ROUND=COMPATIBLE, RC - ties round away from zero
};

// Fortran label. Must be in [1..99999].
using Label = std::uint64_t;

// Fortran arrays may have up to 15 dimensions (See Fortran 2018 section 5.4.6).
static constexpr int maxRank{15};

// Fortran names may have up to 63 characters (See Fortran 2018 C601).
static constexpr int maxNameLen{63};
} // namespace Fortran::common
#endif // FORTRAN_COMMON_FORTRAN_H_
