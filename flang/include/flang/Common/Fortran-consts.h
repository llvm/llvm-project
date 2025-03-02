//===-- include/flang/Common/Fortran-consts.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_FORTRAN_CONSTS_H_
#define FORTRAN_COMMON_FORTRAN_CONSTS_H_

#include "enum-set.h"
#include <cstdint>

namespace Fortran::common {

// Fortran has five kinds of standard intrinsic data types, the Unsigned
// extension, and derived types.
ENUM_CLASS(
    TypeCategory, Integer, Unsigned, Real, Complex, Character, Logical, Derived)
ENUM_CLASS(VectorElementCategory, Integer, Unsigned, Real)

ENUM_CLASS(IoStmtKind, None, Backspace, Close, Endfile, Flush, Inquire, Open,
    Print, Read, Rewind, Wait, Write)

// Defined I/O variants
ENUM_CLASS(
    DefinedIo, ReadFormatted, ReadUnformatted, WriteFormatted, WriteUnformatted)

// Fortran arrays may have up to 15 dimensions (See Fortran 2018 section 5.4.6).
static constexpr int maxRank{15};

// Floating-point rounding modes; these are packed into a byte to save
// room in the runtime's format processing context structure.  These
// enumerators are defined with the corresponding values returned from
// llvm.get.rounding.
enum class RoundingMode : std::uint8_t {
  ToZero, // ROUND=ZERO, RZ - truncation
  TiesToEven, // ROUND=NEAREST, RN - default IEEE rounding
  Up, // ROUND=UP, RU
  Down, // ROUND=DOWN, RD
  TiesAwayFromZero, // ROUND=COMPATIBLE, RC - ties round away from zero
};

} // namespace Fortran::common
#endif /* FORTRAN_COMMON_FORTRAN_CONSTS_H_ */
