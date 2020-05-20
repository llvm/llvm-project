//===-- runtime/character.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines API between compiled code and the CHARACTER
// support functions in the runtime library.

#ifndef FORTRAN_RUNTIME_CHARACTER_H_
#define FORTRAN_RUNTIME_CHARACTER_H_
#include "entry-names.h"
#include <cstddef>

namespace Fortran::runtime {

class Descriptor;

extern "C" {

// Appends the corresponding (or expanded) characters of 'operand'
// to the (elements of) a (re)allocation of 'temp', which must be an
// initialized CHARACTER allocatable scalar or array descriptor -- use
// AllocatableInitCharacter() to set one up.  Crashes when not
// conforming.  Assumes independence of data.
void RTNAME(CharacterConcatenate)(Descriptor &temp, const Descriptor &operand,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Convenience specialization for ASCII scalars.
void RTNAME(CharacterConcatenateScalar1)(
    Descriptor &temp, const char *, std::size_t byteLength);

// Assigns the value(s) of 'rhs' to 'lhs'.  Handles reallocation,
// truncation, or padding ss necessary.  Crashes when not conforming.
// Assumes independence of data.
// Call MoveAlloc() instead as an optimization when a temporary value is
// being assigned to a deferred-length allocatable.
void RTNAME(CharacterAssign)(Descriptor &lhs, const Descriptor &rhs,
    const char *sourceFile = nullptr, int sourceLine = 0);

// CHARACTER comparisons.  The kinds must match.  Like std::memcmp(),
// the result is less than zero, zero, or greater than zero if the first
// argument is less than the second, equal to the second, or greater than
// the second, respectively.  The shorter argument is treated as if it were
// padded on the right with blanks.
// N.B.: Calls to the restricted specific intrinsic functions LGE, LGT, LLE,
// & LLT are converted into calls to these during lowering; they don't have
// to be able to be passed as actual procedure arguments.
int RTNAME(CharacterCompareScalar)(const Descriptor &, const Descriptor &);
int RTNAME(CharacterCompareScalar1)(
    const char *x, const char *y, std::size_t xBytes, std::size_t yBytes);
int RTNAME(CharacterCompareScalar2)(const char16_t *x, const char16_t *y,
    std::size_t xBytes, std::size_t yBytes);
int RTNAME(CharacterCompareScalar4)(const char32_t *x, const char32_t *y,
    std::size_t xBytes, std::size_t yBytes);

// General CHARACTER comparison; the result is a LOGICAL(KIND=1) array that
// is established and populated.
void RTNAME(CharacterCompare)(
    Descriptor &result, const Descriptor &, const Descriptor &);

// Special-case support for optimized ASCII scalar expressions.

// Copies data from 'rhs' to the remaining space (lhsLength - offset)
// in 'lhs', if any.  Returns the new offset.  Assumes independence.
std::size_t RTNAME(CharacterAppend1)(char *lhs, std::size_t lhsBytes,
    std::size_t offset, const char *rhs, std::size_t rhsBytes);

// Appends any necessary spaces to a CHARACTER(KIND=1) scalar.
void RTNAME(CharacterPad1)(char *lhs, std::size_t bytes, std::size_t offset);
}
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_CHARACTER_H_
