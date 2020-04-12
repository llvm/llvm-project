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
#include "descriptor.h"
#include "entry-names.h"
#include <cstddef>

namespace Fortran::runtime {
extern "C" {

// Appends the corresponding (or expanded) characters of 'operand'
// to the (elements of) a (re)allocation of 'temp', which must be an
// initialized CHARACTER allocatable scalar or array descriptor -- use
// AllocatableInitCharacter() to set one up.  Crashes when not
// conforming.  Assumes independence of data.
void RTNAME(CharacterConcatenate)(Descriptor &temp, const Descriptor &operand,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Convenience specialization for character scalars.
void RTNAME(CharacterConcatenateScalar)(
    Descriptor &temp, const char *, std::size_t byteLength);

// Assigns the value(s) of 'rhs' to 'lhs'.  Handles reallocation,
// truncation, or padding ss necessary.  Crashes when not conforming.
// Assumes independence of data.
// Call MoveAlloc() instead as an optimization when a temporary value is
// being assigned to a deferred-length allocatable.
void RTNAME(CharacterAssign)(Descriptor &lhs, const Descriptor &rhs,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Special-case support for optimized scalar CHARACTER concatenation
// expressions.

// Copies data from 'rhs' to the remaining space (lhsLength - offset)
// in 'lhs', if any.  Returns the new offset.  Assumes independence.
std::size_t RTNAME(CharacterAppend)(char *lhs, std::size_t lhsLength,
    std::size_t offset, const char *rhs, std::size_t rhsLength);

// Appends any necessary spaces to a CHARACTER(KIND=1) scalar.
void RTNAME(CharacterPad)(char *lhs, std::size_t length, std::size_t offset);
}
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_CHARACTER_H_
