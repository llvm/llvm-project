//===-- include/flang/Runtime/array-constructor-consts.h --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_ARRAY_CONSTRUCTOR_CONSTS_H_
#define FORTRAN_RUNTIME_ARRAY_CONSTRUCTOR_CONSTS_H_

#include "flang/Runtime/descriptor-consts.h"
#include "flang/Runtime/entry-names.h"
#include <cstdint>

namespace Fortran::runtime {
struct ArrayConstructorVector;

// Max sizeof(ArrayConstructorVector) and sizeof(ArrayConstructorVector) for any
// target.
// TODO: Use target-specific size/alignment instead of overapproximation.
constexpr std::size_t MaxArrayConstructorVectorSizeInBytes = 2 * 40;
constexpr std::size_t MaxArrayConstructorVectorAlignInBytes = 8;

// This file defines an API to "push" an evaluated array constructor value
// "from" into some storage "to" of an array constructor. It can be seen as a
// form of std::vector::push_back() implementation for Fortran array
// constructors. In the APIs and ArrayConstructorVector struct above:
//
// - "to" is a ranked-1 descriptor whose declared type is already set to the
// array constructor derived type. It may be already allocated, even before the
// first call to this API, or it may be unallocated. "to" extent is increased
// every time a "from" is pushed past its current extent. At this end of the
// API calls, its extent is the extent of the array constructor. If "to" is
// unallocated and its extent is not null, it is assumed this is the final array
// constructor extent value, and the first allocation already "reserves" storage
// space accordingly to avoid reallocations.
//  - "from" is a scalar or array descriptor for the evaluated array
//  constructor value that must be copied into the storage of "to" at
//  "nextValuePosition".
//  - "useValueLengthParameters" must be set to true if the array constructor
//  has length parameters and no type spec. If it is true and "to" is
//  unallocated, "to" will take the length parameters of "from". If it is true
//  and "to" is an allocated character array constructor, it will be checked
//  that "from" length matches the one from "to". When it is false, the
//  character length must already be set in "to" before the first call to this
//  API and "from" character lengths are allowed to mismatch from "to".
// - "nextValuePosition" is the zero based sequence position of "from" in the
// array constructor. It is updated after this call by the number of "from"
// elements. It should be set to zero by the caller of this API before the first
// call.
// - "actualAllocationSize" is the current allocation size of "to" storage. It
// may be bigger than "to" extent for reallocation optimization purposes, but
// should never be smaller, unless this is the first call and "to" is
// unallocated. It is updated by the runtime after each successful allocation or
// reallocation. It should be set to "to" extent if "to" is allocated before the
// first call of this API, and can be left undefined otherwise.
//
// Note that this API can be used with "to" being a variable (that can be
// discontiguous). This can be done when the variable is the left hand side of
// an assignment from an array constructor as long as:
//  - none of the ac-value overlaps with the variable,
//  - this is an intrinsic assignment that is not a whole allocatable
//  assignment, *and* for a type that has no components requiring user defined
//  assignments,
//  - the variable is properly finalized before using this API if its need to,
//  - "useValueLengthParameters" should be set to false in this case, even if
//  the array constructor has no type-spec, since the variable may have a
//  different character length than the array constructor values.

extern "C" {
// API to initialize an ArrayConstructorVector before any values are pushed to
// it. Inlined code is only expected to allocate the "ArrayConstructorVector"
// class instance storage with sufficient size
// (MaxArrayConstructorVectorSizeInBytes is expected to be large enough for all
// supported targets). This avoids the need for the runtime to maintain a state,
// or to use dynamic allocation for it.
void RTDECL(InitArrayConstructorVector)(ArrayConstructorVector &vector,
    Descriptor &to, bool useValueLengthParameters,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Generic API to push any kind of entity into the array constructor (any
// Fortran type and any rank).
void RTDECL(PushArrayConstructorValue)(
    ArrayConstructorVector &vector, const Descriptor &from);

// API to push scalar array constructor value of:
//   - a numerical or logical type,
//   - or a derived type that has no length parameters, and no allocatable
//   component (that would require deep copies).
// It requires no descriptor for the value that is passed via its base address.
void RTDECL(PushArrayConstructorSimpleScalar)(
    ArrayConstructorVector &vector, void *from);
} // extern "C"
} // namespace Fortran::runtime

#endif /* FORTRAN_RUNTIME_ARRAY_CONSTRUCTOR_CONSTS_H_ */
