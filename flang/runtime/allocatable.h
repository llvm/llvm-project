//===-- runtime/allocatable.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines APIs for Fortran runtime library support of code generated
// to manipulate and query allocatable variables, dummy arguments, & components.
#ifndef FORTRAN_RUNTIME_ALLOCATABLE_H_
#define FORTRAN_RUNTIME_ALLOCATABLE_H_

#include "descriptor.h"
#include "entry-names.h"

namespace Fortran::runtime {

extern "C" {

// Initializes the descriptor for an allocatable of intrinsic or derived type.
// The incoming descriptor is treated as (and can be) uninitialized garbage.
// Must be called for each allocatable variable as its scope comes into being.
// The storage for the allocatable's descriptor must have already been
// allocated to a size sufficient for the rank, corank, and type.
// A descriptor must be initialized before being used for any purpose,
// but needs reinitialization in a deallocated state only when there is
// a change of type, rank, or corank.
void RTNAME(AllocatableInitIntrinsic)(
    Descriptor &, TypeCategory, int kind, int rank = 0, int corank = 0);
void RTNAME(AllocatableInitCharacter)(Descriptor &, SubscriptValue length = 0,
    int kind = 1, int rank = 0, int corank = 0);
void RTNAME(AllocatableInitDerived)(
    Descriptor &, const typeInfo::DerivedType &, int rank = 0, int corank = 0);

// Checks that an allocatable is not already allocated in statements
// with STAT=.  Use this on a value descriptor before setting bounds or
// type parameters.  Not necessary on a freshly initialized descriptor.
// (If there's no STAT=, the error will be caught later anyway, but
// this API allows the error to be caught before descriptor is modified.)
// Return 0 on success (deallocated state), else the STAT= value.
int RTNAME(AllocatableCheckAllocated)(Descriptor &,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

// For MOLD= allocation; sets bounds, cobounds, and length type
// parameters from another descriptor.  The destination descriptor must
// be initialized and deallocated.
void RTNAME(AllocatableApplyMold)(Descriptor &, const Descriptor &mold);

// Explicitly sets the bounds and length type parameters of an initialized
// deallocated allocatable.
void RTNAME(AllocatableSetBounds)(
    Descriptor &, int zeroBasedDim, SubscriptValue lower, SubscriptValue upper);

// The upper cobound is ignored for the last codimension.
void RTNAME(AllocatableSetCoBounds)(Descriptor &, int zeroBasedCoDim,
    SubscriptValue lower, SubscriptValue upper = 0);

// Length type parameters are indexed in declaration order; i.e., 0 is the
// first length type parameter in the deepest base type.  (Not for use
// with CHARACTER; see above.)
void RTNAME(AllocatableSetDerivedLength)(
    Descriptor &, int which, SubscriptValue);

// When an explicit type-spec appears in an ALLOCATE statement for an
// allocatable with an explicit (non-deferred) length type paramater for
// a derived type or CHARACTER value, the explicit value has to match
// the length type parameter's value.  This API checks that requirement.
// Returns 0 for success, or the STAT= value on failure with hasStat==true.
int RTNAME(AllocatableCheckLengthParameter)(Descriptor &,
    int which /* 0 for CHARACTER length */, SubscriptValue other,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Allocates an allocatable.  The allocatable descriptor must have been
// initialized and its bounds and length type parameters set and must be
// in a deallocated state.
// On failure, if hasStat is true, returns a nonzero error code for
// STAT= and (if present) fills in errMsg; if hasStat is false, the
// image is terminated.  On success, leaves errMsg alone and returns zero.
// Successfully allocated memory is initialized if the allocatable has a
// derived type, and is always initialized by AllocatableAllocateSource().
// Performs all necessary coarray synchronization and validation actions.
int RTNAME(AllocatableAllocate)(Descriptor &, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);
int RTNAME(AllocatableAllocateSource)(Descriptor &, const Descriptor &source,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Implements the intrinsic subroutine MOVE_ALLOC (16.9.137 in F'2018,
// but note the order of first two arguments is reversed for consistency
// with the other APIs for allocatables.)  The destination descriptor
// must be initialized.
int RTNAME(MoveAlloc)(Descriptor &to, const Descriptor &from,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Deallocates an allocatable.  Finalizes elements &/or components as needed.
// The allocatable is left in an initialized state suitable for reallocation
// with the same bounds, cobounds, and length type parameters.
int RTNAME(AllocatableDeallocate)(Descriptor &, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

// Variant of above that does not finalize; for intermediate results
void RTNAME(AllocatableDeallocateNoFinal)(
    Descriptor &, const char *sourceFile = nullptr, int sourceLine = 0);
} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_ALLOCATABLE_H_
