//===-- include/flang/Runtime/pointer.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines APIs for Fortran runtime library support of code generated
// to manipulate and query data pointers.

#ifndef FORTRAN_RUNTIME_POINTER_H_
#define FORTRAN_RUNTIME_POINTER_H_

#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime {
extern "C" {

// Data pointer initialization for NULLIFY(), "p=>NULL()`, & for ALLOCATE().

// Initializes a pointer to a disassociated state for NULLIFY() or "p=>NULL()".
void RTNAME(PointerNullifyIntrinsic)(
    Descriptor &, TypeCategory, int kind, int rank = 0, int corank = 0);
void RTNAME(PointerNullifyCharacter)(Descriptor &, SubscriptValue length = 0,
    int kind = 1, int rank = 0, int corank = 0);
void RTNAME(PointerNullifyDerived)(
    Descriptor &, const typeInfo::DerivedType &, int rank = 0, int corank = 0);

// Explicitly sets the bounds of an initialized disassociated pointer.
// The upper cobound is ignored for the last codimension.
void RTNAME(PointerSetBounds)(
    Descriptor &, int zeroBasedDim, SubscriptValue lower, SubscriptValue upper);
void RTNAME(PointerSetCoBounds)(Descriptor &, int zeroBasedCoDim,
    SubscriptValue lower, SubscriptValue upper = 0);

// Length type parameters are indexed in declaration order; i.e., 0 is the
// first length type parameter in the deepest base type.  (Not for use
// with CHARACTER; see above.)
void RTNAME(PointerSetDerivedLength)(Descriptor &, int which, SubscriptValue);

// For MOLD= allocation: acquires information from another descriptor
// to initialize a null data pointer.
void RTNAME(PointerApplyMold)(Descriptor &, const Descriptor &mold);

// Data pointer association for "p=>TARGET"

// Associates a scalar pointer with a simple scalar target.
void RTNAME(PointerAssociateScalar)(Descriptor &, void *);

// Associates a pointer with a target of the same rank, possibly with new lower
// bounds, which are passed in a vector whose length must equal the rank.
void RTNAME(PointerAssociate)(Descriptor &, const Descriptor &target);
void RTNAME(PointerAssociateLowerBounds)(
    Descriptor &, const Descriptor &target, const Descriptor &lowerBounds);

// Associates a pointer with a target with bounds remapping.  The target must be
// simply contiguous &/or of rank 1.  The bounds constitute a [2,newRank]
// integer array whose columns are [lower bound, upper bound] on each dimension.
void RTNAME(PointerAssociateRemapping)(Descriptor &, const Descriptor &target,
    const Descriptor &bounds, const char *sourceFile = nullptr,
    int sourceLine = 0);

// Data pointer allocation and deallocation

// When an explicit type-spec appears in an ALLOCATE statement for an
// pointer with an explicit (non-deferred) length type paramater for
// a derived type or CHARACTER value, the explicit value has to match
// the length type parameter's value.  This API checks that requirement.
// Returns 0 for success, or the STAT= value on failure with hasStat==true.
int RTNAME(PointerCheckLengthParameter)(Descriptor &,
    int which /* 0 for CHARACTER length */, SubscriptValue other,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Allocates a data pointer.  Its descriptor must have been initialized
// and its bounds and length type parameters set.  It need not be disassociated.
// On failure, if hasStat is true, returns a nonzero error code for
// STAT= and (if present) fills in errMsg; if hasStat is false, the
// image is terminated.  On success, leaves errMsg alone and returns zero.
// Successfully allocated memory is initialized if the pointer has a
// derived type, and is always initialized by PointerAllocateSource().
// Performs all necessary coarray synchronization and validation actions.
int RTNAME(PointerAllocate)(Descriptor &, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);
int RTNAME(PointerAllocateSource)(Descriptor &, const Descriptor &source,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Deallocates a data pointer, which must have been allocated by
// PointerAllocate(), possibly copied with PointerAssociate().
// Finalizes elements &/or components as needed.  The pointer is left
// in an initialized disassociated state suitable for reallocation
// with the same bounds, cobounds, and length type parameters.
int RTNAME(PointerDeallocate)(Descriptor &, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

// Association inquiries for ASSOCIATED()

// True when the pointer is not disassociated.
bool RTNAME(PointerIsAssociated)(const Descriptor &);

// True when the pointer is associated with a specific target.
bool RTNAME(PointerIsAssociatedWith)(
    const Descriptor &, const Descriptor *target);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_POINTER_H_
