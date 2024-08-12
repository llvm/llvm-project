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
void RTDECL(PointerNullifyIntrinsic)(
    Descriptor &, TypeCategory, int kind, int rank = 0, int corank = 0);
void RTDECL(PointerNullifyCharacter)(Descriptor &, SubscriptValue length = 0,
    int kind = 1, int rank = 0, int corank = 0);
void RTDECL(PointerNullifyDerived)(
    Descriptor &, const typeInfo::DerivedType &, int rank = 0, int corank = 0);

// Explicitly sets the bounds of an initialized disassociated pointer.
// The upper cobound is ignored for the last codimension.
void RTDECL(PointerSetBounds)(
    Descriptor &, int zeroBasedDim, SubscriptValue lower, SubscriptValue upper);
void RTDECL(PointerSetCoBounds)(Descriptor &, int zeroBasedCoDim,
    SubscriptValue lower, SubscriptValue upper = 0);

// Length type parameters are indexed in declaration order; i.e., 0 is the
// first length type parameter in the deepest base type.  (Not for use
// with CHARACTER; see above.)
void RTDECL(PointerSetDerivedLength)(Descriptor &, int which, SubscriptValue);

// For MOLD= allocation: acquires information from another descriptor
// to initialize a null data pointer.
void RTDECL(PointerApplyMold)(
    Descriptor &, const Descriptor &mold, int rank = 0);

// Data pointer association for "p=>TARGET"

// Associates a scalar pointer with a simple scalar target.
void RTDECL(PointerAssociateScalar)(Descriptor &, void *);

// Associates a pointer with a target of the same rank, possibly with new lower
// bounds, which are passed in a vector whose length must equal the rank.
void RTDECL(PointerAssociate)(Descriptor &, const Descriptor &target);
void RTDECL(PointerAssociateLowerBounds)(
    Descriptor &, const Descriptor &target, const Descriptor &lowerBounds);

// Associates a pointer with a target with bounds remapping.  The target must be
// simply contiguous &/or of rank 1.  The bounds constitute a [2,newRank]
// integer array whose columns are [lower bound, upper bound] on each dimension.
void RTDECL(PointerAssociateRemapping)(Descriptor &, const Descriptor &target,
    const Descriptor &bounds, const char *sourceFile = nullptr,
    int sourceLine = 0);

// Data pointer allocation and deallocation

// When an explicit type-spec appears in an ALLOCATE statement for an
// pointer with an explicit (non-deferred) length type paramater for
// a derived type or CHARACTER value, the explicit value has to match
// the length type parameter's value.  This API checks that requirement.
// Returns 0 for success, or the STAT= value on failure with hasStat==true.
int RTDECL(PointerCheckLengthParameter)(Descriptor &,
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
int RTDECL(PointerAllocate)(Descriptor &, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);
int RTDECL(PointerAllocateSource)(Descriptor &, const Descriptor &source,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Deallocates a data pointer, which must have been allocated by
// PointerAllocate(), possibly copied with PointerAssociate().
// Finalizes elements &/or components as needed. The pointer is left
// in an initialized disassociated state suitable for reallocation
// with the same bounds, cobounds, and length type parameters.
int RTDECL(PointerDeallocate)(Descriptor &, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

// Same as PointerDeallocate but also set the dynamic type as the declared type
// as mentioned in 7.3.2.3 note 7.
int RTDECL(PointerDeallocatePolymorphic)(Descriptor &,
    const typeInfo::DerivedType *, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

// Association inquiries for ASSOCIATED()

// True when the pointer is not disassociated.
bool RTDECL(PointerIsAssociated)(const Descriptor &);

// True when the pointer is associated with a specific target.
bool RTDECL(PointerIsAssociatedWith)(
    const Descriptor &, const Descriptor *target);

// Fortran POINTERs are allocated with an extra validation word after their
// payloads in order to detect erroneous deallocations later.
RT_API_ATTRS void *AllocateValidatedPointerPayload(std::size_t);
RT_API_ATTRS bool ValidatePointerPayload(const ISO::CFI_cdesc_t &);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_POINTER_H_
